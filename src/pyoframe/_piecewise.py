"""Piecewise linear function support for Pyoframe."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Sequence

import polars as pl

from pyoframe._constants import PyoframeError

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe._core import Expression, Variable


def piecewise_linear(
    x: Variable,
    breakpoints: Sequence[float],
    values: Sequence[float],
    *,
    name: str,
    method: Literal["sos2", "incremental", "auto"] = "sos2",
) -> Expression:
    """Create a piecewise linear function ``y = f(x)`` defined by breakpoints and values.

    The function is defined by linear interpolation between the given breakpoints.
    This adds auxiliary variables and constraints to the model that ``x`` belongs to.

    Parameters:
        x:
            The input variable. Must already be added to a model.
        breakpoints:
            Strictly increasing x-coordinates of the breakpoints.
        values:
            The y-coordinates at each breakpoint (same length as ``breakpoints``).
        name:
            A prefix used for naming the auxiliary variables and constraints added to the model.
        method:
            The formulation to use:

            - ``"sos2"``: Convex combination with SOS2 constraints. Works for any breakpoints.
            - ``"incremental"``: Delta formulation. Pure LP (no SOS2), requires strictly increasing breakpoints.
            - ``"auto"``: Picks ``"incremental"`` if breakpoints are strictly increasing, ``"sos2"`` otherwise.

    Returns:
        An expression representing ``y = f(x)`` with the same dimensions as ``x``.
    """
    # Validation
    if len(breakpoints) != len(values):
        raise PyoframeError(
            f"Length of breakpoints ({len(breakpoints)}) must equal length of values ({len(values)})."
        )
    if len(breakpoints) < 2:
        raise PyoframeError("At least 2 breakpoints are required.")

    breakpoints = list(breakpoints)
    values = list(values)

    if not x._has_ids:
        raise PyoframeError(
            "Variable must be added to a model before using piecewise_linear(). "
            "Assign the variable to a model first (e.g. m.x = pf.Variable(...))."
        )

    model = x._model
    assert model is not None

    is_strictly_increasing = all(
        breakpoints[i] < breakpoints[i + 1] for i in range(len(breakpoints) - 1)
    )

    if method == "auto":
        method = "incremental" if is_strictly_increasing else "sos2"

    if method == "incremental":
        if not is_strictly_increasing:
            raise PyoframeError(
                "Incremental method requires strictly increasing breakpoints."
            )
        return _incremental_formulation(x, breakpoints, values, name, model)
    elif method == "sos2":
        return _sos2_formulation(x, breakpoints, values, name, model)
    else:
        raise PyoframeError(f"Unknown method: '{method}'. Use 'sos2', 'incremental', or 'auto'.")


def _sos2_formulation(
    x: Variable,
    breakpoints: list[float],
    values: list[float],
    name: str,
    model,
) -> Expression:
    """SOS2 convex combination formulation."""
    import pyoframe as pf
    from pyoframe._sos import SOS2 as _SOS2

    n_bp = len(breakpoints)
    x_dims = x._dimensions_unsafe
    bp_dim = f"__{name}_bp"

    # Create breakpoint set
    bp_set = pf.Set(**{bp_dim: list(range(n_bp))})

    # Lambda variables: indexed by [*x_dims, bp_dim], each in [0, 1]
    if x_dims:
        lam_sets = [x.data.select(x_dims), bp_set]
    else:
        lam_sets = [bp_set]

    lam_name = f"__{name}_lambda"
    lam = pf.Variable(*lam_sets, lb=0, ub=1)
    setattr(model, lam_name, lam)

    # Convexity constraint: sum of lambdas == 1 (over bp_dim)
    convexity_name = f"__{name}_convexity"
    convexity = lam.sum(bp_dim) == 1
    setattr(model, convexity_name, convexity)

    # Breakpoint parameter: indexed by bp_dim
    bp_param = pf.Param(pl.DataFrame({bp_dim: list(range(n_bp)), "value": breakpoints}))
    val_param = pf.Param(pl.DataFrame({bp_dim: list(range(n_bp)), "value": values}))

    # Interpolation constraint: x == sum(lambda_i * bp_i) over bp_dim
    interp_name = f"__{name}_interp"
    interp = x == (lam * bp_param).sum(bp_dim)
    setattr(model, interp_name, interp)

    # SOS2 constraint on lambda, grouped by x_dims
    sos_name = f"__{name}_sos2"
    lam_var = getattr(model, lam_name)
    if x_dims:
        sos = _SOS2(lam_var, by=x_dims)
    else:
        sos = _SOS2(lam_var)
    setattr(model, sos_name, sos)

    # Return y = sum(lambda_i * value_i) over bp_dim
    y_expr = (lam * val_param).sum(bp_dim)
    return y_expr


def _incremental_formulation(
    x: Variable,
    breakpoints: list[float],
    values: list[float],
    name: str,
    model,
) -> Expression:
    """Incremental (delta) formulation — pure LP, no SOS2."""
    import pyoframe as pf
    from pyoframe._core import Expression

    n_segments = len(breakpoints) - 1
    x_dims = x._dimensions_unsafe
    seg_dim = f"__{name}_seg"

    # Delta variables: indexed by [*x_dims, seg_dim], each in [0, 1]
    seg_set = pf.Set(**{seg_dim: list(range(1, n_segments + 1))})

    if x_dims:
        delta_sets = [x.data.select(x_dims), seg_set]
    else:
        delta_sets = [seg_set]

    delta_name = f"__{name}_delta"
    delta = pf.Variable(*delta_sets, lb=0, ub=1)
    setattr(model, delta_name, delta)

    # Filling-order constraints: delta_{i+1} <= delta_i for all i
    # Build using shifted expressions so all ordering constraints are vectorized
    if n_segments > 1:
        delta_var = getattr(model, delta_name)
        delta_data = delta_var.to_expr().data

        # "next" = delta at seg=2..n, with seg shifted down by 1
        next_data = delta_data.filter(pl.col(seg_dim) >= 2).with_columns(
            (pl.col(seg_dim) - 1).alias(seg_dim)
        )
        # "curr" = delta at seg=1..n-1
        curr_data = delta_data.filter(pl.col(seg_dim) <= n_segments - 1)

        next_expr = Expression(next_data, name=f"{name}_delta_next")
        curr_expr = Expression(curr_data, name=f"{name}_delta_curr")
        order_name = f"__{name}_order"
        setattr(model, order_name, next_expr <= curr_expr)

    # Build segment widths and value increments as parameters
    widths = [breakpoints[i + 1] - breakpoints[i] for i in range(n_segments)]
    val_increments = [values[i + 1] - values[i] for i in range(n_segments)]

    width_param = pf.Param(
        pl.DataFrame({seg_dim: list(range(1, n_segments + 1)), "value": widths})
    )
    val_inc_param = pf.Param(
        pl.DataFrame({seg_dim: list(range(1, n_segments + 1)), "value": val_increments})
    )

    # Link constraint: x == b_0 + sum(delta_i * width_i)
    link_name = f"__{name}_link"
    link = x == breakpoints[0] + (delta * width_param).sum(seg_dim)
    setattr(model, link_name, link)

    # Return y = v_0 + sum(delta_i * val_increment_i)
    y_expr = values[0] + (delta * val_inc_param).sum(seg_dim)
    return y_expr
