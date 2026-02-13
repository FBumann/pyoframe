"""Defines SOS (Special Ordered Set) constraint types for Pyoframe."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import polars as pl
import pyoptinterface as poi

from pyoframe._constants import (
    CONSTRAINT_KEY,
    VAR_KEY,
    Config,
    PyoframeError,
    VType,
)
from pyoframe._model_element import BaseBlock

if TYPE_CHECKING:  # pragma: no cover
    from pyoframe._core import Variable
    from pyoframe._model import Model


class SOSConstraint(BaseBlock):
    """A Special Ordered Set (SOS) constraint.

    SOS constraints restrict which variables in a set can be simultaneously non-zero.

    - **SOS1**: At most one variable in the set can be non-zero.
    - **SOS2**: At most two variables can be non-zero, and they must be adjacent
      (according to the weight ordering).

    Parameters:
        variable:
            The variable to constrain. Must already be added to a model.
        sos_type:
            The type of SOS constraint (``poi.SOSType.SOS1`` or ``poi.SOSType.SOS2``).
        by:
            Dimension name(s) defining groups. One SOS set is created per group.
            If ``None``, all variable entries form a single SOS set.
        weights:
            Ordering weights for the variables within each SOS set.
            If ``None``, sequential weights ``1, 2, 3, ...`` are assigned within each group.
        big_m:
            Big-M value for reformulation on solvers without native SOS support.
            If ``None``, the upper bound of the variable is used.

    See Also:
        :class:`SOS1`, :class:`SOS2` for convenience constructors.
    """

    def __init__(
        self,
        variable: Variable,
        sos_type: poi.SOSType,
        by: str | Sequence[str] | None = None,
        weights: Sequence[float] | None = None,
        big_m: float | None = None,
    ):
        if not variable._has_ids:
            raise PyoframeError(
                "Variable must be added to a model before creating an SOS constraint. "
                "Assign the variable to a model first (e.g. m.x = pf.Variable(...))."
            )

        self._variable = variable
        self._sos_type = sos_type
        self._big_m = big_m

        if isinstance(by, str):
            by = [by]
        self._by = by

        # Validate 'by' dimensions
        var_dims = variable._dimensions_unsafe
        if by is not None:
            for dim in by:
                if dim not in var_dims:
                    raise PyoframeError(
                        f"Dimension '{dim}' specified in 'by' is not a dimension of the variable. "
                        f"Variable dimensions: {var_dims}"
                    )

        # Determine the "within-group" dimensions (dims not in 'by')
        if by is not None:
            member_dims = [d for d in var_dims if d not in by]
        else:
            member_dims = var_dims

        # Build the data DataFrame — this will hold group dims + CONSTRAINT_KEY after _assign_ids
        if by is not None:
            data = variable.data.select(by).unique(maintain_order=True)
        else:
            data = pl.DataFrame()

        self._weights = weights
        self._member_dims = member_dims

        super().__init__(data)

    @classmethod
    def _get_id_column_name(cls) -> str:
        return CONSTRAINT_KEY

    def _on_add_to_model(self, model: Model, name: str):
        super()._on_add_to_model(model, name)

        if model.solver.supports_sos:
            self._assign_ids()
        elif model.solver.supports_integer_variables:
            self._reformulate_big_m()
        else:
            raise PyoframeError(
                f"Solver '{model.solver.name}' does not support SOS constraints and "
                "cannot use Big-M reformulation (no integer variable support)."
            )

    def _compute_big_m(self) -> float:
        variable = self._variable

        # Validate lower bound is non-negative
        if variable.lb is None or variable.lb < 0:
            lb_str = "None (unbounded)" if variable.lb is None else str(variable.lb)
            raise PyoframeError(
                f"Big-M reformulation requires non-negative lower bounds on the variable. "
                f"Variable '{variable.name}' has lb={lb_str}. "
                f"Set lb=0 (or another non-negative value)."
            )

        has_scalar_ub = variable.ub is not None and variable.ub < 1e100

        if self._big_m is not None and has_scalar_ub:
            return min(self._big_m, variable.ub)
        elif self._big_m is not None:
            return self._big_m
        elif has_scalar_ub:
            return variable.ub
        elif variable._ub_expr is not None:
            raise PyoframeError(
                "Variable has expression-based upper bounds. "
                "Provide big_m= to use Big-M reformulation."
            )
        else:
            raise PyoframeError(
                f"Cannot determine Big-M value for variable '{variable.name}'. "
                "Set finite upper bounds on the variable or provide big_m= parameter."
            )

    def _reformulate_big_m(self):
        if self._sos_type == poi.SOSType.SOS1:
            self._reformulate_sos1()
        else:
            self._reformulate_sos2()

    def _reformulate_sos1(self):
        import pyoframe as pf
        from pyoframe._core import Expression

        model = self._model
        assert model is not None
        name = self.name
        variable = self._variable
        M = self._compute_big_m()

        member_dims = self._member_dims
        var_dims = variable._dimensions_unsafe

        # Binary indicators y — same dimensions as the original variable
        if var_dims:
            y_sets: list = [variable.data.select(var_dims)]
        else:
            y_sets = []

        y_name = f"__{name}_y"
        setattr(model, y_name, pf.Variable(*y_sets, vtype=VType.BINARY))
        y = getattr(model, y_name)

        # Upper linking: variable <= M * y
        upper_name = f"__{name}_upper"
        setattr(model, upper_name, variable <= M * y)

        # Cardinality: sum(y) over member_dims <= 1
        card_name = f"__{name}_card"
        if member_dims:
            setattr(model, card_name, y.sum(*member_dims) <= 1)
        else:
            # Dimensionless: single variable, y <= 1 (redundant for BINARY but correct)
            setattr(model, card_name, y <= 1)

    def _reformulate_sos2(self):
        import pyoframe as pf
        from pyoframe._core import Expression

        model = self._model
        assert model is not None
        name = self.name
        variable = self._variable
        M = self._compute_big_m()

        member_dims = self._member_dims
        by = self._by

        # For SOS2 Big-M, we need a single ordering dimension
        if len(member_dims) == 0:
            # Dimensionless variable — SOS2 trivially satisfied
            return

        if len(member_dims) != 1:
            raise PyoframeError(
                "Big-M reformulation for SOS2 requires exactly one member dimension "
                f"(the ordering dimension), but found {len(member_dims)}: {member_dims}. "
                "Use 'by' to specify grouping dimensions."
            )

        member_dim = member_dims[0]
        var_data = variable.data

        # Get ordered member values
        if by is not None:
            members_df = var_data.select(by + [member_dim]).unique(maintain_order=True)
            # All groups should have the same members; get from first group
            first_group = var_data.select(by).unique(maintain_order=True).head(1)
            member_values = (
                var_data.join(first_group, on=by)
                .select(member_dim)
                .unique(maintain_order=True)
                .get_column(member_dim)
                .to_list()
            )
        else:
            member_values = (
                var_data.select(member_dim)
                .unique(maintain_order=True)
                .get_column(member_dim)
                .to_list()
            )

        n = len(member_values)

        if n <= 2:
            # SOS2 with 2 or fewer variables is trivially satisfied
            return

        # Sort by weights if provided
        if self._weights is not None:
            weights = list(self._weights)
            if len(weights) != n:
                raise PyoframeError(
                    f"Length of weights ({len(weights)}) must match number of variables ({n})."
                )
            sorted_pairs = sorted(zip(weights, member_values), key=lambda x: x[0])
            member_values = [v for _, v in sorted_pairs]

        n_segments = n - 1
        seg_dim = f"__{name}_seg"

        # Create segment indicator variables z
        seg_set = pf.Set(**{seg_dim: list(range(n_segments))})
        if by is not None:
            z_sets: list = [var_data.select(by).unique(maintain_order=True), seg_set]
        else:
            z_sets = [seg_set]

        z_name = f"__{name}_z"
        setattr(model, z_name, pf.Variable(*z_sets, vtype=VType.BINARY))
        z_var = getattr(model, z_name)

        # Build linking constraints using dimension remapping
        z_data = z_var.to_expr().data

        member_dtype = var_data[member_dim].dtype
        seg_dtype = z_data[seg_dim].dtype

        # Left map: seg i -> member at position i (left endpoint of segment)
        left_map = pl.DataFrame({
            seg_dim: list(range(n_segments)),
            member_dim: member_values[:-1],
        }).cast({member_dim: member_dtype, seg_dim: seg_dtype})

        z_left_data = z_data.join(left_map, on=seg_dim).drop(seg_dim)

        # Right map: seg i -> member at position i+1 (right endpoint of segment)
        right_map = pl.DataFrame({
            seg_dim: list(range(n_segments)),
            member_dim: member_values[1:],
        }).cast({member_dim: member_dtype, seg_dim: seg_dtype})

        z_right_data = z_data.join(right_map, on=seg_dim).drop(seg_dim)

        # Combine: each member gets contributions from adjacent segments
        z_combined_data = pl.concat([z_left_data, z_right_data])
        z_combined_expr = Expression(z_combined_data, name=f"{name}_z_combined")

        # Linking constraint: variable <= M * z_combined
        link_name = f"__{name}_link"
        setattr(model, link_name, variable <= M * z_combined_expr)

        # Cardinality: sum(z) over seg_dim <= 1
        card_name = f"__{name}_card"
        setattr(model, card_name, z_var.sum(seg_dim) <= 1)

    def _assign_ids(self):
        assert self._model is not None

        var_data = self._variable.data
        var_dims = self._variable._dimensions_unsafe
        by = self._by

        if by is None:
            # Single SOS set: all variables in one group
            var_ids = var_data.get_column(VAR_KEY).to_list()
            variables = [poi.VariableIndex(v) for v in var_ids]

            if self._weights is not None:
                weights = list(self._weights)
                if len(weights) != len(variables):
                    raise PyoframeError(
                        f"Length of weights ({len(weights)}) must match number of variables ({len(variables)})."
                    )
            else:
                weights = list(range(1, len(variables) + 1))

            result = self._model.poi.add_sos_constraint(
                variables, self._sos_type, weights
            )
            self._data = pl.DataFrame(
                {CONSTRAINT_KEY: [result.index]}
            ).cast({CONSTRAINT_KEY: Config.id_dtype})
        else:
            # Grouped SOS sets: one per group
            # Sort by group dims to enable splitting
            df = var_data.sort(by, maintain_order=True)

            # Find split points using is_first_distinct on group columns
            split = (
                df.lazy()
                .with_row_index()
                .filter(pl.struct(by).is_first_distinct())
                .select("index")
                .collect()
                .to_series()
                .to_list()
            ) + [df.height]

            var_ids_all = df.get_column(VAR_KEY).to_list()
            group_keys = df.select(by).unique(maintain_order=True)

            ids = []
            for s0, s1 in _pairwise(split):
                group_var_ids = var_ids_all[s0:s1]
                variables = [poi.VariableIndex(v) for v in group_var_ids]

                if self._weights is not None:
                    weights = list(self._weights)
                    if len(weights) != len(variables):
                        raise PyoframeError(
                            f"Length of weights ({len(weights)}) must match number of variables per group ({len(variables)})."
                        )
                else:
                    weights = list(range(1, len(variables) + 1))

                result = self._model.poi.add_sos_constraint(
                    variables, self._sos_type, weights
                )
                ids.append(result.index)

            try:
                self._data = group_keys.with_columns(
                    pl.Series(ids, dtype=Config.id_dtype).alias(CONSTRAINT_KEY)
                )
            except TypeError as e:
                raise TypeError(
                    f"Number of SOS constraints exceeds the current data type ({Config.id_dtype}). "
                    "Consider increasing the data type by changing Config.id_dtype."
                ) from e


class SOS1(SOSConstraint):
    """An SOS Type 1 constraint: at most one variable in the set can be non-zero.

    Parameters:
        variable:
            The variable to constrain. Must already be added to a model.
        by:
            Dimension name(s) defining groups. One SOS1 set per group.
            If ``None``, all variable entries form a single SOS1 set.
        weights:
            Ordering weights. If ``None``, sequential ``1, 2, 3, ...`` within each group.
        big_m:
            Big-M value for reformulation on solvers without native SOS support.
            If ``None``, the upper bound of the variable is used.

    Examples:
        >>> m = pf.Model("gurobi")
        >>> m.x = pf.Variable(pf.Set(i=[1, 2, 3]), lb=0, ub=10)
        >>> m.sos = pf.SOS1(m.x)
    """

    def __init__(
        self,
        variable: Variable,
        by: str | Sequence[str] | None = None,
        weights: Sequence[float] | None = None,
        big_m: float | None = None,
    ):
        super().__init__(variable, poi.SOSType.SOS1, by=by, weights=weights, big_m=big_m)


class SOS2(SOSConstraint):
    """An SOS Type 2 constraint: at most two variables can be non-zero, and they must be adjacent.

    Parameters:
        variable:
            The variable to constrain. Must already be added to a model.
        by:
            Dimension name(s) defining groups. One SOS2 set per group.
            If ``None``, all variable entries form a single SOS2 set.
        weights:
            Ordering weights. If ``None``, sequential ``1, 2, 3, ...`` within each group.
        big_m:
            Big-M value for reformulation on solvers without native SOS support.
            If ``None``, the upper bound of the variable is used.

    Examples:
        >>> m = pf.Model("gurobi")
        >>> m.x = pf.Variable(pf.Set(i=[1, 2, 3]), lb=0, ub=1)
        >>> m.sos = pf.SOS2(m.x)
    """

    def __init__(
        self,
        variable: Variable,
        by: str | Sequence[str] | None = None,
        weights: Sequence[float] | None = None,
        big_m: float | None = None,
    ):
        super().__init__(variable, poi.SOSType.SOS2, by=by, weights=weights, big_m=big_m)


def _pairwise(iterable):
    """Polyfill for itertools.pairwise (Python 3.10+)."""
    a, b = iter(iterable), iter(iterable)
    next(b, None)
    return zip(a, b)
