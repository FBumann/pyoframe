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

    See Also:
        :class:`SOS1`, :class:`SOS2` for convenience constructors.
    """

    def __init__(
        self,
        variable: Variable,
        sos_type: poi.SOSType,
        by: str | Sequence[str] | None = None,
        weights: Sequence[float] | None = None,
    ):
        if not variable._has_ids:
            raise PyoframeError(
                "Variable must be added to a model before creating an SOS constraint. "
                "Assign the variable to a model first (e.g. m.x = pf.Variable(...))."
            )

        self._variable = variable
        self._sos_type = sos_type

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

        if not model.solver.supports_sos:
            raise PyoframeError(
                f"Solver '{model.solver.name}' does not support SOS constraints."
            )

        self._assign_ids()

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
    ):
        super().__init__(variable, poi.SOSType.SOS1, by=by, weights=weights)


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
    ):
        super().__init__(variable, poi.SOSType.SOS2, by=by, weights=weights)


def _pairwise(iterable):
    """Polyfill for itertools.pairwise (Python 3.10+)."""
    a, b = iter(iterable), iter(iterable)
    next(b, None)
    return zip(a, b)
