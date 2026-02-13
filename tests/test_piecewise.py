"""Tests for piecewise linear functions."""

import pytest
from pytest import approx

import pyoframe as pf
from pyoframe._constants import SUPPORTED_SOLVERS


def _sos_capable_solvers():
    """Return solvers that support SOS constraints (natively or via Big-M reformulation)."""
    return [s for s in SUPPORTED_SOLVERS
            if s.supports_sos or s.supports_integer_variables]


@pytest.fixture(params=_sos_capable_solvers(), ids=lambda s: s.name)
def sos_solver(request):
    from tests.conftest import _installed_solvers

    if request.param not in _installed_solvers:
        return pytest.skip("Solver not installed.")
    return request.param


def test_pwl_sos2_basic(sos_solver):
    """Triangular PWL function with SOS2: maximize finds the peak."""
    # f(x) = triangular: (0,0), (5,10), (10,0)
    m = pf.Model(sos_solver)
    m.x = pf.Variable(lb=0, ub=10)
    y = pf.piecewise_linear(
        m.x,
        breakpoints=[0, 5, 10],
        values=[0, 10, 0],
        name="f",
    )
    m.maximize = y

    m.optimize()
    assert m.x.solution == approx(5, abs=1e-4)
    assert m.maximize.value == approx(10, abs=1e-4)


def test_pwl_incremental_basic(sos_solver):
    """Same triangular function with incremental method."""
    # f(x) = triangular: (0,0), (5,10), (10,0)
    m = pf.Model(sos_solver)
    m.x = pf.Variable(lb=0, ub=10)
    y = pf.piecewise_linear(
        m.x,
        breakpoints=[0, 5, 10],
        values=[0, 10, 0],
        name="f",
        method="incremental",
    )
    m.maximize = y

    m.optimize()
    assert m.x.solution == approx(5, abs=1e-4)
    assert m.maximize.value == approx(10, abs=1e-4)


def test_pwl_sos2_monotone(sos_solver):
    """Monotone increasing PWL with SOS2: bounded variable should be at upper bound."""
    # f(x) = piecewise: (0,0), (3,1), (6,5), (10,6)
    m = pf.Model(sos_solver)
    m.x = pf.Variable(lb=0, ub=10)
    y = pf.piecewise_linear(
        m.x,
        breakpoints=[0, 3, 6, 10],
        values=[0, 1, 5, 6],
        name="f",
    )
    m.maximize = y

    m.optimize()
    assert m.x.solution == approx(10, abs=1e-4)
    assert m.maximize.value == approx(6, abs=1e-4)


def test_pwl_incremental_monotone(sos_solver):
    """Monotone increasing PWL with incremental: bounded variable at upper bound."""
    m = pf.Model(sos_solver)
    m.x = pf.Variable(lb=0, ub=10)
    y = pf.piecewise_linear(
        m.x,
        breakpoints=[0, 3, 6, 10],
        values=[0, 1, 5, 6],
        name="f",
        method="incremental",
    )
    m.maximize = y

    m.optimize()
    assert m.x.solution == approx(10, abs=1e-4)
    assert m.maximize.value == approx(6, abs=1e-4)


def test_pwl_dimensioned(sos_solver):
    """Piecewise linear on a dimensioned variable."""
    # f(x) = (0,0), (5,10), (10,0) — each time period gets its own PWL
    m = pf.Model(sos_solver)
    m.x = pf.Variable(pf.Set(t=[1, 2, 3]), lb=0, ub=10)
    y = pf.piecewise_linear(
        m.x,
        breakpoints=[0, 5, 10],
        values=[0, 10, 0],
        name="f",
    )
    m.maximize = y.sum()

    m.optimize()
    # Each x[t] should be at 5 to maximize f(x[t]) = 10
    assert m.maximize.value == approx(30, abs=1e-4)
    sol = m.x.solution
    for t in [1, 2, 3]:
        assert sol.filter(t=t)["solution"].item() == approx(5, abs=1e-4)


def test_pwl_dimensioned_incremental(sos_solver):
    """Piecewise linear on a dimensioned variable with incremental method."""
    m = pf.Model(sos_solver)
    m.x = pf.Variable(pf.Set(t=[1, 2, 3]), lb=0, ub=10)
    y = pf.piecewise_linear(
        m.x,
        breakpoints=[0, 5, 10],
        values=[0, 10, 0],
        name="f",
        method="incremental",
    )
    m.maximize = y.sum()

    m.optimize()
    assert m.maximize.value == approx(30, abs=1e-4)
    sol = m.x.solution
    for t in [1, 2, 3]:
        assert sol.filter(t=t)["solution"].item() == approx(5, abs=1e-4)


def test_pwl_auto_method(sos_solver):
    """Auto method picks incremental for strictly increasing breakpoints."""
    m = pf.Model(sos_solver)
    m.x = pf.Variable(lb=0, ub=10)
    # Auto should pick incremental (breakpoints strictly increasing)
    y = pf.piecewise_linear(
        m.x,
        breakpoints=[0, 5, 10],
        values=[0, 10, 0],
        name="f",
        method="auto",
    )
    m.maximize = y

    m.optimize()
    assert m.x.solution == approx(5, abs=1e-4)
    assert m.maximize.value == approx(10, abs=1e-4)


def test_pwl_validation_length_mismatch():
    """Error when breakpoints and values have different lengths."""
    m = pf.Model("gurobi")
    m.x = pf.Variable(lb=0, ub=10)
    with pytest.raises(pf.PyoframeError, match="Length of breakpoints"):
        pf.piecewise_linear(
            m.x, breakpoints=[0, 5, 10], values=[0, 10], name="f"
        )


def test_pwl_validation_too_few_breakpoints():
    """Error when fewer than 2 breakpoints."""
    m = pf.Model("gurobi")
    m.x = pf.Variable(lb=0, ub=10)
    with pytest.raises(pf.PyoframeError, match="At least 2 breakpoints"):
        pf.piecewise_linear(
            m.x, breakpoints=[5], values=[10], name="f"
        )


def test_pwl_validation_non_increasing_incremental():
    """Error when incremental method used with non-increasing breakpoints."""
    m = pf.Model("gurobi")
    m.x = pf.Variable(lb=0, ub=10)
    with pytest.raises(pf.PyoframeError, match="strictly increasing"):
        pf.piecewise_linear(
            m.x, breakpoints=[0, 5, 3], values=[0, 10, 5], name="f",
            method="incremental",
        )


def test_pwl_variable_not_on_model():
    """Error when variable hasn't been added to a model."""
    x = pf.Variable(lb=0, ub=10)
    with pytest.raises(pf.PyoframeError, match="must be added to a model"):
        pf.piecewise_linear(
            x, breakpoints=[0, 5, 10], values=[0, 10, 0], name="f"
        )


def test_pwl_two_breakpoints(sos_solver):
    """PWL with exactly 2 breakpoints (single linear segment)."""
    m = pf.Model(sos_solver)
    m.x = pf.Variable(lb=0, ub=10)
    y = pf.piecewise_linear(
        m.x, breakpoints=[0, 10], values=[0, 20], name="f"
    )
    m.maximize = y

    m.optimize()
    assert m.x.solution == approx(10, abs=1e-4)
    assert m.maximize.value == approx(20, abs=1e-4)
