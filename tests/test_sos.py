"""Tests for SOS (Special Ordered Set) constraints."""

import pytest
from pytest import approx

import pyoframe as pf
from pyoframe._constants import SUPPORTED_SOLVERS, _Solver


def _sos_solvers():
    """Return solvers that support SOS constraints."""
    return [s for s in SUPPORTED_SOLVERS if s.supports_sos]


@pytest.fixture(params=_sos_solvers(), ids=lambda s: s.name)
def sos_solver(request):
    from tests.conftest import _installed_solvers

    if request.param not in _installed_solvers:
        return pytest.skip("Solver not installed.")
    return request.param


def test_sos1_basic(sos_solver):
    """SOS1: at most one variable can be non-zero."""
    m = pf.Model(sos_solver)
    m.x = pf.Variable(pf.Set(i=[1, 2, 3]), lb=0, ub=10)
    m.sos = pf.SOS1(m.x)
    m.maximize = m.x.sum()

    m.optimize()
    sol = m.x.solution
    # With SOS1, at most one variable is non-zero. The max sum is 10.
    assert sol["solution"].sum() == approx(10, abs=1e-4)
    # Count non-zero values
    non_zero = (sol["solution"].abs() > 1e-6).sum()
    assert non_zero <= 1


def test_sos1_dimensioned(sos_solver):
    """SOS1 with 'by' grouping: one SOS1 per group."""
    m = pf.Model(sos_solver)
    m.x = pf.Variable(
        pf.Set(time=[1, 2], item=["a", "b", "c"]),
        lb=0,
        ub=10,
    )
    # One SOS1 per time period — within each time, at most one item non-zero
    m.sos = pf.SOS1(m.x, by="time")
    m.maximize = m.x.sum()

    m.optimize()
    sol = m.x.solution

    # Each time group can have at most 1 non-zero, so max sum = 10 * 2 = 20
    assert sol["solution"].sum() == approx(20, abs=1e-4)
    # Check per-group
    for t in [1, 2]:
        group = sol.filter(time=t)
        non_zero = (group["solution"].abs() > 1e-6).sum()
        assert non_zero <= 1


def test_sos2_basic(sos_solver):
    """SOS2: at most two adjacent variables can be non-zero."""
    m = pf.Model(sos_solver)
    m.x = pf.Variable(pf.Set(i=[1, 2, 3, 4]), lb=0, ub=1)
    m.sos = pf.SOS2(m.x)
    # Try to maximize x[1] + x[4] — can't have both non-zero (not adjacent)
    coefs = pf.Param({"i": [1, 2, 3, 4], "value": [10, 0, 0, 10]})
    m.maximize = (coefs * m.x).sum()

    m.optimize()
    # Can't have x[1] and x[4] both non-zero, so best is one of them at 1
    assert m.maximize.value == approx(10, abs=1e-4)


def test_sos_custom_weights(sos_solver):
    """SOS2 with custom weights changes adjacency order."""
    m = pf.Model(sos_solver)
    m.x = pf.Variable(pf.Set(i=[1, 2, 3]), lb=0, ub=1)
    # Custom weights: [1, 3, 2] makes i=1 and i=3 adjacent (weights 1,2)
    m.sos = pf.SOS2(m.x, weights=[1, 3, 2])
    # Maximize x[1] + x[3] — adjacent under custom weights
    coefs = pf.Param({"i": [1, 2, 3], "value": [10, 0, 10]})
    m.maximize = (coefs * m.x).sum()

    m.optimize()
    # x[1] and x[3] are adjacent (weights 1 and 2), so both can be 1
    assert m.maximize.value == approx(20, abs=1e-4)


def test_sos_variable_not_on_model():
    """Error when variable hasn't been added to a model."""
    x = pf.Variable(pf.Set(i=[1, 2, 3]), lb=0)
    with pytest.raises(pf.PyoframeError, match="must be added to a model"):
        pf.SOS1(x)


def test_sos_invalid_by_dimension(sos_solver):
    """Error when 'by' references a non-existent dimension."""
    m = pf.Model(sos_solver)
    m.x = pf.Variable(pf.Set(i=[1, 2, 3]), lb=0)
    with pytest.raises(pf.PyoframeError, match="not a dimension"):
        pf.SOS1(m.x, by="nonexistent")


def test_sos_unsupported_solver():
    """Error when solver doesn't support SOS."""
    unsupported = [s for s in SUPPORTED_SOLVERS if not s.supports_sos]
    if not unsupported:
        pytest.skip("All solvers support SOS")

    from tests.conftest import _installed_solvers

    solver = None
    for s in unsupported:
        if s in _installed_solvers:
            solver = s
            break
    if solver is None:
        pytest.skip("No unsupported SOS solver installed")

    m = pf.Model(solver)
    m.x = pf.Variable(pf.Set(i=[1, 2, 3]), lb=0)
    with pytest.raises(pf.PyoframeError, match="does not support SOS"):
        m.sos = pf.SOS1(m.x)


def test_sos_model_tracking(sos_solver):
    """SOS constraints are tracked in model.sos_constraints."""
    m = pf.Model(sos_solver)
    m.x = pf.Variable(pf.Set(i=[1, 2, 3]), lb=0)
    assert len(m.sos_constraints) == 0
    m.sos = pf.SOS1(m.x)
    assert len(m.sos_constraints) == 1
