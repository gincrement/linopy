"""Tests specific to the GAMSPy solver backend."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from linopy import Model
from linopy.constants import TerminationCondition
from linopy.solvers import GAMSPy, Solver, _gamspy_termination_condition

pytestmark = pytest.mark.skipif(
    not GAMSPy.is_available() or not GAMSPy.license_status().ok,
    reason="gamspy not installed or not licensed",
)


def _lp(sense: str = "min") -> Model:
    m = Model()
    x = m.add_variables(0, 10, coords=[pd.RangeIndex(3, name="i")], name="x")
    y = m.add_variables(-5, 5, name="y")
    m.add_constraints(x + y >= 4, name="lower")
    m.add_constraints(2 * x - y <= 20, name="upper")
    m.add_constraints(x.sum() == 9, name="total")
    m.add_objective((x * np.array([1, 2, 3])).sum() + 2 * y, sense=sense)
    return m


def _assert_matches_highs(m: Model, ref: Model, duals: bool = True) -> None:
    assert m.objective.value == pytest.approx(ref.objective.value, abs=1e-6)
    for name in m.variables:
        np.testing.assert_allclose(
            m.variables[name].solution, ref.variables[name].solution, atol=1e-6
        )
    if duals:
        for name in m.constraints:
            np.testing.assert_allclose(
                m.constraints[name].dual, ref.constraints[name].dual, atol=1e-6
            )


def _highs(m: Model) -> Model:
    pytest.importorskip("highspy")
    m.solve("highs", output_flag=False)
    return m


@pytest.mark.parametrize("sense", ["min", "max"])
def test_lp_matches_highs(sense: str) -> None:
    m = _lp(sense)
    status, condition = m.solve("gamspy", io_api="direct")
    assert (status, condition) == ("ok", "optimal")
    _assert_matches_highs(m, _highs(_lp(sense)))


def _mixed(m: Model) -> Model:
    """Add binary, integer and semi-continuous columns to ``m``."""
    x = m.variables["x"]
    z = m.add_variables(coords=x.coords, name="z", binary=True)
    k = m.add_variables(
        0, 7, coords=[pd.RangeIndex(2, name="k")], name="k", integer=True
    )
    s = m.add_variables(2, 6, name="s", semi_continuous=True)
    m.add_constraints(x <= 10 * z, name="link")
    m.add_constraints(k.sum() + s >= 3.5, name="cover")
    m.objective = m.objective.expression + 5 * z.sum() + k.sum() + 0.5 * s
    return m


def test_mixed_integer_matches_highs() -> None:
    m = _mixed(_lp())
    assert m.solve("gamspy", io_api="direct") == ("ok", "optimal")
    _assert_matches_highs(m, _highs(_mixed(_lp())), duals=False)
    assert m.solver is not None and m.solver.report is not None
    assert m.solver.report.dual_bound == pytest.approx(m.objective.value, rel=1e-4)


def test_mixed_integer_has_no_duals() -> None:
    m = _mixed(_lp())
    m.solve("gamspy", io_api="direct")
    with pytest.raises(AttributeError, match="does not have dual values"):
        m.constraints["lower"].dual


def test_mixed_integer_quadratic_with_cross_term() -> None:
    # (x - k)^2 + (k - 2.6)^2 without its constant 6.76: optimum at x = k = 3
    m = Model()
    x = m.add_variables(-10, 10, name="x")
    k = m.add_variables(0, 5, name="k", integer=True)
    m.add_constraints(x + k >= 0, name="c")
    m.add_objective(x * x - 2 * x * k + 2 * k * k - 5.2 * k)
    assert m.solve("gamspy", io_api="direct") == ("ok", "optimal")
    assert m.objective.value == pytest.approx(0.16 - 6.76, abs=1e-6)
    assert float(x.solution) == pytest.approx(3, abs=1e-6)
    assert float(k.solution) == pytest.approx(3)


def test_quadratic_duals_match_highs() -> None:
    def build() -> Model:
        m = _lp()
        x = m.variables["x"]
        m.objective = m.objective.expression + (x * x).sum()
        return m

    m = build()
    assert m.solve("gamspy", io_api="direct") == ("ok", "optimal")
    _assert_matches_highs(m, _highs(build()))


def test_semi_continuous_zero_branch() -> None:
    m = Model()
    x = m.add_variables(1, 10, name="x", semi_continuous=True)
    m.add_constraints(x <= 0.5, name="cap")
    m.add_objective(x, sense="max")
    assert m.solve("gamspy", io_api="direct") == ("ok", "optimal")
    assert float(x.solution) == pytest.approx(0)


def test_infeasible_and_unbounded() -> None:
    m = _lp()
    m.add_constraints(m.variables["x"].sum() >= 100, name="impossible")
    assert m.solve("gamspy", io_api="direct") == ("warning", "infeasible")


def _unbounded() -> Model:
    m = Model()
    x = m.add_variables(name="x")
    m.add_constraints(x >= 0, name="c")
    m.add_objective(x, sense="max")
    return m


def test_unbounded_is_reported_after_cplex_rerun() -> None:
    # GAMS has no "infeasible or unbounded" status: CPLEX reports such a
    # presolve result as infeasible unless it reruns without presolve.
    m = _unbounded()
    _, condition = m.solve("gamspy", io_api="direct", solver="cplex")
    assert condition == "infeasible"
    _, condition = m.solve(
        "gamspy", io_api="direct", solver="cplex", solver_options={"rerun": "yes"}
    )
    assert condition == "unbounded"


def test_violated_row_without_coefficients_is_infeasible() -> None:
    m = _lp()
    x = m.variables["x"]
    m.add_constraints(0 * x.sel(i=0) >= 1, name="empty")
    status, condition = m.solve("gamspy", io_api="direct", sanitize_zeros=False)
    assert (status, condition) == ("warning", "infeasible")


def test_satisfied_row_without_coefficients_is_ignored() -> None:
    m = _lp()
    x = m.variables["x"]
    m.add_constraints(0 * x.sel(i=0) <= 1, name="empty")
    assert m.solve("gamspy", io_api="direct", sanitize_zeros=False) == (
        "ok",
        "optimal",
    )
    assert float(m.constraints["empty"].dual) == 0


def test_masked_entries_are_skipped() -> None:
    m = Model()
    mask = pd.Series([True, False, True])
    x = m.add_variables(0, 10, coords=[pd.RangeIndex(3, name="i")], name="x", mask=mask)
    m.add_constraints(x >= pd.Series([1.0, 2.0, 3.0]).rename_axis("i"), name="c")
    m.add_objective(x.sum())
    m.solve("gamspy", io_api="direct")
    np.testing.assert_allclose(x.solution.values, [1, np.nan, 3])


def test_file_io_api_is_redirected_to_direct() -> None:
    m = _lp()
    assert m.solve("gamspy") == ("ok", "optimal")
    assert m.solver is not None and m.solver.io_api == "direct"
    _assert_matches_highs(m, _highs(_lp()))


def test_options_are_split_by_destination() -> None:
    import gamspy

    solver = Solver.from_name(
        "gamspy",
        options={
            "solver": "cplex",
            "time_limit": 5,
            "threads": 1,
            "lpmethod": 2,
            "solver_options": {"advind": 0},
        },
    )
    assert isinstance(solver, GAMSPy)
    kwargs = solver._split_options()
    assert kwargs["solver"] == "cplex"
    assert isinstance(kwargs["options"], gamspy.Options)
    assert kwargs["options"].time_limit == 5
    assert kwargs["options"].threads == 1
    assert kwargs["solver_options"] == {"advind": 0, "lpmethod": 2}


def test_options_reach_the_solver() -> None:
    m = _lp()
    out = io.StringIO()
    m.solve("gamspy", io_api="direct", solver="cplex", output=out, lpmethod=2)
    log = out.getvalue()
    assert "CPLEX" in log
    assert "lpmethod 2" in log.lower() or "lpmethod" in log.lower()


def test_log_fn_receives_the_log(tmp_path: Path) -> None:
    log_fn = tmp_path / "gams.log"
    m = _lp()
    m.solve("gamspy", io_api="direct", log_fn=log_fn)
    assert "Optimal" in log_fn.read_text() or "optimal" in log_fn.read_text()


def test_unknown_gams_solver_raises() -> None:
    m = _lp()
    with pytest.raises(Exception, match="(?i)solver"):
        m.solve("gamspy", io_api="direct", solver="no_such_solver")


@pytest.mark.parametrize(
    ("model_status", "solve_status", "expected"),
    [
        (1, 1, TerminationCondition.optimal),
        (8, 1, TerminationCondition.optimal),
        (8, 3, TerminationCondition.time_limit),
        (14, 3, TerminationCondition.time_limit),
        (7, 1, TerminationCondition.suboptimal),
        (19, 1, TerminationCondition.infeasible),
        (10, 1, TerminationCondition.infeasible),
        (18, 1, TerminationCondition.unbounded),
        (13, 10, TerminationCondition.internal_solver_error),
        (11, 7, TerminationCondition.licensing_problems),
        (14, 8, TerminationCondition.user_interrupt),
    ],
)
def test_status_mapping(
    model_status: int, solve_status: int, expected: TerminationCondition
) -> None:
    import gamspy

    condition = _gamspy_termination_condition(
        gamspy.ModelStatus(model_status), gamspy.SolveStatus(solve_status)
    )
    assert condition == expected
