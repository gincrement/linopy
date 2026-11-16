"""Tests specific to the GAMSPy solver backend."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from linopy import Model, Variable
from linopy.constants import TerminationCondition
from linopy.persistent import RebuildReason, RebuildRequiredError
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


def _sos_model(sos_type: Literal[1, 2], coeffs: list[float]) -> tuple[Model, Variable]:
    m = Model()
    locations = pd.Index([0, 1, 2], name="locations")
    build = m.add_variables(coords=[locations], name="build", binary=True)
    m.add_sos_constraints(build, sos_type=sos_type, sos_dim="locations")
    m.add_objective(build * np.array(coeffs), sense="max")
    return m, build


@pytest.mark.parametrize(
    ("sos_type", "coeffs", "expected", "objective"),
    [
        (1, [1, 2, 3], [0, 0, 1], 3),
        (2, [1, 2, 3], [0, 1, 1], 5),
        (2, [2, 1, 3], [0, 1, 1], 4),
    ],
)
def test_sos_binary_maximize(
    sos_type: Literal[1, 2],
    coeffs: list[float],
    expected: list[float],
    objective: float,
) -> None:
    m, build = _sos_model(sos_type, coeffs)
    assert m.solve("gamspy", io_api="direct") == ("ok", "optimal")
    np.testing.assert_allclose(build.solution.values, expected, atol=1e-6)
    assert m.objective.value == pytest.approx(objective)


def test_sos2_adjacency_follows_declared_order() -> None:
    # linopy orders SOS members as declared unless the coords ascend, so with
    # coords 0, 2, 1 the neighbours are positions (0, 1) and (1, 2): the best
    # pair is coords 0.0 and 2.0, worth 3 + 1.5. Adjacency by coordinate value
    # would instead only allow pairs worth 3 + 0.5.
    m = Model()
    seg = pd.Index([0.0, 2.0, 1.0], name="seg")
    lam = m.add_variables(0, 1, coords=[seg], name="lam")
    with pytest.warns(UserWarning, match="neither ascend nor descend"):
        m.add_sos_constraints(lam, sos_type=2, sos_dim="seg")
    m.add_constraints(lam.sum() <= 1.5, name="budget")
    m.add_objective(lam * np.array([3.0, 3.0, 1.0]), sense="max")
    assert m.solve("gamspy", io_api="direct") == ("ok", "optimal")
    assert m.objective.value == pytest.approx(4.5)
    assert float(lam.solution.sel(seg=1.0)) == pytest.approx(0, abs=1e-9)


def test_sos2_adjacency_follows_ascending_coords() -> None:
    # Ascending coords are the weights; position and weight order agree, so
    # the non-neighbours 0.0 and 2.0 cannot both be non-zero.
    m = Model()
    seg = pd.Index([0.0, 1.0, 2.0], name="seg")
    lam = m.add_variables(0, 1, coords=[seg], name="lam")
    m.add_sos_constraints(lam, sos_type=2, sos_dim="seg")
    m.add_constraints(lam.sum() <= 1.5, name="budget")
    m.add_objective(lam * np.array([3.0, 1.0, 3.0]), sense="max")
    assert m.solve("gamspy", io_api="direct") == ("ok", "optimal")
    assert m.objective.value == pytest.approx(3.5)
    ends = lam.solution.sel(seg=0.0) * lam.solution.sel(seg=2.0)
    assert float(ends) == pytest.approx(0, abs=1e-9)


def test_sos1_continuous_with_quadratic_objective() -> None:
    m = Model()
    seg = pd.Index([0, 1, 2], name="seg")
    x = m.add_variables(lower=0, upper=10, coords=[seg], name="x")
    m.add_sos_constraints(x, sos_type=1, sos_dim="seg")
    m.add_constraints(x.sum() >= 5)
    m.add_objective((x * x).sum() + (x * np.array([0.0, -10.0, 0.0])).sum())
    assert m.solve("gamspy", io_api="direct") == ("ok", "optimal")
    np.testing.assert_allclose(x.solution.values, [0, 5, 0], atol=1e-5)
    assert m.objective.value == pytest.approx(-25)


def test_grouped_sos_sets_on_integer_variables() -> None:
    m = Model()
    groups = pd.Index(["a", "b"], name="group")
    seg = pd.Index([0, 1, 2], name="seg")
    n = m.add_variables(0, 4, coords=[groups, seg], name="n", integer=True)
    m.add_sos_constraints(n, sos_type=1, sos_dim="seg")
    m.add_objective(n.sum(), sense="max")
    assert m.solve("gamspy", io_api="direct") == ("ok", "optimal")
    assert m.objective.value == pytest.approx(8)
    assert ((n.solution > 0).sum("seg") == 1).all()


# ---------------------------------------------------------------- in-place updates


def _persistent(m: Model, **options: object) -> GAMSPy:
    solver = Solver.from_name(
        "gamspy", m, io_api="direct", track_updates=True, options=dict(options)
    )
    assert isinstance(solver, GAMSPy)
    solver.solve(assign=True)
    return solver


def _fresh_objective(m: Model) -> float:
    fresh = Solver.from_name("gamspy", m, io_api="direct")
    result = fresh.solve()
    assert result.solution is not None
    return float(result.solution.objective)


def _update_sequence(m: Model) -> list:
    """Updates of every in-place kind, applied one after the other."""
    x, y = m.variables["x"], m.variables["y"]
    return [
        lambda: m.constraints["lower"].update(rhs=4.5),
        lambda: x.lower.values.__setitem__(..., 1.0),
        lambda: y.upper.values.__setitem__(..., 2.0),
        lambda: m.constraints["upper"].update(coeffs=m.constraints["upper"].coeffs * 3),
        lambda: setattr(m.objective, "expression", (3 * x).sum() + 5 * y),
        lambda: setattr(m.objective, "sense", "max"),
        lambda: m.constraints["lower"].update(rhs=5.0),
    ]


@pytest.mark.parametrize("freeze", [True, False])
def test_sequential_updates_match_fresh_builds(freeze: bool) -> None:
    m = _lp()
    solver = _persistent(m, freeze=freeze)
    for step, update in enumerate(_update_sequence(m)):
        update()
        result = solver.solve(m, assign=True, disallow_rebuild=True)
        assert result.status.is_ok, step
        assert m.objective.value == pytest.approx(_fresh_objective(m), abs=1e-6), step
        assert solver.solver_model.frozen is freeze
    assert solver._in_place_updates == len(_update_sequence(m))
    assert solver._rebuilds == 0


def test_duals_after_frozen_updates_match_highs() -> None:
    m = _lp()
    solver = _persistent(m)
    m.constraints["lower"].update(rhs=6.0)
    m.variables["x"].upper.values[...] = 4.0
    solver.solve(m, assign=True, disallow_rebuild=True)
    assert solver.solver_model.frozen

    ref = _lp()
    ref.constraints["lower"].update(rhs=6.0)
    ref.variables["x"].upper.values[...] = 4.0
    _assert_matches_highs(m, _highs(ref))


def test_mixed_integer_updates_in_place() -> None:
    m = _mixed(_lp())
    solver = _persistent(m)
    m.constraints["cover"].update(rhs=6.5)
    m.variables["k"].upper.values[...] = 2.0
    solver.solve(m, assign=True, disallow_rebuild=True)
    assert m.objective.value == pytest.approx(_fresh_objective(m), abs=1e-6)
    assert solver.solver_model.frozen


def _row_with_coeffs(coeffs: list[float]) -> Model:
    m = Model()
    x = m.add_variables(0, 10, coords=[pd.RangeIndex(3, name="i")], name="x")
    weights = xr.DataArray(coeffs, coords=[x.coords["i"]])
    m.add_constraints((weights * x).sum() >= 4, name="row")
    m.add_objective(x.sum())
    return m


def test_new_nonzero_rebuilds_and_solves() -> None:
    solver = _persistent(_row_with_coeffs([1.0, 0.0, 1.0]))
    result = solver.solve(_row_with_coeffs([1.0, 4.0, 1.0]))
    assert solver._rebuilds == 1
    assert result.solution is not None
    assert result.solution.objective == pytest.approx(1.0)


def test_coefficient_pattern_is_fixed_at_build() -> None:
    m = _row_with_coeffs([1.0, 0.0, 1.0])
    solver = _persistent(m)
    gm = solver.solver_model
    row, absent, present = np.array([0]), np.array([1]), np.array([2])
    assert gm.in_pattern(row, present, np.array([5.0]))
    assert gm.in_pattern(row, absent, np.array([0.0]))
    assert not gm.in_pattern(row, absent, np.array([5.0]))

    gm.set_coefficients(np.array([0, 0]), np.array([1, 2]), np.array([0.0, 4.0]))
    result = solver.solve()
    assert result.solution is not None
    assert result.solution.objective == pytest.approx(1.0)


def test_variable_type_change_rebuilds() -> None:
    def build(integer: bool) -> Model:
        m = Model()
        x = m.add_variables(0, 10, coords=[range(3)], name="x", integer=integer)
        m.add_constraints(x >= 1.5, name="c1")
        m.add_objective(x.sum())
        return m

    solver = _persistent(build(False))
    with pytest.raises(RebuildRequiredError):
        solver.solve(build(True), disallow_rebuild=True)
    result = solver.solve(build(True))
    assert solver._last_rebuild_reason is RebuildReason.BACKEND_REJECTED
    assert result.solution is not None
    assert result.solution.objective == pytest.approx(6.0)


def test_frozen_rhs_update_with_row_without_coefficients() -> None:
    # Rows without coefficients are not in the frozen instance; their rhs
    # must neither reach it (unmatched records) nor be forgotten.
    m = _lp()
    m.add_constraints(0 * m.variables["x"].sel(i=0) <= 1, name="empty")
    solver = Solver.from_name("gamspy", m, io_api="direct", track_updates=True)
    solver.solve(assign=True)
    m.constraints["lower"].update(rhs=4.5)
    m.constraints["empty"].update(rhs=2.0)
    result = solver.solve(m, assign=True, disallow_rebuild=True)
    assert solver.solver_model.frozen
    assert result.status.is_ok
    m.constraints["empty"].update(rhs=-1.0)
    result = solver.solve(m, disallow_rebuild=True)
    assert result.status.termination_condition == "infeasible"


def test_freeze_adds_coefficients_only_once_they_change() -> None:
    m = _lp()
    solver = _persistent(m)
    gm = solver.solver_model
    m.constraints["lower"].update(rhs=4.5)
    solver.solve(m, disallow_rebuild=True)
    assert gm.frozen and not gm._frozen_with_coefficients

    upper = m.constraints["upper"]
    upper.update(coeffs=upper.coeffs * 3)
    result = solver.solve(m, assign=True, disallow_rebuild=True)
    assert gm.frozen and gm._frozen_with_coefficients
    assert result.solution is not None
    assert result.solution.objective == pytest.approx(_fresh_objective(m), abs=1e-6)


def test_close_releases_the_frozen_instance() -> None:
    m = _lp()
    solver = _persistent(m)
    m.constraints["lower"].update(rhs=6.0)
    solver.solve(m, disallow_rebuild=True)
    gm = solver.solver_model
    assert gm.frozen
    solver.close()
    assert not gm.frozen
    assert solver.solver_model is None


def _solve_windows(
    solver_name: str, warm_start: bool, **options: Any
) -> tuple[list[float], Solver]:
    """Objectives of three consecutive windows, solved in place after the first."""
    m = Model()
    t = pd.RangeIndex(12, name="t")
    demand = pd.Series(np.linspace(1, 6, 12), index=t)
    gen = m.add_variables(0, 10, coords=[t], name="gen")
    m.add_constraints(gen >= demand, name="demand")
    m.add_objective((gen * pd.Series(np.arange(1.0, 13.0), index=t)).sum())

    windows = [pd.RangeIndex(i, i + 4, name="t") for i in (0, 4, 8)]
    w = m.restrict(t=windows[0])
    objectives = []
    for i, window in enumerate(windows):
        if i:
            w.advance(t=window)
        w.solve(
            solver_name=solver_name,
            disallow_rebuild=i > 0,
            warm_start=warm_start,
            **options,
        )
        assert w.objective.value is not None
        objectives.append(w.objective.value)
    assert w.solver is not None
    return objectives, w.solver


@pytest.mark.parametrize("warm_start", [True, False])
def test_matrix_windows_update_in_place(warm_start: bool) -> None:
    pytest.importorskip("highspy")
    objectives, solver = _solve_windows("gamspy", warm_start)
    # a cold start needs a generated model, so only warm windows freeze
    assert solver.solver_model.frozen is warm_start
    reference, _ = _solve_windows("highs", warm_start, output_flag=False)
    np.testing.assert_allclose(objectives, reference, atol=1e-6)


# ---------------------------------------------------------- GUSS scenario solves


def _variants() -> list[Model]:
    """Variants of ``_lp`` changing each kind of scenario data."""
    rhs = _lp()
    rhs.constraints["lower"].update(rhs=4.5)
    bounds = _lp()
    bounds.variables["x"].upper.values[...] = 3.5
    bounds.variables["y"].lower.values[...] = 0.5
    cost = _lp()
    x, y = cost.variables["x"], cost.variables["y"]
    cost.objective.expression = (3 * x).sum() + 5 * y
    coef = _lp()
    upper = coef.constraints["upper"]
    upper.update(coeffs=upper.coeffs * 3)
    mixed = _lp()
    mixed.constraints["lower"].update(rhs=5.0)
    mixed.variables["y"].upper.values[...] = 2.0
    return [_lp(), rhs, bounds, cost, coef, mixed]


def _scenario_solver(m: Model) -> GAMSPy:
    solver = Solver.from_name("gamspy", m, io_api="direct")
    assert isinstance(solver, GAMSPy)
    return solver


def test_scenarios_match_separate_solves() -> None:
    pytest.importorskip("highspy")
    solver = _scenario_solver(_lp())
    variants = _variants()
    results = solver.solve_scenarios(variants)
    assert len(results) == len(variants)
    for k, (variant, result) in enumerate(zip(variants, results)):
        assert result.status.is_ok, k
        variant.assign_result(result, solver=solver)
        _assert_matches_highs(variant, _highs(_variants()[k]))


def test_infeasible_scenario_does_not_affect_the_others() -> None:
    feasible = _lp()
    feasible.constraints["lower"].update(rhs=4.5)
    infeasible = _lp()
    infeasible.variables["x"].upper.values[...] = 1.0  # sum(x) == 9 impossible
    empty_row = _lp()

    solver = _scenario_solver(_lp())
    results = solver.solve_scenarios([feasible, infeasible, empty_row])
    assert results[0].status.is_ok
    assert results[1].status.termination_condition == "infeasible"
    assert results[1].solution is not None and np.isnan(results[1].solution.objective)
    assert results[2].status.is_ok
    assert results[0].solution is not None and results[2].solution is not None
    assert results[0].solution.objective > results[2].solution.objective


def test_scenario_violating_a_row_without_coefficients() -> None:
    def build(rhs: float) -> Model:
        m = _lp()
        m.add_constraints(0 * m.variables["x"].sel(i=0) <= rhs, name="empty")
        return m

    solver = _scenario_solver(build(1.0))
    ok, violated = solver.solve_scenarios([build(2.0), build(-1.0)])
    assert ok.status.is_ok
    assert violated.status.termination_condition == "infeasible"


def test_mixed_integer_scenarios() -> None:
    base = _mixed(_lp())
    variant = _mixed(_lp())
    variant.constraints["cover"].update(rhs=6.5)
    variant.variables["k"].upper.values[...] = 2.0

    solver = _scenario_solver(base)
    results = solver.solve_scenarios([_mixed(_lp()), variant])
    for model, result in zip((_mixed(_lp()), variant), results):
        assert result.status.is_ok
        assert result.solution is not None
        assert result.solution.dual.size == 0
        assert result.solution.objective == pytest.approx(
            _fresh_objective(model), abs=1e-6
        )


def test_models_that_are_no_variants_are_rejected() -> None:
    solver = _scenario_solver(_lp())

    added = _lp()
    added.add_constraints(added.variables["y"] <= 1, name="extra")
    with pytest.raises(ValueError, match="Model 1 is no variant"):
        solver.solve_scenarios([_lp(), added])

    flipped = _lp("max")
    with pytest.raises(ValueError, match="objective sense"):
        solver.solve_scenarios([flipped])

    signed = _lp()
    signed.constraints["lower"].update(sign="<=")
    with pytest.raises(ValueError, match="sign"):
        solver.solve_scenarios([signed])


def test_repeated_scenario_runs_and_regular_solves() -> None:
    m = _lp()
    solver = _scenario_solver(m)
    first = solver.solve_scenarios(_variants())
    second = solver.solve_scenarios(_variants()[::-1])
    for a, b in zip(first, second[::-1]):
        assert a.solution is not None and b.solution is not None
        assert a.solution.objective == pytest.approx(b.solution.objective)

    # the built model is untouched by the scenario data
    result = solver.solve()
    assert result.solution is not None
    assert result.solution.objective == pytest.approx(_fresh_objective(_lp()))


def test_scenarios_on_a_frozen_model() -> None:
    m = _lp()
    solver = _persistent(m)
    m.constraints["lower"].update(rhs=4.5)
    solver.solve(m, disallow_rebuild=True)
    assert solver.solver_model.frozen

    (result,) = solver.solve_scenarios([_variants()[2]])
    assert result.solution is not None
    assert result.solution.objective == pytest.approx(
        _fresh_objective(_variants()[2]), abs=1e-6
    )


# ---------------------------------------------------------------- remote backends


def test_backend_and_client_are_passed_to_gamspy(monkeypatch: Any) -> None:
    client = object()
    m = _lp()
    solver = Solver.from_name(
        "gamspy",
        m,
        io_api="direct",
        track_updates=True,
        options={"backend": "neos", "client": client, "solver": "cplex"},
    )
    gm = solver.solver_model
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(gm.model, "solve", lambda **kwargs: calls.append(kwargs))

    gm.freeze_requested = True
    solver.solve()
    assert calls[-1]["backend"] == "neos"
    assert calls[-1]["client"] is client
    assert calls[-1]["solver"] == "cplex"
    # frozen models only run locally
    assert not gm.frozen


def test_sigint_handler_is_restored_after_gams_runs() -> None:
    # GAMSPy replaces the process-wide SIGINT handler when it starts GAMS;
    # Ctrl-C must keep working for other solvers once GAMS has run.
    import signal

    before = signal.getsignal(signal.SIGINT)
    m = _lp()
    solver = _persistent(m)
    m.constraints["lower"].update(rhs=4.5)
    solver.solve(m, disallow_rebuild=True)  # freezes and runs frozen
    solver.solve_scenarios([_lp()])
    assert signal.getsignal(signal.SIGINT) is before
    GAMSPy._license_probe()
    assert signal.getsignal(signal.SIGINT) is before
