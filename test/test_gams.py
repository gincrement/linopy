"""Tests specific to the GAMS command line solver backend."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from linopy import Model, Variable
from linopy.constants import TerminationCondition
from linopy.solvers import GAMS, Solver, _gams_termination_condition

pytestmark = pytest.mark.skipif(
    not GAMS.is_available() or not GAMS.license_status().ok,
    reason="no licensed GAMS system found",
)

FILE_IO_APIS = ["lp", "mps"]


def _lp(sense: str = "min") -> Model:
    m = Model()
    x = m.add_variables(0, 10, coords=[pd.RangeIndex(3, name="i")], name="x")
    y = m.add_variables(-5, 5, name="y")
    m.add_constraints(x + y >= 4, name="lower")
    m.add_constraints(2 * x - y <= 20, name="upper")
    m.add_constraints(x.sum() == 9, name="total")
    m.add_objective((x * np.array([1, 2, 3])).sum() + 2 * y, sense=sense)
    return m


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


def _highs(m: Model) -> Model:
    pytest.importorskip("highspy")
    m.solve("highs", output_flag=False)
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


@pytest.mark.parametrize("io_api", FILE_IO_APIS)
@pytest.mark.parametrize("sense", ["min", "max"])
def test_lp_matches_highs(sense: str, io_api: str) -> None:
    m = _lp(sense)
    assert m.solve("gams", io_api=io_api) == ("ok", "optimal")
    _assert_matches_highs(m, _highs(_lp(sense)))


@pytest.mark.parametrize("io_api", ["lp"])
def test_mixed_integer_matches_highs(io_api: str) -> None:
    m = _mixed(_lp())
    assert m.solve("gams", io_api=io_api) == ("ok", "optimal")
    _assert_matches_highs(m, _highs(_mixed(_lp())), duals=False)
    assert m.solver is not None and m.solver.report is not None
    assert m.solver.report.dual_bound == pytest.approx(m.objective.value, rel=1e-4)
    assert m.solver.report.mip_gap == pytest.approx(0, abs=1e-4)


def test_mixed_integer_has_no_duals() -> None:
    m = _mixed(_lp())
    m.solve("gams")
    with pytest.raises(AttributeError, match="does not have dual values"):
        m.constraints["lower"].dual


@pytest.mark.parametrize("io_api", FILE_IO_APIS)
def test_quadratic_duals_match_highs(io_api: str) -> None:
    def build() -> Model:
        m = _lp()
        x = m.variables["x"]
        m.objective = m.objective.expression + (x * x).sum()
        return m

    m = build()
    assert m.solve("gams", io_api=io_api) == ("ok", "optimal")
    _assert_matches_highs(m, _highs(build()))


def test_semi_continuous_zero_branch() -> None:
    m = Model()
    x = m.add_variables(1, 10, name="x", semi_continuous=True)
    m.add_constraints(x <= 0.5, name="cap")
    m.add_objective(x, sense="max")
    assert m.solve("gams") == ("ok", "optimal")
    assert float(x.solution) == pytest.approx(0)


@pytest.mark.parametrize("io_api", FILE_IO_APIS)
@pytest.mark.parametrize("fixval", [0, 1])
def test_fixed_binary_is_held(io_api: str, fixval: int) -> None:
    # mps2gms resets the bounds of binaries in an LP file; linopy restores them
    m = Model()
    v = m.add_variables(name="v", binary=True)
    x = m.add_variables(0, 10, name="x")
    m.add_constraints(x + v >= 2, name="c")
    m.add_objective(x + 3 * v if fixval else x - 3 * v)
    v.fix(fixval)
    assert m.solve("gams", io_api=io_api) == ("ok", "optimal")
    assert float(v.solution) == pytest.approx(fixval)


def test_infeasible() -> None:
    m = _lp()
    m.add_constraints(m.variables["x"].sum() >= 100, name="impossible")
    assert m.solve("gams") == ("warning", "infeasible")


def test_violated_row_without_coefficients_is_infeasible(tmp_path: Path) -> None:
    # linopy writes no empty rows, but other LP files may hold them; GAMS then
    # aborts the solve instead of passing the model to a solver
    problem_fn = tmp_path / "empty_row.lp"
    problem_fn.write_text(
        "min\nobj: x0\nst\nc0: x0 >= 1\nc1: 0 x0 >= 1\nbounds\nx0 <= 10\nend\n"
    )
    result = Solver.from_file("gams", problem_fn).solve()
    assert result.status.termination_condition == TerminationCondition.infeasible


def test_masked_entries_are_skipped() -> None:
    m = Model()
    mask = pd.Series([True, False, True])
    x = m.add_variables(0, 10, coords=[pd.RangeIndex(3, name="i")], name="x", mask=mask)
    m.add_constraints(x >= pd.Series([1.0, 2.0, 3.0]).rename_axis("i"), name="c")
    m.add_objective(x.sum())
    m.solve("gams")
    np.testing.assert_allclose(x.solution.values, [1, np.nan, 3])


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
    assert m.solve("gams", io_api="lp", reformulate_sos=True) == ("ok", "optimal")
    np.testing.assert_allclose(build.solution.values, expected, atol=1e-6)
    assert m.objective.value == pytest.approx(objective)


def test_sos_on_binaries_is_rejected_by_mps2gms() -> None:
    # GAMS SOS variables are continuous; reformulate_sos=True solves this
    m, _ = _sos_model(1, [1, 2, 3])
    with pytest.raises(RuntimeError, match="not continuous"):
        m.solve("gams", io_api="lp")


def test_sos2_adjacency_follows_declared_order() -> None:
    # with coords 0, 2, 1 the neighbours are positions (0, 1) and (1, 2): the
    # best pair is coords 0.0 and 2.0, worth 3 + 1.5
    m = Model()
    seg = pd.Index([0.0, 2.0, 1.0], name="seg")
    lam = m.add_variables(0, 1, coords=[seg], name="lam")
    with pytest.warns(UserWarning, match="neither ascend nor descend"):
        m.add_sos_constraints(lam, sos_type=2, sos_dim="seg")
    m.add_constraints(lam.sum() <= 1.5, name="budget")
    m.add_objective(lam * np.array([3.0, 3.0, 1.0]), sense="max")
    assert m.solve("gams", io_api="lp") == ("ok", "optimal")
    assert m.objective.value == pytest.approx(4.5)
    assert float(lam.solution.sel(seg=1.0)) == pytest.approx(0, abs=1e-9)


def test_sos2_adjacency_follows_ascending_coords() -> None:
    m = Model()
    seg = pd.Index([0.0, 1.0, 2.0], name="seg")
    lam = m.add_variables(0, 1, coords=[seg], name="lam")
    m.add_sos_constraints(lam, sos_type=2, sos_dim="seg")
    m.add_constraints(lam.sum() <= 1.5, name="budget")
    m.add_objective(lam * np.array([3.0, 1.0, 3.0]), sense="max")
    assert m.solve("gams", io_api="lp") == ("ok", "optimal")
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
    assert m.solve("gams", io_api="lp") == ("ok", "optimal")
    np.testing.assert_allclose(x.solution.values, [0, 5, 0], atol=1e-5)
    assert m.objective.value == pytest.approx(-25)


def test_from_file_solves_without_model(tmp_path: Path) -> None:
    problem_fn = tmp_path / "problem.lp"
    _lp().to_file(problem_fn)
    result = Solver.from_file("gams", problem_fn).solve()
    assert result.status.is_ok
    assert result.solution is not None
    assert result.solution.objective == pytest.approx(_highs(_lp()).objective.value)


def test_solver_options_reach_the_option_file(tmp_path: Path) -> None:
    log_fn = tmp_path / "gams.log"
    m = _lp()
    status = m.solve(
        "gams", solver="cplex", solver_options={"lpmethod": 2}, log_fn=log_fn
    )
    assert status == ("ok", "optimal")
    log = log_fn.read_text()
    assert "CPLEX" in log
    assert "Setting 'lpmethod': 2" in log


def test_solver_options_need_a_solver() -> None:
    with pytest.raises(ValueError, match="solver"):
        _lp().solve("gams", solver_options={"lpmethod": 2})


def test_command_line_parameters_are_passed(tmp_path: Path) -> None:
    log_fn = tmp_path / "gams.log"
    m = _lp()
    assert m.solve("gams", reslim=123, log_fn=log_fn) == ("ok", "optimal")
    assert "ResLim 123" in log_fn.read_text()


def test_solution_fn_receives_the_gdx(tmp_path: Path) -> None:
    solution_fn = tmp_path / "out" / "solution.gdx"
    m = _lp()
    m.solve("gams", solution_fn=solution_fn, keep_files=True)
    assert solution_fn.stat().st_size > 0


def test_unknown_gams_solver_raises() -> None:
    with pytest.raises(RuntimeError, match="(?i)solver"):
        _lp().solve("gams", solver="no_such_solver")


def test_missing_sysdir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No GAMS system"):
        _lp().solve("gams", sysdir=tmp_path)


def test_runs_without_gamspy(tmp_path: Path) -> None:
    script = textwrap.dedent(
        """
        import sys
        sys.modules["gamspy"] = None  # any import of gamspy fails
        import linopy
        m = linopy.Model()
        x = m.add_variables(0, 10, name="x")
        m.add_constraints(x >= 2, name="c")
        m.add_objective(x)
        assert m.solve("gams") == ("ok", "optimal")
        assert float(x.solution) == 2
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=tmp_path)


@pytest.mark.parametrize(
    ("model_status", "solve_status", "expected"),
    [
        (1, 1, TerminationCondition.optimal),
        (8, 1, TerminationCondition.optimal),
        (8, 3, TerminationCondition.time_limit),
        (19, 1, TerminationCondition.infeasible),
        (18, 1, TerminationCondition.unbounded),
        (11, 7, TerminationCondition.licensing_problems),
        (None, None, TerminationCondition.unknown),
    ],
)
def test_status_mapping(
    model_status: int | None, solve_status: int | None, expected: TerminationCondition
) -> None:
    assert _gams_termination_condition(model_status, solve_status) == expected
