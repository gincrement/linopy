"""
A generic GAMSPy model built from linopy's flat matrices.

GAMS has no matrix interface: models are indexed equations over sets. A
:class:`GamsMatrixModel` therefore encodes a :class:`~linopy.matrices.Matrices`
as data. The columns form the set ``i`` and the rows the set ``j``; the
sparsity pattern of ``A`` is a static two-dimensional set and its values a
parameter, so one generic equation per constraint sign covers every row.

GAMS fixes the type of a variable per symbol, not per element. Columns are
therefore grouped by kind into one variable symbol each (continuous, integer,
semi-continuous), and every matrix-valued piece of data (``A``, ``c``, ``Q``)
is split along the same groups.

Keeping the pattern in sets rather than in ``$`` conditions on the values is
what allows a frozen GAMSPy model: the values can then be declared modifiable
and changed in place between solves.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import gamspy as gp

    from linopy.matrices import Matrices

SosSet = tuple[int, np.ndarray, np.ndarray]

# GAMS variable type of each column group, keyed by group name.
_GROUP_TYPES: dict[str, str] = {"c": "free", "i": "integer", "s": "semicont"}
# Column group of each linopy variable type (binaries are bounded integers).
_VTYPE_GROUPS: dict[str, str] = {"C": "c", "I": "i", "B": "i", "S": "s"}
_SIGNS: dict[str, str] = {"<": "le", ">": "ge", "=": "eq"}

# Model statuses for which GAMS returns a primal point.
_STATUSES_WITH_SOLUTION = frozenset({1, 2, 3, 6, 7, 8, 9, 15, 16, 17})


@dataclass
class _Group:
    """The columns of one variable kind and the symbols defined over them."""

    name: str
    positions: np.ndarray
    var: gp.Variable
    cols: gp.Set
    c: gp.Parameter
    nz: gp.Set | None = None
    A: gp.Parameter | None = None
    # entries of A in this group, sorted by their key ``row * n_cols + col``
    keys: np.ndarray | None = None
    rows: np.ndarray | None = None
    cols_of_entries: np.ndarray | None = None
    values: np.ndarray | None = None


def _total(terms: Iterable[Any]) -> Any:
    """Sum of GAMSPy expressions, without the integer start value of ``sum``."""
    return functools.reduce(operator.add, terms)


def _scatter(records: pd.DataFrame | None, column: str, size: int) -> np.ndarray:
    """
    Dense array of ``records[column]`` indexed by the integer element label.

    GAMS omits records at their default value, so missing positions stay 0.
    """
    out = np.zeros(size, dtype=float)
    if records is None or records.empty:
        return out
    labels = records.iloc[:, 0]
    if isinstance(labels.dtype, pd.CategoricalDtype):
        categories = labels.cat.categories.astype(np.int64)
        positions = np.asarray(categories)[labels.cat.codes.to_numpy()]
    else:
        positions = labels.to_numpy().astype(np.int64)
    out[positions] = records[column].to_numpy(dtype=float)
    return out


class GamsMatrixModel:
    """
    A GAMSPy container and model holding one linopy matrix problem.

    Parameters
    ----------
    M : Matrices
        The flat problem, in solver-scaled units.
    sos : sequence of (type, positions, weights), optional
        SOS sets over column positions.

    Attributes
    ----------
    container : gamspy.Container
        The GAMSPy container holding all symbols.
    model : gamspy.Model
        The GAMSPy model, solvable with any GAMS solver.
    problem : str
        GAMS model type: ``LP``, ``MIP``, ``QCP`` or ``MIQCP``.
    """

    def __init__(self, M: Matrices, *, sos: Sequence[SosSet] = ()) -> None:
        import gamspy as gp

        self.n_cols = len(M.vlabels)
        self.n_rows = len(M.clabels)
        if not self.n_cols:
            raise ValueError("Model has no variables, cannot export to GAMSPy.")
        self.objective_sense = M.objective_sense
        # Frozen-instance state; see freeze() and discard_start().
        self.frozen = False
        self.freeze_requested = False
        self.cold_start = False
        self._coefficients_changed = False
        self._frozen_with_coefficients = False
        # Copies of the data that in-place updates rewrite symbol-wise.
        self._lb = np.asarray(M.lb, dtype=float).copy()
        self._ub = np.asarray(M.ub, dtype=float).copy()
        self._c = np.asarray(M.c, dtype=float).copy()

        size = max([self.n_cols, self.n_rows, len(sos), *(len(p) for _, p, _ in sos)])
        # Element labels are the decimal positions, shared by every set so
        # that GAMS orders them numerically wherever they appear.
        self._uels = pd.Index(np.arange(size).astype(str))

        ct = gp.Container()
        self.container = ct
        self.i = gp.Set(ct, "i", records=self._elements(np.arange(self.n_cols)))
        self.j = gp.Set(ct, "j", records=self._elements(np.arange(self.n_rows)))
        self.ii = gp.Alias(ct, "ii", self.i)

        self._group_of = np.array([_VTYPE_GROUPS[str(t)] for t in M.vtypes], dtype="U1")
        self.groups: dict[str, _Group] = {}
        for name in _GROUP_TYPES:
            positions = np.flatnonzero(self._group_of == name)
            if positions.size:
                self.groups[name] = self._add_group(name, positions, M)

        A = M.A
        self._row_nnz = np.zeros(self.n_rows, dtype=np.int64)
        if A is not None and A.nnz:
            self._add_matrix(A.tocsr())
        self._b = np.asarray(M.b, dtype=float).copy()
        # Rows without coefficients stay out of the equations (see
        # trivially_infeasible), and out of b: a frozen instance rejects
        # records it has no row for.
        active = np.flatnonzero(self._row_nnz > 0)
        self.b = gp.Parameter(
            ct, "b", domain=self.j, records=self._values(active, self._b[active], "j")
        )
        self._sense = np.asarray(M.sense).astype("U1")

        self.equations: dict[str, gp.Equation] = {}
        for sign, suffix in _SIGNS.items():
            rows = np.flatnonzero((self._sense == sign) & (self._row_nnz > 0))
            if rows.size:
                self.equations[sign] = self._add_equation(sign, suffix, rows)

        self.sos_equations: list[gp.Equation] = []
        if sos:
            self._add_sos(sos, M)

        objective = self._objective(M)
        discrete = bool(set(self.groups) - {"c"}) or bool(sos)
        quadratic = M.Q is not None and M.Q.nnz > 0
        if quadratic:
            self.problem = "MIQCP" if discrete else "QCP"
        else:
            self.problem = "MIP" if discrete else "LP"
        self.model = gp.Model(
            ct,
            "linopy",
            equations=[*self.equations.values(), *self.sos_equations],
            problem=self.problem,
            sense="MAX" if M.objective_sense == "max" else "MIN",
            objective=objective,
        )

    # ------------------------------------------------------------------ records

    def _elements(self, positions: np.ndarray) -> pd.Categorical:
        return pd.Categorical.from_codes(
            np.asarray(positions, dtype=np.int64), categories=self._uels
        )

    def _values(
        self, positions: np.ndarray, values: np.ndarray, dim: str
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {dim: self._elements(positions), "value": np.asarray(values, float)}
        )

    def _pairs(
        self,
        first: np.ndarray,
        second: np.ndarray,
        dims: tuple[str, str],
        values: np.ndarray | None = None,
    ) -> pd.DataFrame:
        data: dict[str, Any] = {
            dims[0]: self._elements(first),
            dims[1]: self._elements(second),
        }
        if values is not None:
            data["value"] = np.asarray(values, dtype=float)
        return pd.DataFrame(data)

    def _bounds(
        self, positions: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "i": self._elements(positions),
                "lower": np.asarray(lower, dtype=float),
                "upper": np.asarray(upper, dtype=float),
            }
        )

    # ------------------------------------------------------------------- build

    def _add_group(self, name: str, positions: np.ndarray, M: Matrices) -> _Group:
        import gamspy as gp

        ct = self.container
        cols = gp.Set(
            ct, f"cols_{name}", domain=self.i, records=self._elements(positions)
        )
        var = gp.Variable(ct, f"x_{name}", type=_GROUP_TYPES[name], domain=self.i)
        var.setRecords(self._bounds(positions, M.lb[positions], M.ub[positions]))
        c = gp.Parameter(
            ct,
            f"c_{name}",
            domain=self.i,
            records=self._values(positions, M.c[positions], "i"),
        )
        return _Group(name, positions, var, cols, c)

    def _add_matrix(self, A: Any) -> None:
        import gamspy as gp

        rows = np.repeat(np.arange(A.shape[0]), np.diff(A.indptr))
        cols = np.asarray(A.indices)
        data = np.asarray(A.data, dtype=float)
        np.add.at(self._row_nnz, rows, 1)
        group_of_entry = self._group_of[cols]
        for name, group in self.groups.items():
            mask = group_of_entry == name
            if not mask.any():
                continue
            keys = rows[mask] * self.n_cols + cols[mask]
            order = np.argsort(keys, kind="stable")
            group.keys = keys[order]
            group.rows = rows[mask][order]
            group.cols_of_entries = cols[mask][order]
            group.values = data[mask][order]
            group.nz = gp.Set(
                self.container,
                f"nz_{name}",
                domain=[self.j, self.i],
                records=self._pairs(rows[mask], cols[mask], ("j", "i")),
            )
            group.A = gp.Parameter(
                self.container,
                f"A_{name}",
                domain=[self.j, self.i],
                records=self._pairs(rows[mask], cols[mask], ("j", "i"), data[mask]),
            )

    def _add_equation(self, sign: str, suffix: str, rows: np.ndarray) -> gp.Equation:
        import gamspy as gp

        ct = self.container
        js = gp.Set(ct, f"rows_{suffix}", domain=self.j, records=self._elements(rows))
        eq = gp.Equation(ct, f"e_{suffix}", domain=self.j)
        lhs = _total(
            gp.Sum(g.nz[js, self.i], g.A[js, self.i] * g.var[self.i])
            for g in self.groups.values()
            if g.nz is not None and g.A is not None
        )
        if sign == "<":
            eq[js] = lhs <= self.b[js]
        elif sign == ">":
            eq[js] = lhs >= self.b[js]
        else:
            eq[js] = lhs == self.b[js]
        return eq

    def _add_sos(self, sos: Sequence[SosSet], M: Matrices) -> None:
        """
        Add SOS1/SOS2 sets through auxiliary SOS variables linked to the columns.

        GAMS declares SOS membership by variable type, with the last index
        running over the members of one set. Each linopy set becomes a row
        ``k`` of an auxiliary ``x_sos{type}(k, m)`` variable, whose member index
        ``m`` is the rank by weight, since SOS2 adjacency follows the element
        order. An equality ties every member to its column; the auxiliary
        variables are free, so the column keeps its own bounds.
        """
        import gamspy as gp

        ct = self.container
        k = gp.Set(ct, "sos_k", records=self._elements(np.arange(len(sos))))
        m = gp.Set(
            ct,
            "sos_m",
            records=self._elements(np.arange(max(len(p) for _, p, _ in sos))),
        )
        for sos_type in sorted({t for t, _, _ in sos}):
            members = [
                (index, positions[np.argsort(weights, kind="stable")])
                for index, (t, positions, weights) in enumerate(sos)
                if t == sos_type
            ]
            set_index = np.concatenate([np.full(p.size, index) for index, p in members])
            rank = np.concatenate([np.arange(p.size) for _, p in members])
            column = np.concatenate([p for _, p in members])

            var = gp.Variable(
                ct, f"x_sos{sos_type}", type=f"sos{sos_type}", domain=[k, m]
            )
            var.setRecords(
                pd.DataFrame(
                    {
                        "sos_k": self._elements(set_index),
                        "sos_m": self._elements(rank),
                        "lower": -np.inf,
                        "upper": np.inf,
                    }
                )
            )
            group_of_member = self._group_of[column]
            for name, group in self.groups.items():
                mask = group_of_member == name
                if not mask.any():
                    continue
                link = gp.Set(
                    ct,
                    f"sos{sos_type}_map_{name}",
                    domain=[k, m, self.i],
                    records=pd.DataFrame(
                        {
                            "sos_k": self._elements(set_index[mask]),
                            "sos_m": self._elements(rank[mask]),
                            "i": self._elements(column[mask]),
                        }
                    ),
                )
                eq = gp.Equation(
                    ct, f"sos{sos_type}_link_{name}", domain=[k, m, self.i]
                )
                eq[k, m, self.i].where[link[k, m, self.i]] = (
                    var[k, m] == group.var[self.i]
                )
                self.sos_equations.append(eq)

    def _objective(self, M: Matrices) -> Any:
        import gamspy as gp

        terms: list[Any] = [
            gp.Sum(g.cols, g.c[g.cols] * g.var[g.cols]) for g in self.groups.values()
        ]
        if M.Q is not None and M.Q.nnz:
            Q = M.Q.tocoo()
            rows, cols, data = (
                np.asarray(Q.row),
                np.asarray(Q.col),
                np.asarray(Q.data, float),
            )
            keep = data != 0
            rows, cols, data = rows[keep], cols[keep], data[keep]
            row_group, col_group = self._group_of[rows], self._group_of[cols]
            for gk, first in self.groups.items():
                for gl, second in self.groups.items():
                    mask = (row_group == gk) & (col_group == gl)
                    if not mask.any():
                        continue
                    pairs = (rows[mask], cols[mask], ("i", "ii"))
                    qnz = gp.Set(
                        self.container,
                        f"qnz_{gk}{gl}",
                        domain=[self.i, self.ii],
                        records=self._pairs(*pairs),
                    )
                    q = gp.Parameter(
                        self.container,
                        f"Q_{gk}{gl}",
                        domain=[self.i, self.ii],
                        records=self._pairs(*pairs, data[mask]),
                    )
                    terms.append(
                        0.5
                        * gp.Sum(
                            qnz[self.i, self.ii],
                            q[self.i, self.ii]
                            * first.var[self.i]
                            * second.var[self.ii],
                        )
                    )
        return _total(terms)

    # ------------------------------------------------------------------- solve

    def trivially_infeasible(self, tol: float = 1e-9) -> bool:
        """
        Whether a row without coefficients cannot hold.

        Such rows are left out of the GAMS equations, where GAMS would abort
        with an execution error instead of reporting infeasibility.
        """
        empty = self._row_nnz == 0
        if not empty.any():
            return False
        b, sense = self._b[empty], self._sense[empty]
        violated = (
            ((sense == "<") & (b < -tol))
            | ((sense == ">") & (b > tol))
            | ((sense == "=") & (np.abs(b) > tol))
        )
        return bool(violated.any())

    def solve(self, **kwargs: Any) -> Any:
        """Solve the GAMSPy model; keyword arguments go to ``gamspy.Model.solve``."""
        return self.model.solve(**kwargs)

    # ---------------------------------------------------------- in-place updates

    def set_bounds(
        self, positions: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> None:
        """Change the bounds of the columns at ``positions``."""
        self._lb[positions] = lower
        self._ub[positions] = upper
        for g in self._groups_of(positions):
            g.var.setRecords(
                self._bounds(g.positions, self._lb[g.positions], self._ub[g.positions])
            )

    def set_rhs(self, rows: np.ndarray, values: np.ndarray) -> None:
        """Change the right-hand side of the rows at ``rows``."""
        self._b[rows] = values
        active = np.flatnonzero(self._row_nnz > 0)
        self.b.setRecords(self._values(active, self._b[active], "j"))

    def set_objective(self, positions: np.ndarray, values: np.ndarray) -> None:
        """Change the linear objective coefficients of the columns at ``positions``."""
        self._c[positions] = values
        for g in self._groups_of(positions):
            g.c.setRecords(self._values(g.positions, self._c[g.positions], "i"))

    def _locate(
        self, rows: np.ndarray, cols: np.ndarray
    ) -> list[tuple[_Group, np.ndarray, np.ndarray]]:
        """Per group: which of the entries lie in it, and their slots there."""
        keys = np.asarray(rows, dtype=np.int64) * self.n_cols + np.asarray(cols)
        group_of_entry = self._group_of[np.asarray(cols)]
        found = []
        for name, g in self.groups.items():
            mask = group_of_entry == name
            if not mask.any() or g.keys is None:
                found.append((g, mask, np.full(int(mask.sum()), -1)))
                continue
            slots = np.searchsorted(g.keys, keys[mask])
            slots[slots >= g.keys.size] = 0
            slots[g.keys[slots] != keys[mask]] = -1
            found.append((g, mask, slots))
        return found

    def in_pattern(
        self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray
    ) -> bool:
        """
        Whether all non-zero entries lie in the sparsity pattern of ``A``.

        The pattern is fixed when the model is built; zeros outside it are
        no-ops and need no slot.
        """
        values = np.asarray(values)
        for _, mask, slots in self._locate(rows, cols):
            if ((slots < 0) & (values[mask] != 0)).any():
                return False
        return True

    def set_coefficients(
        self, rows: np.ndarray, cols: np.ndarray, values: np.ndarray
    ) -> None:
        """Change entries of ``A`` within its pattern (see :meth:`in_pattern`)."""
        values = np.asarray(values, dtype=float)
        for g, mask, slots in self._locate(rows, cols):
            hit = slots >= 0
            if not hit.any():
                continue
            assert g.values is not None and g.A is not None
            assert g.rows is not None and g.cols_of_entries is not None
            g.values[slots[hit]] = values[mask][hit]
            self._coefficients_changed = True
            g.A.setRecords(self._pairs(g.rows, g.cols_of_entries, ("j", "i"), g.values))

    def set_sense(self, sense: str) -> None:
        """
        Change the objective sense to ``"min"`` or ``"max"``.

        A frozen instance fixes the sense, so it is released; the next
        :meth:`freeze` generates it again.
        """
        import gamspy as gp

        self.close()
        self.model.sense = gp.Sense.MAX if sense == "max" else gp.Sense.MIN
        self.objective_sense = sense

    def _groups_of(self, positions: np.ndarray) -> list[_Group]:
        names = set(np.unique(self._group_of[np.asarray(positions)]).tolist())
        return [g for name, g in self.groups.items() if name in names]

    def freeze(self) -> None:
        """
        Freeze the model, so that later solves update the generated instance.

        The data that :meth:`set_bounds`, :meth:`set_rhs` and
        :meth:`set_objective` change is declared modifiable, which saves GAMS
        the model generation on every solve. The matrix ``A`` is only declared
        modifiable once :meth:`set_coefficients` has changed it: GAMS generates
        a modifiable coefficient as a placeholder, which makes the instance
        larger, so a model whose coefficients never change is frozen without.
        Calling it again refreezes only when ``A`` has to be added.
        """
        coefficients = self._coefficients_changed
        if self.frozen and (self._frozen_with_coefficients or not coefficients):
            return
        self.close()
        modifiables: list[Any] = [self.b]
        for g in self.groups.values():
            modifiables += [g.c, g.var.lo, g.var.up]
            if coefficients and g.A is not None:
                modifiables.append(g.A)
        self.model.freeze(modifiables=modifiables)
        self.frozen = True
        self._frozen_with_coefficients = coefficients

    def discard_start(self) -> None:
        """
        Make the next solve start from scratch instead of the last solution.

        A frozen instance always restarts from its last solution, so it is
        released; the next solve then generates the model anew and rejects
        any advanced basis (``basis_detection_threshold=1``, GAMS ``bratio``).
        """
        self.close()
        self.cold_start = True

    def close(self) -> None:
        """Release the frozen model instance, if any."""
        if self.frozen:
            self.model.unfreeze()
            self.frozen = False

    @property
    def has_solution(self) -> bool:
        status = self.model.status
        return status is not None and status.value in _STATUSES_WITH_SOLUTION

    def primal(self) -> np.ndarray:
        """Column values by position."""
        out = np.zeros(self.n_cols, dtype=float)
        for g in self.groups.values():
            values = _scatter(g.var.records, "level", self.n_cols)
            out[g.positions] = values[g.positions]
        return out

    def dual(self) -> np.ndarray:
        """Row marginals by position; rows without coefficients get 0."""
        out = np.zeros(self.n_rows, dtype=float)
        for eq in self.equations.values():
            out += _scatter(eq.records, "marginal", self.n_rows)
        return out
