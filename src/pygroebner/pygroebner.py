"""Algebra over F2 based on a Groebner basis.

Monomials are modeled by sparse vectors ((g1, e1), (g2, e2), ...).
The Groebner basis is truncated by a predicate function."""

from __future__ import annotations

import copy
import sqlite3
import time
from bisect import bisect_left
from collections import defaultdict
from itertools import (
    chain,
    repeat,
    combinations,
    groupby,
    combinations_with_replacement,
)
from typing import (
    Type,
    Iterable,
    NamedTuple,
    Callable,
    Any,
    Optional,
)

from . import base_algebra as BA
from .mymath import (
    Vector,
    add_dtuple,
    sub_dtuple,
    div_mod_dtuple,
    tex_parenthesis,
    tex_pow,
    two_expansion,
    get_one_element,
)


class Gen(NamedTuple):
    name: str
    deg: Vector


class DgaGen(NamedTuple):
    name: str
    deg: Vector
    diff: set


# ---------- Predicate functions/factories ----------
def pred_always_true(_):
    return True


def pred_d0(d: int):
    """Factory for predicate functions."""

    def pred(deg):
        return deg[0] <= d

    return pred


def pred_di(d: int, i: int):
    """Factory for predicate functions."""

    def pred(deg):
        return deg[i] <= d

    return pred


# ---------- Key functions ----------
def key_lex(m):
    return tuple((-i, -e) for i, e in m)


class AlgGb(BA.AlgebraMod2):
    """A factory for algebras based on Groebner basis.

    `new_alg()` creates a new algebra which is a subclass of `AlgGb` nad has
    its own generators and relations.
    """

    gens = None  # type: dict[int, Gen]
    rels = None  # type: dict[tuple, set]
    _rels_buffer = None  # type: dict[int, list[set]]
    key_mo = None  # type: Callable[[tuple], Any]
    pred = None  # type: Callable[[tuple], Any]
    dim_grading = None  # type: int

    _index_subclass = 0

    @classmethod
    def copy_alg(cls) -> Type[AlgGb]:
        """Return a copy of current algebra."""
        A = new_alg(key_mo=cls.key_mo, pred=cls.pred)
        A.gens = cls.gens.copy()
        A.rels = copy.deepcopy(cls.rels)
        A._rels_buffer = copy.deepcopy(cls._rels_buffer)
        A.dim_grading = cls.dim_grading
        return A

    @classmethod
    def save_alg(
        cls, filename: str, tablename: str, grading: list[str] = None
    ):  # TODO: implement
        """Save to a database."""
        conn = sqlite3.connect(filename)
        c = conn.cursor()

        if grading is None:
            grading = [f"d{i}" for i in range(cls.dim_grading)]
        assert len(grading) == cls.dim_grading
        sql_grading = ", ".join(f"{d} SMALL INT" for d in grading)

        c.execute("CREATE TABLE IF NOT EXISTS info (key TEXT PRIMARY KEY, value TEXT)")
        c.execute(
            f"CREATE TABLE IF NOT EXISTS {tablename}_generators (gen_id INTEGER PRIMARY KEY, gen_name TEXT UNIQUE, {sql_grading})"
        )
        c.execute(
            f"CREATE TABLE IF NOT EXISTS {tablename}_relations (leading_term TEXT, basis TEXT, {sql_grading})"
        )
        c.execute(f"DELETE FROM info")
        c.execute(f"DELETE FROM {tablename}_generators")
        c.execute(f"DELETE FROM {tablename}_relations")

        t = time.localtime()
        sql_date = f"{t.tm_mon:02}/{t.tm_mday:02}/{t.tm_year} {t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}"
        sql_mo = {None: "Revlex", "Lex": "Lex"}
        info = {
            "version": "0.0",
            "mo": (sql_mo[cls.key_mo] if cls.key_mo in sql_mo else ""),
            "date": sql_date,
            "grading": ",".join(grading),
        }
        c.executemany("INSERT INTO info (key, value) VALUES (?, ?);", info.items())

        c.executemany(
            f"INSERT INTO {tablename}_generators (gen_id, gen_name, {', '.join(grading)}) values ({','.join('?' for i in range(cls.dim_grading + 2))})",
            ((i, g.name, *g.deg) for i, g in cls.gens.items()),
        )
        c.executemany(
            f"INSERT INTO {tablename}_relations (leading_term, basis, {', '.join(grading)}) values ({','.join('?' for i in range(cls.dim_grading + 2))})",
            (
                (cls.str_sqlite3_mon(m), cls.str_sqlite3_data(b), *cls.deg_mon(m))
                for m, b in cls.rels.items()
            ),
        )

        c.close()
        conn.commit()
        conn.close()

    @classmethod
    def latex_alg(cls) -> str:
        """For latex."""
        result = "\\section{Gens}\n\n"
        gens = defaultdict(list)
        for g in cls.gens.values():
            result += f"${g.name}$ {g.deg}\n\n"
        result += "\n\\section{Relations}\n\n"
        for m in cls.rels:
            result += f"${cls(m)} = {cls(cls.rels[m])}$\n\n"
        return result

    @classmethod
    def to_dga(cls, deg_diff: tuple) -> Type["DgaGb"]:
        class_name = f"GbDGA_{DgaGb._index_subclass}"
        DgaGb._index_subclass += 1
        dct = {
            "gens": {i: DgaGen(*g, None) for i, g in cls.gens.items()},
            "rels": copy.deepcopy(cls.rels),
            "_rels_buffer": copy.deepcopy(cls._rels_buffer),
            "key_mo": cls.key_mo,
            "pred": cls.pred,
            "dim_grading": cls.dim_grading,
            "deg_diff": Vector(deg_diff),
        }
        return type(class_name, (DgaGb,), dct)

    # --------- SQLite3 Interfaces --------------
    @classmethod
    def str_sqlite3_mon(cls, mon: tuple):
        """Convert mon to a string in a database."""
        return ",".join(f"{i},{-e}" for i, e in mon)

    @classmethod
    def str_sqlite3_data(cls, data: set):
        """Convert self to a string to be stored in sqlite."""
        return ";".join(
            cls.str_sqlite3_mon(mon)
            for mon in sorted(data, key=cls.key_mo, reverse=True)
        )

    @classmethod
    def mon_sqlite3(cls, s: str):
        """Load mon from a string from a database"""
        if not s:
            return tuple()
        mon = tuple(map(int, s.split(",")))
        it_zip = zip((it := iter(mon)), it)
        return tuple((i, -e) for i, e in it_zip)

    @classmethod
    def data_sqlite3(cls, s: str):
        """Load data from a string from a database"""
        if not s:
            return cls.zero_data()
        data = set()
        for str_mon in s.split(";"):
            mon = tuple(map(int, str_mon.split(",")))
            it_zip = zip((it := iter(mon)), it)
            mon = tuple((i, -e) for i, e in it_zip)
            data.add(mon)
        return data

    # ---------- AlgebraMod2 ----------
    @classmethod
    def mul_mons(cls, mon1: tuple, mon2: tuple):
        m = add_dtuple(mon1, mon2)
        return cls.reduce_data({m})

    @classmethod
    def str_mon(cls, mon: tuple):
        if mon:
            return "".join(tex_pow(cls.gens[i].name, -e) for i, e in mon)
        else:
            return "1"

    @classmethod
    def repr_mon(cls, mon: tuple, clsname: str):
        """clsname should be the variable name of the algebra"""
        if mon:
            return " * ".join(
                f'{clsname}.gen("{cls.gens[i].name}") ** {-e}'
                if -e > 1
                else f'{clsname}.gen("{cls.gens[i].name}")'
                for i, e in mon
            )
        else:
            return f"{clsname}.unit()"

    # ---------- Degree ----------
    @classmethod
    def deg_mon(cls, mon: tuple) -> Vector:
        return sum(
            (cls.gens[i].deg * (-e) for i, e in mon), Vector.zero(cls.dim_grading)
        )

    @classmethod
    def deg0_mon(cls, mon: tuple) -> int:
        return sum((cls.gens[i].deg[0] * (-e) for i, e in mon))

    @classmethod
    def deg_data(cls, data: set) -> Optional[Vector]:
        """Return the degree of `data`."""
        for m in data:
            return cls.deg_mon(m)

    @classmethod
    def deg0_data(cls, data: set) -> int:
        """Return the first degree of `data`."""
        for m in data:
            return cls.deg0_mon(m)

    def deg(self) -> Optional[Vector]:
        """Require `self` to be homogeneous."""
        for m in self.data:
            return self.deg_mon(m)

    def deg0(self) -> Optional[Vector]:
        """Require `self` to be homogeneous."""
        for m in self.data:
            return self.deg0_mon(m)

    # ---------- Generators ----------
    @classmethod
    def gen(cls, k: int | str):
        """Return a generator."""
        if type(k) is int:
            return cls(((k, -1))).reduce()
        for i, g in cls.gens.items():
            if g.name == k:
                m = ((i, -1),)
                return cls(m).reduce()
        else:
            raise BA.MyKeyError(f"No generator named {k}")

    @classmethod
    def add_gens(cls, name_deg_s):
        """Add gens. name_deg_s is a list of tuples (name, deg)."""
        id = max(cls.gens, default=-1) + 1
        for name, deg in name_deg_s:
            if type(deg) is int:
                deg = Vector((deg,))
            if cls.dim_grading is None:
                cls.dim_grading = len(deg)
            if cls.pred(deg):
                cls.gens[id].append(Gen(name, deg))
                id += 1

    @classmethod
    def add_gen(cls, name: str, deg: int | tuple | Vector) -> AlgGb | None:
        """Add a new generator and return it.

        Return None if `deg` is too big."""
        if type(deg) is int:
            deg = Vector((deg,))
        elif type(deg) is tuple:
            deg = Vector(deg)
        cls.dim_grading = len(deg)
        if cls.pred(deg):
            id = max(cls.gens, default=-1) + 1
            cls.gens[id] = Gen(name, deg)
            m = ((id, -1),)
            return cls(m).reduce()

    @classmethod
    def remove_gen(cls, name: str):
        """Remove a generator.

        If the generator `name` equals zero in the algebra whose relations are simplified,
        call this function to remove this generator."""
        assert len(cls._rels_buffer) == 0
        for i, g in cls.gens.items():
            if g.name == name:
                break
        else:
            raise ValueError(f"no generator named {name}")
        m = ((i, -1),)
        assert cls.rels[m] == set()
        del cls.rels[m]
        del cls.gens[i]

    @classmethod
    def rename_gen(cls, old_name, new_name):
        """Rename a generator."""
        for i, g in cls.gens.items():
            if g.name == old_name:
                break
        else:
            raise ValueError(f"no generator named {old_name}")
        cls.gens[i] = Gen(new_name, g.deg)

    @classmethod
    def reorder_gens(cls, index_map: dict = None, key_mo=None):
        """Reorganize the relations by a new ordering of gens and a new key function.
        The old i'th generator is the new `index_map[i]`'th generator."""
        A = new_alg(pred=cls.pred, key_mo=key_mo)
        num_gens = len(cls.gens)
        if index_map:

            def f(m):
                return tuple(sorted((index_map[_i], _e) for _i, _e in m))

            assert num_gens == len(index_map)
            rels_A = [{f(m)} | {f(m1) for m1 in cls.rels[m]} for m in cls.rels]

            index_map_inv = {}
            for i, fi in index_map.items():
                index_map_inv[fi] = i
            A.gens = {index_map_inv[i]: cls.gens[i] for i in cls.gens}
        else:
            A.gens = cls.gens.copy()
        A.add_rels_data(rels_A, clear_cache=True)
        return A

    def is_gen(self):
        if len(self.data) == 1:
            for m in self.data:
                if sum(-e for i, e in m) == 1:
                    return True
        return False

    # ---------- Relations ----------
    @classmethod
    def add_rels_buffer(cls, d_max: int = None):
        """Add relations from cache up to degree `d_max`."""
        while cls._rels_buffer:
            d = min(cls._rels_buffer)
            if d_max and d > d_max:
                break

            # reduce new relations in degree d
            rels_reduced = []
            leads = []
            for rel in map(cls.reduce_data, cls._rels_buffer[d]):
                for i, m in enumerate(leads):
                    if m in rel:
                        rel ^= rels_reduced[i]
                if rel:
                    rels_reduced.append(rel)
                    leads.append(cls.get_lead(rel))

            # add these relations
            for m, r in zip(leads, rels_reduced):
                for m1, b1 in cls.rels.items():
                    if gcd_nonzero_dtuple(m, m1):
                        lcm = max_dtuple(m, m1)
                        deg_lcm = cls.deg_mon(lcm)
                        if cls.pred(deg_lcm):
                            dif = sub_dtuple(lcm, m)
                            dif1 = sub_dtuple(lcm, m1)
                            new_rel = ({add_dtuple(_m, dif) for _m in r} - {lcm}) ^ {
                                add_dtuple(_m, dif1) for _m in b1
                            }
                            if new_rel:
                                cls._rels_buffer[deg_lcm[0]].append(new_rel)
                cls.rels[m] = r - {m}

            del cls._rels_buffer[d]

    @classmethod
    def add_rel_data(cls, rel: set, d_max=None):
        """Add relations."""
        if rel:
            deg = cls.deg_data(rel)
            if cls.pred(deg):
                cls._rels_buffer[deg[0]].append(rel)
                cls.add_rels_buffer(d_max)

    @classmethod
    def add_rels_data(cls, rels: Iterable[set], d_max=None):
        """Add relations."""
        for rel in sorted(rels, key=cls.deg_data):
            cls.add_rel_data(rel, cls.deg_data(rel)[0])
        cls.add_rels_buffer(d_max)

    @classmethod
    def add_rel(cls, rel: AlgGb, d_max=None):
        """Add a relation."""
        if not rel.is_homo():
            raise ValueError(f"Relation {rel} not homogeneous!")
        cls.add_rel_data(rel.data, d_max)

    @classmethod
    def add_rels(cls, rels: Iterable[AlgGb], d_max=None):
        """Add a relation."""
        for rel in rels:
            if not rel.is_homo():
                raise ValueError(f"relation {rel} not homogeneous!")
        cls.add_rels_data((rel.data for rel in rels), d_max)

    # ---------- Reduction ----------
    @classmethod
    def get_lead(cls, data):
        """Return the leading term of `data`."""
        return max(data, key=cls.key_mo) if cls.key_mo else max(data)

    @classmethod
    def reduce_data(cls, data: set) -> set:
        """Return reduced `data`. `data` will not be changed."""
        s = data.copy()
        result = set()
        while s:
            mon = cls.get_lead(s)
            s.remove(mon)
            for m in cls.rels:
                if le_dtuple(m, mon):
                    q, r = div_mod_dtuple(mon, m)
                    m_to_q = cls.pow_data(cls.rels[m], q)
                    s ^= {add_dtuple(r, m1) for m1 in m_to_q}
                    break
            else:
                result ^= {mon}
        return result

    def reduce(self) -> AlgGb:
        """Simplify self by relations."""
        self.data = self.reduce_data(self.data)
        return self

    @classmethod
    def reduce_rels(cls):
        """Simplify `cls.rels`."""
        assert len(cls._rels_buffer) == 0
        rels, cls.rels = cls.rels, {}
        for m in sorted(rels, key=cls.deg0_mon):
            rel = cls.reduce_data({m} | rels[m])
            if rel:
                m1 = cls.get_lead(rel)
                cls.rels[m1] = rel - {m1}

    @classmethod
    def basis_mons(cls, pred=None, basis: dict[Vector, list[tuple]] = None):
        """Return a list of basis grouped by degree."""
        pred = pred or cls.pred
        result = basis or defaultdict(list, {Vector.zero(cls.dim_grading): [()]})
        old_ds = set(result)
        leadings = sorted(cls.rels, key=lambda _m: _m[-1][0])
        leadings = {
            index: list(g) for index, g in groupby(leadings, key=lambda _m: _m[-1][0])
        }
        for id, gen in cls.gens.items():
            ds = list(result)
            for d in ds:
                if (d_ := d + gen.deg) not in old_ds and pred(d_):
                    for m in result[d]:
                        i_m = m[-1][0] if m else -1
                        if id == i_m and d in old_ds:
                            e = 1
                            while pred(d1 := d + gen.deg * e):
                                m1 = m[:-1] + ((m[-1][0], m[-1][1] - e),)
                                if id in leadings and any(
                                    map(le_dtuple, leadings[id], repeat(m1))
                                ):
                                    break
                                result[d1].append(m1)
                                e += 1
                        elif id > i_m:
                            e = 1
                            while pred(d1 := d + gen.deg * e):
                                m1 = m + ((id, -e),)
                                if id in leadings and any(
                                    map(le_dtuple, leadings[id], repeat(m1))
                                ):
                                    break
                                elif d1 in result:
                                    result[d1].append(m1)
                                else:
                                    result[d1] = [m1]
                                e += 1
        return result

    @classmethod
    def basis(cls, pred=None):
        if type(pred) is int:
            basis = cls.basis_mons(pred=lambda d3d: d3d[0] <= pred)
        else:
            basis = cls.basis_mons(pred)
        for basis_d in basis.values():
            for m in basis_d:
                yield cls(m)

    @classmethod
    def is_reducible(cls, mon):
        """Determine if mon is reducible by `cls.rels`."""
        return any(le_dtuple(m, mon) for m in cls.rels)

    # ---------- Algorithms ----------
    @classmethod
    def indecomposables(cls, ideal: list[set]):
        """Return the minimal generating set of `ideal`."""
        A = cls.copy_alg()
        rel_gens = []
        for rel in sorted(ideal, key=A.deg_data):
            A.add_rels_buffer(A.deg_data(rel))
            rel = A.reduce_data(rel)
            if rel:
                rel_gens.append(rel)
                A.add_rel_data(rel)
        return rel_gens

    @classmethod
    def indecomposables_An(
        cls, vectors: list[list[tuple["AlgGb", str, int]]], *, inplace=False
    ):
        """Return the minimal generating set of `vectors`, which form an A-submodule of A^n.

        `vectors` is a list of [(ele, name, deg), ...].
        The names should not overlap with existing generator names of `cls`."""
        A = cls if inplace else cls.copy_alg()
        rels = []  # type: list[tuple[int, AlgGb]]
        added_names = set()
        for index, v in enumerate(vectors):
            rel = A.zero()
            for ele, name, d in v:
                name = f"v_{{{name}}}"
                x = A.gen(name) if name in added_names else A.add_gen(name, d)
                added_names.add(name)
                rel += x * ele
            if rel:
                rels.append((index, rel))
        rels_module = []
        for i1, i2 in combinations_with_replacement(
            added_names, 2
        ):  # TODO: add pred in add_rels()
            rels_module.append(A.gen(i1) * A.gen(i2))
        A.add_rels(rels_module)
        result = []
        for index, rel in sorted(rels, key=lambda _x: _x[1].deg()):
            A.add_rels_buffer(rel.deg())
            rel.simplify()
            if rel:
                result.append(index)
                A.add_rel(rel)
        return [vectors[i] for i in result]

    @classmethod
    def ann(cls, x):
        """Return the generators of the ideal ann(x)."""
        annihilators = cls.ann_seq([(x, "")])
        return [a[0][0].data for a in annihilators]

    @classmethod
    def ann_seq(cls, ele_names: list[tuple["AlgGb", str]]):
        """Return relations among elements: $\\sum a_ie_i=0$."""
        A = cls.copy_alg()
        index = max(A.gens, default=-1)
        if cls.key_mo:
            A.key_mo = lambda _m: (
                not (_m1 := _m[bisect_left(_m, (index, 0)) :]),
                _m1,
                cls.key_mo(_m),
            )
        else:
            A.key_mo = lambda _m: (
                not (_m1 := _m[bisect_left(_m, (index, 0)) :]),
                _m1,
                _m,
            )
        rels_new = []
        for ele, name in ele_names:
            x = A.add_gen(name, ele.deg())
            rels_new.append(x + ele)
        A.add_rels(rels_new, clear_cache=True)
        annilators = []
        for m in A.rels:
            if m[-1][0] > index:
                a = []
                for m1 in chain((m,), A.rels[m]):
                    gen = A.gens[m1[-1][0]]
                    m11 = (
                        m1[:-1] + ((m1[-1][0], m1[-1][1] + 1),)
                        if m1[-1][1] + 1 != 0
                        else m1[:-1]
                    )
                    a.append((m11, gen.name, gen.deg))
                annilators.append(a)
        for en1, en2 in combinations(ele_names, 2):
            ele1, name1 = en1
            deg1 = ele1.deg()
            ele2, name2 = en2
            deg2 = ele2.deg()
            a = []
            for m1 in ele1.data:
                a.append((m1, name2, deg2))
            for m2 in ele2.data:
                a.append((m2, name1, deg1))
            annilators.append(a)
        if cls.key_mo:

            def key(_m):
                return A.deg0_mon(_m[bisect_left(_m, (index, 0)) :]), cls.key_mo(_m)

        else:

            def key(_m):
                return A.deg0_mon(_m[bisect_left(_m, (index, 0)) :]), _m

        A = A.reorder_gens(key_mo=key)
        annilators = [
            [(cls(A.reduce({_m})), name, deg) for _m, name, deg in a]
            for a in annilators
        ]
        return cls.indecomposables_An(annilators)

    @staticmethod
    def latex_annilators(annilators: list[list[tuple["AlgGb", str, int]]]):
        """Display the annilator in a readable form."""
        from IPython.display import Latex

        result = "\\begin{align*}\n"
        for a in annilators:
            s = "+".join(f"{name}{tex_parenthesis(c)}" for c, name, d in a)
            result += f"& {s}=0\\\\\n"
        result += "\\end{align*}"
        return Latex(result)

    @classmethod
    def subalgebra(cls, ele_names: list[tuple["AlgGb", str]], *, key_mo=None):
        """Return the subalgebra generated by `ele_names`."""
        A = cls.copy_alg()
        index = max(A.gens, default=-1)

        def key1(_m):
            _m1, _m2 = _m[: (i := bisect_left(_m, (index, 0)))], _m[i:]
            return (
                cls.deg0_mon(_m1),
                (cls.key_mo(_m1) if cls.key_mo else _m1),
                (key_mo(_m2) if key_mo else _m2),
            )

        A.key_mo = key1
        for ele, name in ele_names:
            x = A.add_gen(name, ele.deg())
            A.add_rel(x + ele)
        A.add_rels_buffer()
        A.gens = {i: v for i, v in A.gens.items() if i > index}
        A.key_mo = key_mo
        rels, A.rels = A.rels, {}
        for m in rels:
            if m[0][0] > index:
                rel_subalg = {m[bisect_left(m, (index, 0)) :]} | {
                    _m[bisect_left(_m, (index, 0)) :] for _m in rels[m]
                }
                A.add_rel_data(rel_subalg)
        A.add_rels_buffer()
        return A

    def evaluation(self, image_gens: dict[str, "AlgGb"]):
        """Return f(self) where f is an algebraic map determined by `image_gens`."""
        for v in image_gens.values():
            R = type(v)
            break
        else:
            raise ValueError("empty image_gens")
        zero = R.zero() if issubclass(R, BA.Algebra) else 0
        unit = R.unit() if issubclass(R, BA.Algebra) else 1
        result = zero
        for m in self.data:
            fm = unit
            for i, e in m:
                fm *= image_gens[self.gens[i].name] ** (-e)
            result += fm
        return result


class DgaGb(AlgGb):
    """A factory for DGA over F_2."""

    gens = None  # type: dict[int, DgaGen]
    deg_diff = None

    _index_subclass = 0

    @classmethod
    def copy_alg(cls) -> "Type[DgaGb]":
        """Return a copy of current algebra."""
        class_name = f"GbDga_{cls._index_subclass}"
        cls._index_subclass += 1
        dct = {
            "gens": cls.gens.copy(),
            "rels": copy.deepcopy(cls.rels),
            "_rels_buffer": copy.deepcopy(cls._rels_buffer),
            "key_mo": cls.key_mo,
            "pred": cls.pred,
            "deg_diff": cls.deg_diff,
        }
        return type(class_name, (DgaGb,), dct)

    # setters ----------------------------
    @classmethod
    def add_gen(cls, name: str, deg: int | Vector, diff=None):
        """Add a new generator and return it."""
        if type(deg) is int:
            deg = Vector((deg,))
        cls.dim_grading = len(deg)

        index = max(cls.gens, default=-1) + 1
        if diff is None:
            diff = set()
        elif type(diff) is not set:
            diff = diff.data
        if diff and cls.deg_data(diff) - deg != cls.deg_diff:
            raise BA.MyDegreeError("inconsistent differential degree")
        cls.gens[index] = DgaGen(name, deg, diff)
        m = ((index, -1),)
        return cls(m).reduce()

    @classmethod
    def set_diff(cls, gen_name: str, diff: None | set | DgaGb):
        """Define the differential of gen_name."""
        for i, gen in cls.gens.items():
            if gen.name == gen_name:
                break
        else:
            raise BA.MyKeyError(f"generator {gen_name} not found")

        if type(diff) is not set and diff is not None:
            diff = diff.data
        if diff and (
            not cls(diff).is_homo() or cls.deg_data(diff) - gen.deg != cls.deg_diff
        ):
            raise BA.MyDegreeError("inconsistent differential degree")
        gen = cls.gens[i]
        cls.gens[i] = DgaGen(gen.name, gen.deg, diff)

    def diff(self):
        """Return the boundary of the chain."""
        result = set()
        for m in self.data:
            for (i, (index, e)) in enumerate(m):
                if e % 2:
                    m1 = (
                        m[:i] + ((index, e + 1),) + m[i + 1 :]
                        if e + 1
                        else m[:i] + m[i + 1 :]
                    )
                    m1_by_dg_i = {add_dtuple(m1, _m) for _m in self.gens[index].diff}
                    result ^= m1_by_dg_i
        return type(self)(result).reduce()

    @classmethod
    def rename_gen(cls, old_name, new_name):
        """Rename a generator."""
        for i, gen in cls.gens.items():
            if gen.name == old_name:
                break
        else:
            raise ValueError(f"no generator named {old_name}")
        cls.gens[i] = DgaGen(new_name, gen.deg, gen.diff)

    # getters ----------------------------
    @classmethod
    def homology():  # TODO: implement this
        pass

    @classmethod
    def is_differentiable_mon(cls, mon):
        for i, e in mon:
            if cls.gens[i].diff is None:
                return False
        return True

    @classmethod
    def is_differentiable_data(cls, data):
        return all(map(cls.is_differentiable_mon, data))

    def is_differentiable(self):
        return self.is_differentiable_data(self.data)

    @staticmethod
    def contains_gen_mon(mon, gen_id):
        """Return if the monomial contains the generator."""
        for i, e in mon:
            if i == gen_id:
                return True
        return False

    @staticmethod
    def contains_gen_data(data, gen_id):
        """Return if the data contains the generator."""
        return any(map(DgaGb.contains_gen_mon, data, repeat(gen_id)))

    @classmethod
    def determine_diff(cls, g: str | int, basis: dict, image_gens=None):
        """Determine differentials by relations."""
        if type(g) is str:
            for id, gen in cls.gens.items():
                if gen.name == g:
                    break
            else:
                raise BA.MyKeyError(f"generator {g} not found")
        else:
            id, gen = g, cls.gens[g]
            g = gen.name
        deg_target = gen.deg + cls.deg_diff
        if deg_target not in basis:
            cls.set_diff(g, set())
            print(f"set d({g})=0")
            return
        # print("Possible summands:")
        # for m in basis[deg_target]:
        #     print(cls(m))
        rels = []
        cls.set_diff(g, set())
        for m in cls.rels:
            if cls.is_differentiable_mon(m) and all(
                map(cls.is_differentiable_mon, cls.rels[m])
            ):
                if cls.contains_gen_mon(m, id) or any(
                    map(cls.contains_gen_mon, cls.rels[m], repeat(id))
                ):
                    rels.append({m} | cls.rels[m])
        possible_diffs = []
        for n in range(1 << len(basis[deg_target])):
            data = {basis[deg_target][i] for i in two_expansion(n)}
            if all(map(cls.is_differentiable_mon, data)) and cls(data).diff():
                continue
            cls.set_diff(g, data)
            compatible = True
            for rel in rels:
                if cls(rel).diff():
                    compatible = False
                    break
            if (
                image_gens
                and image_gens[gen.name].is_differentiable()
                and cls(data).evaluation(image_gens) != image_gens[gen.name].diff()
            ):
                compatible = False
            if compatible:
                possible_diffs.append(data)
        if len(possible_diffs) == 1:
            cls.set_diff(g, possible_diffs[0])
            print(f"set d({g})={cls(possible_diffs[0])}")
        elif len(possible_diffs) == 0:
            raise BA.MyClassError(f"Invalid DGA. d({g})=?")
        else:
            for data in possible_diffs:
                print(f"d({g})={cls(data)} is possible.")
            cls.set_diff(g, None)

    @classmethod
    def determine_diffs(cls, basis: dict, image_gens=None):
        for i in sorted(
            filter(lambda _i: cls.gens[_i].diff is None, cls.gens),
            key=lambda _i: cls.gens[_i].deg[0],
        ):
            cls.determine_diff(i, basis, image_gens)

    @classmethod
    def latex_alg(cls, show_gb=False):
        """For latex."""
        result = super().latex_alg(show_gb)
        result += "\\section{Differentials}\n\n"
        for gen in cls.gens.values():
            result += f"$d({gen.name})={cls(gen.diff)}$\\vspace{{3pt}}\n\n"


def new_alg(*, key_mo: str | Callable = None, pred=None) -> Type[AlgGb]:
    """Return a dynamically created subclass of AlgGb.

    When `key_mo=None`, use revlex ordering by default."""
    class_name = f"AlgGb_{AlgGb._index_subclass}"
    AlgGb._index_subclass += 1
    if key_mo == "Lex" or key_mo == "lex":
        key_mo = key_lex
    elif key_mo == "Revlex" or key_mo == "revlex":
        key_mo = None
    else:
        raise BA.MyError("unknown monomial ordering")
    dct = {
        "gens": {},
        "rels": {},
        "_rels_buffer": defaultdict(list),
        "key_mo": key_mo,
        "pred": pred or pred_always_true,
        "dim_grading": None,
    }
    return type(class_name, (AlgGb,), dct)


def load_alg(
    filename, tablename, *, key_mo=None, pred=None
) -> Type["AlgGb"] | Type["DgaGb"]:
    """load an algebra from a database."""
    conn = sqlite3.connect(filename)
    c = conn.cursor()
    version = get_one_element(c.execute('SELECT value FROM info WHERE key="version"'))[
        0
    ]
    if version == "0.0":
        if key_mo is None:
            mo = get_one_element(c.execute('SELECT value FROM info WHERE key="mo"'))[0]
            if mo == "Revlex":
                mo = None
            elif mo == "Lex":
                mo == key_lex
            else:
                raise BA.MyError("Can not determine the key function")
        grading = get_one_element(
            c.execute('SELECT value FROM info WHERE key="grading"')
        )[0]

        A = new_alg(key_mo=mo, pred=pred)
        A.dim_grading = grading.count(",") + 1
        if version == "0.0":
            for id, name, *deg in c.execute(
                f"SELECT gen_id, gen_name, {grading} FROM {tablename}_generators ORDER BY gen_id"
            ):
                A.gens[id] = Gen(name, Vector(deg))
            for m, b in c.execute(
                f"SELECT leading_term, basis FROM {tablename}_relations"
            ):
                A.rels[AlgGb.mon_sqlite3(m)] = AlgGb.data_sqlite3(b)
    else:
        raise BA.MyError("Version unknown")

    c.close()
    conn.commit()
    conn.close()
    return A


def new_dga(*, key_mo=None, pred=None, deg_diff=None) -> Type[DgaGb]:
    """Return a dynamically created subclass of GbDga.

    When key_mo=None, use revlex ordering by default."""
    class_name = f"GbDga_{DgaGb._index_subclass}"
    DgaGb._index_subclass += 1
    if deg_diff is not None:
        deg_diff = Vector(deg_diff)
    else:
        raise BA.MyDegreeError("degree of differential not supplied")
    dct = {
        "gens": {},
        "rels": {},
        "_rels_buffer": {},
        "key_mo": key_mo,
        "pred": pred or pred_always_true,
        "dim_grading": None,
        "deg_diff": deg_diff,
    }
    return type(class_name, (DgaGb,), dct)


# operations for monomials with negative exponents
def le_dtuple(d1, d2):
    """Return if d1_i <= d2_i as sparse vectors."""
    d2_dict = dict(d2)
    return all(gen in d2_dict and exp >= d2_dict[gen] for gen, exp in d1)


def min_dtuple(d1, d2):
    """return (min(d1_i, d2_i), ...)."""
    d1_dict = dict(d1)
    result = {}
    for gen, exp in d2:
        if gen in d1_dict:
            result[gen] = max(exp, d1_dict[gen])
    return tuple(sorted(result.items()))


def max_dtuple(d1, d2):
    """return (max(d1_i, d2_i), ...)."""
    result = dict(d1)
    for gen, exp in d2:
        result[gen] = min(exp, result[gen]) if gen in result else exp
    return tuple(sorted(result.items()))


def gcd_nonzero_dtuple(d1, d2):
    """return (min(d1_i, d2_i), ...)."""
    d1_dict = dict(d1)
    for gen, exp in d2:
        if gen in d1_dict:
            return True
    return False
