from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Hashable, Any
import operator
from itertools import product, repeat
from functools import reduce
# TODO: try __slots__


class Algebra(ABC):
    # methods -------------------
    def __repr__(self) -> str:
        return self.__str__()

    def __bool__(self) -> bool:
        return bool(self.data)

    def __eq__(self, other):
        return self.data == other.data

    def __add__(self, other):
        return type(self)(self.add_data(self.data, other.data))

    def __sub__(self, other):
        return type(self)(self.sub_data(self.data, other.data))

    def __mul__(self, other):
        return type(self)(self.mul_data(self.data, other.data))

    def __pow__(self, n: int):
        power = self.data
        pro = self.unit_data()
        while n:
            if n & 1:
                pro = self.mul_data(pro, power)
            n >>= 1
            if n:
                power = self.square_data(power)
        return type(self)(pro)
    
    @classmethod
    def pow_data(cls, data: set, n: int):
        power = data
        pro = cls.unit_data()
        while n:
            if n & 1:
                pro = cls.mul_data(pro, power)
            n >>= 1
            if n:
                power = cls.square_data(power)
        return pro

    def _repr_markdown_(self):
        return f"${self}$"

    def copy(self):
        return type(self)(self.data.copy())

    @classmethod
    def unit(cls):
        return cls(cls.unit_data())

    @classmethod
    def zero(cls):
        return cls(cls.zero_data())

    def square(self):
        return type(self)(self.square_data(self.data))

    def inverse(self, d_max) -> list:
        """Return 1/self, listed by degrees up to `d_max`."""
        list_homo = self.split_homo(d_max)
        if not list_homo or list_homo[0] != self.unit():
            raise ValueError("not monic")
        result = [self.unit()]
        for d in range(1, d_max + 1):
            term_d = -sum((result[i] * list_homo[d - i] for i in range(0, d)), self.zero())
            result.append(term_d)
        return result

    def deg(self) -> None | int:
        """Return the deg of the polynomial. Return None if it is zero."""
        return max(map(self.deg_mon, self.data)) if self.data else None

    # abstract -----------------
    @abstractmethod
    def __init__(self, data):
        self.data = data  # type: Any
        raise NotImplementedError

    @abstractmethod
    def __str__(self) -> str: pass

    def __neg__(self): pass

    def __iadd__(self, other): pass

    def __isub__(self, other): pass

    @abstractmethod
    def repr_(self, clsname: str) -> str:
        """Return the representation (functions as the actual `__repr__`)."""
        pass

    @staticmethod
    @abstractmethod
    def unit_data(): pass

    @staticmethod
    @abstractmethod
    def zero_data(): pass

    @staticmethod
    @abstractmethod
    def mul_mons(mon1, mon2):
        """Return the product of two monomials."""
        pass

    @staticmethod
    @abstractmethod
    def add_data(data1, data2):
        """Return the sum as data"""
        pass

    @staticmethod
    @abstractmethod
    def sub_data(data1, data2):
        """Return the sum as data"""
        pass

    @staticmethod
    @abstractmethod
    def mul_data(data1, data2):
        """Return product as data."""
        pass

    @staticmethod
    @abstractmethod
    def square_data(data):
        """Return the square of monomials."""
        pass

    @staticmethod
    @abstractmethod
    def str_mon(mon) -> str:
        """Return the str for the monomial."""
        pass

    @staticmethod
    @abstractmethod
    def repr_mon(mon, clsname) -> str:
        """Return the representation for the monomial."""
        pass

    @abstractmethod
    def _sorted_mons(self) -> list:
        """Sort the monomials for __str__()."""
        pass

    @abstractmethod
    def homo(self, d) -> Algebra:
        """Return the degree d homogeneous part."""
        pass

    @abstractmethod
    def split_homo(self, d_max) -> list:
        """Return up to degree d homogeneous parts."""
        pass

    @staticmethod
    @abstractmethod
    def deg_mon(mon: Hashable):
        """Return the degree of mon."""
        pass


class AlgebraMod2(Algebra, ABC):
    """ self.data is a set of monomials """
    def __init__(self, data: set | tuple):
        if type(data) is set:
            self.data = data # type: set[tuple]
        elif type(data) is tuple:
            self.data = {data}
        else:
            raise TypeError(f"{data} of type {type(data)} can not initialize {type(self).__name__}.")

    # -- Algebra -----------
    def __str__(self):
        result = " + ".join(map(self.str_mon, self._sorted_mons()))
        return result if result else "0"

    def repr_(self, clsname):
        result = " + ".join(map(self.repr_mon, self._sorted_mons(), repeat(clsname)))
        return result if result else f"{clsname}.zero()"

    def _sorted_mons(self) -> list:
        return sorted(self.data, key=lambda m: (self.deg_mon(m), m), reverse=True)

    def __neg__(self):
        return self.copy()

    def __iadd__(self, other):
        self.data ^= other.data
        return self

    def __isub__(self, other):
        self.data ^= other.data
        return self

    @staticmethod
    def add_data(data1, data2):
        return data1 ^ data2

    @staticmethod
    def sub_data(data1, data2):
        return data1 ^ data2

    @classmethod
    def mul_data(cls, data1, data2):
        return reduce(operator.xor, (pro if type(pro := cls.mul_mons(m, n)) is set else {pro}
                                     for m, n in product(data1, data2)), set())

    @staticmethod
    def unit_data() -> set:
        return {()}

    @staticmethod
    def zero_data() -> set:
        return set()

    @classmethod
    def square_data(cls, data):
        """Warning: non-commutative algebra should overwrite this."""
        return reduce(operator.xor, (pro if type(pro := cls.mul_mons(m, m)) is set else {pro}
                                     for m in data), set())

    def homo(self, d):
        data = set(m for m in self.data if self.deg_mon(m) == d)
        return type(self)(data)

    def split_homo(self, d_max):
        list_homo = [self.zero() for _ in range(d_max + 1)]
        for m in self.data:
            if self.deg_mon(m) <= d_max:
                list_homo[self.deg_mon(m)].data.add(m)
        return list_homo

    def is_homo(self):
        prev_deg = None
        for m in self.data:
            if prev_deg is None:
                prev_deg = self.deg_mon(m)
            elif self.deg_mon(m) != prev_deg:
                return False
        return True


class MyError(Exception):
    pass
