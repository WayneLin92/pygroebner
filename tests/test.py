import unittest
import os, sys

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) + "/src/"
sys.path.append(src_dir)


class GroebnerTestCase(unittest.TestCase):
    def setUp(self):
        print(sys.path)
        from groebner import new_alg
        from itertools import permutations

        self.new_alg = new_alg
        self.permutations = permutations

    def alg_B(self, n_max):
        gens = []
        for i in range(n_max):
            for j in range(i + 1, n_max + 1):
                gens.append((f"R_{{{i}{j}}}", (2 ** j - 2 ** i, 1, j - i)))
        gens.sort(key=lambda _x: _x[1][2])

        E1 = self.new_alg()
        for name, deg in gens:
            E1.add_gen(name, deg)

        def R(S: int | tuple, T: int | tuple):
            if type(S) is int:
                return E1.gen(f"R_{{{S}{T}}}")
            assert len(S) == len(T)
            S, T = sorted(S), sorted(T)
            result = E1.zero()
            for T1 in self.permutations(T):
                if all(t - s > 0 for s, t in zip(S, T1)):
                    pro = E1.unit()
                    for x in map(R, S, T1):
                        pro *= x
                    result += pro
            return result

        rels = []
        for d in range(2, n_max + 1):
            for i in range(n_max + 1 - d):
                j = i + d
                rel = sum((R(i, k) * R(k, j) for k in range(i + 1, j)), E1.zero())
                rels.append(rel)
        rels.sort(key=lambda x: x.deg())
        E1.add_rels(rels)
        return E1, R

    def test_AlgGb(self):
        E1, _ = self.alg_B(7)
        self.assertEqual(65, len(E1.rels))

    def test_subalgebra(self):
        n_max = 4
        E1, R = self.alg_B(n_max)
        ele_names = []
        for i in range(n_max):
            ele_names.append((R(i, i + 1), f"h_{i}"))
        for i in range(n_max - 2):
            ele_names.append((R((i, i + 1), (i + 2, i + 3)), f"h_{i}(1)"))
        for i in range(n_max - 4):
            ele_names.append(
                (R((i, i + 1, i + 3), (i + 2, i + 4, i + 5)), f"h_{i}(1, 3)")
            )
        for i in range(n_max - 4):
            ele_names.append(
                (R((i, i + 1, i + 2), (i + 3, i + 4, i + 5)), f"h_{i}(1, 2)")
            )
        for d in range(2, n_max + 1):
            for i in range(n_max + 1 - d):
                j = i + d
                ele_names.append((R(i, j) * R(i, j), f"b_{{{i}{j}}}"))
        HX = E1.subalgebra(ele_names, key=E1.key)
        HX.reduce_rels()
        self.assertEqual(15, len(HX.rels))

    @classmethod
    def tearDownClass(cls):
        pass

    if __name__ == "__main__":
        unittest.main()
