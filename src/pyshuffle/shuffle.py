from functools import reduce
from operator import add
from .vector import Vector
from .concat import Concat


class Shuffle(Vector):
    def __init__(self, terms=[]):
        super().__init__(terms)

    def coprod(self):
        @Shuffle.linear_map
        def deconc_basis(w):
            u = w[0]
            if u == []:
                return Shuffle.to_vec(([], [], w[1:]))

            terms = [Shuffle.to_vec((u[:k], u[k:], w[1:])) for k in range(len(u) + 1)]
            return reduce(add, terms, Shuffle.zero())

        return deconc_basis(self)

    def __mul__(self, other):
        if self.terms == [] or other.terms == []:
            return Shuffle.zero()
        if self.terms[0][1] == ([],):
            return other
        if other.terms[0][1] == ([],):
            return self

        terms = reduce(
            add,
            (
                r
                * s
                * (Shuffle.to_vec(u) * Shuffle.to_vec(v[0][:-1])).__conc(
                    Shuffle.to_vec([v[0][-1]])
                )
                for (r, u) in self.terms
                for (s, v) in other.terms
            ),
            Shuffle.zero(),
        )

        terms += reduce(
            add,
            (
                r
                * s
                * (Shuffle.to_vec(u[0][:-1]) * Shuffle.to_vec(v[0])).__conc(
                    Shuffle.to_vec([u[0][-1]])
                )
                for (r, u) in self.terms
                for (s, v) in other.terms
            ),
            Shuffle.zero(),
        )

        return terms

    def __conc(self, other):
        x = Concat(self.terms)
        y = Concat(other.terms)
        return Shuffle((x * y).terms)

    @staticmethod
    def conv(f, g):
        @Shuffle.linear_map
        def conv_basis(w):
            a, b = w
            return f(Shuffle.to_vec((a,))).__mul__(g(Shuffle.to_vec((b,))))

        return lambda x: conv_basis(x.coprod())

    @staticmethod
    def J(x):
        @Shuffle.linear_map
        def J_basis(b):
            return Shuffle.to_vec(b) if b != ([],) else Shuffle.zero()

        return J_basis(x)

    @staticmethod
    def Y(x):
        @Shuffle.linear_map
        def Y_basis(b):
            return len(b[0]) * Shuffle.to_vec(b)

        return Y_basis(x)

    @staticmethod
    def Yinv(x):
        @Shuffle.linear_map
        def Y_basis(b):
            return Shuffle.to_vec(b) / len(b[0])

        return Y_basis(x)

    @staticmethod
    def S(x):
        @Shuffle.linear_map
        def S_basis(b):
            return (-1) ** (len(b[0])) * Shuffle.to_vec(b[0][::-1])

        return S_basis(x)

    @staticmethod
    def eulerian(x):
        @Shuffle.linear_map
        def e_basis(b):
            g = Shuffle.J
            res = Shuffle.zero()
            for k in range(len(b[0])):
                res += (-1) ** k * g(Shuffle.to_vec(b)) / (k + 1)
                g = Shuffle.conv(g, Shuffle.J)
            return res

        return e_basis(x)

    @staticmethod
    def D(x):
        return Shuffle.Yinv(Shuffle.conv(Shuffle.Y, Shuffle.S)(x))
