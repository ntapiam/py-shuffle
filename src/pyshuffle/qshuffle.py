from functools import reduce
from operator import add
from .vector import Vector
from .concat import Concat


class Monomial:
    __table = {
            "0": u"\u2070",
            "1": u"\u00B9",
            "2": u"\u00B2",
            "3": u"\u00B3",
            "4": u"\u2074",
            "5": u"\u2075",
            "6": u"\u2076",
            "7": u"\u2077",
            "8": u"\u2078",
            "9": u"\u2079",
            }

    def __init__(self, exps={}):
        self.exps = exps

    def weight(self):
        return sum(self.exps.values())

    def __mul__(self, other):
        res = self.exps.copy()
        for k in other.exps:
            res[k] = self.exps.get(k, 0) + other.exps[k]
        return Monomial(res)

    def __eq__(self, other):
        return self.exps == other.exps

    def __lt__(self, other):
        for k in self.exps:
            if self.exps[k] > other.exps.get(k, 0):
                return False
        return True

    def __repr__(self):
        res = "("
        for k, v in sorted(self.exps.items()):
            if v == 0:
                continue
            else:
                digits = str(v)
                res += str(k) + "".join([Monomial.__table[d] for d in digits])

        return res + ")"


class QShuffle(Vector):
    def __init__(self, terms=[]):
        super().__init__(terms)

    def coprod(self):
        @QShuffle.linear_map
        def deconc_basis(w):
            u = w[0]
            if u == []:
                return QShuffle.to_vec(([], [], w[1:]))

            terms = [QShuffle.to_vec((u[:k], u[k:], w[1:])) for k in range(len(u) + 1)]
            return reduce(add, terms, QShuffle.zero())

        return deconc_basis(self)

    def __mul__(self, other):
        if self.terms == [] or other.terms == []:
            return QShuffle.zero()
        if self.terms[0][1] == ([],):
            return other
        if other.terms[0][1] == ([],):
            return self

        terms = reduce(
            add,
            (
                r
                * s
                * (QShuffle.to_vec(u) * QShuffle.to_vec(v[0][:-1])).__conc(
                    QShuffle.to_vec([v[0][-1]])
                )
                for (r, u) in self.terms
                for (s, v) in other.terms
            ),
            QShuffle.zero(),
        )

        terms += reduce(
            add,
            (
                r
                * s
                * (QShuffle.to_vec(u[0][:-1]) * QShuffle.to_vec(v[0])).__conc(
                    QShuffle.to_vec([u[0][-1]])
                )
                for (r, u) in self.terms
                for (s, v) in other.terms
            ),
            QShuffle.zero(),
        )

        terms += reduce(
            add,
            (
                r
                * s
                * (QShuffle.to_vec(u[0][:-1]) * QShuffle.to_vec(v[0][:-1])).__conc(
                    QShuffle.to_vec([u[0][-1] * v[0][-1]])
                )
                for (r, u) in self.terms
                for (s, v) in other.terms
            ),
            QShuffle.zero(),
        )

        return terms

    def __conc(self, other):
        x = Concat(self.terms)
        y = Concat(other.terms)
        return QShuffle((x * y).terms)

    def conv(self, f, g):
        @QShuffle.linear_map
        def conv_basis(w):
            a, b = w
            return f(QShuffle.to_vec((a,))).__mul__(g(QShuffle.to_vec((b,))))

        tensors = self.coprod()
        return conv_basis(tensors)

    def J(self):
        @QShuffle.linear_map
        def J_basis(b):
            return QShuffle.to_vec(b) if b != ([],) else QShuffle.zero()

        return J_basis(self)

    def Y(self):
        @QShuffle.linear_map
        def Y_basis(b):
            return b[0][0].weight() * QShuffle.to_vec(b)

        return Y_basis(self)

    def S(self):
        @QShuffle.linear_map
        def S_basis(b):
            return (-1) ** (len(b[0])) * QShuffle.to_vec(b[0][::-1])

        return S_basis(self)
