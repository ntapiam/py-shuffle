from functools import reduce
from operator import add
from .vector import Vector
from .concat import Concat


class Monomial:
    __table = {
        "0": "\u2070",
        "1": "\u00b9",
        "2": "\u00b2",
        "3": "\u00b3",
        "4": "\u2074",
        "5": "\u2075",
        "6": "\u2076",
        "7": "\u2077",
        "8": "\u2078",
        "9": "\u2079",
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

    def __R(self):
        @QShuffle.linear_map
        def R_basis(b):
            b = b[0]
            if b == []:
                return QShuffle.zero()

            return QShuffle.to_vec(b[::-1])
        return R_basis(self)

    def __T(self):
        @QShuffle.linear_map
        def T_basis(b):
            b = b[0]
            if b == []:
                return QShuffle.zero()

            return (-1) ** len(b) * QShuffle.to_vec(b)
        return T_basis(self)

    def __contract(self, m):
        @QShuffle.linear_map
        def contract_basis(b):
            b = b[0]
            if b == []:
                return QShuffle.zero()

            return QShuffle.to_vec([m * b[0]]).__conc(QShuffle.to_vec(b[1:]))
        return contract_basis(self)

    def __Sigma(self):
        @QShuffle.linear_map
        def Sigma_basis(b):
            b = b[0]
            if b == []:
                return QShuffle.zero()

            if len(b) == 1:
                return QShuffle.to_vec(b)

            w = QShuffle.to_vec(b[1:]).__Sigma()
            a = QShuffle.to_vec([b[0]])
            return a.__conc(w) + w.__contract(b[0])
                
        return Sigma_basis(self)

    @staticmethod
    def conv(f, g):
        @QShuffle.linear_map
        def conv_basis(w):
            a, b = w
            return f(QShuffle.to_vec((a,))).__mul__(g(QShuffle.to_vec((b,))))

        return lambda x: conv_basis(x.coprod())

    @staticmethod
    def J(x):
        @QShuffle.linear_map
        def J_basis(b):
            return QShuffle.to_vec(b) if b != ([],) else QShuffle.zero()

        return J_basis(x)

    @staticmethod
    def Y(x):
        @QShuffle.linear_map
        def Y_basis(b):
            if b[0] == []:
                return QShuffle.zero()
            we = sum(map(lambda x: x.weight(), b[0]))
            return we * QShuffle.to_vec(b)

        return Y_basis(x)

    @staticmethod
    def Yinv(x):
        @QShuffle.linear_map
        def Y_basis(b):
            if b[0] == []:
                return QShuffle.zero()
            we = sum(map(lambda x: x.weight(), b[0]))
            return QShuffle.to_vec(b) / we

        return Y_basis(x)

    @staticmethod
    def S(x):
        return x.__T().__Sigma().__R()

    @staticmethod
    def eulerian(x):
        @QShuffle.linear_map
        def e_basis(b):
            g = QShuffle.J
            res = QShuffle.zero()
            for k in range(len(b[0])):
                res += (-1) ** k * g(QShuffle.to_vec(b[0])) / (k + 1)
                g = QShuffle.conv(g, QShuffle.J)
            return res

        return e_basis(x)

    @staticmethod
    def D(x):
        return QShuffle.Yinv(QShuffle.conv(QShuffle.Y, QShuffle.S)(x))
