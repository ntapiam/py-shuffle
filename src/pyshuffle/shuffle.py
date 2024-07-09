from fractions import Fraction
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

    def conv(self, f, g):
        @Vector.linear_map
        def conv_basis(w):
            a, b = w
            return f(Shuffle.to_vec((a,))).__mul__(g(Shuffle.to_vec((b,))))

        tensors = self.coprod()
        return conv_basis(tensors)

    def J(self):
        @Vector.linear_map
        def J_basis(b):
            return Shuffle.to_vec(b) if b != ([],) else Shuffle.zero()

        return J_basis(self)

    def Y(self):
        @Vector.linear_map
        def Y_basis(b):
            return len(b[0]) * Shuffle.to_vec(b)

        return Y_basis(self)

    def S(self):
        @Vector.linear_map
        def S_basis(b):
            return (-1) ** (len(b[0])) * Shuffle.to_vec(b[0][::-1])

        return S_basis(self)


def sh_conv(f, g):
    def inner(v):
        return v.shuffle_conv(f, g)

    return inner


def cat_conv(f, g):
    def inner(v):
        return v.cat_conv(f, g)

    return inner


@Vector.linear_map
def sh_eulerian(b):
    g = J
    out = Shuffle.zero()
    for k in range(len(b[0])):
        out += Fraction((-1) ** k, k + 1) * g(Shuffle.to_vec(b))
        g = sh_conv(g, J)

    return out


@Vector.linear_map
def cat_eulerian(b):
    g = J
    out = Vector.zero()
    for k in range(len(b[0])):
        out += Fraction((-1) ** k, k + 1) * g(Vector.to_vec(b))
        g = cat_conv(g, J)

    return out


def sh_D(x):
    return x.conv(Y, S)


def cat_D(x):
    return x.conv(Y, S)


if __name__ == "__main__":
    for k in range(1, 6):
        x = Shuffle.to_vec(list(range(1, k + 1)))
        a = sh_eulerian(x)
        c = sh_D(x)

        print(f"Vector: {x}")
        print("In the (ш, Δ) Hopf algebra:")
        print(f"\te₁(x) = {a}\r\n\tD(x) = {c}\n\r")
