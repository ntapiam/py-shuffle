"""MIT License

Copyright (c) 2023 Nikolas Tapia

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from fractions import Fraction
from functools import reduce
from operator import add, mul


class Vector:
    def __init__(self, terms=[]):
        self.terms = terms
        self.__normalize()

    def __normalize(self):
        if self.terms == []:
            return
        self.terms.sort(key=lambda x: x[1])
        out = [self.terms[0]]
        for (s, b) in self.terms[1:]:
            c = out[-1]
            if b == c[1]:
                out[-1] = (s + c[0], b)
            else:
                out.append((s, b))

        self.terms = list(filter(lambda x: x[0] != 0, out))
        self.terms = [
            (s, tuple(Vector.__flatten(b))) if isinstance(b, tuple) else (s, (b,))
            for (s, b) in self.terms
        ]

    @staticmethod
    def linear_map(func):
        def lin_ext(v):
            return Vector(
                [(s * a, u) for (s, b) in v.terms for (a, u) in func(b).terms]
            )

        return lin_ext

    @staticmethod
    def to_vec(b):
        return Vector([(Fraction(1), b)])

    @staticmethod
    def zero(b=[]):
        return Vector([(Fraction(0), b)])

    def deconc(self):
        @Vector.linear_map
        def deconc_basis(w):
            if len(w) > 1:
                raise ValueError("method not defined for this basis")
            w = w[0]
            if w == []:
                return Vector.to_vec(([], []))

            terms = [Vector.to_vec((w[:k], w[k:])) for k in range(len(w) + 1)]
            return reduce(add, terms, Vector.zero())

        return deconc_basis(self)

    def unshuf(self):
        @Vector.linear_map
        def unshuf_basis(w):
            if len(w) > 1:
                raise ValueError("method not definied for this basis")
            w = w[0]
            if w == []:
                return Vector.to_vec(([], []))

            terms = [Vector.to_vec(([], [a])) + Vector.to_vec(([a], [])) for a in w]
            return reduce(mul, terms, Vector.to_vec(([], [])))

        return unshuf_basis(self)

    def shuffle(self, other):
        if self.terms == [] or other.terms == []:
            return Vector.zero()
        if self.terms[0][1] == ([],):
            return other
        if other.terms[0][1] == ([],):
            return self

        terms = reduce(
            add,
            (
                r
                * s
                * Vector.to_vec(u).shuffle(Vector.to_vec(v[0][:-1]))
                * Vector.to_vec([v[0][-1]])
                for (r, u) in self.terms
                for (s, v) in other.terms
            ),
            Vector.zero(),
        )

        terms += reduce(
            add,
            (
                r
                * s
                * Vector.to_vec(u[0][:-1]).shuffle(Vector.to_vec(v[0]))
                * Vector.to_vec([u[0][-1]])
                for (r, u) in self.terms
                for (s, v) in other.terms
            ),
            Vector.zero(),
        )

        return terms

    def shuffle_conv(self, f, g):
        @Vector.linear_map
        def conv_basis(w):
            a, b = w
            return f(Vector.to_vec((a,))).shuffle(g(Vector.to_vec((b,))))

        tensors = self.deconc()
        return conv_basis(tensors)

    def cat_conv(self, f, g):
        @Vector.linear_map
        def conv_basis(w):
            a, b = w
            return f(Vector.to_vec((a,))) * g(Vector.to_vec((b,)))

        tensors = self.unshuf()
        return conv_basis(tensors)

    def __flatten(t):
        for i in t:
            yield from [i] if not isinstance(i, tuple) else Vector.__flatten(i)

    def outer(self, other):
        @Vector.linear_map
        def outer_basis(b):
            return Vector(
                [(s, tuple(Vector.__flatten((a, b)))) for (s, a) in self.terms]
            )

        return outer_basis(other)

    def __add__(self, other):
        if isinstance(other, (int, float, Fraction)):
            other = Vector([(other, [])])

        return Vector(self.terms + other.terms)

    def __mul__(self, other):
        @Vector.linear_map
        def mul_basis(b):
            n = len(b)
            if n % 2 != 0:
                raise ValueError("can only concatenate with same tensor order?")
            return Vector.to_vec(
                tuple(u + v for (u, v) in zip(b[: n // 2], b[n // 2 :]))
            )

        return mul_basis(self.outer(other))

    def __rmul__(self, s):
        return Vector([(r * s, b) for (r, b) in self.terms])

    def __eq__(self, other):
        return self.terms == other.terms

    def __repr__(self):
        def coef_to_string(k, s):
            if k == 0:
                if s >= 0:
                    return f"{s}⋅" if s != 1 else ""
                else:
                    return f"{-s}⋅" if s != -1 else " -"

            else:
                if s >= 0:
                    return f" + {s}⋅" if s != 1 else " + "
                else:
                    return f" - {-s}⋅" if s != -1 else " - "

        strings = [
            f"{coef_to_string(k, s)}{'⊗'.join(map(str, b))}"
            for (k, (s, b)) in enumerate(self.terms)
        ]
        return "".join(strings)


@Vector.linear_map
def J(b):
    return Vector.to_vec(b) if b != ([],) else Vector.zero()


@Vector.linear_map
def Y(b):
    return len(b[0]) * Vector.to_vec(b)


@Vector.linear_map
def S(b):
    return (-1) ** (len(b[0])) * Vector.to_vec(b[0][::-1])


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
    out = Vector.zero()
    for k in range(len(b[0])):
        out += Fraction((-1) ** k, k + 1) * g(Vector.to_vec(b))
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
    return x.shuffle_conv(S, Y)


def cat_D(x):
    return x.cat_conv(S, Y)


if __name__ == "__main__":
    for k in range(1, 6):
        x = Vector.to_vec(list(range(1, k + 1)))
        a = sh_eulerian(x)
        b = cat_eulerian(x)
        c = sh_D(x)
        d = cat_D(x)

        print(f"Vector: {x}")
        print("In the (ш, Δ) Hopf algebra:")
        print(f"\te₁(x) = {a}\r\n\tD(x) = {c}\n\r")
        print("In the (⊗, Δ_ш) Hopf algbera:")
        print(f"\te₁(x) = {b}\r\n\tD(x) = {c}\n\r")
