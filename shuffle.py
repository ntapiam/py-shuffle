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


class Vector:
    def __init__(self, terms):
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

    @staticmethod
    def to_vec(b):
        return Vector([(Fraction(1), b)])

    @staticmethod
    def __deconc(w):
        return [(w[:k], w[k:]) for k in range(len(w) + 1)]

    @staticmethod
    def __deconc_reduced(w):
        return [(w[:k], w[k:]) for k in range(1, len(w))]

    def deconc(self):
        return Tensor([(s, d) for (s, w) in self.terms for d in Vector.__deconc(w)])

    def deconc_reduced(self):
        return Tensor(
            [(s, d) for (s, w) in self.terms for d in Vector.__deconc_reduced(w)]
        )

    def outer(self, other):
        return Tensor(
            [(a * b, (u, v)) for (a, u) in self.terms for (b, v) in other.terms]
        )

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        if isinstance(other, (int, float, Fraction)):
            other = Vector([(other, [])])

        return Vector(self.terms + other.terms)

    def conv(self, f, g):
        copr = self.deconc()
        img = Tensor(
            [
                (s * a * b, (u, v))
                for (s, (x, y)) in copr.terms
                for (a, u) in f(x).terms
                for (b, v) in g(y).terms
            ]
        )
        return img.shuffle()

    def __repr__(self):
        strings = [f"{s}⋅{b}" for (s, b) in self.terms]
        return "+".join(strings)

    def __rmul__(self, a):
        return a * self

    def __mul__(self, a):
        return Vector([(a * s, b) for (s, b) in self.terms])

    def __neg__(self):
        return Vector([(-s, b) for (s, b) in self.terms])

    def __len__(self):
        return len(self.terms)

    @staticmethod
    def __eulerian(w):
        g = J
        terms = []

        for k in range(len(w)):
            terms.append(g(w) * Fraction((-1) ** k, k + 1))
            g = conv(g, J)

        return sum(terms)

    def eulerian(self):
        return Vector(
            [
                (s * a, u)
                for (s, b) in self.terms
                for (a, u) in Vector.__eulerian(b).terms
            ]
        )


class Tensor(Vector):
    def __init__(self, terms):
        Vector.__init__(self, terms)

    def __repr__(self):
        strings = [f"{s}⋅{b[0]}⊗{b[1]}" for (s, b) in self.terms]
        return " ".join(strings)

    @staticmethod
    def __shuffle(u, v):
        if u == []:
            return [v]

        if v == []:
            return [u]

        result = [w + [u[-1]] for w in Tensor.__shuffle(u[:-1], v)]
        result += [w + [v[-1]] for w in Tensor.__shuffle(u, v[:-1])]

        return result

    def shuffle(self):
        return Vector(
            [(s, w) for (s, (u, v)) in self.terms for w in Tensor.__shuffle(u, v)]
        )


def J(x):
    return Vector.to_vec(x) if x != [] else Vector([(Fraction(0), [])])


def conv(f, g):
    def u(x):
        if not isinstance(x, Vector):
            x = Vector.to_vec(x)
        return x.conv(f, g)

    return u


if __name__ == "__main__":
    from pprint import pprint

    x = Vector([(Fraction(1), [1, 1, 2, 2])])
    pprint(x.eulerian())
