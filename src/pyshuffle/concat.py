from functools import reduce
from operator import mul
from .vector import Vector


class Concat(Vector):
    def __init__(self, terms=[]):
        super().__init__(terms)

    def coprod(self):
        @Concat.linear_map
        def unshuf_basis(w):
            if len(w) > 1:
                raise ValueError("method not definied for this basis")
            w = w[0]
            if w == []:
                return Concat.to_vec(([], []))

            terms = [Concat.to_vec(([], [a])) + Concat.to_vec(([a], [])) for a in w]
            return reduce(mul, terms, Concat.to_vec(([], [])))

        return unshuf_basis(self)

    def conv(self, f, g):
        @Concat.linear_map
        def conv_basis(w):
            a, b = w
            return f(Concat.to_vec((a,))) * g(Concat.to_vec((b,)))

        tensors = self.coprod()
        return conv_basis(tensors)

    def __mul__(self, other):
        @Concat.linear_map
        def mul_basis(b):
            n = len(b)
            if n % 2 != 0:
                raise ValueError("can only concatenate with same tensor order?")
            return Concat.to_vec(
                tuple(u + v for (u, v) in zip(b[: n // 2], b[n // 2 :]))
            )

        return mul_basis(self.outer(other))

    def J(self):
        @Concat.linear_map
        def J_basis(b):
            return Concat.to_vec(b) if b != ([],) else Concat.zero()

        return J_basis(self)

    def Y(self):
        @Concat.linear_map
        def Y_basis(b):
            return len(b[0]) * Concat.to_vec(b)

        return Y_basis(self)

    def S(self):
        @Concat.linear_map
        def S_basis(b):
            return (-1) ** (len(b[0])) * Concat.to_vec(b[0][::-1])

        return S_basis(self)
