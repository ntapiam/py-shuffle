from shuffle import Vector
from fractions import Fraction
from functools import reduce
from operator import add


def test_add():
    x = Vector.to_vec([1, 2, 3])
    y = Vector.to_vec([3, 2, 1])
    z = x + y
    assert z == Vector([(Fraction(1), [1, 2, 3]), (Fraction(1), [3, 2, 1])])


def test_lin_ext():
    x = Vector.to_vec([1, 2])

    @Vector.linear_map
    def f(w):
        return reduce(add, [Vector.to_vec((w[0][:k], w[0][k:])) for k in range(len(w[0]) + 1)], 0 * Vector.to_vec(([], [])))

    assert f(x) == Vector.to_vec(([], [1, 2])) + \
        Vector.to_vec(([1], [2])) + Vector.to_vec(([1, 2], []))


def test_outer():
    x = Vector.to_vec([1, 2]) + 3 * Vector.to_vec([3, 4])
    y = Vector.to_vec([1, 2])

    assert x.outer(y) == Vector.to_vec(
        ([1, 2], [1, 2])) + 3 * Vector.to_vec(([3, 4], [1, 2]))
