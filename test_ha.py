from shuffle import Vector

@Vector.linear_map
def J(x):
    return Vector.to_vec(x) if x != ([],) else Vector.zero()

def test_deconc():
    x = Vector.to_vec([1, 2, 3])
    r = Vector.to_vec(([], [1, 2, 3])) + Vector.to_vec(([1], [2, 3])) \
        + Vector.to_vec(([1, 2], [3])) + Vector.to_vec(([1, 2, 3], []))

    assert x.deconc() == r


def test_unshuf():
    x = Vector.to_vec([1, 2, 3])
    r = Vector.to_vec(([], [1, 2, 3])) + Vector.to_vec(([1], [2, 3])) \
        + Vector.to_vec(([2], [1, 3])) + Vector.to_vec(([3], [1, 2])) \
        + Vector.to_vec(([1, 2], [3])) + Vector.to_vec(([1, 3], [2])) \
        + Vector.to_vec(([2, 3], [1])) + Vector.to_vec(([1, 2, 3], []))

    assert x.unshuf() == r


def test_conc():
    x = Vector.to_vec(([1, 2], [3]))
    y = Vector.to_vec(([4], [5, 6]))
    r = Vector.to_vec(([1, 2, 4], [3, 5, 6]))

    assert x * y == r


def test_shuf():
    x = Vector.to_vec([1, 2])
    y = Vector.to_vec([3])
    r = Vector.to_vec([1, 2, 3]) + Vector.to_vec([1, 3, 2]) \
        + Vector.to_vec([3, 1, 2])

    assert x.shuffle(y) == r


def test_sh_conv():
    x = Vector.to_vec([1, 2])

    r = Vector.to_vec([1, 2]) + Vector.to_vec([2, 1])
    assert x.shuffle_conv(J, J) == r


def test_cat_conv():
    x = Vector.to_vec([1, 2])

    r = Vector.to_vec([1, 2]) + Vector.to_vec([2, 1])

    assert x.cat_conv(J, J) == r
