from pyshuffle.shuffle import Shuffle
from pyshuffle.concat import Concat


@Shuffle.linear_map
def J_sh(x):
    return Shuffle.to_vec(x) if x != ([],) else Shuffle.zero()

@Concat.linear_map
def J_cat(x):
    return Concat.to_vec(x) if x != ([],) else Concat.zero()

def test_deconc():
    x = Shuffle.to_vec([1, 2, 3])
    r = (
        Shuffle.to_vec(([], [1, 2, 3]))
        + Shuffle.to_vec(([1], [2, 3]))
        + Shuffle.to_vec(([1, 2], [3]))
        + Shuffle.to_vec(([1, 2, 3], []))
    )

    assert x.coprod() == r


def test_unshuf():
    x = Concat.to_vec([1, 2, 3])
    r = (
        Concat.to_vec(([], [1, 2, 3]))
        + Concat.to_vec(([1], [2, 3]))
        + Concat.to_vec(([2], [1, 3]))
        + Concat.to_vec(([3], [1, 2]))
        + Concat.to_vec(([1, 2], [3]))
        + Concat.to_vec(([1, 3], [2]))
        + Concat.to_vec(([2, 3], [1]))
        + Concat.to_vec(([1, 2, 3], []))
    )

    assert x.coprod() == r


def test_conc():
    x = Concat.to_vec(([1, 2], [3]))
    y = Concat.to_vec(([4], [5, 6]))
    r = Concat.to_vec(([1, 2, 4], [3, 5, 6]))

    assert x * y == r


def test_shuf():
    x = Shuffle.to_vec([1, 2])
    y = Shuffle.to_vec([3])
    z = x * y
    r = Shuffle.to_vec([1, 2, 3]) + Shuffle.to_vec([1, 3, 2]) + Shuffle.to_vec([3, 1, 2])

    assert z == r


def test_sh_conv():
    x = Shuffle.to_vec([1, 2])

    r = Shuffle.to_vec([1, 2]) + Shuffle.to_vec([2, 1])
    assert x.conv(J_sh, J_sh) == r


def test_cat_conv():
    x = Concat.to_vec([1, 2])

    r = Concat.to_vec([1, 2]) + Concat.to_vec([2, 1])

    assert x.conv(J_cat, J_cat) == r
