from pyshuffle.concat import Concat


def test_coprod():
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


def test_prod():
    x = Concat.to_vec(([1, 2], [3]))
    y = Concat.to_vec(([4], [5, 6]))
    r = Concat.to_vec(([1, 2, 4], [3, 5, 6]))

    assert x * y == r


def test_conv():
    x = Concat.to_vec([1, 2])

    r = Concat.to_vec([1, 2]) + Concat.to_vec([2, 1])

    assert Concat.conv(Concat.J, Concat.J)(x) == r
