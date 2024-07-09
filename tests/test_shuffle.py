from pyshuffle.shuffle import Shuffle



def test_coprod():
    x = Shuffle.to_vec([1, 2, 3])
    r = (
        Shuffle.to_vec(([], [1, 2, 3]))
        + Shuffle.to_vec(([1], [2, 3]))
        + Shuffle.to_vec(([1, 2], [3]))
        + Shuffle.to_vec(([1, 2, 3], []))
    )

    assert x.coprod() == r

def test_prod():
    x = Shuffle.to_vec([1, 2])
    y = Shuffle.to_vec([3])
    z = x * y
    r = Shuffle.to_vec([1, 2, 3]) + Shuffle.to_vec([1, 3, 2]) + Shuffle.to_vec([3, 1, 2])

    assert z == r

def test_conv():
    x = Shuffle.to_vec([1, 2])

    r = Shuffle.to_vec([1, 2]) + Shuffle.to_vec([2, 1])
    assert x.conv(lambda a: a.J(), lambda a: a.J()) == r
