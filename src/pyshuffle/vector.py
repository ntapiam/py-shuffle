from fractions import Fraction


class Vector:
    def __init__(self, terms=[]):
        self.terms = terms
        self.__normalize()

    def __normalize(self):
        if self.terms == []:
            return
        self.terms.sort(key=lambda x: x[1])
        out = [self.terms[0]]
        for s, b in self.terms[1:]:
            c = out[-1]
            if b == c[1]:
                out[-1] = (s + c[0], b)
            else:
                out.append((s, b))

        self.terms = list(filter(lambda x: x[0] != 0, out))
        self.terms = [
            (s, tuple(self.__class__.__flatten(b)))
            if isinstance(b, tuple)
            else (s, (b,))
            for (s, b) in self.terms
        ]

    @classmethod
    def linear_map(cls, func):
        def lin_ext(v):
            return cls([(s * a, u) for (s, b) in v.terms for (a, u) in func(b).terms])

        return lin_ext

    @classmethod
    def to_vec(cls, b):
        return cls([(Fraction(1), b)])

    @classmethod
    def zero(cls, b=[]):
        return cls([(Fraction(0), b)])

    @classmethod
    def __flatten(cls, t):
        for i in t:
            yield from [i] if not isinstance(i, tuple) else cls.__flatten(i)

    def outer(self, other):
        @self.__class__.linear_map
        def outer_basis(b):
            return self.__class__(
                [(s, tuple(self.__class__.__flatten((a, b)))) for (s, a) in self.terms]
            )

        return outer_basis(other)

    def __add__(self, other):
        if isinstance(other, (int, float, Fraction)):
            other = self.__class__([(other, [])])

        return self.__class__(self.terms + other.terms)

    def __sub__(self, other):
        if isinstance(other, (int, float, Fraction)):
            other = self.__class__([(other, [])])

        return self + Fraction(-1) * other

    def __neg__(self):
        return Fraction(-1) * self

    def __eq__(self, other):
        return self.terms == other.terms

    def __repr__(self):
        def coef_to_string(k, s):
            if k == 0:
                if s >= 0:
                    return f"{s}⋅" if s != 1 else ""
                else:
                    return f"{s}⋅" if s != -1 else "-"

            else:
                if s >= 0:
                    return f" + {s}⋅" if s != 1 else " + "
                else:
                    return f" - {-s}⋅" if s != -1 else " - "

        strings = [
            f"{coef_to_string(k, s)}{'⊗'.join(map(lambda a: ''.join(map(str, a)), b))}"
            for (k, (s, b)) in enumerate(self.terms)
        ]
        return "".join(strings) if strings else "0"

    def __rmul__(self, s):
        s = Fraction(s) if isinstance(s, int) else s
        return self.__class__([(r * s, b) for (r, b) in self.terms])

    def __truediv__(self, s):
        s = 1 / Fraction(s) if isinstance(s, (int, Fraction)) else 1 / s
        return self.__class__([(r * s, b) for (r, b) in self.terms])
