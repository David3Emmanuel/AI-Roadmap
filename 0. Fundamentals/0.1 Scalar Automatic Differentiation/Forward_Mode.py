import math


class FValue:
    def __init__(self, val, grad=0.) -> None:
        self.val = val
        self.grad = grad

    def __add__(self, other) -> 'FValue':
        if not isinstance(other, type(self)): other = FValue(other)
        val = self.val + other.val
        grad = self.grad + other.grad
        return FValue(val, grad)

    def __radd__(self, other) -> 'FValue':
        return self + other

    def __sub__(self, other) -> 'FValue':
        if not isinstance(other, type(self)): other = FValue(other)
        val = self.val - other.val
        grad = self.grad - other.grad
        return FValue(val, grad)

    def __rsub__(self, other) -> 'FValue':
        if not isinstance(other, type(self)): other = FValue(other)
        return other - self

    def __mul__(self, other) -> 'FValue':
        if not isinstance(other, type(self)): other = FValue(other)
        val = self.val * other.val
        grad = self.grad * other.val + self.val * other.grad
        return FValue(val, grad)

    def __rmul__(self, other) -> 'FValue':
        return self * other

    def __truediv__(self, other) -> 'FValue':
        if not isinstance(other, type(self)): other = FValue(other)
        val = self.val / other.val
        grad = (self.grad * other.val - self.val * other.grad) / (other.val ** 2)
        return FValue(val, grad)

    def __rtruediv__(self, other) -> 'FValue':
        if not isinstance(other, type(self)): other = FValue(other)
        return other / self

    def __pow__(self, other) -> 'FValue':
        if not isinstance(other, type(self)): other = FValue(other)
        val = self.val ** other.val
        grad = (other.val * self.val ** (other.val - 1)) * self.grad + \
               (self.val ** other.val * math.log(self.val)) * other.grad
        return FValue(val, grad)

    def __rpow__(self, other) -> 'FValue':
        if not isinstance(other, type(self)): other = FValue(other)
        return other ** self

    def __repr__(self) -> str:
        return f'Value({self.val}, {self.grad})'


def Input(val):
    return FValue(val, 1)


def f(x, y):
    return (2 * x + y) / (5 * x - y)


if __name__ == '__main__':
    z1 = f(Input(1), FValue(1))
    print(z1)
    z2 = f(FValue(1), Input(1))
    print(z2)
