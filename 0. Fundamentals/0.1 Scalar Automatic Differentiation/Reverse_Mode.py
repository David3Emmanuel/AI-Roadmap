import math

class RValue:
    _prev: list[tuple['RValue', float]]

    def __init__(self, val) -> None:
        self._prev = []
        self.val = val
        self.grad = 0.0

    def backward(self):
        ordered_nodes: list[RValue] = []
        visited = set()

        def build_graph(node: RValue):
            if node in visited:
                return
            node.grad = 0.0
            visited.add(node)
            for parent, _ in node._prev:
                build_graph(parent)
            ordered_nodes.append(node)

        build_graph(self)
        ordered_nodes.reverse()

        self.grad = 1.0
        for node in ordered_nodes:
            for parent, parent_grad in node._prev:
                parent.grad += node.grad * parent_grad

    def __add__(self, other) -> 'RValue':
        if not isinstance(other, type(self)):
            other = RValue(other)
        out = RValue(self.val + other.val)
        out._prev = [(self, 1.0), (other, 1.0)]
        return out

    def __radd__(self, other) -> 'RValue':
        return self + other

    def __sub__(self, other) -> 'RValue':
        if not isinstance(other, type(self)):
            other = RValue(other)
        out = RValue(self.val - other.val)
        out._prev = [(self, 1.0), (other, -1.0)]
        return out

    def __rsub__(self, other) -> 'RValue':
        if not isinstance(other, type(self)):
            other = RValue(other)
        return other - self

    def __mul__(self, other) -> 'RValue':
        if not isinstance(other, type(self)):
            other = RValue(other)
        out = RValue(self.val * other.val)
        out._prev = [(self, other.val), (other, self.val)]
        return out

    def __rmul__(self, other) -> 'RValue':
        return self * other

    def __truediv__(self, other) -> 'RValue':
        if not isinstance(other, type(self)):
            other = RValue(other)
        out = RValue(self.val / other.val)
        out._prev = [(self, 1 / other.val), (other, -self.val / (other.val ** 2))]
        return out

    def __rtruediv__(self, other) -> 'RValue':
        if not isinstance(other, type(self)):
            other = RValue(other)
        return other / self

    def __pow__(self, other) -> 'RValue':
        if not isinstance(other, type(self)):
            other = RValue(other)
        out = RValue(self.val ** other.val)
        out._prev = [
            (self, other.val * self.val ** (other.val - 1)),
            (other, self.val ** other.val * math.log(self.val)),
        ]
        return out

    def __rpow__(self, other) -> 'RValue':
        if not isinstance(other, type(self)):
            other = RValue(other)
        return other ** self

    def __repr__(self) -> str:
        return f'RValue({self.val}, {self.grad})'

def Output(out: RValue) -> RValue:
    out.backward()
    return out

if __name__ == "__main__":
    x = RValue(1)
    y = RValue(1)

    z = Output((2 * x + y) / (5 * x - y))

    print(x, y, z)
