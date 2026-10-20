from fifth import grad
from fifth import subval


def test_single_argument_grad():
    def square(x):
        return x*x

    derivative = grad(square)
    assert callable(derivative)
    assert derivative(3.0) == 6.0
    assert derivative(5.0) == 10.0

def test_subval():
    original = (3.0, 4.0, 5.0)
    marker = object()
    changed = subval(original, 1, marker)

    assert isinstance(changed, tuple)
    assert changed[0] == 3.0
    assert changed[1] is marker
    assert changed[2] == 5.0
    assert original == (3.0, 4.0, 5.0)

def test_partial_derivatives():
    def f(x, y):
        return x*x*y + y

    assert grad(f, argnum=0)(3.0, 4.0) == 24.0
    assert grad(f, argnum=1)(3.0, 4.0) == 10.0
    assert grad(f)(3.0, 4.0) == 24.0

def test_keyword_configuration():
    def scaled_square(x, *, scale=1.0):
        return scale*x*x
    assert grad(scaled_square)(3.0, scale=2.0) == 12.0

def test_graph_edge_cases():
    assert grad(lambda x: x)(3.0) == 1.0
    assert grad(lambda x: 7.0)(3.0) == 0.0
    def f(x, y):
        return x*x
    assert grad(f, argnum=1)(3.0, 4.0) == 0.0

def test_repeated_calls():
    calls = 0

    def cube(x):
        nonlocal calls
        calls += 1
        return x*x*x

    derivative = grad(cube)
    assert calls == 0

    assert derivative(2.0) == 12.0
    assert calls == 1

    assert derivative(3.0) == 27.0
    assert calls == 2

    assert derivative(2.0) == 12.0
    assert calls == 3

if __name__ == '__main__':
    test_single_argument_grad()
    test_subval()
    test_partial_derivatives()
    test_keyword_configuration()
    test_graph_edge_cases()
    test_repeated_calls()
