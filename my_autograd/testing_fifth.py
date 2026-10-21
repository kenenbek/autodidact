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

def expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")

def test_errors():
    expect(TypeError, lambda: grad(lambda x: x, argnum="0"))
    expect(TypeError, lambda: grad(lambda x: x, argnum=True))
    expect(ValueError, lambda: grad(lambda x: x, argnum=-1))
    expect(ValueError, lambda: grad(lambda x, y: x*y, argnum=1)(3.0))
    expect(ValueError, lambda: grad(lambda x: x)(x=3.0))

def numerical_derivative(function, x, epsilon=1e-6):
    return (function(x + epsilon) - function(x - epsilon)) / (2*epsilon)

def test_against_finite_differences():
    def f(x):
        return x*x*x + 2.0*x + 1.0

    derivative = grad(f)
    for point in (-2.0, 0.0, 0.5, 3.0):
        expected = numerical_derivative(f, point)
        assert abs(derivative(point) - expected) < 1e-6

def loss(weight, input_value, target):
    prediction = weight * input_value
    error = prediction + (-1)*target
    return error*error

def test_loss():
    weight_gradient = grad(loss, argnum=0)
    gradient = weight_gradient(2., 3., 10.)

    assert gradient == -24.0

    learning_rate = 0.01
    updated_weight = 2.0 - learning_rate*gradient

    assert loss(updated_weight, 3.0, 10.0) < loss(2.0, 3.0, 10.0)

if __name__ == '__main__':
    test_single_argument_grad()
    test_subval()
    test_partial_derivatives()
    test_keyword_configuration()
    test_graph_edge_cases()
    test_repeated_calls()
    test_errors()
    test_against_finite_differences()
