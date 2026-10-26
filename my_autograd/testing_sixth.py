import numpy as onp
from third import unbroadcast

def test_unbroadcast():
    g = onp.ones((2, 3))
    onp.testing.assert_allclose(unbroadcast(onp.zeros(3), g), [2, 2, 2])
    onp.testing.assert_allclose(unbroadcast(onp.zeros((2, 1)), g), [[3], [3]])
    scalar_gradient = unbroadcast(0.0, g)
    assert onp.shape(scalar_gradient) == ()


if __name__ == '__main__':
    test_unbroadcast()
