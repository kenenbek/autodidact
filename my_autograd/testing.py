from third import multiply, add

from third import get_vjp_rule, primitive_vjps, defvjp


def test_get_vjp_rule_for_multiply():
    left_rule = get_vjp_rule(multiply, 0)
    right_rule = get_vjp_rule(multiply, 1)

    assert left_rule is primitive_vjps[multiply][0]
    assert right_rule is primitive_vjps[multiply][1]

    assert left_rule(1.0, 6.0, 2.0, 3.0) == 3.0
    assert right_rule(1.0, 6.0, 2.0, 3.0) == 2.0


if __name__ == '__main__':

    test_get_vjp_rule_for_multiply()