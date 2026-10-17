from third import multiply, add
from third import trace
from third import get_vjp_rule, primitive_vjps, defvjp


def test_get_vjp_rule_for_multiply():
    left_rule = get_vjp_rule(multiply, 0)
    right_rule = get_vjp_rule(multiply, 1)

    assert left_rule is primitive_vjps[multiply][0]
    assert right_rule is primitive_vjps[multiply][1]

    assert left_rule(1.0, 6.0, 2.0, 3.0) == 3.0
    assert right_rule(1.0, 6.0, 2.0, 3.0) == 2.0


def test_trace_connected_output():
    value, end_node, start_node = trace(lambda x: x*x, 3.0)

    assert value == 9
    assert end_node is not None
    assert start_node.recipe is None
    assert end_node.recipe.function is multiply


def test_trace_constant_output():
    value, end_node, start_node = trace(lambda x: 7.0, 3.0)

    assert value == 7.0
    assert end_node is None
    assert start_node.recipe is None

if __name__ == '__main__':

    test_trace_connected_output()
    test_trace_constant_output()
