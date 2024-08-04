from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # TODO: Implement for Task 1.1.
    f1 = f(*vals[:arg], vals[arg] + epsilon, *vals[arg + 1 :])
    f2 = f(*vals)
    return (f1 - f2) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    var_set = set()
    vars = []
    def visit(n: Variable):
        if n.unique_id in var_set or n.is_constant():
            return
        for node in n.parents:
            visit(node)
        var_set.add(n.unique_id)
        vars.append(n)

    visit(variable)
    yield from vars[::-1]
    


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # var id to its right vars
    right_derivs = defaultdict(float)
    right_derivs[variable.unique_id] = deriv
    # Should I really be yielding variable?
    # Yes, this assumes that deriv isn't actually the deriv of variable, but
    # a value to propagate. And, glancing at the code, that's the assumption. Ok, weird code.
    for var in topological_sort(variable):
        deriv = right_derivs[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(deriv)
        else:
            var_derivs = var.chain_rule(deriv)
            for (_, dv), vp in zip(var_derivs, var.parents):
                right_derivs[vp.unique_id] += dv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
