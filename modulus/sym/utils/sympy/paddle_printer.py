# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper functions for converting sympy equations to paddle
"""

from sympy import lambdify, Symbol, Derivative, Function, Basic, Add, Max, Min
from sympy.printing.str import StrPrinter
import paddle
import numpy as np
import functools
from typing import List, Dict

from modulus.sym.constants import diff_str, tf_dt


def paddle_lambdify(f, r, separable=False):
    """
    generates a Paddle function from a sympy equation

    Parameters
    ----------
    f : Sympy Exp, float, int, bool
      the equation to convert to paddle.
      If float, int, or bool this gets converted
      to a constant function of value `f`.
    r : list, dict
      A list of the arguments for `f`. If dict then
      the keys of the dict are used.

    Returns
    -------
    paddle_f : Paddle function
    """

    try:
        f = float(f)
    except:
        pass
    if isinstance(f, (float, int, bool)):  # constant function

        def loop_lambda(constant):
            return lambda **x: paddle.zeros_like(next(iter(x.items()))[1]) + constant

        lambdify_f = loop_lambda(f)
    else:
        vars = [k for k in r] if separable else [[k for k in r]]
        try:  # NOTE this fixes a very odd bug in SymPy TODO add issue to SymPy
            lambdify_f = lambdify(
                vars, f, [{**PADDLE_SYMPY_PRINTER, **PADDLE_CUSTOM_SYMPY_PRINTER}]
            )
        except:
            lambdify_f = lambdify(
                vars, f, [{**PADDLE_SYMPY_PRINTER, **PADDLE_CUSTOM_SYMPY_PRINTER}]
            )
    return lambdify_f


def _where_paddle(conditions, x, y):
    if isinstance(x, (int, float)):
        x = float(x) * paddle.ones(conditions.get_shape())
    if isinstance(y, (int, float)):
        y = float(y) * paddle.ones(conditions.get_shape())
    return paddle.where(conditions, x, y)


def _heaviside_paddle(x, values=0):
    return paddle.maximum(
        paddle.sign(x),
        paddle.zeros(
            [
                1,
            ]
        ),
    )


def _sqrt_paddle(x):
    return paddle.sqrt((x - 1e-6) * _heaviside_paddle(x - 1e-6) + 1e-6)


def _or_paddle(*x):
    return_value = x[0]
    for value in x:
        return_value = paddle.logical_or(return_value, value)
    return return_value


def _and_paddle(*x):
    return_value = x[0]
    for value in x:
        return_value = paddle.logical_and(return_value, value)
    return return_value


def _min_jit(x: List[paddle.Tensor]):
    assert len(x) > 0
    min_tensor = x[0]
    for i in range(1, len(x)):
        min_tensor = paddle.minimum(min_tensor, y=x[i])
    return min_tensor


def _min_paddle(*x):
    # method 1
    # assert isinstance(x[0], (int, float))
    # result = paddle.clip(x[1], max=x[0])
    # return result

    # method 2
    # get tensor shape
    for value in x:
        if not isinstance(value, (int, float)):
            tensor_shape = list(map(int, value.shape))

    # convert all floats and ints to tensor
    x_only_tensors = []
    for value in x:
        if isinstance(value, (int, float)):
            value = paddle.full(tensor_shape, value)
        x_only_tensors.append(value)

    min_tensor = x_only_tensors[0]
    for tmp in x_only_tensors[1:]:
        min_tensor = paddle.minimum(min_tensor, tmp)

    # jit option
    # return _min_jit(x_only_tensors)

    # method 3
    # min_tensor = paddle.min(x=paddle.stack(x=x_only_tensors, axis=-1), axis=-1)

    return min_tensor


def _max_jit(x: List[paddle.Tensor]):
    assert len(x) > 0
    max_tensor = x[0]
    for i in range(1, len(x)):
        max_tensor = paddle.maximum(max_tensor, x[i])
    return max_tensor


def _max_paddle(*x):
    # method 1
    # return paddle.clip(x[1], min=x[0])

    # method 2
    # get tensor shape
    for value in x:
        if not isinstance(value, (int, float)):
            tensor_shape = list(map(int, value.shape))

    # convert all floats and ints to tensor
    x_only_tensors = []
    for value in x:
        if isinstance(value, (int, float)):
            value = paddle.full(tensor_shape, value)
        x_only_tensors.append(value)

    max_tensor = x_only_tensors[0]
    for tmp in x_only_tensors[1:]:
        max_tensor = paddle.maximum(max_tensor, tmp)

    # method 3
    # paddle.max 高阶微分不支持
    # max_tensor = paddle.max(x=paddle.stack(x=x_only_tensors, axis=-1), axis=-1)
    return max_tensor

    # jit option
    # return _max_jit(x_only_tensors)


def custom_exp(x, e=paddle.to_tensor(np.e)):
    return paddle.pow(e, x)


def _dirac_delta_paddle(x):
    return paddle.equal(x=x, y=0.0)


def softplus(x):
    # Numeric stable version of softplus
    THRESHOLD = 20.0
    gt_mask = (x > THRESHOLD).astype(x.dtype)
    le_mask = 1 - gt_mask
    x_le = le_mask * x
    y = gt_mask * x + le_mask * (  # keep the original value for x > THRESHOLD
        paddle.log(1 + paddle.exp(x_le))
    )  # compute 1+e^x for x <= THRESHOLD
    return y


PADDLE_SYMPY_PRINTER = {
    "abs": paddle.abs,
    "Abs": paddle.abs,
    "sign": paddle.sign,
    "ceiling": paddle.ceil,
    "floor": paddle.floor,
    "log": paddle.log,
    "exp": paddle.exp,
    "sqrt": _sqrt_paddle,
    "cos": paddle.cos,
    "acos": paddle.acos,
    "sin": paddle.sin,
    "asin": paddle.asin,
    "tan": paddle.tan,
    "atan": paddle.atan,
    "atan2": paddle.atan2,
    "cosh": paddle.cosh,
    "acosh": paddle.acosh,
    "sinh": paddle.sinh,
    "asinh": paddle.asinh,
    "tanh": paddle.tanh,
    "atanh": paddle.atanh,
    "erf": paddle.erf,
    "loggamma": paddle.lgamma,
    "Min": _min_paddle,
    "Max": _max_paddle,
    "Heaviside": _heaviside_paddle,
    "DiracDelta": _dirac_delta_paddle,
    "logical_or": _or_paddle,
    "logical_and": _and_paddle,
    "where": _where_paddle,
    "pi": np.pi,
    "conjugate": paddle.conj,
}


PADDLE_CUSTOM_SYMPY_PRINTER = {
    "softplus": softplus,
}


class CustomDerivativePrinter(StrPrinter):
    def _print_Function(self, expr):
        """
        Custom printing of the SymPy Derivative class.
        Instead of:
        D(x(t), t)
        We will print:
        x__t
        """
        return expr.func.__name__

    def _print_Derivative(self, expr):
        """
        Custom printing of the SymPy Derivative class.
        Instead of:
        D(x(t), t)
        We will print:
        x__t
        """
        prefix = str(expr.args[0].func)
        for expr in expr.args[1:]:
            prefix += expr[1] * (diff_str + str(expr[0]))
        return prefix


def _subs_derivatives(expr):
    while True:
        try:
            deriv = expr.atoms(Derivative).pop()
            new_fn_name = str(deriv)
            expr = expr.subs(deriv, Function(new_fn_name)(*deriv.free_symbols))
        except:
            break
    while True:
        try:
            fn = {
                fn
                for fn in expr.atoms(Function)
                if (
                    fn.class_key()[1] == 0
                    and fn.name not in PADDLE_CUSTOM_SYMPY_PRINTER
                )
            }.pop()  # check if standard Sympy Eq (TODO better check)
            new_symbol_name = str(fn)
            expr = expr.subs(fn, Symbol(new_symbol_name))
        except:
            break
    return expr


# Override the __str__ method of to use CustromStrPrinter
Basic.__str__ = lambda self: CustomDerivativePrinter().doprint(self)


# Class to compile and evaluate a sympy expression in Paddle
# Cannot currently script this module because self.paddle_expr is unknown
class SympyToPaddle(paddle.nn.Layer):
    def __init__(
        self,
        sympy_expr,
        name: str,
        freeze_terms: List[int] = [],
        detach_names: List[str] = [],
    ):
        super().__init__()
        # Sort keys to guarantee ordering
        self.keys = sorted([k.name for k in sympy_expr.free_symbols])
        self.freeze_terms = freeze_terms
        if not self.freeze_terms:
            self.paddle_expr = paddle_lambdify(sympy_expr, self.keys)
        else:
            assert all(
                x < len(Add.make_args(sympy_expr)) for x in freeze_terms
            ), "The freeze term index cannot be larger than the total terms in the expression"
            self.paddle_expr = []
            for i in range(len(Add.make_args(sympy_expr))):
                self.paddle_expr.append(
                    paddle_lambdify(Add.make_args(sympy_expr)[i], self.keys)
                )
            self.freeze_list = list(self.paddle_expr[i] for i in freeze_terms)
        self.name = name
        self.detach_names = detach_names

    def forward(self, var: Dict[str, paddle.Tensor]) -> Dict[str, paddle.Tensor]:
        args = [
            var[k].detach() if k in self.detach_names else var[k] for k in self.keys
        ]
        if not self.freeze_terms:
            output = self.paddle_expr(args)
        else:
            output = paddle.zeros_like(var[self.keys[0]])
            for i, expr in enumerate(self.paddle_expr):
                if expr in self.freeze_list:
                    output += expr(args).detach()
                else:
                    output += expr(args)

        return {self.name: output}
