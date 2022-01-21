#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from mathics.core.expression import Expression
from mathics.core.atoms import Number, Real
from mathics.core.symbols import SymbolList
from mathics.core.systemsymbols import SymbolAutomatic, SymbolInfinity
from mathics.builtin.numeric import apply_N

from scipy.optimize import (
    minimize_scalar,
    minimize,
    basinhopping,
    least_squares,
    curve_fit,
    root_scalar,
    root,
)


def get_tolerance_and_maxit(opts: dict, scale=0, evaluation: "Evaluation" = None):
    """
    Looks at an opts dictionary and tries to determine the numeric values of
    Accuracy and Precision goals. If not available, returns None.
    """
    acc_goal = opts.get("System`AccuracyGoal", None)
    if acc_goal:
        acc_goal = apply_N(acc_goal, evaluation)
        if acc_goal is SymbolAutomatic:
            acc_goal = Real(12.0)
        elif acc_goal is SymbolInfinity:
            acc_goal = None
        elif not isinstance(acc_goal, Number):
            acc_goal = None

    prec_goal = opts.get("System`PrecisionGoal", None)
    if prec_goal:
        prec_goal = apply_N(prec_goal, evaluation)
        if prec_goal is SymbolAutomatic:
            prec_goal = Real(12.0)
        elif prec_goal is SymbolInfinity:
            prec_goal = None
        elif not isinstance(prec_goal, Number):
            prec_goal = None

    tol = 0.0
    if acc_goal:
        tol = 10 ** (-acc_goal.to_python())
    if prec_goal and scale:
        tol = tol + scale * 10 ** (-prec_goal.to_python())
    if tol == 0.0:
        tol = None

    maxit_parm = opts["System`MaxIterations"]
    if maxit_parm is SymbolAutomatic:
        maxit = 100
    else:
        if not isinstance(maxit_parm, Number):
            maxit_parm = apply_N(maxit_parm, evaluation)
        maxit = maxit_parm.get_int_value()
    return tol, maxit


def compile_fn(f, x, opts, evaluation):
    """produces a compiled version of f, which is callable from Python"""
    if opts["_isfindmaximum"]:
        f = -f
    comp_func = Expression("Compile", Expression(SymbolList, x), f).evaluate(evaluation)
    return comp_func._leaves[2].cfunc


def process_result_1d_opt(result, opts, evaluation):
    """Process the results"""
    x0 = Real(result.x)
    fopt = Real(result.fun)
    if opts["_isfindmaximum"]:
        fopt = -fopt
    return (x0, fopt), result.success


def find_minimum_brent(
    f: "Expression",
    x0: "Expression",
    x: "Expression",
    opts: dict,
    evaluation: "Evaluation",
) -> (Number, bool):
    """
    This implements the Brent's optimizer
    """
    comp_fun = compile_fn(f, x, opts, evaluation)

    boundary = opts.get("_x0", None)
    if boundary and len(boundary) == 2:
        a, b = sorted(u.to_python() for u in boundary)
    else:
        x0 = apply_N(x0, evaluation)
        b = abs(x0.to_python())
        b = 1 if b == 0 else b
        a = -b
    tol, maxit = get_tolerance_and_maxit(opts, b - a, evaluation)
    result = minimize_scalar(
        comp_fun, (a, b), method="brent", tol=tol, options={"maxiter": maxit}
    )

    return process_result_1d_opt(result, opts, evaluation)


def find_minimum_golden(
    f: "Expression",
    x0: "Expression",
    x: "Expression",
    opts: dict,
    evaluation: "Evaluation",
) -> (Number, bool):
    """
    This implements the Brent's optimizer
    """
    comp_fun = compile_fn(f, x, opts, evaluation)
    boundary = opts.get("_x0", None)
    if boundary and len(boundary) == 2:
        a, b = sorted(u.to_python() for u in boundary)
    else:
        x0 = apply_N(x0, evaluation)
        b = abs(x0.to_python())
        b = 1 if b == 0 else b
        a = -b

    tol, maxit = get_tolerance_and_maxit(opts, b - a, evaluation)

    result = minimize_scalar(
        comp_fun, (a, b), method="golden", tol=tol, options={"maxiter": maxit}
    )

    return process_result_1d_opt(result, opts, evaluation)


scipy_optimizer_methods = {
    "brent": find_minimum_brent,
    "golden": find_minimum_brent,
}
