"""Work out what string a snippet would build, without running it.

An attacker who can put code in a prompt can put anything in a prompt, so
this never executes what it reads. It parses to an AST and walks a closed
whitelist of node types, computing values itself. There is no `exec`, no
`eval`, no `compile`, and no builtins lookup anywhere in this module: the
handful of callables it understands are matched by name against a table and
implemented here, so a snippet naming `open` or `__import__` finds nothing to
call rather than finding the real one.

Two bounds keep a hostile snippet from being expensive rather than dangerous.
A step budget stops an expansion that would otherwise run for a long time,
and a length cap stops one that would otherwise allocate a lot. Both are
counted as the walk proceeds, so a snippet cannot spend the budget before the
check notices.

Where this differs from `benchmark.transformations.code_snippet`: that module
recognises the three shapes its own generator emits, which is all a benchmark
needs. A defence does not get to assume the attacker used the generator, so
this evaluates the subset rather than pattern-matching it, and reports what
it rejected so a caller can tell "not code I understand" from "code I refuse
to run".
"""

import ast
from collections.abc import Callable, Sequence
from typing import Any

from tipguard.base import FrozenModel

#: Longest source this will parse. Parsing is itself attacker-controlled
#: work, so the bound comes before `ast.parse`, not after.
MAX_SOURCE_LENGTH = 20_000

#: Longest string any intermediate value may reach. Checked at every step
#: that can grow one, so a snippet cannot build a huge value and be caught
#: only at the end.
MAX_STRING_LENGTH = 10_000

#: How many AST nodes may be evaluated before the walk gives up.
MAX_STEPS = 10_000

#: Longest sequence a comprehension may iterate.
MAX_ITERATIONS = 10_000

BUDGET_EXCEEDED = "budget_exceeded"


class CodeAnalysis(FrozenModel):
    """What a snippet builds, or why this would not work it out.

    Exactly one of `result` and `rejected_reason` is set. `operations` names
    the whitelisted calls the snippet used, in the order they were reduced,
    so a trace can say what a snippet did without quoting the snippet.
    """

    result: str | None
    operations: tuple[str, ...]
    rejected_reason: str | None


class UnsupportedCodeError(Exception):
    """A snippet outside the subset, or one that exceeded a bound."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _Budget:
    """Step and size limits, checked as the walk goes rather than after."""

    __slots__ = ("steps",)

    def __init__(self) -> None:
        self.steps = 0

    def step(self) -> None:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise UnsupportedCodeError(BUDGET_EXCEEDED)

    @staticmethod
    def check_string(value: str) -> str:
        if len(value) > MAX_STRING_LENGTH:
            raise UnsupportedCodeError(BUDGET_EXCEEDED)
        return value

    @staticmethod
    def check_length(length: int) -> None:
        """Refuse an oversized result before it is built.

        Every string operation here can compute its own output length from
        its inputs, so the cap is applied to that number rather than to the
        finished value. Checking afterwards still allocates first: a
        `replace` that expands its input reached 4.5 MB before being
        refused, which is the cost the cap exists to avoid paying.
        """
        if length > MAX_STRING_LENGTH:
            raise UnsupportedCodeError(BUDGET_EXCEEDED)

    @staticmethod
    def check_sequence(value: list[object]) -> list[object]:
        """Bound a list the way a string is bounded.

        Only strings were capped, so a chain of list concatenations doubled
        its way to 213 MB before the step budget noticed: twenty-one
        doublings is twenty-one steps, and each one allocates rather than
        looping. A list is as cheap to write and as expensive to hold.
        """
        if len(value) > MAX_ITERATIONS:
            raise UnsupportedCodeError(BUDGET_EXCEEDED)
        return value


def _require_str(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise UnsupportedCodeError(f"{what} expects a string")
    return value


def _require_int(value: object, what: str) -> int:
    # `bool` is an `int` subclass; a snippet writing `chr(True)` is outside
    # anything the generator emits and is not worth admitting.
    if not isinstance(value, int) or isinstance(value, bool):
        raise UnsupportedCodeError(f"{what} expects an integer")
    return value


def _call_join(target: object, args: Sequence[object]) -> str:
    separator = _require_str(target, "join")
    if len(args) != 1 or not isinstance(args[0], (list, tuple)):
        raise UnsupportedCodeError("join expects one sequence")
    parts = [_require_str(part, "join") for part in args[0]]
    if parts:
        _Budget.check_length(sum(len(part) for part in parts) + len(separator) * (len(parts) - 1))
    return separator.join(parts)


def _call_replace(target: object, args: Sequence[object]) -> str:
    text = _require_str(target, "replace")
    if len(args) != 2:
        raise UnsupportedCodeError("replace expects two arguments")
    old = _require_str(args[0], "replace")
    new = _require_str(args[1], "replace")
    if not old:
        # `"abc".replace("", "x")` inserts between every character, which is
        # a cheap way to multiply a string; nothing legitimate needs it.
        raise UnsupportedCodeError("replace with an empty pattern")
    _Budget.check_length(len(text) + text.count(old) * (len(new) - len(old)))
    return text.replace(old, new)


def _call_chr(_target: object, args: Sequence[object]) -> str:
    if len(args) != 1:
        raise UnsupportedCodeError("chr expects one argument")
    code = _require_int(args[0], "chr")
    if not 0 <= code <= 0x10FFFF:
        raise UnsupportedCodeError("chr out of range")
    return chr(code)


def _call_ord(_target: object, args: Sequence[object]) -> int:
    if len(args) != 1:
        raise UnsupportedCodeError("ord expects one argument")
    text = _require_str(args[0], "ord")
    if len(text) != 1:
        raise UnsupportedCodeError("ord expects one character")
    return ord(text)


def _call_reversed(_target: object, args: Sequence[object]) -> list[object]:
    if len(args) != 1:
        raise UnsupportedCodeError("reversed expects one argument")
    value = args[0]
    if isinstance(value, str):
        return list(reversed(value))
    if isinstance(value, (list, tuple)):
        return list(reversed(value))
    raise UnsupportedCodeError("reversed expects a string or sequence")


def _call_split(target: object, args: Sequence[object]) -> list[str]:
    text = _require_str(target, "split")
    if not args:
        return text.split()
    if len(args) != 1:
        raise UnsupportedCodeError("split expects at most one argument")
    separator = _require_str(args[0], "split")
    if not separator:
        raise UnsupportedCodeError("split on an empty separator")
    return text.split(separator)


def _nullary(name: str, method: Callable[[str], str]) -> Callable[..., str]:
    """A no-argument string method, bound by reference rather than by name.

    Deliberately not `getattr(text, name)`: that is a dynamic attribute
    lookup on a value, which is the one shape this module exists to avoid.
    Holding the unbound method directly means the table cannot be made to
    reach a method that is not in it.
    """

    def call(target: object, args: Sequence[object]) -> str:
        if args:
            raise UnsupportedCodeError(f"{name} expects no arguments")
        return method(_require_str(target, name))

    return call


#: Every call this understands, by name. A snippet naming anything else --
#: `open`, `__import__`, `eval` -- finds no entry and is rejected; the name is
#: never resolved against real builtins, so there is nothing to reach even if
#: the table were wrong.
_METHODS: dict[str, Callable[[object, Sequence[object]], Any]] = {
    "join": _call_join,
    "replace": _call_replace,
    "upper": _nullary("upper", str.upper),
    "lower": _nullary("lower", str.lower),
    "strip": _nullary("strip", str.strip),
    "split": _call_split,
}

_FUNCTIONS: dict[str, Callable[[object, Sequence[object]], Any]] = {
    "chr": _call_chr,
    "ord": _call_ord,
    "reversed": _call_reversed,
}


class RestrictedCodeAnalyzer:
    """Evaluates the whitelisted subset and refuses everything else."""

    name = "restricted_code_analyzer"

    def analyze(self, source: str) -> CodeAnalysis:
        """What `source` builds, or why this would not work it out.

        Never raises for a bad snippet: a rejection is a value, because a
        prompt is allowed to contain nonsense and the pipeline has to carry
        on with its other views.
        """
        operations: list[str] = []
        try:
            module = self._parse(source)
            result = self._run(module, operations)
        except UnsupportedCodeError as exc:
            return CodeAnalysis(
                result=None, operations=tuple(operations), rejected_reason=exc.reason
            )
        except RecursionError:
            # A deeply nested expression can exhaust the walk's own recursion
            # before the step budget notices. That is an unsupported snippet,
            # not a crash.
            return CodeAnalysis(
                result=None, operations=tuple(operations), rejected_reason=BUDGET_EXCEEDED
            )
        return CodeAnalysis(result=result, operations=tuple(operations), rejected_reason=None)

    @staticmethod
    def _parse(source: str) -> ast.Module:
        if len(source) > MAX_SOURCE_LENGTH:
            raise UnsupportedCodeError(BUDGET_EXCEEDED)
        try:
            return ast.parse(source)
        except (SyntaxError, ValueError, MemoryError) as exc:
            raise UnsupportedCodeError("not parseable as Python") from exc

    def _run(self, module: ast.Module, operations: list[str]) -> str:
        budget = _Budget()
        scope: dict[str, object] = {}
        last: object = None
        for statement in module.body:
            last = self._statement(statement, scope, budget, operations)
        if isinstance(last, str):
            return last
        # The generator's snippets end by assigning the message to a name, so
        # fall back to that before giving up.
        target = scope.get("msg")
        if isinstance(target, str):
            return target
        raise UnsupportedCodeError("snippet does not build a string")

    def _statement(
        self,
        node: ast.stmt,
        scope: dict[str, object],
        budget: _Budget,
        operations: list[str],
    ) -> object:
        budget.step()
        if isinstance(node, ast.Assign):
            value = self._expression(node.value, scope, budget, operations)
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    raise UnsupportedCodeError("only plain name assignment")
                scope[target.id] = value
            return value
        if isinstance(node, ast.Expr):
            return self._expression(node.value, scope, budget, operations)
        raise UnsupportedCodeError(f"statement {type(node).__name__}")

    def _expression(
        self,
        node: ast.expr,
        scope: dict[str, object],
        budget: _Budget,
        operations: list[str],
    ) -> object:
        budget.step()
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return _Budget.check_string(node.value)
            if isinstance(node.value, int) and not isinstance(node.value, bool):
                return node.value
            raise UnsupportedCodeError("only string and integer constants")
        if isinstance(node, ast.Name):
            if node.id not in scope:
                # Never resolved against builtins: an unknown name is a
                # rejection, not a lookup.
                raise UnsupportedCodeError(f"unknown name {node.id!r}")
            return scope[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return _Budget.check_sequence(
                [self._expression(item, scope, budget, operations) for item in node.elts]
            )
        if isinstance(node, ast.BinOp):
            return self._binop(node, scope, budget, operations)
        if isinstance(node, ast.Call):
            return self._call(node, scope, budget, operations)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            return self._comprehension(node, scope, budget, operations)
        raise UnsupportedCodeError(f"expression {type(node).__name__}")

    def _binop(
        self,
        node: ast.BinOp,
        scope: dict[str, object],
        budget: _Budget,
        operations: list[str],
    ) -> object:
        # Only string concatenation. Multiplication is deliberately absent:
        # `"a" * 10**7` is one node that allocates megabytes, and the length
        # cap would catch it only after the allocation.
        if not isinstance(node.op, ast.Add):
            raise UnsupportedCodeError(f"operator {type(node.op).__name__}")
        left = self._expression(node.left, scope, budget, operations)
        right = self._expression(node.right, scope, budget, operations)
        if isinstance(left, str) and isinstance(right, str):
            _Budget.check_length(len(left) + len(right))
            return left + right
        if isinstance(left, list) and isinstance(right, list):
            if len(left) + len(right) > MAX_ITERATIONS:
                # Checked before the concatenation, not after: the point is
                # not to allocate the oversized value in the first place.
                raise UnsupportedCodeError(BUDGET_EXCEEDED)
            return [*left, *right]
        raise UnsupportedCodeError("addition of unsupported types")

    def _call(
        self,
        node: ast.Call,
        scope: dict[str, object],
        budget: _Budget,
        operations: list[str],
    ) -> object:
        if node.keywords:
            raise UnsupportedCodeError("keyword arguments")
        args = [self._expression(arg, scope, budget, operations) for arg in node.args]
        if isinstance(node.func, ast.Name):
            handler = _FUNCTIONS.get(node.func.id)
            if handler is None:
                raise UnsupportedCodeError(f"call to {node.func.id!r}")
            operations.append(node.func.id)
            return handler(None, args)
        if isinstance(node.func, ast.Attribute):
            method = _METHODS.get(node.func.attr)
            if method is None:
                raise UnsupportedCodeError(f"method {node.func.attr!r}")
            # The receiver is evaluated through this same walk, so it can only
            # ever be a value this module built. An attribute on a module --
            # `os.system` -- fails here because `os` is not a known name.
            target = self._expression(node.func.value, scope, budget, operations)
            operations.append(node.func.attr)
            return method(target, args)
        raise UnsupportedCodeError("call to a computed value")

    def _comprehension(
        self,
        node: ast.ListComp | ast.GeneratorExp,
        scope: dict[str, object],
        budget: _Budget,
        operations: list[str],
    ) -> list[object]:
        if len(node.generators) != 1:
            raise UnsupportedCodeError("only one generator")
        generator = node.generators[0]
        if generator.ifs or generator.is_async:
            raise UnsupportedCodeError("filtered or async comprehension")
        if not isinstance(generator.target, ast.Name):
            raise UnsupportedCodeError("only a plain loop variable")
        source = self._expression(generator.iter, scope, budget, operations)
        items = self._iterable(source)
        if len(items) > MAX_ITERATIONS:
            raise UnsupportedCodeError(BUDGET_EXCEEDED)
        # A fresh scope per comprehension, so the loop variable cannot leak
        # into the enclosing snippet or shadow one of its names.
        results: list[object] = []
        for item in items:
            inner = {**scope, generator.target.id: item}
            results.append(self._expression(node.elt, inner, budget, operations))
        return _Budget.check_sequence(results)

    @staticmethod
    def _iterable(source: object) -> list[object]:
        if isinstance(source, list):
            return source
        if isinstance(source, str):
            return list(source)
        raise UnsupportedCodeError("comprehension over an unsupported value")
