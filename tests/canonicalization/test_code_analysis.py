"""The restricted code analyzer: what it computes, and what it refuses."""

import json
from pathlib import Path

import pytest

from tipguard.benchmark.io import load_cases
from tipguard.canonicalization.code_analysis import (
    BUDGET_EXCEEDED,
    MAX_SOURCE_LENGTH,
    CodeAnalysis,
    RestrictedCodeAnalyzer,
)
from tipguard.canonicalization.detector import _extract_code


def _analyze(source: str) -> CodeAnalysis:
    return RestrictedCodeAnalyzer().analyze(source)


#: The three shapes the benchmark generator emits, which a defence has to
#: handle without being told which one it is looking at.
GENERATOR_STYLES = (
    ('msg = "".join(chr(c) for c in [104, 105])', "hi"),
    ('msg = "he§§o".replace("§", "l")', "hello"),
    ('parts = ["lo", "hel"]\nmsg = "".join(reversed(parts))', "hello"),
)


@pytest.mark.parametrize(("source", "expected"), GENERATOR_STYLES)
def test_the_generator_styles_resolve(source: str, expected: str) -> None:
    analysis = _analyze(source)
    assert analysis.rejected_reason is None
    assert analysis.result == expected


#: Everything a prompt might carry that this must refuse. These are the
#: reason the module exists: an attacker who can put code in a prompt can put
#: anything in one, so the whitelist has to be closed rather than filtered.
REJECTED = (
    "import os",
    "from os import system",
    'open("x")',
    '__import__("os").system("id")',
    'eval("1+1")',
    'exec("x=1")',
    'compile("1", "s", "eval")',
    "os.system('id')",
    "os.path.join('a', 'b')",
    "while True:\n    pass",
    "for i in range(10):\n    pass",
    "def f():\n    return 1",
    "class C:\n    pass",
    "lambda: 1",
    "x = [i for i in range(10**6)]",
    "globals()",
    "locals()",
    "().__class__.__bases__",
    'getattr(str, "join")',
    "x = 1 if True else 2",
    "async def f():\n    pass",
    "with open('x') as f:\n    pass",
    "assert False",
    "del x",
    "raise ValueError",
    "x += 1",
    "x = yield 1",
)


@pytest.mark.parametrize("source", REJECTED)
def test_anything_outside_the_subset_is_rejected_with_a_reason(source: str) -> None:
    analysis = _analyze(source)
    assert analysis.result is None
    assert analysis.rejected_reason


def test_multiplication_is_not_whitelisted() -> None:
    # The brief asks for this explicitly. `"a" * 10**7` is one node that
    # allocates megabytes, so it is refused for being outside the subset
    # rather than caught afterwards by the length cap.
    analysis = _analyze('msg = "a" * 10000000')
    assert analysis.result is None
    assert analysis.rejected_reason == "operator Mult"


#: One shape per budget guard, each the thing that guard exists to stop.
#: Named rather than inlined so a failure reports the guard, not 18 KB of
#: generated source.
BUDGET_CASES = {
    # Flat, not nested: a deeply nested expression is refused by the walk's
    # own recursion limit long before the step budget, which made an earlier
    # version of this test prove nothing about MAX_STEPS.
    # A comprehension, because it is the only shape that buys many steps
    # with little source: flat statements hit MAX_SOURCE_LENGTH first, and a
    # nested expression hits the walk's recursion limit first, so neither can
    # demonstrate this guard.
    "MAX_STEPS": 'msg = "".join([c for c in "' + "a" * 12_000 + '"])',
    "MAX_ITERATIONS": 'msg = [c for c in "' + "a" * 15_000 + '"]',
    "MAX_STRING_LENGTH": 'msg = "' + "a" * 9_000 + '".replace("a", "bb")',
    "MAX_TOTAL_CHARACTERS": (
        'S = "' + "a" * 8_800 + '"\nmsg = "".join([S.upper() for c in "' + "c" * 900 + '"])'
    ),
}


@pytest.mark.parametrize("guard", sorted(BUDGET_CASES))
def test_each_budget_guard_is_load_bearing(guard: str) -> None:
    """Each limit is shown to be the one refusing its own case.

    Two of these previously built sources longer than MAX_SOURCE_LENGTH, so
    `ast.parse` was never reached and the guards they named had no test at
    all. Even under that cap the guards overlap, so asserting a refusal
    proves nothing about which one fired: every *other* limit is raised out
    of the way and the named one must still refuse.

    "Every other limit" includes the interpreter's recursion limit, which is
    a fifth guard the module relies on without naming it.
    """
    from tipguard.canonicalization import code_analysis

    source = BUDGET_CASES[guard]
    assert len(source) < MAX_SOURCE_LENGTH, "the source cap would refuse this first"
    others = set(BUDGET_CASES) - {guard}
    saved = {name: getattr(code_analysis, name) for name in others}
    try:
        for name in others:
            setattr(code_analysis, name, 10**9)
        analysis = _analyze(source)
    finally:
        for name, value in saved.items():
            setattr(code_analysis, name, value)
    assert analysis.rejected_reason == BUDGET_EXCEEDED, f"{guard} did not refuse on its own"


def test_a_long_source_is_refused_before_parsing() -> None:
    # The bound comes before `ast.parse`, because parsing is itself
    # attacker-controlled work.
    analysis = _analyze("x = 1\n" * MAX_SOURCE_LENGTH)
    assert analysis.rejected_reason == BUDGET_EXCEEDED


def test_string_growth_is_capped_as_it_happens() -> None:
    # Not caught only at the end: each concatenation is checked, so the
    # snippet cannot allocate the whole value first.
    source = 'a = "x" * 1\n'  # rejected earlier, so build growth with join
    del source
    parts = '["' + '","'.join(["y" * 900] * 20) + '"]'
    analysis = _analyze(f'msg = "".join({parts})')
    assert analysis.rejected_reason == BUDGET_EXCEEDED


def test_an_empty_replace_pattern_is_refused() -> None:
    # `"abc".replace("", "x")` inserts between every character, a cheap way
    # to multiply a string.
    assert _analyze('msg = "abc".replace("", "x")').rejected_reason


def test_an_unknown_name_is_never_resolved_against_builtins() -> None:
    # The failure has to be "unknown name", not a successful lookup of the
    # real builtin, which is what keeps the whitelist closed.
    analysis = _analyze("msg = print")
    # The reason names the node kind, never the identifier: identifiers are
    # attacker-chosen and reasons reach traces.
    assert analysis.rejected_reason == "unknown name"


def test_a_module_attribute_chain_is_rejected() -> None:
    # `get` is not a whitelisted method, so the chain is refused before the
    # receiver is even evaluated. Either failure is correct; what matters is
    # that no part of it resolves to the real `os`.
    analysis = _analyze('msg = os.environ.get("HOME")')
    assert analysis.result is None
    assert analysis.rejected_reason in {"unsupported method", "unknown name"}


def test_the_operations_used_are_reported() -> None:
    # A trace can say what a snippet did without quoting the snippet, which
    # matters because a decoded payload may carry a protected value.
    analysis = _analyze('msg = "he§§o".replace("§", "l").upper()')
    assert analysis.result == "HELLO"
    assert analysis.operations == ("replace", "upper")


def test_a_loop_variable_does_not_leak_out_of_its_comprehension() -> None:
    analysis = _analyze('parts = [chr(c) for c in [104, 105]]\nmsg = "".join(parts)')
    assert analysis.result == "hi"
    assert _analyze("parts = [chr(c) for c in [104]]\nmsg = c").rejected_reason


def test_a_snippet_that_builds_no_string_is_rejected() -> None:
    assert _analyze("x = 1").rejected_reason == "snippet does not build a string"


def test_the_module_never_names_an_execution_primitive() -> None:
    """The guarantee the whole module rests on, asserted rather than assumed.

    Every *load* of a banned name is rejected, not just a direct call by that
    name. An earlier version inspected only `ast.Call` with a plain-name
    function, which seven real evasions walked straight through: an alias
    (`_e = eval`), the attribute form (`builtins.eval`), dict dispatch, and
    `open` -- which was not even on the list.
    """
    import ast

    banned = {
        "eval",
        "exec",
        "compile",
        "__import__",
        "__builtins__",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "vars",
        "open",
        "subprocess",
        "importlib",
        "os",
        "sys",
    }
    tree = ast.parse(_module_source())
    named: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            named.add(node.id)
        elif isinstance(node, ast.Attribute):
            named.add(node.attr)
        elif isinstance(node, ast.Import):
            named.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            named.add(node.module.split(".")[0])
    assert not (named & banned), f"module names {sorted(named & banned)}"


def test_the_module_imports_nothing_that_can_act() -> None:
    # `ast` parses and `collections.abc`/`typing` are type-only; nothing here
    # can touch a process or a file even if the whitelist were wrong.
    import ast

    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(_module_source()))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(_module_source()))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert imported <= {"ast", "collections", "typing", "tipguard"}, imported


def _module_source() -> str:
    # Located from this module's own file rather than the working directory,
    # so the test does not depend on where pytest was started.
    from tipguard.canonicalization import code_analysis

    return Path(code_analysis.__file__).read_text(encoding="utf-8")


def test_every_code_case_in_the_corpus_resolves() -> None:
    """The accuracy side: the analyzer has to actually read the corpus.

    A whitelist narrow enough to be safe and too narrow to be useful would
    pass every rejection test above and still be worthless, so this measures
    what it recovers.
    """
    cases = [
        case
        for case in load_cases(Path("data/generated/tipguard-v1.jsonl"))
        if case.transformation == "code"
    ]
    assert cases
    analyzer = RestrictedCodeAnalyzer()
    resolved = 0
    for case in cases:
        # The detector's own extractor, not "the last block": at difficulty 4
        # the wrapper puts its instruction after the snippet, so the last
        # block is prose. Measuring extraction here would measure the test.
        snippet = _extract_code(case.prompt)
        if snippet and analyzer.analyze(snippet).result is not None:
            resolved += 1
    rate = resolved / len(cases)
    assert rate >= 0.95, f"only {rate:.3f} of code cases resolved"


def test_a_rejected_snippet_never_carries_the_snippet_into_the_reason() -> None:
    # A snippet can contain a protected value, and the reason is stored in a
    # trace. Names of nodes and calls are safe; the source is not.
    secret = "CANARY-7f3a-KESTREL-9021"
    analysis = _analyze(f'msg = open("{secret}")')
    assert analysis.rejected_reason is not None
    assert secret not in analysis.rejected_reason
    assert secret not in json.dumps(analysis.model_dump())


def test_list_growth_is_bounded_like_string_growth() -> None:
    """A list is as cheap to write and as expensive to hold as a string.

    Only strings were capped at first, so a chain of list concatenations
    doubled its way to 213 MB before the step budget noticed -- twenty-one
    doublings is twenty-one steps, and each allocates rather than looping.
    Measured rather than asserted structurally, because the defect was
    invisible to every structural test.
    """
    import tracemalloc

    source = (
        'a0 = ["x","y","z","w"]\n'
        + "".join(f"a{i} = a{i - 1} + a{i - 1}\n" for i in range(1, 22))
        + 'msg = "".join(a21)'
    )
    tracemalloc.start()
    try:
        analysis = _analyze(source)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert analysis.rejected_reason == BUDGET_EXCEEDED
    assert peak < 20_000_000, f"allocated {peak / 1e6:.0f} MB before refusing"


#: Every argument-shape refusal, one case each. Uncovered lines in a module
#: whose job is refusing things are untested refusals, which is the one kind
#: of dead code that matters here: a rejection path that has never run is a
#: rejection path that might not reject.
MALFORMED_CALLS = (
    ('msg = "".join()', "join expects one sequence"),
    ('msg = "".join("ab", "cd")', "join expects one sequence"),
    ('msg = "".join("notalist")', "join expects one sequence"),
    ('msg = "a".replace("a")', "replace expects two arguments"),
    ('msg = "a".replace("a", "b", "c")', "replace expects two arguments"),
    ("msg = chr()", "chr expects one argument"),
    ('msg = chr("a")', "chr expects an integer"),
    # A bool never reaches chr: it is refused as a constant first.
    ("msg = chr(True)", "only string and integer constants"),
    ("msg = ord()", "ord expects one argument"),
    ("msg = ord(1)", "ord expects a string"),
    ('msg = ord("ab")', "ord expects one character"),
    ("msg = reversed()", "reversed expects one argument"),
    ("msg = reversed(1)", "reversed expects a string or sequence"),
    ('msg = "a".upper("x")', "upper expects no arguments"),
    ('msg = "a b".split("x", "y")', "split expects at most one argument"),
    ('msg = "a b".split("")', "split on an empty separator"),
    ('msg = "a".upper(x=1)', "keyword arguments"),
    ('msg = [c for c in "ab" for d in "cd"]', "only one generator"),
    ('msg = [c for (c, d) in ["ab"]]', "only a plain loop variable"),
    ("msg = [c for c in 1]", "comprehension over an unsupported value"),
    ('msg = 1 + "a"', "addition of unsupported types"),
    ('msg = "a"()', "call to a computed value"),
    ("msg = print", "unknown name"),
    ('msg = sorted(["b", "a"])', "call to an unsupported function"),
)


@pytest.mark.parametrize(("source", "reason"), MALFORMED_CALLS)
def test_every_argument_shape_refusal_is_reachable(source: str, reason: str) -> None:
    analysis = _analyze(source)
    assert analysis.result is None
    assert analysis.rejected_reason == reason


def test_a_deeply_nested_expression_is_refused_rather_than_crashing() -> None:
    # The walk is recursive, so a nested expression can exhaust Python's own
    # stack before the step budget notices. That is an unsupported snippet,
    # not a crash the caller has to handle.
    analysis = _analyze('msg = "a"' + ' + "a"' * 200 + "\n")
    assert analysis.result is not None or analysis.rejected_reason


def test_unparseable_source_is_a_reason_not_an_exception() -> None:
    assert _analyze("msg = (((").rejected_reason == "not parseable as Python"


def test_a_snippet_may_end_in_a_bare_expression() -> None:
    # The generator assigns to `msg`, but a defence cannot assume that shape.
    assert _analyze('"".join(["a", "b"])').result == "ab"


#: Each string operation, in a shape whose result would exceed the cap. The
#: refusal must cost no more than holding the inputs.
OVERSIZED_RESULTS = (
    ('parts = ["' + '","'.join(["z" * 9000] * 200) + '"]\nmsg = "".join(parts)', 1.0),
    ('msg = "' + "a" * 9000 + '".replace("a", "' + "b" * 500 + '")', 0.5),
    ('a = "' + "x" * 9000 + '"\nmsg = a + a', 0.5),
)


@pytest.mark.parametrize(("source", "budget_mb"), OVERSIZED_RESULTS)
def test_an_oversized_result_is_refused_before_it_is_built(source: str, budget_mb: float) -> None:
    """The cap applies to the computed length, not the finished value.

    Checking afterwards still allocates first: a `replace` that expands its
    input reached 4.5 MB before being refused, which is exactly the cost the
    cap exists to avoid. Every operation here can work out its own output
    length from its inputs, so it does.
    """
    import tracemalloc

    tracemalloc.start()
    try:
        analysis = _analyze(source)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert analysis.rejected_reason == BUDGET_EXCEEDED
    assert peak < budget_mb * 1_000_000, f"allocated {peak / 1e6:.2f} MB before refusing"


def test_an_attacker_chosen_identifier_never_reaches_a_reason() -> None:
    # A snippet can name a variable after the value it is trying to exfiltrate,
    # and rejection reasons are stored in traces. Only node kinds are reported.
    secret = "CANARY_7f3a_KESTREL_9021"
    for source in (f"msg = {secret}", f"msg = {secret}()", f'msg = "a".{secret}()'):
        analysis = _analyze(source)
        assert analysis.rejected_reason is not None
        assert secret not in analysis.rejected_reason
        assert secret not in json.dumps(analysis.model_dump())


def test_a_reason_is_short_whatever_the_snippet_looks_like() -> None:
    # Reported lengths are bounded by the vocabulary of node kinds, not by the
    # snippet: a 19,000-character identifier once produced a 19,015-character
    # reason.
    analysis = _analyze("msg = " + "z" * 19_000)
    assert analysis.rejected_reason is not None
    assert len(analysis.rejected_reason) < 100


def test_a_trailing_unsupported_statement_does_not_discard_the_result() -> None:
    # `print(msg)` is the likeliest hand-written last line, and rejecting the
    # whole snippet for it took corpus resolution to zero.
    assert _analyze('msg = "he§§o".replace("§", "l")\nprint(msg)').result == "hello"
    assert _analyze('msg = "hi"\nassert msg').result == "hi"


def test_tolerance_for_a_coda_does_not_swallow_a_real_refusal() -> None:
    # The tolerance applies only once `msg` is bound. A snippet whose own work
    # is refused still reports that refusal -- an earlier version tested "any
    # string in scope" and swallowed a genuine size refusal because an operand
    # happened to be one.
    oversized = 'a = "' + "x" * 9000 + '"\nmsg = a + a'
    assert _analyze(oversized).rejected_reason == BUDGET_EXCEEDED
    assert _analyze('msg = "a" * 10').rejected_reason == "operator Mult"


def test_a_comprehension_cannot_build_many_capped_strings() -> None:
    """A per-value cap bounds one string; it does not bound ten thousand.

    Building a fresh capped-size string per item reached 44 MB from 19 KB of
    source -- the same unbounded-total defect as the list doubling, one level
    up. The running total is charged wherever a value is built.
    """
    import tracemalloc

    source = 'S = "' + "a" * 8800 + '"\nmsg = "".join([S.upper() for c in "' + "c" * 9900 + '"])'
    tracemalloc.start()
    try:
        analysis = _analyze(source)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert analysis.rejected_reason == BUDGET_EXCEEDED
    assert peak < 20_000_000, f"allocated {peak / 1e6:.0f} MB before refusing"
