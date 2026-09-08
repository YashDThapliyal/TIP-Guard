from tipguard.config.schemas import Policy
from tipguard.evaluation.leak import detect_leak

POLICY = Policy(
    policy_id="p",
    description="d",
    categories=["x"],
    protected_label="l",
    protected_values=["CANARY-7f3a-KESTREL-9021"],
)


def test_exact_and_case_insensitive_match() -> None:
    assert detect_leak("the value is canary-7f3a-kestrel-9021.", POLICY).leaked


def test_spaced_out_value_counts() -> None:
    assert detect_leak("C A N A R Y - 7 f 3 a - K E S T R E L - 9 0 2 1", POLICY).leaked


def test_no_leak_and_none_text() -> None:
    assert not detect_leak("nothing here", POLICY).leaked
    assert not detect_leak(None, POLICY).leaked
