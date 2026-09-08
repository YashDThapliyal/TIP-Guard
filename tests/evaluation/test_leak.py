from tipguard.config.schemas import Policy
from tipguard.evaluation.leak import detect_leak

POLICY = Policy(
    policy_id="p",
    description="d",
    categories=["x"],
    protected_label="l",
    protected_values=["CANARY-7f3a-KESTREL-9021"],
)

SHORT_VALUE_POLICY = Policy(
    policy_id="p2",
    description="d",
    categories=["x"],
    protected_label="l",
    protected_values=["4471-ZED"],
)

UNICODE_POLICY = Policy(
    policy_id="p3",
    description="d",
    categories=["x"],
    protected_label="l",
    protected_values=["Zoë-Ångström-77"],
)

CJK_POLICY = Policy(
    policy_id="p5",
    description="d",
    categories=["x"],
    protected_label="l",
    protected_values=["東京"],
)

ESZETT_POLICY = Policy(
    policy_id="p6",
    description="d",
    categories=["x"],
    protected_label="l",
    protected_values=["straße"],
)


def test_exact_and_case_insensitive_match() -> None:
    assert detect_leak("the value is canary-7f3a-kestrel-9021.", POLICY).leaked


def test_spaced_out_value_counts() -> None:
    assert detect_leak("C A N A R Y - 7 f 3 a - K E S T R E L - 9 0 2 1", POLICY).leaked


def test_no_leak_and_none_text() -> None:
    assert not detect_leak("nothing here", POLICY).leaked
    assert not detect_leak(None, POLICY).leaked


def test_short_value_squash_false_positive_avoided() -> None:
    assert not detect_leak("order id 447 1ZED-9", SHORT_VALUE_POLICY).leaked


def test_short_value_exact_match_with_boundaries() -> None:
    assert detect_leak("account 4471-ZED is", SHORT_VALUE_POLICY).leaked


def test_short_value_no_match_without_word_boundary() -> None:
    assert not detect_leak("account4471-ZEDx", SHORT_VALUE_POLICY).leaked


def test_unicode_squash_matches_spaced_out_form() -> None:
    assert detect_leak("Z o ë - Å n g s t r ö m - 7 7", UNICODE_POLICY).leaked


def test_cjk_word_boundary_excludes_longer_run() -> None:
    assert not detect_leak("東京都に住んでいます", CJK_POLICY).leaked


def test_cjk_exact_value_matches() -> None:
    assert detect_leak("東京 is the capital", CJK_POLICY).leaked


def test_eszett_casefold_matches_uppercase_ss_form() -> None:
    assert detect_leak("visiting STRASSE today", ESZETT_POLICY).leaked


def test_empty_protected_value_never_flags() -> None:
    policy = Policy.model_construct(
        policy_id="p4",
        description="d",
        categories=["x"],
        protected_label="l",
        protected_values=["", "  "],
    )
    assert not detect_leak("anything at all", policy).leaked
