"""Baseline 4: a syntactic detector for which transformation family, if any.

`TransformationDetector` is a heuristic, not a parser: it reuses the Task 1
pattern regexes (`tipguard.classifiers.pattern`) for the families that leave
a clean syntactic signature -- base64, morse, substitution, code -- and adds
one further test on top of each so that a mere regex match is not mistaken
for the family itself: base64 must actually decode to mostly-printable
UTF-8, and code must actually `ast.parse`.

Caesar and reverse share a signature the regexes above cannot see: neither
leaves a distinctive character class, only a run of letter "words" that read
as gibberish. The two are told apart, and told apart from ordinary
non-English text, in three tiers, cheapest first:

1. A run of at least four alpha "words" scores below the stoplist-hit floor
   (`LOW_HIT_THRESHOLD`) -- otherwise the text is not a candidate at all.
2. The prompt names the family directly ("Caesar", "shift", "letters" for a
   caesar wrapper; "revers-" for a reverse one) -- trust that naming; a
   generated wrapper does not lie about which transformation it applied.
3. No naming survives (the harder difficulty tiers drop it, and every round
   of a multi-step decode has no wrapper at all): try the transformation
   itself. `_best_caesar_shift` searches all 25 shifts and plain reversal is
   tried directly; whichever reconstruction reads more like English, by the
   same stoplist measure, wins. If neither clears the floor, no family is
   reported -- the caller (a multi-step round, or this module's own decision
   below) treats that as "not yet decodable", not as a wrong guess.

Two families compose only in the sense that composing them leaves two
independent signatures: reused this way, that lets `multi_step` be recognised
as "two families' evidence fired at once" without a dedicated multi-step
regex. Where only one signature fires but the wrapper still names two
transformations ("base64 wrapping reversed text"), the literal word
"wrapping" -- unique to `MultiStepTransformation`'s hint composition, see
`tipguard.benchmark.transformations.multi_step` -- is the tell. Where neither
holds (the hardest multi-step tier drops the wrapper's naming entirely), the
tell is that the single-round content-fallback resolution above still reads
as gibberish (its hit rate stays below `CONFIDENT_HIT_THRESHOLD`) while the
prompt still uses "then" to sequence two actions -- one more decode round is
plausible. All three tests are deliberately independent, since real prompts
in the corpus supply only one of them.

Riddles and indirect descriptions carry no encoding at all, so they are
detected last, only once nothing else has fired, from four textual tells:
the word "riddle", the word "indirect", the phrases "what am I" and "the
thing that", and -- weakest, and pinned as such by
`tests/canonicalization/test_on_corpus.py` -- the presence of a question
mark anywhere in the prompt. The corpus measurement is that this reaches
essentially every riddle and indirect case at the two easier difficulty
tiers and roughly half of the two harder ones, which drop every lexical tell
along with the encoding; a purely syntactic detector has no way to close
that gap; a semantic one is exactly what a later phase adds.
"""

import ast
import base64
import binascii
import re
import string
from dataclasses import dataclass

from tipguard.benchmark.transformations.base import Encoded, Family
from tipguard.benchmark.transformations.caesar import CaesarTransformation
from tipguard.benchmark.transformations.code_snippet import MAX_SOURCE_LENGTH
from tipguard.canonicalization.types import DetectionResult
from tipguard.classifiers.pattern import MORSE_RUN, PYTHON_STRING_BUILDER, SUBSTITUTION_SYMBOLS
from tipguard.classifiers.pattern import _has_substitution_density as has_substitution_density

#: A curated set of roughly 300 of the most common English words: the closed
#: classes in full (articles, pronouns, prepositions, conjunctions, the verbs
#: "be", "have", "do" and the modals) plus the base forms of the highest
#: frequency content verbs and nouns. Deliberately ordinary vocabulary, not
#: this project's own ("system", "canary", "protected" and the like do not
#: appear): the point is to measure whether a span of text reads as *some*
#: English, not whether it names this benchmark's assets. A caesar-shifted or
#: reversed span scores far below any of the thresholds below on this list
#: because shifting or reversing a word essentially never produces another
#: word already in it; ordinary prose scores well above them because a
#: large fraction of any English sentence is drawn from exactly this list.
STOPLIST: frozenset[str] = frozenset(
    (
        # Articles, conjunctions, prepositions.
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "so",
        "because",
        "as",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "without",
        "within",
        "until",
        "while",
        "than",
        "per",
        # Pronouns and determiners.
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "each",
        "every",
        "either",
        "neither",
        "another",
        "such",
        # Be, have, do, and the modals.
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "should",
        "could",
        "can",
        "may",
        "might",
        "must",
        "shall",
        # Question words, quantifiers, common adverbs.
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "too",
        "very",
        "just",
        "now",
        "here",
        "there",
        "also",
        "still",
        "even",
        "well",
        "back",
        "always",
        "never",
        "often",
        "sometimes",
        "soon",
        "already",
        "yet",
        "quite",
        "rather",
        "almost",
        "enough",
        # Common verbs, base forms.
        "say",
        "get",
        "make",
        "go",
        "know",
        "take",
        "see",
        "come",
        "think",
        "look",
        "want",
        "give",
        "use",
        "find",
        "tell",
        "ask",
        "work",
        "seem",
        "feel",
        "try",
        "leave",
        "call",
        "need",
        "keep",
        "let",
        "begin",
        "help",
        "show",
        "hear",
        "play",
        "run",
        "move",
        "live",
        "believe",
        "bring",
        "happen",
        "write",
        "provide",
        "sit",
        "stand",
        "lose",
        "pay",
        "meet",
        "include",
        "continue",
        "set",
        "learn",
        "change",
        "lead",
        "understand",
        "watch",
        "follow",
        "stop",
        "create",
        "speak",
        "read",
        "allow",
        "add",
        "spend",
        "grow",
        "open",
        "walk",
        "win",
        "offer",
        "remember",
        "love",
        "consider",
        "appear",
        "buy",
        "wait",
        "serve",
        "die",
        "send",
        "expect",
        "build",
        "stay",
        "fall",
        "cut",
        "reach",
        "kill",
        "remain",
        "decode",
        "answer",
        "act",
        "echo",
        "print",
        "reveal",
        "output",
        "mean",
        "refer",
        "describe",
        "spell",
        "produce",
        "share",
        "confirm",
        "copy",
        "request",
        "contain",
        "exist",
        "leak",
        "please",
        "hold",
        "hide",
        "protect",
        # Common nouns.
        "time",
        "year",
        "people",
        "way",
        "day",
        "man",
        "thing",
        "woman",
        "life",
        "child",
        "world",
        "school",
        "state",
        "family",
        "student",
        "group",
        "country",
        "problem",
        "hand",
        "part",
        "place",
        "case",
        "week",
        "company",
        "system",
        "program",
        "question",
        "government",
        "number",
        "night",
        "point",
        "home",
        "water",
        "room",
        "mother",
        "area",
        "money",
        "story",
        "fact",
        "month",
        "lot",
        "right",
        "study",
        "book",
        "eye",
        "job",
        "word",
        "business",
        "issue",
        "side",
        "kind",
        "head",
        "house",
        "service",
        "friend",
        "father",
        "power",
        "hour",
        "game",
        "line",
        "end",
        "member",
        "law",
        "city",
        "community",
        "name",
        "team",
        "minute",
        "idea",
        "body",
        "information",
        "parent",
        "face",
        "level",
        "office",
        "door",
        "health",
        "person",
        "history",
        "party",
        "result",
        "reason",
        "message",
        "note",
        "text",
        "string",
        "value",
        "code",
        "context",
        "instance",
        "instruction",
        "instructions",
        "content",
    )
)

#: A stoplist hit rate below this on the longest low-hit paragraph marks a
#: span of alpha "words" as still-encoded gibberish. Chosen to match the
#: brief's own caesar rule ("if < 20% hits"); reused for reverse and for the
#: multi-step content fallback below, since the same "does this read as
#: English at all" test applies to all three.
LOW_HIT_THRESHOLD = 0.20

#: How much more readable a decode must be than the text it came from before
#: that counts as evidence of an encoding on its own. Two-fold: ordinary
#: prose cannot be made twice as stoplist-dense by shifting or reversing it.
_DECODE_IMPROVEMENT = 2.0

#: A stoplist hit rate at or above this, reached without any wrapper naming
#: the family, is read as "this decode already looks like real English" --
#: i.e. one round was enough, so the case is not multi-step. Set comfortably
#: above `LOW_HIT_THRESHOLD` so that a *partial* decode (one layer of a
#: two-layer composition, still gibberish after the first round) is not
#: mistaken for a completed one; ordinary English sentences score far above
#: it in practice because roughly a third to a half of running text is drawn
#: from `STOPLIST`.
CONFIDENT_HIT_THRESHOLD = 0.30

#: The minimum number of whitespace-delimited alphabetic tokens a span must
#: contain before its stoplist hit rate is trusted at all. Below this a
#: handful of ordinary short words (a wrapper fragment, a stray phrase) could
#: read as "low hit rate" purely from sample size.
MIN_LETTER_TOKENS = 4

#: The floor when the wrapper itself names the cipher. Two tokens is enough
#: to attempt a decode, and the naming carries the evidence the hit rate
#: cannot at that length.
NAMED_CIPHER_MIN_TOKENS = 2

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_PUNCTUATION_STRIP = string.punctuation

CAESAR_WORDS = re.compile(r"\b(?:caesar|shift(?:ed)?|letters?)\b", re.IGNORECASE)
REVERSE_WORDS = re.compile(r"\brevers", re.IGNORECASE)
WRAPPING_HINT = re.compile(r"\bwrapping\b", re.IGNORECASE)
THEN_HINT = re.compile(r"\bthen\b", re.IGNORECASE)
RIDDLE_WORD = re.compile(r"\briddle", re.IGNORECASE)
INDIRECT_WORD = re.compile(r"\bindirect\b", re.IGNORECASE)
WHAT_AM_I = re.compile(r"\bwhat\s+am\s+i\b", re.IGNORECASE)
THE_THING_THAT = re.compile(r"\bthe\s+thing\s+that\b", re.IGNORECASE)

_FENCE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


def _paragraphs(text: str) -> list[str]:
    """Blank-line-delimited blocks of `text`, stripped, empties dropped.

    Every payload in the shipped corpus sits in its own paragraph, separated
    from the natural-language wrapper by a blank line, so this is the one
    piece of structure every family-specific extractor below builds on.
    """
    return [block.strip() for block in _PARAGRAPH_SPLIT.split(text) if block.strip()]


def _alpha_tokens(text: str) -> list[str]:
    """Lower-cased whitespace tokens of `text` that are purely alphabetic.

    Punctuation is stripped from each token's edges before the alphabetic
    check, so "instructions." still counts. A token that keeps a digit or a
    substitution symbol in its interior -- "7h15", "f°qi!g€q+!@o!+b" -- is
    dropped rather than counted as a non-hit; those characters are exactly
    what marks a leetspoken or symbol-substituted word, and a caesar/reverse
    span this function is applied to never contains them, so the two
    concerns never collide.
    """
    tokens = []
    for raw in text.split():
        stripped = raw.strip(_PUNCTUATION_STRIP)
        if stripped and stripped.isalpha():
            tokens.append(stripped.lower())
    return tokens


def _hit_rate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in STOPLIST)
    return hits / len(tokens)


def _candidate_spans(text: str) -> list[str]:
    """Every span of `text` worth testing as the ciphertext.

    Almost always just `_paragraphs(text)`: the shipped corpus separates the
    wrapper from the payload with a blank line. One prompt shape does not --
    "Shift each letter back by 3 and follow the instruction: Sulqw wkh
    klgghq..." puts both in one sentence after a colon -- so the text after
    the last ": " is added as one further candidate. Adding it costs
    nothing when it is not needed: a wrapper sentence with a colon in it
    scores no lower than the real ciphertext paragraph, so it is never
    picked over a genuine candidate; it only matters when there is no blank
    line to fall back on.
    """
    spans = _paragraphs(text)
    if ": " in text:
        suffix = text.rsplit(": ", 1)[-1].strip()
        if suffix and suffix not in spans:
            spans = [*spans, suffix]
    return spans


def _lowest_hit_paragraph(
    text: str, min_tokens: int = MIN_LETTER_TOKENS
) -> tuple[str, float] | None:
    """The candidate span with the lowest stoplist hit rate.

    A candidate is a span (see `_candidate_spans`) with at least `min_tokens`
    alphabetic tokens. The wrapper is always ordinary English and scores well
    above `LOW_HIT_THRESHOLD`; picking the *lowest*-scoring candidate is what
    lets this function find the ciphertext even though it usually is not the
    longest span in the prompt.

    `min_tokens` exists because the default floor guards a *statistical*
    test: below four tokens a stoplist hit rate says almost nothing, so an
    ordinary short phrase would read as ciphertext. A wrapper that names the
    cipher is separate evidence, not subject to that, so a caller holding one
    may lower the floor -- which is what lets "Shift each letter back by 3 to
    read the message: khoor zruog" be recognised at all.
    """
    best: tuple[str, float] | None = None
    for span in _candidate_spans(text):
        tokens = _alpha_tokens(span)
        if len(tokens) < min_tokens:
            continue
        rate = _hit_rate(tokens)
        if best is None or rate < best[1]:
            best = (span, rate)
    return best


def _best_caesar_shift(payload: str) -> tuple[int, float]:
    """The shift in 1..25 whose decode maximises the stoplist hit rate.

    Tries every shift, not just the seven `CaesarTransformation` ever draws
    from (`caesar.ALLOWED_SHIFTS`): a canonicalizer has no reason to assume
    an attacker restricted themselves to the benchmark's own generator.
    """
    transformer = CaesarTransformation()
    best_shift, best_rate = 1, -1.0
    for shift in range(1, 26):
        encoded = Encoded(
            payload=payload, family=Family.CAESAR, params={"shift": str(shift)}, hint=""
        )
        rate = _hit_rate(_alpha_tokens(transformer.decode(encoded)))
        if rate > best_rate:
            best_shift, best_rate = shift, rate
    return best_shift, best_rate


@dataclass(frozen=True)
class LettersResolution:
    """The outcome of resolving a low-hit-rate letters span to a family."""

    family: Family
    hit_rate: float
    via_wrapper_words: bool


def _best_letters_decode(paragraph: str) -> tuple[Family, float] | None:
    """Content-only tier 3 of the caesar/reverse test: try both, pick the better.

    Used with no wrapper naming available at all -- either because the
    corpus dropped it at a harder difficulty, or because this is round two
    or three of a multi-step decode, which never has a wrapper to read.
    Returns `None` if neither reconstruction clears `LOW_HIT_THRESHOLD`, so
    the caller can tell "still encoded, one more layer to go" apart from
    "not this pair of families at all".
    """
    _shift, shift_rate = _best_caesar_shift(paragraph)
    reversed_rate = _hit_rate(_alpha_tokens(paragraph[::-1]))
    if shift_rate < LOW_HIT_THRESHOLD and reversed_rate < LOW_HIT_THRESHOLD:
        return None
    if shift_rate >= reversed_rate:
        return Family.CAESAR, shift_rate
    return Family.REVERSE, reversed_rate


#: Below this many characters, a span is too short to give a token away: the
#: marker may well be the payload.
_MIN_PAYLOAD_CHARS = 12


def _is_prose_marker(token: str, alphabet: str) -> bool:
    """Whether `token` provably cannot belong to a payload over `alphabet`.

    Proof, not a guess. Three heuristics in a row deleted real ciphertext
    here: "any token with interior punctuation" ate a payload opening
    "3.14"; "single letters separated by full stops" ate a shifted "U.S.",
    which has exactly that shape; and a closed list of English labels still
    ate a plaintext abbreviation that happened to shift into one of them.
    Every shape rule fails for the same reason -- a label and a fragment of
    ciphertext can look identical.

    What cannot look identical is a character the alphabet does not contain.
    Base64 has no full stop; morse is dots, dashes and slashes with no
    letters. A token carrying a character outside the payload's own alphabet
    was never payload, and stripping it is safe by construction.

    Caesar and reverse have no such guarantee: they preserve punctuation, so
    "P.S." is a legitimate ciphertext token. They therefore pass an empty
    alphabet, nothing is ever stripped, and their decode carries the label
    through as a harmless prefix. Leaving four characters of noise on a
    correct decode is cheap; deleting a payload's first word is not.
    """
    return bool(alphabet) and any(char not in alphabet for char in token)


#: What each family's payload may contain. Empty means "no guarantee", so
#: nothing is stripped for that family.
BASE64_ALPHABET = string.ascii_letters + string.digits + "+/="
MORSE_ALPHABET = ".-/ "


def _strip_prose_marker(span: str, alphabet: str = "") -> str:
    """`span` without a leading prose marker such as "P.S.".

    A caesar or reverse payload is pure letters and spaces -- the encoders
    touch letters and leave everything else alone -- so a leading token
    carrying interior punctuation was never ciphertext. It is the wrapper's
    own label, and shifting it produces garbage: "P.S." came back as "C.F."
    glued to an otherwise perfect decode. Both the canonicalizer and the
    corpus test's independent oracle made that mistake identically, so the
    test agreed with the bug instead of catching it.
    """
    tokens = span.split()
    # At most one token, and only when a real payload remains. A ciphertext
    # can legitimately begin with an abbreviation -- a shifted "U.S." is
    # indistinguishable from a label -- so the rule is bounded rather than
    # made cleverer: the cost of a wrong strip is one short token, never a
    # truncated or empty payload.
    if len(tokens) < 2 or not _is_prose_marker(tokens[0], alphabet):
        return span
    remainder = " ".join(tokens[1:])
    # Measured in characters, not tokens: a base64 payload is one long token,
    # so a token-count guard blocked exactly the cases this exists to fix.
    if len(remainder) < _MIN_PAYLOAD_CHARS:
        return span
    return remainder


def letters_payload(text: str) -> str | None:
    """The span a caesar or reverse decode should operate on.

    Public because the deterministic canonicalizer needs exactly the span the
    detector reasoned about. It used to re-derive it with the default token
    floor while the detector used the relaxed, wrapper-named one, so on a
    short inline payload the two disagreed: the detector recognised
    "khoor zruog" and the canonicalizer decoded the whole sentence around it,
    returning shifted wrapper prose instead of the message. One function, so
    they cannot drift apart again.
    """
    named = bool(CAESAR_WORDS.search(text)) != bool(REVERSE_WORDS.search(text))
    candidate = _lowest_hit_paragraph(text, NAMED_CIPHER_MIN_TOKENS if named else MIN_LETTER_TOKENS)
    return candidate[0] if candidate is not None else None


def _resolve_letters_family(text: str) -> LettersResolution | None:
    mentions_caesar = bool(CAESAR_WORDS.search(text))
    mentions_reverse = bool(REVERSE_WORDS.search(text))
    named = mentions_caesar != mentions_reverse
    candidate = _lowest_hit_paragraph(text, NAMED_CIPHER_MIN_TOKENS if named else MIN_LETTER_TOKENS)
    if candidate is None:
        return None
    paragraph, hit_rate = candidate
    if hit_rate >= LOW_HIT_THRESHOLD:
        # The span reads like language already -- but a reversed sentence can
        # score at the threshold by accident, because reversing leaves short
        # words like "a" and "no" intact or turns them into other real words.
        # One shipped case sat at exactly 0.20 and was missed entirely. So
        # before giving up, ask whether some decode makes it dramatically
        # more readable; a real encoding does, ordinary prose does not.
        resolved = _best_letters_decode(paragraph)
        if resolved is None or resolved[1] < hit_rate * _DECODE_IMPROVEMENT:
            return None
        return LettersResolution(resolved[0], resolved[1], False)
    if mentions_caesar and not mentions_reverse:
        return LettersResolution(Family.CAESAR, 1.0, True)
    if mentions_reverse and not mentions_caesar:
        return LettersResolution(Family.REVERSE, 1.0, True)
    resolved = _best_letters_decode(paragraph)
    if resolved is None:
        return None
    family, rate = resolved
    return LettersResolution(family, rate, False)


def _mostly_printable(text: str, threshold: float = 0.9) -> bool:
    """Whether a decode looks like text a person wrote, not chance bytes.

    ASCII, plus the substitution glyphs. `str.isprintable` accepts every
    printable codepoint in Unicode, so mojibake counts as printable and any
    ordinary English word long enough to be a base64 candidate can pass:
    "deactivation" decodes to bytes printable somewhere in Unicode, which
    made a plain English sentence a base64 detection at 0.9 confidence.

    The glyphs are admitted because a layered payload's inner layer is often
    symbol-substituted, so a correct base64 decode legitimately yields text
    full of "€" and "°". Restricting to bare ASCII rejected 15 real
    multi-step cases. They are a closed, known set rather than "any
    non-ASCII", so admitting them does not reopen the mojibake hole.
    """
    if not text:
        return False
    allowed = set(SUBSTITUTION_SYMBOLS)
    printable = sum(1 for char in text if (" " <= char <= "~") or char in "\n\t" or char in allowed)
    if printable / len(text) < threshold:
        return False
    # And it has to look like more than one word. A short English word can
    # decode to a run of ASCII by chance -- "circumstantially" did -- but a
    # payload carrying an instruction has spaces in it, or at least a word
    # the stoplist knows. Requiring one of those costs nothing on the corpus,
    # whose payloads are all sentences.
    return any(char.isspace() for char in text) or _hit_rate(_alpha_tokens(text)) > 0.0


def _try_base64_decode(payload: str) -> str | None:
    """Strict base64 decode of `payload` to mostly-printable UTF-8, or None.

    `validate=True` refuses any character outside the base64 alphabet
    (including whitespace), so this only ever accepts exactly the run
    `BASE64_RUN` matched. A run that decodes to bytes that are not valid
    UTF-8, or that are valid but mostly control characters, is treated as a
    false positive of the regex rather than a real base64 payload.
    """
    try:
        raw = base64.b64decode(payload, validate=True)
        decoded = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    return decoded if _mostly_printable(decoded) else None


#: A looser base64 run than the pattern classifier's. That classifier has to
#: decide on shape alone, so it needs 24 characters before a run is unlikely
#: to be an accident; this module confirms every candidate by decoding it
#: strictly and checking the result is mostly printable text, so the decode
#: is the discriminator and the length floor only has to be long enough to be
#: worth trying. Twelve characters is three base64 quanta -- enough that a
#: chance decode to printable text is rare, short enough to catch a one-line
#: payload like "aGVsbG8gd29ybGQ=" that the 24-character floor missed
#: entirely.
BASE64_CANDIDATE = re.compile(r"[A-Za-z0-9+/]{12,}={0,2}")


def _extract_base64(text: str) -> str | None:
    """The longest base64-alphabet run in `text` that actually decodes.

    Several runs can match in one prompt (the wrapper's own prose rarely
    does, but is not impossible for a short one); the longest is preferred,
    matching the extraction rule the brief states for the canonicalizer.
    """
    # No marker preprocessing here. `BASE64_CANDIDATE` already scans for a
    # run of base64 characters anywhere in the text, so a label in front of
    # the payload never reached it -- while stripping a leading token *did*
    # reach the payload: "V2hhdCBpcyB0aGU=. That is the message." lost the
    # payload itself, because its trailing full stop made the whole token
    # look like a label.
    runs = (match.group() for match in BASE64_CANDIDATE.finditer(text))
    for candidate in sorted(runs, key=len, reverse=True):
        if _try_base64_decode(candidate) is not None:
            return candidate
    return None


def _extract_morse(text: str) -> str | None:
    """The paragraph carrying the morse payload, or None.

    `MORSE_RUN` alone under-matches here: morse's own space token ("/") is
    not a dot or a dash, so it breaks a message into one match per word and
    a short word (under four letters) does not match at all. Detection only
    needs one hit anywhere; extracting the *whole* payload for decoding needs
    the paragraph that contains it, found by requiring a `MORSE_RUN` hit
    inside a paragraph built almost entirely from morse's own alphabet.
    """
    best: tuple[str, float] | None = None
    for paragraph in _paragraphs(text):
        if not MORSE_RUN.search(paragraph):
            continue
        paragraph = _strip_prose_marker(paragraph, MORSE_ALPHABET)
        compact = paragraph.replace(" ", "")
        if not compact:
            continue
        purity = sum(1 for char in compact if char in ".-/") / len(compact)
        if best is None or purity > best[1]:
            best = (paragraph, purity)
    return best[0] if best is not None else None


def _code_candidates(text: str) -> list[str]:
    candidates = [match.group(1) for match in _FENCE.finditer(text)]
    candidates.extend(_paragraphs(text))
    return candidates


def _extract_code(text: str) -> str | None:
    """The first candidate block that is valid Python and builds a string.

    A markdown fence is tried before the bare paragraphs it may have come
    from (`_code_candidates` orders them that way); `ast.parse` is what
    decides "valid Python", never `exec`, `eval` or `compile` -- see the
    module docstring of `tipguard.benchmark.transformations.code_snippet`,
    whose `decode_snippet` this detector's positive verdict routes to.
    """
    for candidate in _code_candidates(text):
        if len(candidate) > MAX_SOURCE_LENGTH or not PYTHON_STRING_BUILDER.search(candidate):
            continue
        try:
            # `ast.parse` only builds a syntax tree; nothing here executes a
            # character of the candidate. See the module docstring of
            # `tipguard.benchmark.transformations.code_snippet`, whose
            # execution-free `decode_snippet` this detector's positive
            # verdict is meant to feed.
            ast.parse(candidate)
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            continue
        return candidate
    return None


def _detect_riddle_or_indirect(text: str) -> tuple[Family | None, str | None]:
    if RIDDLE_WORD.search(text):
        return Family.RIDDLE, "riddle_word"
    if INDIRECT_WORD.search(text):
        return Family.INDIRECT, "indirect_word"
    if WHAT_AM_I.search(text):
        return Family.RIDDLE, "what_am_i"
    if THE_THING_THAT.search(text):
        return Family.RIDDLE, "the_thing_that"
    # A bare "?" was tried as a fifth tell and removed. It is what a question
    # looks like, not what a riddle looks like, so it flagged half of every
    # hard negative in the corpus -- 84 of 180 -- and a sixth of the plain
    # direct requests, to buy 40 riddle and indirect cases. Measured over the
    # whole corpus: dropping it takes hard-negative false positives from
    # 0.500 to 0.033 and direct from 0.167 to 0.000, and costs 0.035 of
    # attack detection.
    #
    # That trade is the study's own subject rather than a tuning preference.
    # A defence that fires on half of the hard negatives -- prompts written
    # to look suspicious while being harmless -- has stopped measuring intent
    # and started measuring surface form, which is exactly what the
    # conventional baselines do and what this component exists to beat. The
    # riddle and indirect cases it gave up are the ones a syntactic rule
    # cannot reach; Task 6's LLM canonicalizer is what should reach them.
    return None, None


_STRONG_FAMILIES = frozenset({Family.BASE64, Family.MORSE, Family.CODE})

#: Evidence name for each of the two letters-family outcomes, indexed by
#: family. A lookup rather than an inline ternary at the one call site below,
#: kept short of the line-length limit.
_LETTERS_EVIDENCE: dict[Family, str] = {
    Family.CAESAR: "caesar_low_dict_hit",
    Family.REVERSE: "reverse_low_dict_hit",
}


def _family_confidence(family: Family, via_wrapper_words: bool) -> float:
    if family in _STRONG_FAMILIES:
        return 0.9
    if family is Family.SUBSTITUTION:
        return 0.75
    if via_wrapper_words:
        return 0.7
    return 0.5


@dataclass(frozen=True)
class _FamilyScan:
    """The raw, unescalated result of looking for every family's own signature.

    `families` is in priority order with duplicates removed: code, base64,
    morse, the letters family (caesar or reverse), then substitution last.
    Substitution is deliberately lowest priority even though its regex is
    checked early: `MultiStepTransformation` only ever draws substitution as
    the *inner* step of a composition (see that module's docstring), so
    whenever another family's signature is present too, that other family is
    always the one to decode first. A single-family prompt is unaffected by
    the ordering -- `families` has only one entry either way.
    """

    families: list[Family]
    evidence: list[str]
    implies_code: bool
    letters: LettersResolution | None


def _consume(text: str, payload: str) -> str:
    """`text` with `payload` blanked out, keeping every other offset intact.

    Blanked rather than removed so the surrounding wrapper is unchanged: the
    later tests read whole-text ratios, and deleting a span would shift the
    denominator as well as the content.
    """
    index = text.find(payload)
    if index == -1:
        return text
    return text[:index] + " " * len(payload) + text[index + len(payload) :]


def _scan_families(text: str) -> _FamilyScan:
    """One pass of every per-family syntactic test, with no escalation.

    This is the building block both `TransformationDetector.detect` (which
    layers the multi-step and riddle/indirect decisions on top of it) and
    `first_layer_family` (which a multi-step decode calls once per round, on
    whatever text remains after the previous round's decode) are built from.
    Calling `detect` itself for a later round would be wrong: `detect` would
    re-run the *same* multi-step escalation against text that, by round two,
    may no longer show the two-family or "wrapping"/"then" signature that
    justified it the first time, or -- worse, on round one, when the
    remaining text still *is* the original wrapped prompt -- would escalate
    right back to `MULTI_STEP` and the decode would never progress a single
    round. `first_layer_family` sidesteps that by reading only this
    unescalated scan.
    """
    distinct: list[Family] = []
    evidence: list[str] = []

    # Each span of text may account for one family. A base64 blob is a dense
    # run of mixed-case letters and digits, so it also trips the leet-density
    # test; a Morse run is dots and dashes, which the letters test reads as
    # unpronounceable. Counting those as separate families made every base64
    # case look like two transformations and escalate to MULTI_STEP -- family
    # accuracy on base64 measured 0.119 against a 0.9 floor. A payload
    # confirmed by its own decode therefore consumes its span before the
    # weaker, whole-text tests run on what is left.
    remaining = text

    code_payload = _extract_code(text)
    if code_payload is not None:
        distinct.append(Family.CODE)
        evidence.append("python_string_builder")
        remaining = _consume(remaining, code_payload)

    base64_payload = _extract_base64(text)
    if base64_payload is not None:
        distinct.append(Family.BASE64)
        evidence.append("base64_run")
        remaining = _consume(remaining, base64_payload)

    morse_match = MORSE_RUN.search(text)
    if morse_match:
        distinct.append(Family.MORSE)
        evidence.append("morse_run")
        remaining = _consume(remaining, morse_match.group(0))

    # Substitution is tested before the letter-only families, and ordered
    # ahead of them, because its glyphs are visible on the surface while a
    # caesar shift or a reversal is not. In a layered payload the visible
    # transformation is the outer one, so it is what a decode must peel
    # first. Taking the letters guess first decoded the inner layer against
    # still-substituted text and produced near-miss gibberish -- "tiis is
    # vsheot" for "this is urgent" -- which is worse than not decoding, since
    # it looks almost right.
    substituted = has_substitution_density(remaining)
    if substituted:
        distinct.append(Family.SUBSTITUTION)
        evidence.append("substitution_density")

    letters = _resolve_letters_family(remaining)
    if letters is not None:
        distinct.append(letters.family)
        evidence.append(_LETTERS_EVIDENCE[letters.family])

    implies_code = code_payload is not None
    unique = list(dict.fromkeys(distinct))
    return _FamilyScan(unique, evidence, implies_code, letters)


def first_layer_family(text: str) -> Family | None:
    """The single family whose signature is strongest in `text` right now.

    Used by `tipguard.canonicalization.deterministic` to pick each round's
    layer during a multi-step decode; see `_scan_families` for why that
    module cannot simply call `TransformationDetector.detect` again.
    """
    families = _scan_families(text).families
    return families[0] if families else None


class TransformationDetector:
    """Guesses whether a prompt carries a transformation and which family.

    Purely syntactic: it never decodes for the sake of deciding (base64 is
    the one exception, whose *own* detection test is "does this decode",
    per the brief), and it never reads `canonical_intent` or any other
    dataset field a production caller would not have. See the module
    docstring for the full per-family rationale.
    """

    def detect(self, text: str) -> DetectionResult:
        scan = _scan_families(text)
        letters = scan.letters

        is_multi_step = (
            len(scan.families) >= 2
            or bool(WRAPPING_HINT.search(text))
            or (
                letters is not None
                and not letters.via_wrapper_words
                and letters.hit_rate < CONFIDENT_HIT_THRESHOLD
                and bool(THEN_HINT.search(text))
            )
        )
        if is_multi_step:
            evidence = (*scan.evidence, "multi_step_hint")
            return DetectionResult(
                has_transformation=True,
                family=Family.MULTI_STEP,
                implies_code=scan.implies_code,
                confidence=0.6,
                evidence=evidence,
            )

        if scan.families:
            via_wrapper = letters.via_wrapper_words if letters is not None else False
            family = scan.families[0]
            return DetectionResult(
                has_transformation=True,
                family=family,
                implies_code=scan.implies_code,
                confidence=_family_confidence(family, via_wrapper),
                evidence=tuple(scan.evidence),
            )

        riddle_family, riddle_evidence = _detect_riddle_or_indirect(text)
        if riddle_family is not None and riddle_evidence is not None:
            return DetectionResult(
                has_transformation=True,
                family=riddle_family,
                implies_code=False,
                confidence=0.4 if riddle_evidence != "question_mark" else 0.25,
                evidence=(riddle_evidence,),
            )

        return DetectionResult(
            has_transformation=False,
            family=None,
            implies_code=False,
            confidence=0.0,
            evidence=(),
        )
