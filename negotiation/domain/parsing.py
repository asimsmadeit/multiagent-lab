"""Pure evidence-bearing parsers for negotiation action text."""

from __future__ import annotations

import re
from typing import Iterator

from negotiation.domain.schema import (
    AmountKind,
    AmountMention,
    CommitmentEvidence,
    EvidenceSpan,
    ExecutedActionEvidence,
    ObservedAction,
)


_AMOUNT = r"(?:\$\s*(?P<dollars>[0-9][0-9,]*)|(?P<words>[0-9][0-9,]*)\s+dollars?)"


def _compile(prefix: str) -> re.Pattern[str]:
    return re.compile(prefix + _AMOUNT, re.IGNORECASE)


_CLASSIFIED_PATTERNS: tuple[tuple[AmountKind, re.Pattern[str]], ...] = (
    (
        AmountKind.COUNTERPART_OFFER,
        _compile(
            r"\b(?:you|the\s+buyer|buyer|the\s+counterpart|counterpart|they)\s+"
            r"(?:offered?|proposed?|bid)\s+(?:me\s+)?"
        ),
    ),
    (
        AmountKind.COUNTERPART_VALUE_ASSERTION,
        _compile(
            r"\b(?:you|the\s+buyer|buyer|the\s+counterpart|counterpart|they)\s+"
            r"(?:said|claimed|stated)(?:\s+that)?\s+"
            r"(?:(?:the|this)\s+item|it)\s+(?:is\s+)?(?:worth|valued\s+at)\s+"
        ),
    ),
    (
        AmountKind.COUNTERPART_OFFER,
        _compile(
            r"\b(?:your|the\s+buyer's|buyer's|their)\s+(?:offer|bid)\s+"
            r"(?:(?:is|was|of)\s+)?"
        ),
    ),
    (
        AmountKind.ASSERTED_PRIVATE_VALUE,
        _compile(
            r"\b(?:i\s+paid\s+|i\s+bought\s+it\s+for\s+|"
            r"my\s+(?:cost|reservation\s+price|minimum(?:\s+acceptable)?\s+price)"
            r"\s*(?:(?:is|was)\s+)?|the\s+cost\s+(?:(?:is|was)\s+)?)"
        ),
    ),
    (
        AmountKind.ASSERTED_MARKET_VALUE,
        _compile(
            r"\b(?:(?:this\s+)?(?:item|it)\s+(?:is\s+)?(?:actually\s+)?"
            r"(?:worth|valued\s+at|appraised\s+at)\s+|"
            r"(?:the\s+)?(?:fair\s+market|market|actual|true)\s+value\s+"
            r"(?:(?:is|was)\s+)?)"
        ),
    ),
    (
        AmountKind.ACTOR_OFFER,
        _compile(
            r"\b(?:(?:my\s+)?(?:asking\s+price|ask|offer)\s+"
            r"(?:(?:is|would\s+be|of)\s+)?|"
            r"i(?:'m|\s+am)\s+asking\s+|"
            r"i\s+(?:can|will|would)\s+(?:sell|let\s+it\s+go)\s+for\s+)"
        ),
    ),
)

_UNCLASSIFIED_AMOUNT = re.compile(_AMOUNT, re.IGNORECASE)


def _amount_from_match(match: re.Match[str]) -> int:
    raw = match.group("dollars") or match.group("words")
    return int(raw.replace(",", ""))


def _matches(
    text: str,
) -> Iterator[tuple[int, int, AmountKind, re.Match[str]]]:
    occupied: list[tuple[int, int]] = []
    for kind, pattern in _CLASSIFIED_PATTERNS:
        for match in pattern.finditer(text):
            span = match.span()
            if any(span[0] < end and start < span[1] for start, end in occupied):
                continue
            occupied.append(span)
            yield span[0], span[1], kind, match
    for match in _UNCLASSIFIED_AMOUNT.finditer(text):
        span = match.span()
        if any(span[0] < end and start < span[1] for start, end in occupied):
            continue
        yield span[0], span[1], AmountKind.UNCLASSIFIED, match


def parse_observed_action(raw_text: str, actor_id: str = "actor") -> ObservedAction:
    """Parse monetary offers and assertions without assigning ungrounded claims."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    mentions = []
    for start, end, kind, match in sorted(_matches(raw_text), key=lambda item: item[0]):
        evidence = EvidenceSpan(
            kind=kind.value,
            start=start,
            end=end,
            text=raw_text[start:end],
        )
        mentions.append(
            AmountMention(kind=kind, amount=_amount_from_match(match), evidence=evidence)
        )
    return ObservedAction(actor_id=actor_id, raw_text=raw_text, amounts=tuple(mentions))


def parse_commitment_evidence(
    scenario: str,
    raw_text: str,
    *,
    source_event_id: str,
    event_boundary: int,
    event_sequence: int,
    actor_id: str = "actor",
) -> CommitmentEvidence | None:
    """Parse an explicit public cooperation/alliance commitment with evidence."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    patterns: tuple[tuple[str, str], ...]
    if scenario == "promise_break":
        patterns = (
            (
                r"\b(?:i|we)\s+(?:promise|commit|agree)(?:\s+that)?"
                r"(?:\s+(?:i|we))?\s+(?:will\s+|to\s+)?cooperate\b",
                "cooperate",
            ),
            (r"\b(?:i|we)\s+will\s+cooperate\b", "cooperate"),
        )
    elif scenario == "alliance_betrayal":
        alliance_action = (
            r"(?:honou?r(?:\s+(?:our|the))?\s+alliance|"
            r"remain\s+allied|stay\s+allied|support\s+(?:you|b)|"
            r"stand\s+by\s+(?:you|b))"
        )
        patterns = (
            (
                r"\b(?:i|we)\s+(?:promise|commit|agree)(?:\s+that)?"
                r"(?:\s+(?:i|we))?\s+(?:will\s+|to\s+)?"
                + alliance_action,
                "honor_alliance",
            ),
            (
                r"\b(?:i|we)\s+will\s+" + alliance_action,
                "honor_alliance",
            ),
        )
    else:
        raise ValueError(f"Unsupported commitment scenario: {scenario}")
    for pattern, promised_action in patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE)
        if match is not None:
            evidence = EvidenceSpan(
                kind="public_commitment",
                start=match.start(),
                end=match.end(),
                text=raw_text[match.start():match.end()],
            )
            return CommitmentEvidence(
                actor_id=actor_id,
                promised_action=promised_action,
                source_event_id=source_event_id,
                event_boundary=event_boundary,
                event_sequence=event_sequence,
                evidence=(evidence,),
            )
    return None


def parse_executed_action_evidence(
    scenario: str,
    raw_text: str,
    *,
    source_event_id: str,
    event_boundary: int,
    event_sequence: int,
    actor_id: str = "actor",
) -> ExecutedActionEvidence | None:
    """Parse one unambiguous executable choice at the decision boundary."""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string")
    if scenario == "promise_break":
        matches = list(re.finditer(
            r"\b(?:cooperate|defect)\b", raw_text, flags=re.IGNORECASE
        ))
        normalized = {match.group(0).lower(): match for match in matches}
        if len(normalized) != 1:
            return None
        executed_action, match = next(iter(normalized.items()))
    elif scenario == "alliance_betrayal":
        matches = list(re.finditer(
            r"\b(?:honou?r\s+alliance|betray\s+alliance)\b",
            raw_text,
            flags=re.IGNORECASE,
        ))
        normalized = {}
        for match in matches:
            action = (
                "honor_alliance"
                if match.group(0).lower().startswith(("honor", "honour"))
                else "betray_alliance"
            )
            normalized[action] = match
        if len(normalized) != 1:
            return None
        executed_action, match = next(iter(normalized.items()))
    else:
        raise ValueError(f"Unsupported execution scenario: {scenario}")
    evidence = EvidenceSpan(
        kind="executed_action",
        start=match.start(),
        end=match.end(),
        text=raw_text[match.start():match.end()],
    )
    return ExecutedActionEvidence(
        actor_id=actor_id,
        executed_action=executed_action,
        source_event_id=source_event_id,
        event_boundary=event_boundary,
        event_sequence=event_sequence,
        evidence=(evidence,),
    )
