"""Response-text quality control for probe-training data.

Created 2026-04-21 in response to HF-dataset QC that found dialogue quality
ranging from 23% to 81% clean across model/scenario runs. Five structural
failure modes were identified and are encoded here as regex-based flags:

  - eot_token_leak        — response contains `</s>`, `<|eot_id|>`,
                            `<|end_of_text|>`, or `<|endoftext|>`. Almost
                            always a decode-time bug (skip_special_tokens
                            not set) and means the captured activation is
                            from a token position after end-of-turn.
  - repetition_loop       — same 20-char chunk appears 3+ times. Caused by
                            missing repetition_penalty; activation encodes
                            loop state, not a negotiation decision.
  - narrating_not_speaking — response contains third-person narration
                            (`Negotiator should...`, `Negotiator demonstrated...`).
                            Llama-3.1-8B specifically does this when the chat
                            template is not applied.
  - mc_template_echo      — response parrots the scenario prompt's option
                            list back (`a) ... b) ... c) ... d) ...`).
  - markdown_section_headers / tutorial_steps — the model writes a tutorial
                            instead of speaking.
  - too_short / broken_prefix — obvious text-level corruption.

The filter is applied post-hoc to existing activations (paired with their
transcripts) and also baked into the live experiment loop so runs that
produce <60% clean dialogue can be aborted early.

Usage:

    from interpretability.core.qc_filter import (
        classify_response, filter_samples, qc_report
    )

    flags = classify_response(response_text)
    # -> set of flag names, empty if clean

    kept = filter_samples(samples, drop_probe_rounds=True)
    report = qc_report(samples)
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Set


# Compile once at module load; these run on every sample during a full-dataset
# pass so the overhead matters.
_EOT_PATTERN = re.compile(r'<\|eot_id\|>|<\|endoftext\|>|<\|end_of_text\|>|</s>')
_MC_LIST_PATTERN = re.compile(r'\n[a-d]\)\s|\n[A-D]\.\s')
_MC_CONTEXT_WORDS = ('select', 'option', 'choose', 'move')
_MD_HEADER_PATTERN = re.compile(r'^##\s|\n##\s')
_STEP_PATTERN = re.compile(r'Step\s+\d+:')
_NARRATION_PATTERN = re.compile(
    r'\b(Negotiator|Counterpart|Agent)\s+'
    r'(demonstrated|should|would|could|shows?|acknowledges?|recognizes?)\b'
)
_BROKEN_PREFIX_PATTERNS = ('Negotiator iator', 'iator ')


# A flag name is a short string identifier. Exposed as a tuple so callers can
# pass specific flags to keep_flags without typos.
KNOWN_FLAGS: tuple = (
    'too_short',
    'eot_token_leak',
    'repetition_loop',
    'narrating_not_speaking',
    'mc_template_echo',
    'markdown_section_headers',
    'tutorial_steps',
    'broken_prefix',
)


def _has_repetition_loop(text: str, chunk_size: int = 15, min_hits: int = 3) -> bool:
    """Return True if any `chunk_size`-char substring appears `min_hits`+ times.

    Uses a stride-1 sliding window so phrase repetitions that do not align to
    the chunk boundary (e.g., "Accept the offer now. " repeated 10 times)
    are still caught. O(n) in response length with a Counter; responses are
    capped at ~1000 chars in practice so the overhead is negligible.
    """
    n = len(text)
    if n < chunk_size * min_hits:
        return False
    seen: Counter = Counter()
    for i in range(n - chunk_size + 1):
        chunk = text[i:i + chunk_size]
        seen[chunk] += 1
        if seen[chunk] >= min_hits:
            return True
    return False


def classify_response(text: Optional[str]) -> Set[str]:
    """Return the set of quality flags for a response string.

    Empty set means the response passed all quality checks. See module
    docstring for flag meanings. A `None` input returns `{'too_short'}`
    rather than raising, since legacy samples may have missing responses.
    """
    flags: Set[str] = set()
    if not isinstance(text, str):
        return {'too_short'}

    if len(text) < 20:
        flags.add('too_short')

    if _EOT_PATTERN.search(text):
        flags.add('eot_token_leak')

    if _MC_LIST_PATTERN.search(text) and any(w in text.lower() for w in _MC_CONTEXT_WORDS):
        flags.add('mc_template_echo')

    if _MD_HEADER_PATTERN.search(text):
        flags.add('markdown_section_headers')

    if _STEP_PATTERN.search(text):
        flags.add('tutorial_steps')

    if _has_repetition_loop(text):
        flags.add('repetition_loop')

    if _NARRATION_PATTERN.search(text):
        flags.add('narrating_not_speaking')

    if any(text.startswith(pfx) for pfx in _BROKEN_PREFIX_PATTERNS):
        flags.add('broken_prefix')

    return flags


def _sample_is_negotiation(sample: Any) -> bool:
    """Return True if the sample is a real negotiation turn.

    Prefers the explicit sample_type field (added 2026-04-21); falls back to
    the legacy round_num >= 0 convention for older .pt files.
    """
    sample_type = getattr(sample, 'sample_type', None)
    if sample_type is not None:
        return sample_type == 'negotiation'
    round_num = getattr(sample, 'round_num', None)
    if round_num is None:
        return True
    return round_num >= 0


def filter_samples(
    samples: Iterable[Any],
    *,
    keep_flags: Iterable[str] = (),
    drop_probe_rounds: bool = True,
    response_getter: Callable[[Any], Optional[str]] = lambda s: getattr(s, 'response', None),
) -> List[Any]:
    """Return the subset of `samples` that pass QC.

    Args:
        samples: iterable of sample objects with at least a `response`
            attribute (or customize via response_getter).
        keep_flags: flags that are tolerated; samples with only these flags
            are kept. Useful for permissive filters (e.g., keep
            markdown_section_headers but drop eot_token_leak).
        drop_probe_rounds: if True, exclude pre_verification and
            post_plausibility samples regardless of text quality.
        response_getter: callable extracting the response string from a sample.
    """
    keep_flags = set(keep_flags)
    unknown_flags = keep_flags.difference(KNOWN_FLAGS)
    if unknown_flags:
        raise ValueError(f"Unknown QC flags: {sorted(unknown_flags)}")
    kept: List[Any] = []
    for s in samples:
        if drop_probe_rounds and not _sample_is_negotiation(s):
            continue
        flags = classify_response(response_getter(s))
        if flags - keep_flags:
            continue
        kept.append(s)
    return kept


def qc_report(
    samples: Iterable[Any],
    *,
    response_getter: Callable[[Any], Optional[str]] = lambda s: getattr(s, 'response', None),
) -> Dict[str, Any]:
    """Return a summary dict of quality flag frequencies over a dataset.

    The returned dict includes:
        n_total, n_probe_rounds, n_negotiation, n_clean,
        flag_counts (Counter), pct_clean (float in [0,1]).
    """
    samples_list = list(samples)
    n_total = len(samples_list)
    n_probe = sum(1 for s in samples_list if not _sample_is_negotiation(s))
    n_negotiation = n_total - n_probe

    flag_counts: Counter = Counter()
    n_clean = 0
    for s in samples_list:
        if not _sample_is_negotiation(s):
            continue
        flags = classify_response(response_getter(s))
        if not flags:
            n_clean += 1
        for f in flags:
            flag_counts[f] += 1

    pct_clean = (n_clean / n_negotiation) if n_negotiation > 0 else 0.0
    return {
        'n_total': n_total,
        'n_probe_rounds': n_probe,
        'n_negotiation': n_negotiation,
        'n_clean': n_clean,
        'pct_clean': float(pct_clean),
        'flag_counts': dict(flag_counts),
    }
