"""Utilities for parsing model responses in negotiation components."""

from collections.abc import Mapping, Sequence
import re
from typing import Any, Dict, List, Optional

from config.agents.negotiation import ParsingConfig


_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def parse_named_floats(
    response: str,
    field_bounds: Mapping[str, tuple[float, float]],
) -> Dict[str, float]:
  """Extract and clamp ``name: value`` fields from free-form model text.

  Models commonly return all requested fields on one line. Splitting the line at
  the first colon loses every value after the first, so callers should use this
  helper for compact as well as multiline responses.

  Args:
      response: Raw model response.
      field_bounds: Field names mapped to inclusive ``(minimum, maximum)``
        bounds. Matching is case-insensitive.

  Returns:
      Only the fields that were present with finite numeric values.
  """
  parsed: Dict[str, float] = {}
  for field, (minimum, maximum) in field_bounds.items():
    if minimum > maximum:
      raise ValueError(f"Invalid bounds for {field!r}: {minimum} > {maximum}")
    match = re.search(
        rf"(?<!\w){re.escape(field)}\s*:\s*({_FLOAT_PATTERN})",
        response,
        flags=re.IGNORECASE,
    )
    if match is None:
      continue
    try:
      value = float(match.group(1))
    except ValueError:
      continue
    if value != value or value in (float('inf'), float('-inf')):
      continue
    parsed[field] = max(minimum, min(maximum, value))
  return parsed


def parse_offer_value(text: str) -> Optional[float]:
  """Extract the most likely offer amount from negotiation text.

  Numbers attached to an ``offer`` phrase take precedence over unrelated
  currency amounts. Currency-marked amounts are then used as a fallback.
  """
  number = r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?"
  patterns: Sequence[str] = (
      rf"\b(?:offer|offering|counteroffer|propose|proposal)\D{{0,24}}({number})",
      rf"({number})\D{{0,24}}\b(?:offer|offering|counteroffer|proposal)\b",
      rf"(?:\$|USD\s*)\s*({number})",
      rf"({number})\s*(?:dollars?|USD)\b",
  )
  for pattern in patterns:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match is None:
      continue
    try:
      return float(match.group(1).replace(',', ''))
    except ValueError:
      continue
  return None


def contains_offer(text: str) -> bool:
  """Return whether text explicitly refers to an offer or proposal."""
  return bool(re.search(r"\b(?:counteroffers?|offers?|offering|proposals?)\b", text,
                        flags=re.IGNORECASE))


def signals_agreement(text: str) -> bool:
  """Return whether text contains an explicit, non-negated agreement signal."""
  normalized = ' '.join(text.lower().split())
  negated = (
      r"\b(?:do\s+not|don't|cannot|can't|will\s+not|won't|would\s+not|wouldn't|"
      r"not\s+ready\s+to)\s+(?:accept|agree)\b"
  )
  if re.search(negated, normalized):
    return False
  agreement_patterns = (
      r"\b(?:i|we)\s+accept\b",
      r"\b(?:i|we)\s+agree\b",
      r"\bagreement\s+(?:is\s+)?reached\b",
      r"\b(?:it(?:'s| is)|we\s+have)\s+a\s+deal\b",
      r"\boffer\s+accepted\b",
  )
  return any(re.search(pattern, normalized) for pattern in agreement_patterns)


def parse_structured_response(
    response: str,
    sections: Optional[List[str]] = None,
    default_confidence: Optional[float] = None,
) -> Dict[str, Any]:
  """Parse an LLM response with labeled sections.

  Handles responses in format:
      ANALYSIS: Some analysis text
      RECOMMENDATIONS: First recommendation
      Second recommendation
      CONFIDENCE: 0.8
      KEY_FACTORS: Factor 1
      Factor 2

  Args:
      response: Raw LLM response text
      sections: List of section names to look for. Defaults to common sections.
      default_confidence: Default confidence value if parsing fails.

  Returns:
      Dict mapping section names to their content.
      List sections (RECOMMENDATIONS, KEY_FACTORS, RISKS, OPPORTUNITIES)
      return lists. Other sections return strings.
  """
  if default_confidence is None:
    default_confidence = ParsingConfig.DEFAULT_CONFIDENCE

  if sections is None:
    sections = ParsingConfig.DEFAULT_SECTIONS

  list_sections = {
      'RECOMMENDATIONS', 'KEY_FACTORS', 'RISKS',
      'OPPORTUNITIES', 'SUGGESTIONS'
  }

  parsed: Dict[str, Any] = {
      'analysis': '',
      'recommendations': [],
      'confidence': default_confidence,
      'key_factors': [],
      'risks': [],
      'opportunities': []
  }

  current_section: Optional[str] = None

  for line in response.strip().split('\n'):
    line = line.strip()
    if not line:
      continue

    # Check if line starts a new section
    section_found = False
    for section in sections:
      prefix = f"{section}:"
      if line.upper().startswith(prefix.upper()):
        current_section = section.lower()
        content = line[len(prefix):].strip()

        if section.upper() == 'CONFIDENCE':
          try:
            confidence = float(content)
            parsed['confidence'] = max(0.0, min(1.0, confidence))
          except (ValueError, IndexError):
            parsed['confidence'] = default_confidence
        elif section.upper() in list_sections:
          parsed[current_section] = [content] if content else []
        else:
          parsed[current_section] = content
        section_found = True
        break

    # If not a new section, append to current section
    if not section_found and current_section:
      if current_section == 'confidence':
        continue  # Skip additional lines for confidence
      elif isinstance(parsed.get(current_section), list):
        if line:  # Don't add empty lines
          parsed[current_section].append(line)
      else:
        parsed[current_section] = parsed.get(current_section, '') + ' ' + line

  return parsed


def parse_confidence(response: str, default: float = 0.7) -> float:
  """Extract confidence score from response.

  Args:
      response: LLM response text
      default: Default value if parsing fails

  Returns:
      Confidence value between 0 and 1
  """
  parsed = parse_structured_response(
      response,
      ['CONFIDENCE'],
      default_confidence=default,
  )
  try:
    confidence = float(parsed.get('confidence', default))
    return max(0.0, min(1.0, confidence))
  except (ValueError, TypeError):
    return default


def parse_score(response: str, default: float = 0.5) -> float:
  """Extract numeric score from response.

  Args:
      response: LLM response text
      default: Default value if parsing fails

  Returns:
      Score value between 0 and 1
  """
  parsed = parse_structured_response(response, ['SCORE'])
  try:
    score = float(parsed.get('score', default))
    return max(0.0, min(1.0, score))
  except (ValueError, TypeError):
    return default
