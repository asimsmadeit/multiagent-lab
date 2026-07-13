"""Deterministic offline fixtures for negotiation tests."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from negotiation import base_negotiator
from negotiation.components import theory_of_mind


class DeterministicModel:
    """Small prompt-aware model that satisfies Concordia's model protocol."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def sample_text(self, prompt: str, **_: object) -> str:
        self.prompts.append(prompt)
        if 'Return only the profile name' in prompt:
            return 'western_business'
        if 'Analyze the temporal context' in prompt:
            return (
                '- Interaction type: repeated\n'
                '- Current phase: middle\n'
                '- Future potential: high\n'
                '- Time horizon: long\n'
                '- Key temporal factors: trust, renewal'
            )
        if 'relationship impact' in prompt:
            return 'build|renewal promise|positive|medium'
        if 'extract:' in prompt.lower() and 'counterpart name' in prompt.lower():
            return 'Bob|continued negotiation|trust and future plans|75|unknown'
        if 'Format your response as:' in prompt and 'ANALYSIS:' in prompt:
            return (
                'ANALYSIS: Evidence-based assessment\n'
                'RECOMMENDATIONS: Ask a clarifying question, Offer a package\n'
                'CONFIDENCE: 0.8\n'
                'KEY_FACTORS: price, trust\n'
                'RISKS: deadline\n'
                'OPPORTUNITIES: renewal'
            )
        if 'Analyze this negotiation context for uncertainty' in prompt:
            return (
                'MISSING_INFO: budget\n'
                'UNCERTAINTY_SOURCES: private constraints\n'
                'CONFIDENCE_LEVELS: medium\n'
                'VALUABLE_INFO: reservation value\n'
                'SCENARIOS: agreement, impasse'
            )
        if 'Extract any numerical or quantitative information' in prompt:
            return (
                'BUDGET_INFO: 75 0.9\n'
                'TIMELINE_INFO: 0.7 0.8\n'
                'PRIORITY_INFO: 0.6 0.7\n'
                'MARKET_INFO: 0.2 0.6\n'
                'RELATIONSHIP_INFO: 0.7 0.8'
            )
        if 'strategic characteristics' in prompt:
            return (
                'stakes:0.7 relationship:0.8 time_pressure:0.3 '
                'competitive:0.4 complexity:0.6 uncertainty:0.5'
            )
        if 'emotional content' in prompt:
            return (
                'anger:0.1 fear:0.2 joy:0.6 sadness:0.0 surprise:0.1 '
                'trust:0.8 anticipation:0.5 valence:0.5 arousal:0.4 '
                'confidence:0.8'
            )
        if 'likely goals' in prompt:
            return 'financial:0.8 relationship:0.7 time:0.4 flexibility:0.6'
        if 'personality traits' in prompt:
            return 'openness:0.7 agreeableness:0.8 analytical:0.9'
        return 'Action: I offer $75 for the package and request confirmation.'

    def sample_choice(
        self,
        prompt: str,
        responses: tuple[str, ...] | list[str],
        **_: object,
    ) -> tuple[int, str, dict[str, float]]:
        self.prompts.append(prompt)
        return 0, responses[0], {}


@pytest.fixture
def model() -> DeterministicModel:
    return DeterministicModel()


@pytest.fixture
def memory_factory():
    def make_memory():
        memory_class = base_negotiator.basic_associative_memory.AssociativeMemoryBank
        kwargs: dict[str, object] = {
            'sentence_embedder': lambda _: np.ones(4),
        }
        if 'allow_duplicates' in inspect.signature(memory_class).parameters:
            kwargs['allow_duplicates'] = True
        return memory_class(**kwargs)

    return make_memory


@pytest.fixture
def action_spec():
    return theory_of_mind.entity_lib.free_action_spec(
        call_to_action='What should {name} do in negotiation round 2?'
    )
