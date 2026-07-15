"""Evidence-linked theory-of-mind diagnostics for negotiation agents."""

from __future__ import annotations

import dataclasses
import logging
import re
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from concordia.typing import entity as entity_lib
from concordia.typing import entity_component

from config.agents.negotiation import DeceptionDetectionConfig, TheoryOfMindConfig
from negotiation.components.contracts import (
    ComponentDiagnosticContract,
    ModelCallDeclaration,
)
from negotiation.utils.parsing import parse_named_floats

logger = logging.getLogger(__name__)
LanguageModel = Any


@dataclasses.dataclass
class EmotionalState:
    """An evidence-derived estimate of a counterpart's emotional state."""

    emotions: Dict[str, float]
    valence: float
    arousal: float
    confidence: float
    timestamp: str
    triggers: List[str] = dataclasses.field(default_factory=list)

    def dominant_emotion(self) -> Tuple[str, float]:
        """Return the strongest scored emotion."""
        if not self.emotions:
            return 'neutral', 0.0
        return max(self.emotions.items(), key=lambda item: item[1])

    def emotional_intensity(self) -> float:
        """Return maximum emotion intensity, or zero when unavailable."""
        return max(self.emotions.values()) if self.emotions else 0.0


@dataclasses.dataclass(frozen=True)
class TrustAssessment:
    """Trust is unavailable until an observation contains trust evidence."""

    value: Optional[float]
    available: bool
    method: str
    evidence: Tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class BeliefRecord:
    """A counterpart-specific recursive belief with explicit provenance."""

    counterpart_id: str
    level: int
    proposition: Optional[str]
    confidence: Optional[float]
    available: bool
    method: str
    evidence: Tuple[str, ...] = ()


@dataclasses.dataclass
class RecursiveBelief:
    """Legacy state shape retained solely for snapshot migration."""

    level: int
    believer: str
    content: Dict[str, Any]
    confidence: float
    evidence: List[str]


@dataclasses.dataclass
class MentalModel:
    """Current evidence-bearing model of one counterpart."""

    counterpart_id: str
    goals: Dict[str, float]
    personality_traits: Dict[str, float]
    emotional_state: EmotionalState
    strategies: Dict[str, float]
    constraints: List[str]
    deception_indicators: Dict[str, float]
    trust: TrustAssessment
    evidence: List[str]
    advice: str
    last_updated: str


class TheoryOfMind(entity_component.ContextComponent):
    """Analyze observations once, then expose pure counterpart diagnostics."""

    DIAGNOSTIC_CONTRACT = ComponentDiagnosticContract(
        inputs=('counterpart observation', 'action attempt'),
        outputs=('belief records', 'counterpart diagnostics', 'cached action guidance'),
        state_fields=(
            'mental_models', 'belief_records', 'emotion_history',
            'baseline_patterns', 'deception_cue_history',
            'deception_indicators', 'recursion_depth', 'emotion_sensitivity',
            'empathy_level', 'recent_emotional_trend', 'last_observation',
        ),
        extra_model_calls={
            'pre_act': 0,
            'post_act': 0,
            'pre_observe': 0,
            'post_observe': ModelCallDeclaration(
                0,
                3,
                'three analysis calls only when an observation is staged',
            ),
        },
        logging_fields=(
            'counterpart_id', 'deception_risk', 'trust_available',
            'belief_availability',
        ),
    )

    _POSITIVE_TRUST_CUES = (
        'i trust', 'we trust', 'reliable', 'dependable', 'kept your promise',
        'good faith', 'confidence in you',
    )
    _NEGATIVE_TRUST_CUES = (
        "don't trust", 'do not trust', 'cannot trust', "can't trust",
        'unreliable', 'broke your promise', 'broken promise', 'lied',
        'dishonest', 'bad faith',
    )

    def __init__(
        self,
        model: LanguageModel,
        max_recursion_depth: int = 3,
        emotion_sensitivity: float = 0.7,
        empathy_level: float = 0.8,
        tom_version: str = 'v1',
    ) -> None:
        if tom_version not in ('v1', 'v2'):
            raise ValueError("tom_version must be 'v1' or 'v2'.")
        if max_recursion_depth < 0:
            raise ValueError('max_recursion_depth must be non-negative.')
        if not 0.0 <= emotion_sensitivity <= 1.0:
            raise ValueError('emotion_sensitivity must be between 0 and 1.')
        if not 0.0 <= empathy_level <= 1.0:
            raise ValueError('empathy_level must be between 0 and 1.')
        self._model = model
        self._max_recursion_depth = max_recursion_depth
        self._emotion_sensitivity = emotion_sensitivity
        self._empathy_level = empathy_level
        self._mental_models: Dict[str, MentalModel] = {}
        self._belief_records: Dict[str, List[BeliefRecord]] = {}
        self._emotion_history: deque[EmotionalState] = deque(
            maxlen=TheoryOfMindConfig.EMOTION_HISTORY_SIZE
        )
        self._baseline_patterns: Dict[str, float] = {}
        self._deception_cue_history: deque[Dict[str, float]] = deque(
            maxlen=TheoryOfMindConfig.EMOTION_HISTORY_SIZE
        )
        self._deception_indicators: List[str] = []
        self._last_observation = ''
        # Theory of Mind 2.0 (Plan 3): v1 above stays the frozen baseline.
        # The v2 engine is attached per trial by the experiment runtime,
        # because belief hypothesis spaces and partner contracts are
        # trial-scoped, not builder-scoped.
        self._tom_version = tom_version
        self._v2_engine = None
        self._v2_decision = None
        self._pending_v2_snapshot: Optional[Dict[str, Any]] = None

    def _detect_emotions(self, communication: str) -> EmotionalState:
        """Perform the model-backed emotion analysis for one observation."""
        prompt = f"""Analyze the emotional content of this negotiation communication:

Communication: {communication}

Assess anger, fear, joy, sadness, surprise, trust, anticipation, valence,
arousal, and confidence. Format exactly as named decimal scores:
anger:X.X fear:X.X joy:X.X sadness:X.X surprise:X.X trust:X.X
anticipation:X.X valence:X.X arousal:X.X confidence:X.X"""
        response = self._model.sample_text(prompt)
        emotion_names = (
            'anger', 'fear', 'joy', 'sadness', 'surprise', 'trust',
            'anticipation',
        )
        bounds = {name: (0.0, 1.0) for name in emotion_names}
        bounds.update({
            'valence': (-1.0, 1.0), 'arousal': (0.0, 1.0),
            'confidence': (0.0, 1.0),
        })
        scores = parse_named_floats(response, bounds)
        emotions = {name: scores.get(name, 0.0) for name in emotion_names}
        return EmotionalState(
            emotions=emotions,
            valence=scores.get('valence', 0.0),
            arousal=scores.get('arousal', 0.0),
            confidence=scores.get('confidence', 0.0),
            timestamp='current',
            triggers=self._identify_emotional_triggers(communication),
        )

    @staticmethod
    def _identify_emotional_triggers(communication: str) -> List[str]:
        trigger_patterns = {
            'deadline_pressure': ('deadline', 'time', 'urgent', 'quickly'),
            'price_concern': ('expensive', 'cost', 'budget', 'afford'),
            'trust_issue': ('promise', 'guarantee', 'reliable', 'honest', 'trust'),
            'complexity': ('complicated', 'complex', 'difficult', 'confusing'),
            'progress': ('progress', 'moving forward', 'agreement', 'solution'),
        }
        lowered = communication.lower()
        return [
            trigger for trigger, keywords in trigger_patterns.items()
            if any(keyword in lowered for keyword in keywords)
        ]

    def _infer_goals(self, observations: List[str]) -> Dict[str, float]:
        prompt = f"""Analyze these negotiation observations to infer the counterpart's likely goals:

Observations: {observations}

Format: financial:X.X costs:X.X relationship:X.X precedent:X.X
information:X.X time:X.X reputation:X.X conflict:X.X
capabilities:X.X flexibility:X.X"""
        response = self._model.sample_text(prompt)
        names = (
            'financial', 'costs', 'relationship', 'precedent', 'information',
            'time', 'reputation', 'conflict', 'capabilities', 'flexibility',
        )
        return parse_named_floats(response, {name: (0.0, 1.0) for name in names})

    def _assess_personality(self, observations: List[str]) -> Dict[str, float]:
        prompt = f"""Analyze these communication patterns for personality traits:

Communication patterns: {observations}

Format: openness:X.X conscientiousness:X.X extraversion:X.X
agreeableness:X.X neuroticism:X.X risk_tolerance:X.X patience:X.X
assertiveness:X.X analytical:X.X"""
        response = self._model.sample_text(prompt)
        names = (
            'openness', 'conscientiousness', 'extraversion', 'agreeableness',
            'neuroticism', 'risk_tolerance', 'patience', 'assertiveness',
            'analytical',
        )
        return parse_named_floats(response, {name: (0.0, 1.0) for name in names})

    @staticmethod
    def _raw_deception_cues(statement: str) -> Dict[str, float]:
        lowered = statement.lower()
        words = lowered.split()
        total_words = len(words)
        complexity = 0.0
        if total_words:
            complex_words = sum(
                len(word) > DeceptionDetectionConfig.COMPLEX_WORD_MIN_LENGTH
                for word in words
            )
            complexity = min(
                1.0,
                complex_words / total_words
                * DeceptionDetectionConfig.LINGUISTIC_COMPLEXITY_MULTIPLIER,
            )
        return {
            'linguistic_complexity': complexity,
            'evasiveness': min(
                1.0,
                sum(phrase in lowered for phrase in DeceptionDetectionConfig.EVASIVE_PHRASES)
                * DeceptionDetectionConfig.EVASIVENESS_MULTIPLIER,
            ),
            'over_certainty': min(
                1.0,
                sum(phrase in lowered for phrase in DeceptionDetectionConfig.CERTAINTY_WORDS)
                * DeceptionDetectionConfig.OVER_CERTAINTY_MULTIPLIER,
            ),
            'defensive_language': min(
                1.0,
                sum(phrase in lowered for phrase in DeceptionDetectionConfig.DEFENSIVE_PHRASES)
                * DeceptionDetectionConfig.DEFENSIVE_LANGUAGE_MULTIPLIER,
            ),
            'negative_emotion': min(
                1.0,
                sum(word in words for word in DeceptionDetectionConfig.NEGATIVE_EMOTION_WORDS)
                * DeceptionDetectionConfig.NEGATIVE_EMOTION_MULTIPLIER,
            ),
        }

    def _detect_deception(self, statement: str) -> Dict[str, float]:
        """Calibrate raw linguistic cues against the observed cue baseline."""
        raw = self._raw_deception_cues(statement)
        return {
            name: max(0.0, value - self._baseline_patterns.get(name, 0.0))
            for name, value in raw.items()
        }

    def _update_deception_baseline(self, raw_cues: Dict[str, float]) -> None:
        self._deception_cue_history.append(dict(raw_cues))
        minimum = TheoryOfMindConfig.MIN_OBSERVATIONS_FOR_BASELINE
        if len(self._deception_cue_history) < minimum:
            return
        recent = list(self._deception_cue_history)[-minimum:]
        self._baseline_patterns = {
            name: sum(cues.get(name, 0.0) for cues in recent) / len(recent)
            for name in raw_cues
        }

    @classmethod
    def _assess_trust(
        cls,
        observation: str,
        previous: Optional[TrustAssessment],
    ) -> TrustAssessment:
        lowered = observation.lower()
        negative = tuple(cue for cue in cls._NEGATIVE_TRUST_CUES if cue in lowered)
        scrubbed = lowered
        for cue in negative:
            scrubbed = scrubbed.replace(cue, '')
        positive = tuple(cue for cue in cls._POSITIVE_TRUST_CUES if cue in scrubbed)
        evidence = tuple(
            [f'positive:{cue}' for cue in positive]
            + [f'negative:{cue}' for cue in negative]
        )
        if evidence:
            balance = len(positive) - len(negative)
            value = max(0.0, min(1.0, 0.5 + 0.2 * balance))
            return TrustAssessment(
                value=value,
                available=True,
                method='lexical_trust_cues/v1',
                evidence=evidence,
            )
        if previous is not None and previous.available:
            return TrustAssessment(
                value=previous.value,
                available=True,
                method='carried_forward_from_observed_trust/v1',
                evidence=previous.evidence,
            )
        return TrustAssessment(
            value=None,
            available=False,
            method='no_trust_evidence/v1',
            evidence=(),
        )

    def _build_belief_records(
        self,
        counterpart_id: str,
        observation: str,
    ) -> List[BeliefRecord]:
        records = [BeliefRecord(
            counterpart_id=counterpart_id,
            level=0,
            proposition=observation,
            confidence=TheoryOfMindConfig.BASE_BELIEF_CONFIDENCE,
            available=True,
            method='direct_observation/v1',
            evidence=(observation,),
        )]
        lowered = observation.lower()
        first_order = bool(re.search(
            r'\b(?:i|we)\s+(?:believe|think|expect|assume|know)\b', lowered
        ))
        second_order = bool(re.search(
            r'\b(?:i|we)\s+(?:believe|think|expect|assume)\s+'
            r'(?:you|they)\s+(?:believe|think|expect|assume|know)\b',
            lowered,
        ))
        for level in range(1, self._max_recursion_depth + 1):
            available = (level == 1 and first_order) or (level == 2 and second_order)
            records.append(BeliefRecord(
                counterpart_id=counterpart_id,
                level=level,
                proposition=observation if available else None,
                confidence=(
                    max(
                        TheoryOfMindConfig.MIN_BELIEF_CONFIDENCE,
                        TheoryOfMindConfig.BASE_BELIEF_CONFIDENCE
                        - level * TheoryOfMindConfig.BELIEF_CONFIDENCE_DECAY,
                    )
                    if available else None
                ),
                available=available,
                method=(
                    f'explicit_level_{level}_epistemic_cue/v1'
                    if available else 'insufficient_evidence/v1'
                ),
                evidence=(observation,) if available else (),
            ))
        return records

    @staticmethod
    def _counterpart_id(observation: str) -> str:
        colon_match = re.match(r'^\s*([A-Za-z][\w -]{0,40})\s*:', observation)
        if colon_match:
            candidate = colon_match.group(1).strip()
            if candidate.lower() not in {'self', 'negotiator', 'agent'}:
                return candidate
        name_match = re.match(
            r'^\s*([A-Z][a-z]+)\s+(?:says|said|asks|offers|is|rejects|accepts)\b',
            observation,
        )
        return name_match.group(1) if name_match else 'counterpart'

    def _render_advice(self, model: MentalModel) -> str:
        emotion, intensity = model.emotional_state.dominant_emotion()
        empathic = self._generate_empathic_response(model.emotional_state)
        trust = model.trust
        if not trust.available:
            trust_guidance = 'Trust is unknown; ask for evidence before relying on it.'
        elif trust.value is not None and trust.value < 0.5:
            trust_guidance = 'Address the observed trust concern explicitly.'
        else:
            trust_guidance = 'Preserve the evidence-backed trust signal.'
        return (
            f'{model.counterpart_id}: acknowledge {emotion} ({intensity:.2f}). '
            f'{empathic} {trust_guidance}'
        )

    def _generate_empathic_response(self, emotional_state: EmotionalState) -> str:
        emotion, intensity = emotional_state.dominant_emotion()
        if intensity < TheoryOfMindConfig.LOW_EMOTION_THRESHOLD:
            return 'Maintain a neutral, perspective-taking tone.'
        templates = {
            'anger': 'Validate the frustration before problem-solving.',
            'fear': 'Offer clarity without overstating reassurance.',
            'sadness': 'Acknowledge disappointment and propose alternatives.',
            'joy': 'Match enthusiasm while confirming concrete terms.',
            'surprise': 'Clarify how the proposal was derived.',
        }
        return templates.get(emotion, 'Acknowledge the observed emotional signal.')

    def analyze_observation(
        self,
        observation: str,
        counterpart_id: Optional[str] = None,
    ) -> None:
        """Own all model-backed updates for one counterpart observation."""
        if not observation.strip():
            return
        counterpart = counterpart_id or self._counterpart_id(observation)
        if counterpart.lower() in {'self', 'negotiator', 'agent'}:
            return
        previous = self._mental_models.get(counterpart)
        evidence = list(previous.evidence if previous else [])
        evidence.append(observation)
        evidence = evidence[-10:]
        emotional_state = self._detect_emotions(observation)
        raw_cues = self._raw_deception_cues(observation)
        model = MentalModel(
            counterpart_id=counterpart,
            goals=self._infer_goals(evidence),
            personality_traits=self._assess_personality(evidence),
            emotional_state=emotional_state,
            strategies={},
            constraints=[],
            deception_indicators=self._detect_deception(observation),
            trust=self._assess_trust(
                observation, previous.trust if previous is not None else None
            ),
            evidence=evidence,
            advice='',
            last_updated='current',
        )
        model.advice = self._render_advice(model)
        self._mental_models[counterpart] = model
        self._belief_records[counterpart] = self._build_belief_records(
            counterpart, observation
        )
        self._emotion_history.append(emotional_state)
        self._deception_indicators = [
            name for name, score in model.deception_indicators.items() if score > 0.0
        ]
        self._update_deception_baseline(raw_cues)

    def get_counterpart_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        """Return a JSON-safe, mutation-free view of counterpart evidence."""
        diagnostics: Dict[str, Dict[str, Any]] = {}
        for counterpart, model in sorted(self._mental_models.items()):
            if counterpart.lower() in {'self', 'negotiator', 'agent'}:
                continue
            dominant_emotion, _ = model.emotional_state.dominant_emotion()
            deception_risk = (
                sum(model.deception_indicators.values())
                / len(model.deception_indicators)
                if model.deception_indicators else 0.0
            )
            trust = dataclasses.asdict(model.trust)
            trust['evidence'] = list(model.trust.evidence)
            beliefs = []
            for record in self._belief_records.get(counterpart, []):
                belief = dataclasses.asdict(record)
                belief['evidence'] = list(record.evidence)
                beliefs.append(belief)
            diagnostics[counterpart] = {
                'counterpart_id': counterpart,
                'deception_risk': deception_risk,
                'deception_indicators': dict(model.deception_indicators),
                'emotion_intensity': model.emotional_state.emotional_intensity(),
                'valence': model.emotional_state.valence,
                'dominant_emotion': dominant_emotion,
                'top_goals': [
                    name for name, _ in sorted(
                        model.goals.items(), key=lambda item: item[1], reverse=True
                    )[:3]
                ],
                'trust': trust,
                'beliefs': beliefs,
                'advice': model.advice,
                'evidence': list(model.evidence),
                'last_updated': model.last_updated,
            }
        return diagnostics

    def _render_guidance(self) -> str:
        diagnostics = self.get_counterpart_diagnostics()
        if not diagnostics:
            return '\nTheory of Mind: no counterpart evidence observed yet.'
        lines = ['\nTheory of Mind (cached observation evidence):']
        for counterpart, diagnostic in diagnostics.items():
            lines.append(f"- {diagnostic['advice']}")
            for belief in diagnostic['beliefs']:
                status = 'available' if belief['available'] else 'unknown'
                lines.append(
                    f"  level {belief['level']}: {status} ({belief['method']})"
                )
        return '\n'.join(lines)

    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Render cached diagnostics without model calls or state mutation.

        Under ``tom_version='v2'`` this reads the persisted partner belief
        state (via the attached engine's latest prepared decision), never
        the call-to-action text.
        """
        del action_spec
        if self._tom_version == 'v2':
            return self._render_v2_guidance()
        return self._render_guidance()

    def post_act(self, action_attempt: str) -> str:
        """Do not mix the acting agent's output into counterpart beliefs."""
        del action_attempt
        return ''

    def pre_observe(self, observation: str) -> str:
        """Stage an observation; analysis belongs to :meth:`post_observe`."""
        self._last_observation = observation
        return ''

    def post_observe(self) -> str:
        """Analyze each staged observation exactly once (v1 only).

        The v2 path never analyzes raw text linguistically: typed evidence
        flows through :meth:`tom_engine`\\ 's ``ingest_action`` /
        ``ingest_evidence``, driven by the experiment runtime that owns the
        scenario extractors.
        """
        observation = self._last_observation
        self._last_observation = ''
        if observation and self._tom_version != 'v2':
            self.analyze_observation(observation)
        return ''

    def observe(self, observation: str) -> None:
        """Legacy direct observation entry point (v1 only)."""
        if self._tom_version == 'v2':
            return
        self.analyze_observation(observation)

    # ------------------------------------------------------------------
    # Theory of Mind 2.0 integration surface (Plan 3, Phases 6/7/9)
    # ------------------------------------------------------------------

    @property
    def tom_version(self) -> str:
        return self._tom_version

    @property
    def tom_engine(self):
        """The attached :class:`ToMV2Engine`, or ``None`` for v1/detached."""
        return self._v2_engine

    def attach_v2_engine(self, engine) -> None:
        """Attach the per-trial belief engine (v2 only, exactly once)."""
        from negotiation.components.tom.engine import ToMV2Engine

        if self._tom_version != 'v2':
            raise ValueError('attach_v2_engine requires tom_version v2.')
        if not isinstance(engine, ToMV2Engine):
            raise TypeError('engine must be a ToMV2Engine.')
        if self._v2_engine is not None:
            raise ValueError('a v2 engine is already attached.')
        self._v2_engine = engine
        if self._pending_v2_snapshot is not None:
            snapshot = self._pending_v2_snapshot
            self._pending_v2_snapshot = None
            engine.restore(snapshot['state'], snapshot['updates'])

    def prepare_v2_decision(self, **recommend_kwargs):
        """Compute and cache the advisory for the next acting call."""
        engine = self._require_v2_engine()
        decision = engine.recommend(**recommend_kwargs)
        self._v2_decision = decision
        return decision

    def link_v2_action(self, call_id: str):
        """Bind the cached advisory's decision trace to the acting call."""
        engine = self._require_v2_engine()
        return engine.link_action(call_id)

    def _require_v2_engine(self):
        if self._tom_version != 'v2':
            raise ValueError('this operation requires tom_version v2.')
        if self._v2_engine is None:
            raise ValueError('no v2 engine attached.')
        return self._v2_engine

    def _render_v2_guidance(self) -> str:
        if self._v2_engine is None:
            return (
                '\nTheory of Mind v2: engine not attached; '
                'no partner-model advisory available.'
            )
        if self._v2_decision is not None:
            return '\n' + self._v2_decision.guidance
        state = self._v2_engine.state
        leaning = max(
            zip(state.policy_type.probabilities, state.policy_type.categories)
        )[1]
        return (
            '\nTheory of Mind v2 '
            f'[{self._v2_engine.condition.value}]: partner policy leaning '
            f'{leaning}; no advisory prepared for this turn.'
        )

    def _get_emotional_trend(self) -> str:
        minimum = TheoryOfMindConfig.MIN_HISTORY_FOR_TREND
        if len(self._emotion_history) < minimum:
            return 'insufficient_data'
        recent = [state.valence for state in list(self._emotion_history)[-minimum:]]
        if all(value > TheoryOfMindConfig.POSITIVE_TREND_THRESHOLD for value in recent):
            return 'increasingly_positive'
        if all(value < TheoryOfMindConfig.NEGATIVE_TREND_THRESHOLD for value in recent):
            return 'increasingly_negative'
        if recent[-1] > recent[0]:
            return 'improving'
        if recent[-1] < recent[0]:
            return 'declining'
        return 'stable'

    def get_state(self) -> Dict[str, Any]:
        """Return all evidence-bearing state needed for exact restoration."""
        if self._tom_version == 'v2':
            return self._v2_state()
        mental_models = {}
        for counterpart, model in self._mental_models.items():
            if counterpart.lower() in {'self', 'negotiator', 'agent'}:
                continue
            serialized = dataclasses.asdict(model)
            serialized['trust']['evidence'] = list(model.trust.evidence)
            mental_models[counterpart] = serialized
        belief_records = {}
        for counterpart, records in self._belief_records.items():
            if counterpart.lower() in {'self', 'negotiator', 'agent'}:
                continue
            belief_records[counterpart] = []
            for record in records:
                serialized = dataclasses.asdict(record)
                serialized['evidence'] = list(record.evidence)
                belief_records[counterpart].append(serialized)
        return {
            'mental_models': mental_models,
            'belief_records': belief_records,
            'emotion_history': [
                dataclasses.asdict(emotion) for emotion in self._emotion_history
            ],
            'baseline_patterns': dict(self._baseline_patterns),
            'deception_cue_history': list(self._deception_cue_history),
            'deception_indicators': list(self._deception_indicators),
            'recursion_depth': self._max_recursion_depth,
            'emotion_sensitivity': self._emotion_sensitivity,
            'empathy_level': self._empathy_level,
            'recent_emotional_trend': self._get_emotional_trend(),
            'last_observation': self._last_observation,
        }

    @staticmethod
    def _restore_trust(data: Dict[str, Any]) -> TrustAssessment:
        trust_data = data.get('trust')
        if isinstance(trust_data, dict):
            value = trust_data.get('value')
            return TrustAssessment(
                value=float(value) if value is not None else None,
                available=bool(trust_data.get('available', value is not None)),
                method=str(trust_data.get('method', 'restored/v1')),
                evidence=tuple(str(item) for item in trust_data.get('evidence', [])),
            )
        legacy_value = data.get('trust_level')
        if legacy_value is not None:
            return TrustAssessment(
                value=float(legacy_value), available=True,
                method='legacy_trust_level/v1', evidence=(),
            )
        return TrustAssessment(None, False, 'no_trust_evidence/v1', ())

    def _v2_state(self) -> Dict[str, Any]:
        """Schema-native v2 snapshot retaining every legacy key (Phase 9)."""
        if self._v2_engine is not None:
            base = self._v2_engine.legacy_tom_state()
        else:
            base = {
                'mental_models': {},
                'belief_records': {},
                'emotion_history': [],
                'baseline_patterns': {},
                'deception_cue_history': [],
                'deception_indicators': [],
                'tom_version': 'v2',
            }
            if self._pending_v2_snapshot is not None:
                base['v2'] = dict(self._pending_v2_snapshot)
        base.update({
            'recursion_depth': self._max_recursion_depth,
            'emotion_sensitivity': self._emotion_sensitivity,
            'empathy_level': self._empathy_level,
            'recent_emotional_trend': 'insufficient_data',
            'last_observation': self._last_observation,
        })
        return base

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore a snapshot while accepting the previous snapshot shape."""
        if not isinstance(state, dict):
            raise TypeError('Theory-of-mind state must be a mapping.')
        if self._tom_version == 'v2':
            snapshot = state.get('v2')
            if snapshot is not None:
                if self._v2_engine is not None:
                    self._v2_engine.restore(
                        snapshot['state'], snapshot['updates']
                    )
                else:
                    self._pending_v2_snapshot = dict(snapshot)
            self._last_observation = str(state.get('last_observation', ''))
            return
        self._max_recursion_depth = int(
            state.get('recursion_depth', self._max_recursion_depth)
        )
        self._emotion_sensitivity = float(
            state.get('emotion_sensitivity', self._emotion_sensitivity)
        )
        self._empathy_level = float(state.get('empathy_level', self._empathy_level))
        self._mental_models = {}
        for counterpart, data in state.get('mental_models', {}).items():
            if str(counterpart).lower() in {'self', 'negotiator', 'agent'}:
                continue
            emotional_data = data.get('emotional_state')
            if not isinstance(emotional_data, dict):
                continue
            model = MentalModel(
                counterpart_id=str(data.get('counterpart_id', counterpart)),
                goals={str(k): float(v) for k, v in data.get('goals', {}).items()},
                personality_traits={
                    str(k): float(v)
                    for k, v in data.get('personality_traits', {}).items()
                },
                emotional_state=EmotionalState(**emotional_data),
                strategies={
                    str(k): float(v) for k, v in data.get('strategies', {}).items()
                },
                constraints=[str(item) for item in data.get('constraints', [])],
                deception_indicators={
                    str(k): float(v)
                    for k, v in data.get('deception_indicators', {}).items()
                },
                trust=self._restore_trust(data),
                evidence=[str(item) for item in data.get('evidence', [])],
                advice=str(data.get('advice', '')),
                last_updated=str(data.get('last_updated', 'current')),
            )
            if not model.advice:
                model.advice = self._render_advice(model)
            self._mental_models[str(counterpart)] = model
        self._belief_records = {}
        for counterpart, records in state.get('belief_records', {}).items():
            self._belief_records[str(counterpart)] = [
                BeliefRecord(
                    counterpart_id=str(record.get('counterpart_id', counterpart)),
                    level=int(record['level']),
                    proposition=record.get('proposition'),
                    confidence=(
                        float(record['confidence'])
                        if record.get('confidence') is not None else None
                    ),
                    available=bool(record.get('available', False)),
                    method=str(record.get('method', 'restored/v1')),
                    evidence=tuple(str(item) for item in record.get('evidence', [])),
                )
                for record in records
            ]
        if not self._belief_records and state.get('belief_hierarchy'):
            for level, beliefs in state['belief_hierarchy'].items():
                for belief in beliefs:
                    evidence = tuple(str(item) for item in belief.get('evidence', []))
                    record = BeliefRecord(
                        counterpart_id='counterpart',
                        level=int(level),
                        proposition=str(belief.get('content')) if evidence else None,
                        confidence=float(belief.get('confidence', 0.0)) if evidence else None,
                        available=bool(evidence),
                        method='legacy_snapshot/v1' if evidence else 'insufficient_evidence/v1',
                        evidence=evidence,
                    )
                    self._belief_records.setdefault('counterpart', []).append(record)
        self._emotion_history = deque(
            (EmotionalState(**item) for item in state.get('emotion_history', [])),
            maxlen=TheoryOfMindConfig.EMOTION_HISTORY_SIZE,
        )
        self._baseline_patterns = {
            str(k): float(v) for k, v in state.get('baseline_patterns', {}).items()
        }
        self._deception_cue_history = deque(
            (
                {str(k): float(v) for k, v in item.items()}
                for item in state.get('deception_cue_history', [])
            ),
            maxlen=TheoryOfMindConfig.EMOTION_HISTORY_SIZE,
        )
        self._deception_indicators = [
            str(item) for item in state.get('deception_indicators', [])
        ]
        self._last_observation = str(state.get('last_observation', ''))

    def get_action_attempt(
        self,
        context: Any,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Generate one action using cached evidence and one acting-model call."""
        del context
        prompt = (
            'Generate a negotiation action from this situation and cached '
            f'theory-of-mind evidence.\nSituation: {action_spec.call_to_action}\n'
            f'{self._render_guidance()}\nAction:'
        )
        response = self._model.sample_text(prompt).strip()
        return response[7:].strip() if response.lower().startswith('action:') else response

    def update(self) -> None:
        """Lifecycle no-op: evidence changes only when an observation arrives."""
        return None
