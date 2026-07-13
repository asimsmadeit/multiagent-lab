"""Temporal strategy component for multi-horizon negotiation planning."""

import dataclasses
import datetime
from collections.abc import Callable
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from concordia.typing import entity_component
from concordia.typing import entity as entity_lib
from negotiation.components.contracts import ComponentDiagnosticContract


@dataclasses.dataclass
class RelationshipRecord:
    """Track relationship history with a specific counterpart."""
    counterpart_name: str
    interaction_count: int = 0
    total_value_exchanged: float = 0.0
    trust_score: float = 0.5  # 0-1, starts neutral
    last_interaction: Optional[datetime.datetime] = None
    concession_history: List[float] = dataclasses.field(default_factory=list)
    outcome_history: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    reputation_score: float = 0.5  # 0-1, starts neutral

    def update_interaction(
        self,
        value: float,
        outcome: Dict[str, Any],
        *,
        now: Optional[datetime.datetime] = None,
    ) -> None:
        """Update relationship record with new interaction."""
        self.interaction_count += 1
        self.total_value_exchanged += abs(value)
        self.last_interaction = now or datetime.datetime.now()
        self.outcome_history.append(outcome)

        # Update trust only when the outcome contains explicit evidence.
        if outcome.get('promises_kept') is True:
            self.trust_score = min(1.0, self.trust_score + 0.05)
        elif outcome.get('promises_kept') is False:
            self.trust_score = max(0.0, self.trust_score - 0.1)

    def get_relationship_strength(
        self,
        *,
        now: Optional[datetime.datetime] = None,
    ) -> float:
        """Calculate overall relationship strength."""
        recency_factor = 1.0
        if self.last_interaction:
            current_time = now or datetime.datetime.now()
            days_since = max(0, (current_time - self.last_interaction).days)
            recency_factor = np.exp(-days_since / 30)  # Decay over 30 days

        interaction_factor = min(1.0, self.interaction_count / 10)

        return (self.trust_score * 0.5 +
                interaction_factor * 0.3 +
                recency_factor * 0.2)


@dataclasses.dataclass
class TemporalPlan:
    """Multi-horizon plan for negotiation strategy."""
    short_term_goal: str  # This negotiation
    medium_term_objective: str  # Next 3-5 interactions
    long_term_vision: str  # Ongoing relationship
    key_milestones: List[str] = dataclasses.field(default_factory=list)
    contingencies: Dict[str, str] = dataclasses.field(default_factory=dict)


class TemporalStrategy(entity_component.ContextComponent):
    """Component for temporal planning and relationship management in negotiations."""

    DIAGNOSTIC_CONTRACT = ComponentDiagnosticContract(
        inputs=('action_spec.call_to_action', 'action attempt', 'counterpart observation'),
        outputs=('temporal analysis', 'relationship guidance', 'relationship records'),
        state_fields=(
            'discount_factor', 'reputation_weight',
            'relationship_investment_threshold', 'relationships',
            'global_reputation', 'current_phase', 'phase_history',
            'temporal_plans',
        ),
        extra_model_calls={
            'pre_act': 1, 'post_act': 1, 'pre_observe': 1, 'post_observe': 0,
        },
        logging_fields=('current_phase', 'global_reputation', 'counterpart'),
    )

    def __init__(
        self,
        model: Any,
        discount_factor: float = 0.9,
        reputation_weight: float = 0.3,
        relationship_investment_threshold: float = 0.6,
        clock: Callable[[], datetime.datetime] = datetime.datetime.now,
    ):
        """Initialize temporal strategy component.

        Args:
            model: Language model for analysis
            discount_factor: Future value discount (0-1)
            reputation_weight: Importance of reputation (0-1)
            relationship_investment_threshold: When to invest in relationships
            clock: Injectable timestamp source for deterministic runs
        """
        for name, value in (
            ('discount_factor', discount_factor),
            ('reputation_weight', reputation_weight),
            ('relationship_investment_threshold', relationship_investment_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be between 0 and 1.')
        self._model = model
        self._discount_factor = discount_factor
        self._reputation_weight = reputation_weight
        self._investment_threshold = relationship_investment_threshold
        self._clock = clock

        # State tracking
        self._relationships: Dict[str, RelationshipRecord] = {}
        self._temporal_plans: Dict[str, TemporalPlan] = {}
        self._global_reputation = 0.7  # Start with good reputation
        self._current_phase = "opening"
        self._phase_history: List[str] = []

    def _analyze_temporal_context(self, observation: str) -> Dict[str, Any]:
        """Analyze temporal aspects of current situation."""
        prompt = f"""Analyze the temporal context of this negotiation:

Observation: {observation}

Consider:
1. Is this likely a one-time or repeated interaction?
2. What phase of negotiation are we in? (opening/middle/closing)
3. Are there signs of future interaction potential?
4. What is the strategic time horizon?

Provide analysis in this format:
- Interaction type: [one-time/repeated/long-term]
- Current phase: [opening/middle/closing]
- Future potential: [low/medium/high]
- Time horizon: [short/medium/long]
- Key temporal factors: [list main factors]"""

        response = self._model.sample_text(prompt)

        # Parse response
        lines = response.split('\n')
        analysis = {
            'interaction_type': 'repeated',
            'current_phase': 'opening',
            'future_potential': 'medium',
            'time_horizon': 'medium',
            'temporal_factors': []
        }

        for line in lines:
            normalized = line.strip().lstrip('-').strip()
            label, separator, value = normalized.partition(':')
            if not separator:
                continue
            label = label.strip().lower()
            value = value.strip().lower()
            if label == 'interaction type' and value in {'one-time', 'repeated', 'long-term'}:
                analysis['interaction_type'] = value
            elif label == 'current phase' and value in {'opening', 'middle', 'closing'}:
                analysis['current_phase'] = value
            elif label == 'future potential' and value in {'low', 'medium', 'high'}:
                analysis['future_potential'] = value
            elif label == 'time horizon' and value in {'short', 'medium', 'long'}:
                analysis['time_horizon'] = value
            elif label == 'key temporal factors':
                analysis['temporal_factors'] = [
                    factor.strip() for factor in value.split(',') if factor.strip()
                ]

        return analysis

    def _calculate_temporal_value(
        self,
        immediate_value: float,
        counterpart: str,
        action_type: str
    ) -> float:
        """Calculate total value considering multiple time horizons."""
        # Get relationship data
        relationship = self._relationships.get(
            counterpart,
            RelationshipRecord(counterpart)
        )

        # Short-term value (immediate)
        short_term = immediate_value

        # Medium-term value (relationship building)
        relationship_multiplier = 1.0 + (
            0.5 * relationship.get_relationship_strength(now=self._clock())
        )
        medium_term = immediate_value * relationship_multiplier * self._discount_factor

        # Long-term value (reputation and network effects)
        reputation_multiplier = 1.0 + (self._reputation_weight * self._global_reputation)
        long_term = immediate_value * reputation_multiplier * (self._discount_factor ** 2)

        # Adjust based on action type
        if action_type == 'concession':
            # Concessions build relationships
            medium_term *= 1.2
            long_term *= 1.1
        elif action_type == 'aggressive':
            # Aggressive moves may harm relationships
            medium_term *= 0.8
            long_term *= 0.7

        total_value = short_term + medium_term + long_term

        return total_value

    def _should_invest_in_relationship(self, counterpart: str) -> bool:
        """Determine if we should invest in this relationship."""
        relationship = self._relationships.get(
            counterpart,
            RelationshipRecord(counterpart)
        )

        # Factors for investment decision
        strength = relationship.get_relationship_strength(now=self._clock())
        interaction_frequency = relationship.interaction_count
        trust_level = relationship.trust_score

        # Calculate investment score
        investment_score = (
            strength * 0.4 +
            min(1.0, interaction_frequency / 5) * 0.3 +
            trust_level * 0.3
        )

        return investment_score >= self._investment_threshold

    def _generate_temporal_strategy(
        self,
        context: str,
        analysis: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate strategy considering temporal factors."""
        if analysis is None:
            analysis = self._analyze_temporal_context(context)

        # Create phase-specific guidance
        phase_strategies = {
            'opening': (
                "Focus on relationship building and understanding long-term potential. "
                "Be more generous with information sharing. "
                "Signal willingness for future collaboration."
            ),
            'middle': (
                "Balance immediate gains with relationship preservation. "
                "Reference past positive interactions if applicable. "
                "Create options for future value creation."
            ),
            'closing': (
                "Ensure sustainable agreement that enables future interaction. "
                "Leave relationship door open. "
                "Express appreciation and interest in future collaboration."
            )
        }

        # Create time-horizon specific guidance
        horizon_strategies = {
            'short': (
                "Focus on immediate value capture. "
                "Be efficient and direct. "
                "Minimize relationship investment."
            ),
            'medium': (
                "Balance current and future value. "
                "Build trust through fair dealing. "
                "Create precedents for future interactions."
            ),
            'long': (
                "Prioritize relationship and reputation. "
                "Accept short-term costs for long-term gains. "
                "Invest in mutual success and partnership."
            )
        }

        strategy = f"""Temporal Strategy Guidance:

Current Phase: {analysis['current_phase']}
{phase_strategies.get(analysis['current_phase'], '')}

Time Horizon: {analysis['time_horizon']}
{horizon_strategies.get(analysis['time_horizon'], '')}

Relationship Considerations:
- Future interaction potential is {analysis['future_potential']}
- Reputation impact weight: {self._reputation_weight}
- Consider how current actions affect future negotiations

Key Principles:
1. Short-term gains should not sacrifice long-term relationships
2. Build reputation through consistent fair dealing
3. Create value across time horizons
4. Adapt tactics to relationship stage and potential"""

        return strategy

    def pre_act(
        self,
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Add temporal strategy to action context."""
        # Store action_spec for post_act
        self._last_action_spec = action_spec

        # Get current context
        context = action_spec.call_to_action

        analysis = self._analyze_temporal_context(context)
        strategy = self._generate_temporal_strategy(context, analysis)

        # Update phase tracking
        current_phase = analysis['current_phase']
        if current_phase != self._current_phase:
            self._phase_history.append(self._current_phase)
            self._current_phase = current_phase

        return f"\n{strategy}"

    def post_act(self, action_attempt: str) -> str:
        """Update temporal state after action."""
        # Get stored action_spec context
        context = getattr(self, '_last_action_spec', None)
        context_str = context.call_to_action if context else "negotiation"

        # Analyze action for relationship impact
        prompt = f"""Analyze this negotiation action for relationship impact:

Action: {action_attempt}
Context: {context_str}

Evaluate:
1. Does this build or harm the relationship? [build/neutral/harm]
2. What promises or commitments were made? [list any]
3. How does this affect reputation? [positive/neutral/negative]
4. Estimated immediate value impact: [high/medium/low]

Format: impact_type|promises|reputation|value"""

        response = self._model.sample_text(prompt)

        # Simple parsing (in production, use more robust parsing)
        parts = response.split('|')
        if len(parts) >= 4:
            impact = parts[0].strip()
            promises = parts[1].strip()
            reputation = parts[2].strip()

            # Update global reputation
            if 'positive' in reputation.lower():
                self._global_reputation = min(1.0, self._global_reputation + 0.02)
            elif 'negative' in reputation.lower():
                self._global_reputation = max(0.0, self._global_reputation - 0.05)

        return ""

    def update(self) -> None:
        """Update temporal strategy state."""
        # Decay old relationships
        for relationship in self._relationships.values():
            if relationship.last_interaction:
                days_since = max(
                    0,
                    (self._clock() - relationship.last_interaction).days,
                )
                if days_since > 90:  # 3 months
                    # Slowly decay trust for inactive relationships
                    relationship.trust_score *= 0.99

    def observe(
        self,
        observation: str,
    ) -> None:
        """Populate a counterpart record from one explicit extraction."""
        prompt = f"""From this observation, extract:
1. Counterpart name (if mentioned):
2. Outcome category:
3. Relationship indicators (including promise kept/broken):
4. Numeric value exchanged, or unknown:
5. Numeric concession amount, or unknown:

Observation: {observation}

Format exactly: name|outcome|indicators|numeric_value_or_unknown|numeric_concession_or_unknown"""

        response = self._model.sample_text(prompt)
        parts = response.split('|')

        if (parts and parts[0].strip() and
                parts[0].strip().lower() not in {'unknown', 'none', 'n/a'}):
            counterpart = parts[0].strip()
            if counterpart not in self._relationships:
                self._relationships[counterpart] = RelationshipRecord(counterpart)
            outcome_text = parts[1].strip() if len(parts) > 1 else 'unknown'
            indicators = parts[2].strip() if len(parts) > 2 else ''
            value = self._parse_optional_number(parts[3] if len(parts) > 3 else '')
            concession = self._parse_optional_number(
                parts[4] if len(parts) > 4 else ''
            )
            normalized = f'{outcome_text} {indicators}'.casefold()
            outcome: Dict[str, Any] = {
                'outcome': outcome_text,
                'indicators': indicators,
                'source_observation': observation,
                'extraction_method': 'temporal-observation/1',
            }
            if any(phrase in normalized for phrase in (
                'promise broken', 'broke promise', 'did not keep promise'
            )):
                outcome['promises_kept'] = False
            elif any(phrase in normalized for phrase in (
                'promise kept', 'kept promise'
            )):
                outcome['promises_kept'] = True
            relationship = self._relationships[counterpart]
            relationship.update_interaction(
                value or 0.0,
                outcome,
                now=self._clock(),
            )
            if concession is not None:
                relationship.concession_history.append(concession)
            if 'reputation positive' in normalized:
                relationship.reputation_score = min(
                    1.0, relationship.reputation_score + 0.05
                )
            elif 'reputation negative' in normalized:
                relationship.reputation_score = max(
                    0.0, relationship.reputation_score - 0.1
                )

    @staticmethod
    def _parse_optional_number(value: str) -> Optional[float]:
        normalized = value.strip().replace('$', '').replace(',', '')
        if normalized.casefold() in {'', 'unknown', 'none', 'n/a'}:
            return None
        try:
            parsed = float(normalized)
        except ValueError:
            return None
        return parsed if np.isfinite(parsed) else None

    def pre_observe(self, observation: str) -> str:
        """Feed entity observations into temporal relationship tracking."""
        self.observe(observation)
        return ""

    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            'discount_factor': self._discount_factor,
            'reputation_weight': self._reputation_weight,
            'relationship_investment_threshold': self._investment_threshold,
            'relationships': {
                name: {
                    'trust_score': rec.trust_score,
                    'interaction_count': rec.interaction_count,
                    'total_value_exchanged': rec.total_value_exchanged,
                    'last_interaction': (
                        rec.last_interaction.isoformat()
                        if rec.last_interaction is not None else None
                    ),
                    'concession_history': list(rec.concession_history),
                    'outcome_history': list(rec.outcome_history),
                    'reputation_score': rec.reputation_score,
                }
                for name, rec in self._relationships.items()
            },
            'global_reputation': self._global_reputation,
            'current_phase': self._current_phase,
            'phase_history': self._phase_history,
            'temporal_plans': {
                name: dataclasses.asdict(plan)
                for name, plan in self._temporal_plans.items()
            },
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state."""
        if not isinstance(state, dict):
            raise TypeError('Temporal strategy state must be a mapping.')
        configuration = {
            'discount_factor': float(
                state.get('discount_factor', self._discount_factor)
            ),
            'reputation_weight': float(
                state.get('reputation_weight', self._reputation_weight)
            ),
            'relationship_investment_threshold': float(
                state.get(
                    'relationship_investment_threshold', self._investment_threshold
                )
            ),
        }
        for name, value in configuration.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f'{name} must be between 0 and 1.')
        self._discount_factor = configuration['discount_factor']
        self._reputation_weight = configuration['reputation_weight']
        self._investment_threshold = configuration[
            'relationship_investment_threshold'
        ]
        self._global_reputation = state.get('global_reputation', 0.7)
        self._current_phase = state.get('current_phase', 'opening')
        self._phase_history = [str(phase) for phase in state.get('phase_history', [])]
        self._temporal_plans = {
            str(name): TemporalPlan(**data)
            for name, data in state.get('temporal_plans', {}).items()
        }

        self._relationships = {}
        for name, data in state.get('relationships', {}).items():
            last_interaction = data.get('last_interaction')
            self._relationships[name] = RelationshipRecord(
                counterpart_name=name,
                interaction_count=int(data.get('interaction_count', 0)),
                total_value_exchanged=float(data.get('total_value_exchanged', 0.0)),
                trust_score=float(data.get('trust_score', 0.5)),
                last_interaction=(
                    datetime.datetime.fromisoformat(last_interaction)
                    if last_interaction else None
                ),
                concession_history=[
                    float(value) for value in data.get('concession_history', [])
                ],
                outcome_history=list(data.get('outcome_history', [])),
                reputation_score=float(data.get('reputation_score', 0.5)),
            )

    def get_action_attempt(
        self,
        context: Any,  # ComponentContextMapping
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Generate time-aware negotiation action considering multiple horizons."""
        situation_context = action_spec.call_to_action
        
        # Analyze temporal context
        temporal_analysis = self._analyze_temporal_context(situation_context)
        
        # Update phase tracking
        current_phase = temporal_analysis['current_phase']
        if current_phase != self._current_phase:
            self._phase_history.append(self._current_phase)
            self._current_phase = current_phase
        
        # Determine if we should invest in relationship
        counterpart_name = "counterpart"  # Default name
        should_invest = self._should_invest_in_relationship(counterpart_name)
        
        # Calculate temporal value for different action types
        base_value = 1000  # Base negotiation value
        concession_value = self._calculate_temporal_value(base_value, counterpart_name, "concession")
        aggressive_value = self._calculate_temporal_value(base_value, counterpart_name, "aggressive")
        
        # Generate action based on temporal strategy
        prompt = f"""Based on temporal strategy and multi-horizon thinking, generate a negotiation action:

Situation: {situation_context}

Temporal Analysis:
- Current Phase: {temporal_analysis['current_phase']}
- Interaction Type: {temporal_analysis['interaction_type']}
- Future Potential: {temporal_analysis['future_potential']}
- Time Horizon: {temporal_analysis['time_horizon']}

Strategic Considerations:
- Should invest in relationship: {should_invest}
- Concession approach value: {concession_value:.1f}
- Aggressive approach value: {aggressive_value:.1f}
- Global reputation level: {self._global_reputation:.2f}

Phase-Specific Guidance:
{self._generate_temporal_strategy(situation_context, temporal_analysis)}

Generate a negotiation action that:
1. Aligns with the {temporal_analysis['current_phase']} phase requirements
2. Considers {temporal_analysis['time_horizon']}-term value optimization  
3. {'Invests in relationship building' if should_invest else 'Focuses on immediate value'}
4. Balances short-term gains with long-term reputation
5. Adapts to the {temporal_analysis['interaction_type']} interaction context

Action:"""

        response = self._model.sample_text(prompt)
        
        # Clean up response
        action = response.strip()
        if action.lower().startswith('action:'):
            action = action[7:].strip()
        
        # Add temporal framing based on analysis
        if temporal_analysis['time_horizon'] == 'long' and should_invest:
            action = f"Looking at our long-term partnership potential, {action.lower()}"
        elif temporal_analysis['future_potential'] == 'high':
            action = f"Given the promising future opportunities, {action.lower()}"
        elif temporal_analysis['current_phase'] == 'closing':
            action = f"As we move toward closure, {action.lower()}"
        
        return action
