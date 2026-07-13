"""Swarm intelligence component for collective negotiation decision-making."""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from abc import ABC, abstractmethod

from concordia.typing import entity_component
from concordia.typing import entity as entity_lib

from negotiation.utils.parsing import (
    parse_structured_response,
)
from negotiation.components.contracts import (
    ComponentDiagnosticContract,
    ModelCallDeclaration,
    ModelCallDiagnosticsMixin,
)


@dataclasses.dataclass
class SubAgentAnalysis:
    """Analysis result from a specialized sub-agent."""
    agent_type: str
    analysis: str
    recommendations: List[str]
    confidence: float  # 0-1
    key_factors: List[str]
    risks: List[str]
    opportunities: List[str]


@dataclasses.dataclass
class CollectiveDecision:
    """Result of collective decision-making process."""
    chosen_strategy: str
    confidence: float
    supporting_agents: List[str]
    dissenting_agents: List[str]
    compromise_elements: List[str]
    implementation_notes: str
    consensus_reached: bool = False
    iterations_used: int = 0
    consensus_threshold: float = 0.7
    support_ratio: float = 0.0


class SubAgent(ABC):
    """Abstract base class for specialized sub-agents."""

    def __init__(self, model: Any, agent_type: str):
        self._model = model
        self._agent_type = agent_type
        self._performance_history: List[float] = []
        self._expertise_weight = 1.0

    @abstractmethod
    def analyze_situation(self, context: str) -> SubAgentAnalysis:
        """Analyze situation from agent's specialized perspective."""
        pass

    @abstractmethod
    def get_relevance_score(self, context: str) -> float:
        """Calculate how relevant this agent's expertise is for the context."""
        pass

    def update_performance(self, outcome_score: float):
        """Update performance history and expertise weight."""
        outcome_score = max(0.0, min(1.0, float(outcome_score)))
        self._performance_history.append(outcome_score)
        # Keep last 10 outcomes
        if len(self._performance_history) > 10:
            self._performance_history.pop(0)

        # Update expertise weight based on recent performance
        if self._performance_history:
            recent_performance = np.mean(self._performance_history[-5:])
            self._expertise_weight = float(0.5 + recent_performance)


class MarketAnalysisAgent(SubAgent):
    """Sub-agent specialized in market analysis and economic factors."""

    def __init__(self, model: Any):
        super().__init__(model, "market_analysis")

    def analyze_situation(self, context: str) -> SubAgentAnalysis:
        """Analyze market and economic aspects."""
        prompt = f"""As a market analysis specialist, analyze this negotiation context:

Context: {context}

Provide analysis focusing on:
1. Economic factors and market conditions
2. Pricing benchmarks and competitive landscape
3. Financial risks and opportunities
4. Market trends affecting the negotiation

Format your response as:
ANALYSIS: [Your market analysis]
RECOMMENDATIONS: [3-5 specific recommendations]
CONFIDENCE: [0.0-1.0]
KEY_FACTORS: [List key economic factors]
RISKS: [List financial/market risks]
OPPORTUNITIES: [List market opportunities]"""

        response = self._model.sample_text(prompt)

        # Parse response
        analysis = self._parse_response(response)

        return SubAgentAnalysis(
            agent_type=self._agent_type,
            analysis=analysis.get('analysis', 'Market analysis completed'),
            recommendations=analysis.get('recommendations', []),
            confidence=analysis.get('confidence', 0.7),
            key_factors=analysis.get('key_factors', []),
            risks=analysis.get('risks', []),
            opportunities=analysis.get('opportunities', [])
        )

    def get_relevance_score(self, context: str) -> float:
        """Calculate relevance for market analysis."""
        keywords = ['price', 'cost', 'market', 'economic', 'financial', 'budget', 'value', 'competitive']
        relevance = sum(1 for word in keywords if word.lower() in context.lower())
        return min(1.0, relevance / 4.0)  # Normalize to 0-1

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse structured response from language model."""
        return parse_structured_response(response)


class EmotionalIntelligenceAgent(SubAgent):
    """Sub-agent specialized in emotional intelligence and relationship dynamics."""

    def __init__(self, model: Any):
        super().__init__(model, "emotional_intelligence")

    def analyze_situation(self, context: str) -> SubAgentAnalysis:
        """Analyze emotional and relational aspects."""
        prompt = f"""As an emotional intelligence specialist, analyze this negotiation context:

Context: {context}

Provide analysis focusing on:
1. Emotional states and sentiment analysis
2. Relationship dynamics and trust levels
3. Communication patterns and styles
4. Potential interpersonal risks and opportunities

Format your response as:
ANALYSIS: [Your emotional intelligence analysis]
RECOMMENDATIONS: [3-5 specific recommendations]
CONFIDENCE: [0.0-1.0]
KEY_FACTORS: [List key emotional/relational factors]
RISKS: [List relationship/emotional risks]
OPPORTUNITIES: [List relationship opportunities]"""

        response = self._model.sample_text(prompt)
        analysis = self._parse_response(response)

        return SubAgentAnalysis(
            agent_type=self._agent_type,
            analysis=analysis.get('analysis', 'Emotional analysis completed'),
            recommendations=analysis.get('recommendations', []),
            confidence=analysis.get('confidence', 0.7),
            key_factors=analysis.get('key_factors', []),
            risks=analysis.get('risks', []),
            opportunities=analysis.get('opportunities', [])
        )

    def get_relevance_score(self, context: str) -> float:
        """Calculate relevance for emotional intelligence."""
        keywords = ['relationship', 'trust', 'emotion', 'feeling', 'communication', 'interpersonal', 'conflict', 'satisfaction']
        relevance = sum(1 for word in keywords if word.lower() in context.lower())
        return min(1.0, relevance / 4.0)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse structured response from language model."""
        return parse_structured_response(response)


class GameTheoryStrategist(SubAgent):
    """Sub-agent specialized in game theory and strategic optimization."""

    def __init__(self, model: Any):
        super().__init__(model, "game_theory")

    def analyze_situation(self, context: str) -> SubAgentAnalysis:
        """Analyze strategic and game-theoretic aspects."""
        prompt = f"""As a game theory strategist, analyze this negotiation context:

Context: {context}

Provide analysis focusing on:
1. Strategic positions and payoff structures
2. Optimal moves and counter-moves
3. Nash equilibrium and strategy optimization
4. Risk assessment and mitigation strategies

Format your response as:
ANALYSIS: [Your game theory analysis]
RECOMMENDATIONS: [3-5 specific strategic recommendations]
CONFIDENCE: [0.0-1.0]
KEY_FACTORS: [List key strategic factors]
RISKS: [List strategic risks]
OPPORTUNITIES: [List strategic opportunities]"""

        response = self._model.sample_text(prompt)
        analysis = self._parse_response(response)

        return SubAgentAnalysis(
            agent_type=self._agent_type,
            analysis=analysis.get('analysis', 'Strategic analysis completed'),
            recommendations=analysis.get('recommendations', []),
            confidence=analysis.get('confidence', 0.7),
            key_factors=analysis.get('key_factors', []),
            risks=analysis.get('risks', []),
            opportunities=analysis.get('opportunities', [])
        )

    def get_relevance_score(self, context: str) -> float:
        """Calculate relevance for game theory."""
        keywords = ['strategy', 'optimal', 'competition', 'move', 'counter', 'advantage', 'position', 'tactical']
        relevance = sum(1 for word in keywords if word.lower() in context.lower())
        return min(1.0, relevance / 4.0)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse structured response from language model."""
        return parse_structured_response(response)


class DiplomaticRelationsAgent(SubAgent):
    """Sub-agent specialized in diplomatic relations and win-win solutions."""

    def __init__(self, model: Any):
        super().__init__(model, "diplomatic_relations")

    def analyze_situation(self, context: str) -> SubAgentAnalysis:
        """Analyze diplomatic and relationship aspects."""
        prompt = f"""As a diplomatic relations specialist, analyze this negotiation context:

Context: {context}

Provide analysis focusing on:
1. Relationship preservation and building opportunities
2. Win-win solutions and mutual benefit identification
3. Conflict prevention and de-escalation strategies
4. Long-term partnership potential

Format your response as:
ANALYSIS: [Your diplomatic analysis]
RECOMMENDATIONS: [3-5 specific diplomatic recommendations]
CONFIDENCE: [0.0-1.0]
KEY_FACTORS: [List key diplomatic factors]
RISKS: [List relationship risks]
OPPORTUNITIES: [List partnership opportunities]"""

        response = self._model.sample_text(prompt)
        analysis = self._parse_response(response)

        return SubAgentAnalysis(
            agent_type=self._agent_type,
            analysis=analysis.get('analysis', 'Diplomatic analysis completed'),
            recommendations=analysis.get('recommendations', []),
            confidence=analysis.get('confidence', 0.7),
            key_factors=analysis.get('key_factors', []),
            risks=analysis.get('risks', []),
            opportunities=analysis.get('opportunities', [])
        )

    def get_relevance_score(self, context: str) -> float:
        """Calculate relevance for diplomatic relations."""
        keywords = ['partnership', 'relationship', 'cooperation', 'mutual', 'win-win', 'diplomatic', 'collaborative', 'alliance']
        relevance = sum(1 for word in keywords if word.lower() in context.lower())
        return min(1.0, relevance / 4.0)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse structured response from language model."""
        return parse_structured_response(response)


class SwarmIntelligence(
    ModelCallDiagnosticsMixin,
    entity_component.ContextComponent,
):
    """Component for collective intelligence through specialized sub-agents."""

    DIAGNOSTIC_CONTRACT = ComponentDiagnosticContract(
        inputs=('action_spec.call_to_action', 'action attempt'),
        outputs=('sub-agent analyses', 'collective decision', 'guidance'),
        state_fields=(
            'consensus_threshold', 'max_iterations', 'sub_agents',
            'last_analyses', 'decision_history',
        ),
        extra_model_calls={
            'pre_act': ModelCallDeclaration(
                0,
                4,
                'exactly one call per enabled specialized sub-agent',
            ),
            'post_act': 0,
            'pre_observe': 0,
            'post_observe': 0,
        },
        logging_fields=('supporting_agents', 'dissenting_agents', 'confidence'),
    )

    def __init__(
        self,
        model: Any,
        consensus_threshold: float = 0.7,
        max_iterations: int = 3,
        enable_sub_agents: Optional[List[str]] = None
    ):
        """Initialize swarm intelligence component.

        Args:
            model: Language model for analysis
            consensus_threshold: Minimum agreement level for decisions
            max_iterations: Maximum consensus-building iterations
            enable_sub_agents: List of sub-agents to enable
        """
        if not 0.0 <= consensus_threshold <= 1.0:
            raise ValueError('consensus_threshold must be between 0 and 1.')
        if max_iterations < 1:
            raise ValueError('max_iterations must be at least 1.')
        self._model = model
        self._consensus_threshold = consensus_threshold
        self._max_iterations = max_iterations

        # Initialize sub-agents
        if enable_sub_agents is None:
            enable_sub_agents = ['market_analysis', 'emotional_intelligence', 'game_theory', 'diplomatic_relations']

        self._sub_agents: Dict[str, SubAgent] = {}
        supported_agents = {
            'market_analysis',
            'emotional_intelligence',
            'game_theory',
            'diplomatic_relations',
        }
        unknown_agents = set(enable_sub_agents) - supported_agents
        if unknown_agents:
            raise ValueError(
                f'Unknown sub-agents: {sorted(unknown_agents)}'
            )
        for agent_type in enable_sub_agents:
            if agent_type == 'market_analysis':
                self._sub_agents[agent_type] = MarketAnalysisAgent(model)
            elif agent_type == 'emotional_intelligence':
                self._sub_agents[agent_type] = EmotionalIntelligenceAgent(model)
            elif agent_type == 'game_theory':
                self._sub_agents[agent_type] = GameTheoryStrategist(model)
            elif agent_type == 'diplomatic_relations':
                self._sub_agents[agent_type] = DiplomaticRelationsAgent(model)

        # State tracking
        self._last_analyses: Dict[str, SubAgentAnalysis] = {}
        self._decision_history: List[CollectiveDecision] = []

    def _collect_analyses(self, context: str) -> Dict[str, SubAgentAnalysis]:
        """Collect analysis from all sub-agents."""
        analyses = {}
        for agent_type, agent in self._sub_agents.items():
            try:
                analysis = agent.analyze_situation(context)
                analyses[agent_type] = analysis
            except Exception as e:
                # Fallback analysis if sub-agent fails
                analyses[agent_type] = SubAgentAnalysis(
                    agent_type=agent_type,
                    analysis=f"Analysis failed: {str(e)}",
                    recommendations=["Proceed with caution"],
                    confidence=0.3,
                    key_factors=["Analysis error"],
                    risks=["Unable to analyze"],
                    opportunities=[]
                )

        return analyses

    def _calculate_weights(self, context: str) -> Dict[str, float]:
        """Calculate expertise weights for the current context."""
        weights = {}
        total_relevance = 0

        for agent_type, agent in self._sub_agents.items():
            relevance = agent.get_relevance_score(context)
            expertise = agent._expertise_weight
            combined_weight = relevance * expertise
            weights[agent_type] = combined_weight
            total_relevance += combined_weight

        # Normalize weights
        if total_relevance > 0:
            for agent_type in weights:
                weights[agent_type] /= total_relevance
        else:
            # Equal weights if no relevance detected
            if not weights:
                return {}
            equal_weight = 1.0 / len(weights)
            for agent_type in weights:
                weights[agent_type] = equal_weight

        return weights

    def _build_collective_decision(
        self,
        analyses: Dict[str, SubAgentAnalysis],
        weights: Dict[str, float],
    ) -> CollectiveDecision:
        """Build a bounded consensus from weighted recommendation support."""
        rec_scores: Dict[str, float] = {}
        display_text: Dict[str, str] = {}
        recommendations_by_agent: Dict[str, set[str]] = {}
        for agent_type, analysis in analyses.items():
            weight = weights.get(agent_type, 0.0)
            confidence = analysis.confidence
            combined_weight = weight * confidence
            normalized = {
                ' '.join(recommendation.split()).casefold()
                for recommendation in analysis.recommendations
                if recommendation.strip()
            }
            recommendations_by_agent[agent_type] = normalized
            for recommendation in analysis.recommendations:
                key = ' '.join(recommendation.split()).casefold()
                if not key:
                    continue
                display_text.setdefault(key, recommendation.strip())
                rec_scores[key] = rec_scores.get(key, 0.0) + combined_weight

        selected: List[str] = []
        supporting_agents: List[str] = []
        support_ratio = 0.0
        iterations_used = 0
        total_weight = sum(max(0.0, value) for value in weights.values())
        for iterations_used in range(1, self._max_iterations + 1):
            remaining = [key for key in rec_scores if key not in selected]
            if not remaining:
                break
            selected.append(max(remaining, key=lambda key: (rec_scores[key], key)))
            selected_set = set(selected)
            supporting_agents = [
                agent_type
                for agent_type in analyses
                if recommendations_by_agent.get(agent_type, set()) & selected_set
            ]
            supported_weight = sum(
                max(0.0, weights.get(agent_type, 0.0))
                for agent_type in supporting_agents
            )
            support_ratio = supported_weight / total_weight if total_weight else 0.0
            if support_ratio >= self._consensus_threshold:
                break

        consensus_reached = support_ratio >= self._consensus_threshold
        dissenting_agents = [
            agent_type for agent_type in analyses
            if agent_type not in supporting_agents
        ]
        if selected:
            primary = display_text[selected[0]]
            additions = [display_text[key] for key in selected[1:]]
            chosen_strategy = primary
            if additions:
                chosen_strategy += '; incorporate: ' + '; '.join(additions)
            confidence = support_ratio
        else:
            chosen_strategy = "Proceed with balanced approach"
            confidence = 0.0

        compromise_elements = [
            f"Include supported concern: {display_text[key]}"
            for key in selected[1:]
        ]
        for agent_type in dissenting_agents:
            if analyses[agent_type].recommendations:
                compromise_elements.append(
                    f"Unresolved {agent_type} concern: "
                    f"{analyses[agent_type].recommendations[0]}"
                )

        return CollectiveDecision(
            chosen_strategy=chosen_strategy,
            confidence=confidence,
            supporting_agents=supporting_agents,
            dissenting_agents=dissenting_agents,
            compromise_elements=compromise_elements[:3],  # Limit to 3 elements
            implementation_notes=(
                f"Consensus {'reached' if consensus_reached else 'not reached'} "
                f"after {iterations_used} iteration(s): "
                f"{support_ratio:.3f} support against "
                f"{self._consensus_threshold:.3f} threshold"
            ),
            consensus_reached=consensus_reached,
            iterations_used=iterations_used,
            consensus_threshold=self._consensus_threshold,
            support_ratio=support_ratio,
        )

    def _generate_swarm_guidance(self, decision: CollectiveDecision, analyses: Dict[str, SubAgentAnalysis]) -> str:
        """Generate comprehensive guidance from swarm intelligence."""
        guidance = f"""🧠 Collective Intelligence Analysis

**Recommended Strategy**: {decision.chosen_strategy}
**Confidence Level**: {decision.confidence:.1f}
**Supporting Expertise**: {', '.join(decision.supporting_agents)}

**Multi-Perspective Analysis**:
"""

        for agent_type, analysis in analyses.items():
            guidance += f"\n🔍 **{agent_type.replace('_', ' ').title()}**:\n"
            guidance += f"   Analysis: {analysis.analysis}\n"
            if analysis.recommendations:
                guidance += f"   Key Recommendation: {analysis.recommendations[0]}\n"
            if analysis.risks:
                guidance += f"   Primary Risk: {analysis.risks[0] if analysis.risks else 'None identified'}\n"

        guidance += f"\n**Implementation Considerations**:\n"
        for element in decision.compromise_elements:
            guidance += f"   • {element}\n"

        guidance += f"\n**Collective Wisdom**:\n"
        guidance += f"   • This strategy balances insights from {len(analyses)} specialized perspectives\n"
        guidance += f"   • {len(decision.supporting_agents)} experts directly support this approach\n"
        guidance += f"   • Consider compromise elements to address dissenting viewpoints\n"

        return guidance

    def pre_act(self, action_spec: entity_lib.ActionSpec) -> str:
        """Provide collective intelligence guidance before action."""
        context = action_spec.call_to_action

        # Collect analyses from all sub-agents
        analyses = self._collect_analyses(context)
        self._last_analyses = analyses
        enabled_count = len(self._sub_agents)
        self._record_model_call_branch(
            'pre_act',
            observed_calls=enabled_count,
            expected_calls=enabled_count,
            branch=f'{enabled_count}_enabled_sub_agents',
        )

        # Calculate expertise weights
        weights = self._calculate_weights(context)

        # Build collective decision
        decision = self._build_collective_decision(analyses, weights)
        self._decision_history.append(decision)

        # Generate comprehensive guidance
        guidance = self._generate_swarm_guidance(decision, analyses)

        return f"\n{guidance}"

    def post_act(self, action_attempt: str) -> str:
        """Update swarm intelligence based on action taken."""
        if not self._decision_history:
            return ""

        # Simple performance assessment
        last_decision = self._decision_history[-1]

        # Assess if action followed collective recommendation
        action_alignment = 0.7  # Default moderate alignment
        if last_decision.chosen_strategy.lower() in action_attempt.lower():
            action_alignment = 0.8

        # Update sub-agent performance
        for agent_type, agent in self._sub_agents.items():
            if agent_type in last_decision.supporting_agents:
                agent.update_performance(action_alignment)
            else:
                agent.update_performance(1.0 - action_alignment)

        return ""

    def get_state(self) -> Dict[str, Any]:
        """Return a JSON-safe snapshot that can reproduce future behavior."""
        return {
            'consensus_threshold': self._consensus_threshold,
            'max_iterations': self._max_iterations,
            'sub_agents': {
                name: {
                    'expertise_weight': agent._expertise_weight,
                    'performance_history': [
                        float(score) for score in agent._performance_history
                    ],
                }
                for name, agent in self._sub_agents.items()
            },
            'last_analyses': {
                name: dataclasses.asdict(analysis)
                for name, analysis in self._last_analyses.items()
            },
            'decision_history': [
                dataclasses.asdict(decision)
                for decision in self._decision_history
            ],
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`get_state`."""
        if not isinstance(state, dict):
            raise TypeError('Swarm intelligence state must be a mapping.')
        consensus_threshold = float(
            state.get('consensus_threshold', self._consensus_threshold)
        )
        max_iterations = int(state.get('max_iterations', self._max_iterations))
        if not 0.0 <= consensus_threshold <= 1.0:
            raise ValueError('consensus_threshold must be between 0 and 1.')
        if max_iterations < 1:
            raise ValueError('max_iterations must be at least 1.')
        self._consensus_threshold = consensus_threshold
        self._max_iterations = max_iterations
        for name, data in state.get('sub_agents', {}).items():
            if name in self._sub_agents:
                self._sub_agents[name]._expertise_weight = float(
                    data.get('expertise_weight', 1.0)
                )
                self._sub_agents[name]._performance_history = [
                    float(value)
                    for value in data.get('performance_history', [])
                ]
        self._last_analyses = {
            str(name): SubAgentAnalysis(**data)
            for name, data in state.get('last_analyses', {}).items()
        }
        self._decision_history = [
            CollectiveDecision(**data)
            for data in state.get('decision_history', [])
        ]

    def update(self) -> None:
        """Update swarm intelligence state."""
        # Decay old performance history
        for agent in self._sub_agents.values():
            if len(agent._performance_history) > 20:
                agent._performance_history = agent._performance_history[-15:]

    def get_action_attempt(
        self,
        context: Any,  # ComponentContextMapping
        action_spec: entity_lib.ActionSpec,
    ) -> str:
        """Generate collective intelligence-based negotiation action."""
        situation_context = action_spec.call_to_action
        
        # Collect analyses from all sub-agents
        analyses = self._collect_analyses(situation_context)
        
        # Calculate expertise weights for current context
        weights = self._calculate_weights(situation_context)
        
        # Build collective decision
        collective_decision = self._build_collective_decision(analyses, weights)
        
        # Store decision for post_act processing
        self._decision_history.append(collective_decision)
        
        # Generate action based on collective wisdom
        prompt = f"""Based on collective intelligence from multiple specialized agents, generate a negotiation action:

Situation: {situation_context}

Collective Decision: {collective_decision.chosen_strategy}
Confidence Level: {collective_decision.confidence:.2f}
Supporting Expertise: {', '.join(collective_decision.supporting_agents)}

Expert Analysis Summary:"""

        # Add key insights from each sub-agent
        for agent_type, analysis in analyses.items():
            weight = weights.get(agent_type, 0)
            if weight > 0.1:  # Only include agents with significant weight
                prompt += f"\n- {agent_type.replace('_', ' ').title()}: {analysis.recommendations[0] if analysis.recommendations else analysis.analysis}"
        
        prompt += f"""

Compromise Considerations: {'; '.join(collective_decision.compromise_elements) if collective_decision.compromise_elements else 'None needed'}

Generate a negotiation action that:
1. Implements the collective strategy: {collective_decision.chosen_strategy}
2. Balances insights from {len(analyses)} specialized perspectives
3. Addresses any compromise elements from dissenting experts
4. Shows confidence level of {collective_decision.confidence:.1f}
5. Demonstrates sophisticated multi-dimensional thinking

Action:"""

        response = self._model.sample_text(prompt)
        
        # Clean up response
        action = response.strip()
        if action.lower().startswith('action:'):
            action = action[7:].strip()
        
        # Add collective confidence indicator if high confidence
        if collective_decision.confidence > 0.8:
            action = f"Based on our comprehensive analysis, {action.lower()}"
        elif collective_decision.confidence < 0.4:
            action = f"While there are multiple perspectives to consider, {action.lower()}"
        
        return action

    def observe(self, observation: str) -> None:
        """Process observations for swarm intelligence."""
        # Could be used to update sub-agent knowledge
        pass
