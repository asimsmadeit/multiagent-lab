# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collective intelligence module for negotiation game master."""

import dataclasses
from typing import Any, Dict, List, Optional, Set, Tuple

from negotiation.game_master.components import gm_modules


@dataclasses.dataclass
class Coalition:
  """Represents a coalition of negotiating parties."""
  members: Set[str]
  formation_round: int
  shared_goals: List[str]
  coordination_level: float  # 0-1
  power_distribution: Dict[str, float]  # Member -> influence
  stability: float  # 0-1
  active: bool = True


@dataclasses.dataclass
class InformationFlow:
  """Tracks information sharing between parties."""
  sender: str
  recipients: Set[str]
  information_type: str
  sharing_mechanism: str  # 'broadcast', 'targeted', 'coalition'
  round_shared: int
  verification_level: float  # 0-1


@dataclasses.dataclass
class CollectiveDecision:
  """Represents a decision made by multiple parties."""
  participants: Set[str]
  decision_type: str  # 'consensus', 'majority', 'delegation'
  proposal: str
  support_level: Dict[str, float]  # Participant -> support (0-1)
  implementation_commitment: Dict[str, float]  # Participant -> commitment
  round_decided: int


class CollectiveIntelligenceGM(gm_modules.NegotiationGMModule):
  """GM module for managing collective decision-making and coordination."""

  def __init__(
      self,
      name: str = 'collective_intelligence',
      priority: int = 70,
      config: Optional[Dict[str, Any]] = None,
  ):
    """Initialize collective intelligence module."""
    super().__init__(name, priority, config)

    # Coalition tracking
    self._coalitions: List[Coalition] = []
    self._coalition_history: List[Dict[str, Any]] = []

    # Information flow tracking
    self._information_flows: List[InformationFlow] = []
    self._information_networks: Dict[str, Set[str]] = {}

    # Collective decision tracking
    self._collective_decisions: List[CollectiveDecision] = []
    self._consensus_tracking: Dict[str, Dict[str, float]] = {}

    # Coordination patterns
    self._coordination_patterns: Dict[str, List[str]] = {}
    self._emergence_indicators: List[Dict[str, Any]] = []

    # Configuration
    self._track_coalitions = self._config.get('track_coalitions', True)
    self._enable_information_routing = self._config.get('enable_information_routing', True)
    self._consensus_threshold = self._config.get('consensus_threshold', 0.7)
    self._coalition_stability_threshold = self._config.get('coalition_stability_threshold', 0.6)

  def get_supported_agent_modules(self) -> Set[str]:
    """Return agent modules this supports."""
    return {'swarm_intelligence'}

  def detect_coalition_formation(
      self,
      participants: List[str],
      recent_actions: List[Tuple[str, str]],
      context: gm_modules.ModuleContext,
  ) -> Optional[Coalition]:
    """Detect potential coalition formation from interaction patterns."""
    if not self._track_coalitions or len(participants) < 2:
      return None

    # Analyze recent coordination patterns
    coordination_pairs = set()
    shared_positions = {}

    for actor, action in recent_actions[-10:]:  # Last 10 actions
      # Look for coordination keywords
      if any(word in action.lower() for word in ['together', 'jointly', 'coordinate', 'alliance']):
        # Extract mentioned participants
        mentioned = [p for p in participants if p in action and p != actor]
        for mentioned_party in mentioned:
          coordination_pairs.add(tuple(sorted([actor, mentioned_party])))

      # Track similar positions
      if any(word in action.lower() for word in ['agree', 'support', 'same', 'similar']):
        position_key = f"{actor}_position"
        if position_key not in shared_positions:
          shared_positions[position_key] = []
        shared_positions[position_key].append(action)

    # Identify potential coalition members
    potential_members = set()
    for pair in coordination_pairs:
      potential_members.update(pair)

    if len(potential_members) >= 2:
      existing = next(
          (
              coalition for coalition in self._coalitions
              if coalition.active and coalition.members == potential_members
          ),
          None,
      )
      if existing is not None:
        return existing
      # Create coalition
      coalition = Coalition(
          members=potential_members,
          formation_round=context.current_round,
          shared_goals=[],  # Would extract from action analysis
          coordination_level=min(1.0, len(coordination_pairs) * 0.3),
          power_distribution={member: 1.0/len(potential_members) 
                            for member in potential_members},
          stability=0.5,  # Initial stability
      )
      self._coalitions.append(coalition)
      return coalition

    return None

  def track_information_sharing(
      self,
      sender: str,
      action: str,
      context: gm_modules.ModuleContext,
  ) -> None:
    """Track information sharing patterns."""
    if not self._enable_information_routing:
      return

    # Determine recipients
    recipients = set()
    sharing_mechanism = 'broadcast'  # Default

    # Check for targeted sharing
    for participant in context.participants:
      if participant != sender and participant in action:
        recipients.add(participant)
        sharing_mechanism = 'targeted'

    # Check for coalition sharing
    sender_coalitions = [c for c in self._coalitions if sender in c.members and c.active]
    if sender_coalitions and any(word in action.lower() for word in ['our', 'we', 'us']):
      for coalition in sender_coalitions:
        recipients.update(coalition.members - {sender})
        sharing_mechanism = 'coalition'

    # Default to all participants if no specific targeting
    if not recipients:
      recipients = set(context.participants) - {sender}

    # Determine information type
    information_type = 'general'
    if any(word in action.lower() for word in ['preference', 'want', 'need']):
      information_type = 'preference'
    elif any(word in action.lower() for word in ['constraint', 'cannot', 'limit']):
      information_type = 'constraint'
    elif any(word in action.lower() for word in ['strategy', 'plan', 'approach']):
      information_type = 'strategic'

    # Record information flow
    flow = InformationFlow(
        sender=sender,
        recipients=recipients,
        information_type=information_type,
        sharing_mechanism=sharing_mechanism,
        round_shared=context.current_round,
        verification_level=0.8,  # Simplified verification
    )
    self._information_flows.append(flow)

    # Update information networks
    if sender not in self._information_networks:
      self._information_networks[sender] = set()
    self._information_networks[sender].update(recipients)

  def assess_collective_decision_potential(
      self,
      participants: List[str],
      proposal: str,
      context: gm_modules.ModuleContext,
  ) -> Optional[CollectiveDecision]:
    """Assess potential for collective decision-making."""
    if not participants:
      return None
    # Look for consensus indicators
    if not any(word in proposal.lower() for word in ['we', 'all', 'together', 'consensus']):
      return None

    # Determine decision type
    decision_type = 'consensus'
    if 'majority' in proposal.lower():
      decision_type = 'majority'
    elif 'delegate' in proposal.lower() or 'authorize' in proposal.lower():
      decision_type = 'delegation'

    # Initialize support levels (simplified assessment)
    support_level = {}
    commitment_level = {}

    for participant in participants:
      # Base support on recent agreement patterns
      recent_actions = context.shared_data.get('recent_actions', [])
      recent_agreements = sum(
          1 for actor, action in recent_actions[-5:]
          if actor == participant and 'agree' in action.lower()
      )
      support_level[participant] = min(1.0, 0.5 + recent_agreements * 0.1)
      commitment_level[participant] = support_level[participant] * 0.8

    collective_decision = CollectiveDecision(
        participants=set(participants),
        decision_type=decision_type,
        proposal=proposal,
        support_level=support_level,
        implementation_commitment=commitment_level,
        round_decided=context.current_round,
    )

    self._collective_decisions.append(collective_decision)
    return collective_decision

  def update_coalition_stability(self, context: gm_modules.ModuleContext) -> None:
    """Update stability of active coalitions."""
    for coalition in self._coalitions:
      if not coalition.active:
        continue

      # Factors affecting stability
      age = context.current_round - coalition.formation_round
      age_factor = max(0.1, 1.0 - age * 0.05)  # Stability decreases over time

      # Recent coordination actions
      recent_coordination = 0
      for flow in self._information_flows[-10:]:
        if (flow.sender in coalition.members and 
            len(flow.recipients.intersection(coalition.members)) > 0):
          recent_coordination += 1

      coordination_factor = min(1.0, recent_coordination * 0.1)

      # Goal alignment (simplified)
      alignment_factor = 0.7  # Placeholder - would analyze actual goal alignment

      # Update stability
      coalition.stability = (age_factor * 0.3 + 
                           coordination_factor * 0.4 + 
                           alignment_factor * 0.3)

      # Deactivate unstable coalitions
      if coalition.stability < self._coalition_stability_threshold:
        coalition.active = False
        self._coalition_history.append({
            'coalition': coalition.members,
            'dissolved_round': context.current_round,
            'reason': 'low_stability',
            'final_stability': coalition.stability,
        })

  def detect_emergent_behavior(
      self,
      context: gm_modules.ModuleContext,
  ) -> List[str]:
    """Detect emergent collective behaviors."""
    emergent_patterns = []

    # Network density emergence
    if len(self._information_networks) >= 3:
      total_connections = sum(len(connections) for connections in self._information_networks.values())
      possible_connections = len(context.participants) * (len(context.participants) - 1)
      network_density = total_connections / max(1, possible_connections)

      if network_density > 0.8:
        emergent_patterns.append("high_connectivity")
      elif network_density < 0.2:
        emergent_patterns.append("fragmentation")

    # Information cascade detection
    recent_flows = self._information_flows[-15:]
    if len(recent_flows) >= 10:
      # Check for rapid information spreading
      flow_density = len(recent_flows) / max(1, context.current_round)
      if flow_density > 2.0:
        emergent_patterns.append("information_cascade")

    # Consensus convergence
    if self._collective_decisions:
      recent_decisions = [d for d in self._collective_decisions 
                         if d.round_decided > context.current_round - 5]
      if len(recent_decisions) >= 2:
        supported_decisions = [
            decision for decision in recent_decisions
            if decision.support_level
        ]
        avg_support = (
            sum(
                sum(decision.support_level.values()) /
                len(decision.support_level)
                for decision in supported_decisions
            ) / len(supported_decisions)
            if supported_decisions else 0.0
        )
        if avg_support > 0.8:
          emergent_patterns.append("consensus_convergence")

    # Record emergent behaviors
    if emergent_patterns:
      self._emergence_indicators.append({
          'patterns': emergent_patterns,
          'round': context.current_round,
          'context': 'negotiation',
      })

    return emergent_patterns

  def validate_action(
      self,
      actor: str,
      action: str,
      context: gm_modules.ModuleContext,
  ) -> Tuple[bool, Optional[str]]:
    """Validate action for collective intelligence compliance."""
    # Check for coalition constraint violations
    actor_coalitions = [c for c in self._coalitions if actor in c.members and c.active]
    
    for coalition in actor_coalitions:
      # Check if action contradicts coalition goals
      if any(word in action.lower() for word in ['betray', 'abandon', 'defect']):
        return False, f"Action may violate coalition commitment"

    # Check for information sharing appropriateness
    if 'share' in action.lower() or 'tell' in action.lower():
      # Ensure appropriate information sharing within coalitions
      for coalition in actor_coalitions:
        if len(coalition.members) > 1:
          other_members = coalition.members - {actor}
          if not any(member in action for member in other_members):
            # Actor might be withholding from coalition members
            pass  # Allow but note - could flag as coordination issue

    return True, None

  def update_state(
      self,
      event: str,
      actor: str,
      context: gm_modules.ModuleContext,
  ) -> None:
    """Update collective intelligence state based on events."""
    # Track information sharing
    self.track_information_sharing(actor, event, context)

    # Detect coalition formation
    recent_actions = [(actor, event)]  # Simplified - would track more history
    self.detect_coalition_formation(context.participants, recent_actions, context)

    # Update coalition stability
    if context.current_round % 3 == 0:  # Every 3 rounds
      self.update_coalition_stability(context)

    # Assess collective decision potential
    if any(word in event.lower() for word in ['propose', 'suggest', 'all agree']):
      self.assess_collective_decision_potential(context.participants, event, context)

    # Detect emergent behaviors
    if context.current_round % 5 == 0:  # Every 5 rounds
      self.detect_emergent_behavior(context)

  def get_observation_context(
      self,
      observer: str,
      context: gm_modules.ModuleContext,
  ) -> str:
    """Get collective intelligence context for observations."""
    observation = "\nCOLLECTIVE INTELLIGENCE:\n"

    # Active coalitions
    observer_coalitions = [c for c in self._coalitions if observer in c.members and c.active]
    if observer_coalitions:
      observation += f"\nYour active coalitions: {len(observer_coalitions)}\n"
      for coalition in observer_coalitions:
        members = ', '.join(coalition.members - {observer})
        observation += f"- Coalition with {members} (stability: {coalition.stability:.0%})\n"

    # Information network
    if observer in self._information_networks:
      connections = len(self._information_networks[observer])
      observation += f"\nYour information network: {connections} connections\n"

    # Recent collective decisions
    recent_decisions = [d for d in self._collective_decisions 
                       if observer in d.participants and 
                       d.round_decided > context.current_round - 5]
    if recent_decisions:
      observation += f"\nRecent collective decisions: {len(recent_decisions)}\n"
      for decision in recent_decisions[-2:]:  # Show last 2
        support = decision.support_level.get(observer, 0)
        observation += f"- {decision.decision_type}: {support:.0%} support\n"

    # Emergent patterns
    recent_patterns = [e for e in self._emergence_indicators 
                      if e['round'] > context.current_round - 5]
    if recent_patterns:
      all_patterns = set()
      for pattern_dict in recent_patterns:
        all_patterns.update(pattern_dict['patterns'])
      
      if all_patterns:
        observation += f"\nEmerging patterns: {', '.join(all_patterns)}\n"

    # Strategic insights
    if observer_coalitions:
      total_coalition_power = sum(c.coordination_level for c in observer_coalitions)
      if total_coalition_power > 0.7:
        observation += "\n💪 Strong coalition position - coordinate for maximum impact\n"
      elif total_coalition_power < 0.3:
        observation += "\n⚠️ Weak coalition coordination - improve alignment\n"

    # Network position analysis
    if observer in self._information_networks:
      connections = self._information_networks[observer]
      all_participants = set(context.participants) - {observer}
      connectivity = len(connections) / max(1, len(all_participants))
      
      if connectivity > 0.8:
        observation += "\n🌐 High network centrality - information hub position\n"
      elif connectivity < 0.3:
        observation += "\n🔗 Low connectivity - consider expanding network\n"

    return observation

  def get_module_report(self) -> str:
    """Get collective intelligence report."""
    report = "COLLECTIVE INTELLIGENCE REPORT:\n\n"

    # Coalition analysis
    active_coalitions = [c for c in self._coalitions if c.active]
    if active_coalitions:
      report += f"Active Coalitions: {len(active_coalitions)}\n"
      for coalition in active_coalitions:
        report += f"- Members: {', '.join(coalition.members)} "
        report += f"(stability: {coalition.stability:.0%}, "
        report += f"coordination: {coalition.coordination_level:.0%})\n"

    # Coalition history
    if self._coalition_history:
      report += f"\nCoalition Dynamics:\n"
      report += f"- Dissolved coalitions: {len(self._coalition_history)}\n"
      instability_reasons = {}
      for event in self._coalition_history:
        reason = event.get('reason', 'unknown')
        instability_reasons[reason] = instability_reasons.get(reason, 0) + 1
      
      for reason, count in instability_reasons.items():
        report += f"  - {reason}: {count}\n"

    # Information flow analysis
    if self._information_flows:
      report += f"\nInformation Flow: {len(self._information_flows)} exchanges\n"
      
      flow_types = {}
      for flow in self._information_flows:
        flow_types[flow.sharing_mechanism] = flow_types.get(flow.sharing_mechanism, 0) + 1
      
      for mechanism, count in flow_types.items():
        report += f"- {mechanism}: {count}\n"

      # Network density
      if self._information_networks:
        total_connections = sum(
            len(connections)
            for connections in self._information_networks.values()
        )
        nodes = set(self._information_networks)
        for connections in self._information_networks.values():
          nodes.update(connections)
        participants = len(nodes)
        if participants > 1:
          possible_connections = participants * (participants - 1)
          density = total_connections / possible_connections
          report += f"\nNetwork density: {density:.0%}\n"

    # Collective decision analysis
    if self._collective_decisions:
      report += f"\nCollective Decisions: {len(self._collective_decisions)}\n"
      
      decision_types = {}
      for decision in self._collective_decisions:
        decision_types[decision.decision_type] = decision_types.get(decision.decision_type, 0) + 1
      
      for dtype, count in decision_types.items():
        report += f"- {dtype}: {count}\n"

      # Average support levels
      if self._collective_decisions:
        total_support = 0
        total_participants = 0
        for decision in self._collective_decisions:
          total_support += sum(decision.support_level.values())
          total_participants += len(decision.support_level)
        
        if total_participants > 0:
          avg_support = total_support / total_participants
          report += f"\nAverage decision support: {avg_support:.0%}\n"

    # Emergent behavior summary
    if self._emergence_indicators:
      report += f"\nEmergent Behaviors: {len(self._emergence_indicators)} instances\n"
      
      all_patterns = set()
      for indicator in self._emergence_indicators:
        all_patterns.update(indicator['patterns'])
      
      for pattern in all_patterns:
        count = sum(1 for i in self._emergence_indicators if pattern in i['patterns'])
        report += f"- {pattern}: {count} occurrences\n"

    # Intelligence insights
    if active_coalitions and self._information_flows:
      coalition_effectiveness = sum(c.coordination_level * c.stability for c in active_coalitions) / len(active_coalitions)
      
      if coalition_effectiveness > 0.7:
        report += "\n💡 High collective intelligence - effective coordination\n"
      elif coalition_effectiveness < 0.3:
        report += "\n⚠️ Low collective coordination - fragmented decision-making\n"

    return report

  def get_state(self) -> Dict[str, Any]:
    """Return complete collective-intelligence state."""
    return {
        'base': self._get_base_state(),
        'coalitions': [
            {
                **dataclasses.asdict(coalition),
                'members': sorted(coalition.members),
            }
            for coalition in self._coalitions
        ],
        'coalition_history': [
            {
                **event,
                'coalition': sorted(event.get('coalition', [])),
            }
            for event in self._coalition_history
        ],
        'information_flows': [
            {
                **dataclasses.asdict(flow),
                'recipients': sorted(flow.recipients),
            }
            for flow in self._information_flows
        ],
        'information_networks': {
            participant: sorted(connections)
            for participant, connections in self._information_networks.items()
        },
        'collective_decisions': [
            {
                **dataclasses.asdict(decision),
                'participants': sorted(decision.participants),
            }
            for decision in self._collective_decisions
        ],
        'consensus_tracking': {
            topic: dict(values)
            for topic, values in self._consensus_tracking.items()
        },
        'coordination_patterns': {
            participant: list(patterns)
            for participant, patterns in self._coordination_patterns.items()
        },
        'emergence_indicators': [
            dict(item) for item in self._emergence_indicators
        ],
    }

  def set_state(self, state: Dict[str, Any]) -> None:
    """Restore complete collective-intelligence state."""
    if not isinstance(state, dict):
      raise TypeError('Collective GM state must be a mapping.')
    self._set_base_state(state)
    self._coalitions = [
        Coalition(**{**item, 'members': set(item.get('members', []))})
        for item in state.get('coalitions', [])
    ]
    self._coalition_history = [
        {**event, 'coalition': set(event.get('coalition', []))}
        for event in state.get('coalition_history', [])
    ]
    self._information_flows = [
        InformationFlow(
            **{**item, 'recipients': set(item.get('recipients', []))}
        )
        for item in state.get('information_flows', [])
    ]
    self._information_networks = {
        str(participant): set(connections)
        for participant, connections in state.get(
            'information_networks', {}
        ).items()
    }
    self._collective_decisions = [
        CollectiveDecision(
            **{**item, 'participants': set(item.get('participants', []))}
        )
        for item in state.get('collective_decisions', [])
    ]
    self._consensus_tracking = {
        str(topic): {str(name): float(value) for name, value in values.items()}
        for topic, values in state.get('consensus_tracking', {}).items()
    }
    self._coordination_patterns = {
        str(participant): [str(pattern) for pattern in patterns]
        for participant, patterns in state.get(
            'coordination_patterns', {}
        ).items()
    }
    self._emergence_indicators = [
        dict(item) for item in state.get('emergence_indicators', [])
    ]


# Register the module
gm_modules.NegotiationGMModuleRegistry.register(
    'collective_intelligence',
    CollectiveIntelligenceGM
)
