#!/usr/bin/env python3
"""
Basal Ganglia - Reinforcement Learning and Action Selection
Implements detailed basal ganglia structures for motor learning,
habit formation, reinforcement learning, and action selection
"""

import sys
import time
import math
import random
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, deque
import threading
import numpy as np

class BasalGanglia:
    """Basal ganglia for reinforcement learning and action selection"""

    def __init__(self):
        # Core basal ganglia nuclei
        self.striatum = Striatum()
        self.globus_pallidus = GlobusPallidus()
        self.substantia_nigra = SubstantiaNigra()
        self.subthalamic_nucleus = SubthalamicNucleus()

        # Action selection and motor systems
        self.action_selection = ActionSelection()
        self.habit_formation = HabitFormation()
        self.reinforcement_learning = ReinforcementLearning()
        self.motor_learning = MotorLearning()

        # Dopamine systems
        self.dopamine_system = DopamineSystem()
        self.reward_prediction = RewardPrediction()

        # Learning parameters
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2

        # Action repertoire
        self.action_repertoire = self._initialize_action_repertoire()
        self.current_action = None
        self.action_history = []

        # Performance tracking
        self.performance_metrics = {
            'actions_selected': 0,
            'successful_actions': 0,
            'reward_accumulated': 0.0,
            'learning_efficiency': 0.8
        }

        print("🧠 Basal Ganglia initialized - Reinforcement learning active")

    def process_action_selection(self, sensory_input: Dict[str, Any],
                               emotional_state: Dict[str, Any],
                               goal_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process action selection through basal ganglia circuits"""

        # Stage 1: Evaluate current state and goals
        state_evaluation = self._evaluate_current_state(sensory_input, emotional_state, goal_context)

        # Stage 2: Generate action candidates
        action_candidates = self._generate_action_candidates(state_evaluation)

        # Stage 3: Evaluate actions through reinforcement learning
        action_evaluation = self.reinforcement_learning.evaluate_actions(action_candidates, state_evaluation)

        # Stage 4: Select action through basal ganglia circuits
        selected_action = self.action_selection.select_action(action_evaluation, self.dopamine_system)

        # Stage 5: Execute action (simulation)
        execution_result = self._execute_selected_action(selected_action)

        # Stage 6: Learn from outcome
        learning_result = self._learn_from_outcome(selected_action, execution_result, state_evaluation)

        # Stage 7: Update dopamine system
        dopamine_update = self.dopamine_system.update_dopamine(execution_result, state_evaluation)

        # Stage 8: Form/update habits
        habit_update = self.habit_formation.update_habits(selected_action, execution_result)

        # Update performance metrics
        self._update_performance_metrics(selected_action, execution_result)

        return {
            'selected_action': selected_action,
            'execution_result': execution_result,
            'learning_outcome': learning_result,
            'dopamine_response': dopamine_update,
            'habit_strength': habit_update,
            'state_evaluation': state_evaluation,
            'performance_metrics': self.performance_metrics.copy()
        }

    def _evaluate_current_state(self, sensory_input: Dict[str, Any],
                              emotional_state: Dict[str, Any],
                              goal_context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate current state for action selection"""

        # Extract state features
        sensory_features = self._extract_sensory_features(sensory_input)
        emotional_features = self._extract_emotional_features(emotional_state)
        goal_features = self._extract_goal_features(goal_context)

        # Calculate state value
        state_value = self._calculate_state_value(sensory_features, emotional_features, goal_features)

        # Assess action urgency
        action_urgency = self._assess_action_urgency(goal_features, emotional_features)

        return {
            'sensory_features': sensory_features,
            'emotional_features': emotional_features,
            'goal_features': goal_features,
            'state_value': state_value,
            'action_urgency': action_urgency,
            'evaluation_timestamp': time.time()
        }

    def _generate_action_candidates(self, state_evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate candidate actions based on current state"""

        candidates = []

        # Generate actions from repertoire
        for action in self.action_repertoire.values():
            # Evaluate action relevance
            relevance_score = self._calculate_action_relevance(action, state_evaluation)

            if relevance_score > 0.3:  # Threshold for consideration
                candidate = {
                    'action': action,
                    'relevance_score': relevance_score,
                    'expected_reward': self._predict_action_reward(action, state_evaluation),
                    'execution_confidence': self._calculate_execution_confidence(action),
                    'habit_strength': self.habit_formation.get_habit_strength(action['id'])
                }
                candidates.append(candidate)

        # Sort by combined score
        candidates.sort(key=lambda x: x['relevance_score'] + x['expected_reward'] + x['habit_strength'], reverse=True)

        return candidates[:10]  # Return top 10 candidates

    def _execute_selected_action(self, selected_action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the selected action (simulated)"""

        action = selected_action['action']

        # Simulate execution
        execution_time = random.uniform(0.1, 2.0)
        time.sleep(execution_time * 0.1)  # Brief delay for realism

        # Generate execution outcome
        success_probability = selected_action.get('execution_confidence', 0.8)
        success = random.random() < success_probability

        reward_value = random.uniform(-1, 1) if success else random.uniform(-2, 0)

        execution_result = {
            'action_id': action['id'],
            'success': success,
            'execution_time': execution_time,
            'reward_value': reward_value,
            'outcome_description': self._generate_outcome_description(action, success),
            'execution_timestamp': time.time()
        }

        # Store in action history
        self.action_history.append({
            'action': selected_action,
            'result': execution_result,
            'timestamp': time.time()
        })

        return execution_result

    def _learn_from_outcome(self, selected_action: Dict[str, Any],
                          execution_result: Dict[str, Any],
                          state_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from action execution outcome"""

        # Update reinforcement learning
        rl_update = self.reinforcement_learning.update_learning(
            selected_action, execution_result, state_evaluation
        )

        # Update motor learning
        motor_update = self.motor_learning.update_motor_learning(
            selected_action['action'], execution_result
        )

        # Update dopamine prediction
        prediction_update = self.reward_prediction.update_prediction(
            selected_action, execution_result, state_evaluation
        )

        return {
            'reinforcement_learning': rl_update,
            'motor_learning': motor_update,
            'reward_prediction': prediction_update,
            'learning_efficiency': self.learning_rate
        }

    def _initialize_action_repertoire(self) -> Dict[str, Dict[str, Any]]:
        """Initialize the action repertoire"""

        actions = {
            'approach_positive': {
                'id': 'approach_positive',
                'type': 'approach',
                'description': 'Approach positive stimuli',
                'energy_cost': 0.3,
                'success_rate': 0.8,
                'reward_range': (0.5, 1.0)
            },
            'avoid_negative': {
                'id': 'avoid_negative',
                'type': 'avoidance',
                'description': 'Avoid negative stimuli',
                'energy_cost': 0.4,
                'success_rate': 0.9,
                'reward_range': (0.3, 0.8)
            },
            'explore_unknown': {
                'id': 'explore_unknown',
                'type': 'exploration',
                'description': 'Explore unknown environments',
                'energy_cost': 0.6,
                'success_rate': 0.6,
                'reward_range': (-0.2, 0.7)
            },
            'social_engage': {
                'id': 'social_engage',
                'type': 'social',
                'description': 'Engage in social interaction',
                'energy_cost': 0.5,
                'success_rate': 0.7,
                'reward_range': (0.2, 0.9)
            },
            'rest_recover': {
                'id': 'rest_recover',
                'type': 'rest',
                'description': 'Rest and recover energy',
                'energy_cost': -0.8,  # Energy gain
                'success_rate': 0.95,
                'reward_range': (0.1, 0.4)
            },
            'problem_solve': {
                'id': 'problem_solve',
                'type': 'cognitive',
                'description': 'Engage in problem solving',
                'energy_cost': 0.8,
                'success_rate': 0.5,
                'reward_range': (-0.5, 1.2)
            },
            'learn_new': {
                'id': 'learn_new',
                'type': 'learning',
                'description': 'Learn new information or skill',
                'energy_cost': 0.7,
                'success_rate': 0.4,
                'reward_range': (-0.3, 0.8)
            },
            'communicate': {
                'id': 'communicate',
                'type': 'communication',
                'description': 'Communicate with others',
                'energy_cost': 0.4,
                'success_rate': 0.75,
                'reward_range': (0.1, 0.7)
            }
        }

        return actions

    def _extract_sensory_features(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sensory features for state evaluation"""
        features = {}

        # Calculate overall sensory intensity
        if sensory_input:
            intensities = [data.get('intensity', 0) for data in sensory_input.values()]
            features['overall_intensity'] = sum(intensities) / len(intensities)
            features['max_intensity'] = max(intensities)
            features['sensory_diversity'] = len(sensory_input)
        else:
            features['overall_intensity'] = 0.0
            features['max_intensity'] = 0.0
            features['sensory_diversity'] = 0

        # Detect salient features
        features['salient_features'] = []
        for modality, data in sensory_input.items():
            if data.get('salience', 0) > 0.7:
                features['salient_features'].append({
                    'modality': modality,
                    'intensity': data.get('intensity', 0),
                    'salience': data.get('salience', 0)
                })

        return features

    def _extract_emotional_features(self, emotional_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract emotional features"""
        return {
            'primary_emotion': emotional_state.get('primary_emotion', 'neutral'),
            'intensity': emotional_state.get('intensity', 0),
            'valence': emotional_state.get('valence', 0),
            'arousal': emotional_state.get('arousal', 0.5),
            'emotional_stability': 1.0 - abs(emotional_state.get('intensity', 0))
        }

    def _extract_goal_features(self, goal_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract goal-related features"""
        goals = goal_context.get('goals', [])
        return {
            'active_goals': len(goals),
            'goal_urgency': max([g.get('urgency', 0) for g in goals]) if goals else 0,
            'goal_alignment': goal_context.get('goal_alignment', 0.5),
            'progress_toward_goals': goal_context.get('progress', 0)
        }

    def _calculate_state_value(self, sensory: Dict[str, Any],
                             emotional: Dict[str, Any],
                             goal: Dict[str, Any]) -> float:
        """Calculate overall state value"""
        sensory_value = sensory['overall_intensity'] * 0.3
        emotional_value = (emotional['valence'] + 1) / 2 * 0.4  # Convert to 0-1 scale
        goal_value = goal['goal_alignment'] * 0.3

        return sensory_value + emotional_value + goal_value

    def _assess_action_urgency(self, goal_features: Dict[str, Any],
                             emotional_features: Dict[str, Any]) -> float:
        """Assess urgency for action selection"""
        goal_urgency = goal_features.get('goal_urgency', 0)
        emotional_intensity = emotional_features.get('intensity', 0)

        return (goal_urgency + emotional_intensity) / 2

    def _calculate_action_relevance(self, action: Dict[str, Any],
                                  state_evaluation: Dict[str, Any]) -> float:
        """Calculate how relevant an action is to current state"""

        relevance = 0.0

        # Check sensory relevance
        sensory_features = state_evaluation['sensory_features']
        if action['type'] == 'approach' and sensory_features['overall_intensity'] > 0.6:
            relevance += 0.4
        elif action['type'] == 'avoidance' and sensory_features['max_intensity'] > 0.8:
            relevance += 0.4

        # Check emotional relevance
        emotional_features = state_evaluation['emotional_features']
        if action['type'] == 'social' and emotional_features['valence'] > 0:
            relevance += 0.3
        elif action['type'] == 'rest' and emotional_features['arousal'] > 0.8:
            relevance += 0.3

        # Check goal relevance
        goal_features = state_evaluation['goal_features']
        if action['type'] == 'cognitive' and goal_features['active_goals'] > 0:
            relevance += 0.3

        # Add some randomness for exploration
        relevance += random.uniform(0, 0.2)

        return min(1.0, relevance)

    def _predict_action_reward(self, action: Dict[str, Any],
                             state_evaluation: Dict[str, Any]) -> float:
        """Predict reward for action in current state"""
        base_reward = sum(action['reward_range']) / 2

        # Adjust based on state
        state_value = state_evaluation['state_value']
        reward_modifier = (state_value - 0.5) * 0.4  # +/- 0.2

        # Adjust based on habit strength
        habit_bonus = self.habit_formation.get_habit_strength(action['id']) * 0.1

        return base_reward + reward_modifier + habit_bonus

    def _calculate_execution_confidence(self, action: Dict[str, Any]) -> float:
        """Calculate confidence in executing action"""
        base_confidence = action['success_rate']

        # Adjust based on habit strength
        habit_confidence = self.habit_formation.get_habit_strength(action['id']) * 0.2

        return min(1.0, base_confidence + habit_confidence)

    def _generate_outcome_description(self, action: Dict[str, Any], success: bool) -> str:
        """Generate description of action outcome"""
        if success:
            descriptions = [
                f"Successfully {action['description'].lower()}",
                f"Action '{action['id']}' completed successfully",
                f"Goal achieved through {action['description'].lower()}",
                f"Positive outcome from {action['description'].lower()}"
            ]
        else:
            descriptions = [
                f"Failed to {action['description'].lower()}",
                f"Action '{action['id']}' encountered difficulties",
                f"Goal not achieved through {action['description'].lower()}",
                f"Challenges faced during {action['description'].lower()}"
            ]

        return random.choice(descriptions)

    def _update_performance_metrics(self, selected_action: Dict[str, Any],
                                  execution_result: Dict[str, Any]):
        """Update performance metrics"""
        self.performance_metrics['actions_selected'] += 1

        if execution_result['success']:
            self.performance_metrics['successful_actions'] += 1

        self.performance_metrics['reward_accumulated'] += execution_result['reward_value']

        # Calculate learning efficiency
        success_rate = self.performance_metrics['successful_actions'] / self.performance_metrics['actions_selected']
        self.performance_metrics['learning_efficiency'] = success_rate

    def get_status(self) -> Dict[str, Any]:
        """Get basal ganglia status"""
        return {
            'performance_metrics': self.performance_metrics,
            'action_repertoire_size': len(self.action_repertoire),
            'habit_count': len(self.habit_formation.habits),
            'dopamine_levels': self.dopamine_system.get_levels(),
            'reinforcement_learning': self.reinforcement_learning.get_status(),
            'current_action': self.current_action,
            'learning_parameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'exploration_rate': self.exploration_rate
            }
        }


class Striatum:
    """Striatum for action selection and reward processing"""

    def __init__(self):
        self.neurons = []
        self.dopamine_receptors = {}
        self.action_values = {}
        self.reward_sensitivity = 0.8

    def process_action_values(self, action_candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process action values for selection"""
        action_values = {}

        for candidate in action_candidates:
            action_id = candidate['action']['id']
            base_value = candidate['expected_reward']

            # Apply dopamine modulation
            dopamine_modulation = self.dopamine_receptors.get(action_id, 1.0)
            modulated_value = base_value * dopamine_modulation

            # Apply reward sensitivity
            final_value = modulated_value * self.reward_sensitivity

            action_values[action_id] = final_value

        return {
            'action_values': action_values,
            'dominant_action': max(action_values.items(), key=lambda x: x[1])[0] if action_values else None,
            'value_distribution': list(action_values.values())
        }

    def update_action_values(self, action_id: str, reward: float):
        """Update action values based on reward"""
        if action_id not in self.action_values:
            self.action_values[action_id] = 0.0

        # Simple value update
        learning_rate = 0.1
        self.action_values[action_id] += learning_rate * (reward - self.action_values[action_id])

        # Update dopamine receptors
        dopamine_change = reward * 0.2
        self.dopamine_receptors[action_id] = self.dopamine_receptors.get(action_id, 1.0) + dopamine_change


class GlobusPallidus:
    """Globus pallidus for action gating and selection"""

    def __init__(self):
        self.gate_status = 'open'
        self.inhibition_strength = 0.6
        self.gating_history = []

    def gate_actions(self, action_values: Dict[str, Any]) -> Dict[str, Any]:
        """Gate actions based on current state"""

        gated_actions = {}
        inhibited_actions = []

        for action_id, value in action_values.items():
            # Apply gating threshold
            if value < self.inhibition_strength:
                inhibited_actions.append(action_id)
                gated_actions[action_id] = 0.0
            else:
                gated_actions[action_id] = value

        gating_result = {
            'gated_actions': gated_actions,
            'inhibited_actions': inhibited_actions,
            'gate_status': self.gate_status,
            'inhibition_threshold': self.inhibition_strength
        }

        self.gating_history.append(gating_result)
        return gating_result

    def update_gating(self, reward_feedback: float):
        """Update gating parameters based on feedback"""
        if reward_feedback > 0:
            # Positive feedback - relax inhibition
            self.inhibition_strength = max(0.1, self.inhibition_strength - 0.05)
        else:
            # Negative feedback - strengthen inhibition
            self.inhibition_strength = min(0.9, self.inhibition_strength + 0.05)


class SubstantiaNigra:
    """Substantia nigra for dopamine production and reward signaling"""

    def __init__(self):
        self.dopamine_levels = 0.5
        self.reward_signals = []
        self.parkinsonian_risk = 0.1

    def generate_reward_signal(self, reward_value: float) -> Dict[str, Any]:
        """Generate dopamine reward signal"""

        # Calculate dopamine release
        dopamine_release = reward_value * 0.8

        # Apply Parkinson's risk factor
        dopamine_release *= (1 - self.parkinsonian_risk)

        # Update dopamine levels
        self.dopamine_levels = max(0.0, min(1.0, self.dopamine_levels + dopamine_release))

        signal = {
            'dopamine_release': dopamine_release,
            'reward_value': reward_value,
            'dopamine_levels': self.dopamine_levels,
            'signal_strength': abs(dopamine_release),
            'timestamp': time.time()
        }

        self.reward_signals.append(signal)
        return signal

    def update_dopamine_homeostasis(self):
        """Maintain dopamine homeostasis"""
        # Natural decay toward baseline
        baseline = 0.5
        decay_rate = 0.02

        if self.dopamine_levels > baseline:
            self.dopamine_levels -= decay_rate
        elif self.dopamine_levels < baseline:
            self.dopamine_levels += decay_rate

        self.dopamine_levels = max(0.0, min(1.0, self.dopamine_levels))


class SubthalamicNucleus:
    """Subthalamic nucleus for motor control and decision making"""

    def __init__(self):
        self.decision_threshold = 0.6
        self.impulse_control = 0.7
        self.decision_history = []

    def evaluate_decision(self, action_values: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate decision based on action values"""

        # Find best action
        if not action_values:
            return {'decision': 'no_action', 'confidence': 0.0}

        best_action = max(action_values.items(), key=lambda x: x[1])
        action_id, value = best_action

        # Apply decision threshold
        if value < self.decision_threshold:
            decision = 'inhibit_action'
            confidence = 0.0
        else:
            decision = action_id
            confidence = min(1.0, value / self.decision_threshold)

        decision_result = {
            'decision': decision,
            'action_value': value,
            'confidence': confidence,
            'threshold': self.decision_threshold,
            'timestamp': time.time()
        }

        self.decision_history.append(decision_result)
        return decision_result

    def control_impulse(self, decision: str, urgency: float) -> bool:
        """Control impulsive decisions"""
        if urgency > self.impulse_control:
            return False  # Allow impulsive action
        else:
            return True   # Inhibit impulsive action


class ActionSelection:
    """Action selection system using basal ganglia circuits"""

    def __init__(self):
        self.selection_criteria = ['value', 'habit', 'novelty', 'risk']
        self.selection_weights = {'value': 0.4, 'habit': 0.3, 'novelty': 0.2, 'risk': 0.1}

    def select_action(self, action_evaluation: Dict[str, Any],
                     dopamine_system: 'DopamineSystem') -> Dict[str, Any]:
        """Select action using basal ganglia circuits"""

        evaluated_actions = action_evaluation.get('evaluated_options', [])

        if not evaluated_actions:
            return {'action': {'id': 'no_action', 'description': 'No suitable action found'}}

        # Calculate selection scores
        selection_scores = []
        for action_data in evaluated_actions:
            score = self._calculate_selection_score(action_data)
            selection_scores.append((action_data, score))

        # Sort by score
        selection_scores.sort(key=lambda x: x[1], reverse=True)

        # Select best action
        selected_action_data, selection_score = selection_scores[0]

        # Apply dopamine modulation
        dopamine_modulation = dopamine_system.get_current_levels()
        final_score = selection_score * dopamine_modulation

        return {
            'action': selected_action_data['option'],
            'selection_score': selection_score,
            'final_score': final_score,
            'dopamine_modulation': dopamine_modulation,
            'selection_criteria': self.selection_criteria
        }

    def _calculate_selection_score(self, action_data: Dict[str, Any]) -> float:
        """Calculate selection score for action"""
        score = 0.0

        # Value component
        score += action_data.get('expected_value', 0) * self.selection_weights['value']

        # Habit component
        score += action_data.get('habit_strength', 0) * self.selection_weights['habit']

        # Novelty component (inverse of habit strength)
        novelty = 1.0 - action_data.get('habit_strength', 0)
        score += novelty * self.selection_weights['novelty']

        # Risk component (inverse of confidence)
        risk = 1.0 - action_data.get('overall_confidence', 0.5)
        score += risk * self.selection_weights['risk']

        return score


class HabitFormation:
    """Habit formation system"""

    def __init__(self):
        self.habits = {}
        self.habit_threshold = 0.7
        self.formation_rate = 0.05

    def update_habits(self, selected_action: Dict[str, Any],
                     execution_result: Dict[str, Any]) -> float:
        """Update habit strengths based on action execution"""

        action_id = selected_action['action']['id']

        if action_id not in self.habits:
            self.habits[action_id] = {
                'strength': 0.0,
                'execution_count': 0,
                'success_count': 0,
                'last_executed': time.time()
            }

        habit = self.habits[action_id]
        habit['execution_count'] += 1
        habit['last_executed'] = time.time()

        if execution_result['success']:
            habit['success_count'] += 1

        # Calculate success rate
        success_rate = habit['success_count'] / habit['execution_count']

        # Update habit strength
        if success_rate > 0.8 and habit['execution_count'] > 5:
            # Strengthen habit
            habit['strength'] = min(1.0, habit['strength'] + self.formation_rate)
        elif success_rate < 0.5:
            # Weaken habit
            habit['strength'] = max(0.0, habit['strength'] - self.formation_rate * 2)

        return habit['strength']

    def get_habit_strength(self, action_id: str) -> float:
        """Get habit strength for action"""
        return self.habits.get(action_id, {}).get('strength', 0.0)

    def get_strong_habits(self) -> List[str]:
        """Get list of strong habits"""
        return [action_id for action_id, habit in self.habits.items()
                if habit['strength'] > self.habit_threshold]


class ReinforcementLearning:
    """Reinforcement learning system"""

    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.exploration_rate = 0.2

    def evaluate_actions(self, action_candidates: List[Dict[str, Any]],
                        state_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate actions using reinforcement learning"""

        evaluated_options = []

        for candidate in action_candidates:
            action_id = candidate['action']['id']

            # Get Q-value for action
            state_key = self._get_state_key(state_evaluation)
            q_value = self._get_q_value(state_key, action_id)

            # Calculate expected value
            expected_value = q_value + random.uniform(0, self.exploration_rate)

            candidate_copy = candidate.copy()
            candidate_copy['q_value'] = q_value
            candidate_copy['expected_value'] = expected_value

            evaluated_options.append(candidate_copy)

        return {
            'evaluated_options': evaluated_options,
            'exploration_rate': self.exploration_rate,
            'learning_rate': self.learning_rate
        }

    def update_learning(self, selected_action: Dict[str, Any],
                       execution_result: Dict[str, Any],
                       state_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Update reinforcement learning based on outcome"""

        action_id = selected_action['action']['id']
        reward = execution_result['reward_value']
        state_key = self._get_state_key(state_evaluation)

        # Get current Q-value
        current_q = self._get_q_value(state_key, action_id)

        # Calculate temporal difference
        td_error = reward - current_q

        # Update Q-value
        new_q = current_q + self.learning_rate * td_error

        # Store updated Q-value
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action_id] = new_q

        return {
            'td_error': td_error,
            'q_value_update': new_q - current_q,
            'new_q_value': new_q,
            'learning_efficiency': abs(td_error)
        }

    def _get_state_key(self, state_evaluation: Dict[str, Any]) -> str:
        """Generate state key for Q-table"""
        # Create simplified state representation
        sensory_intensity = state_evaluation['sensory_features']['overall_intensity']
        emotional_valence = state_evaluation['emotional_features']['valence']
        goal_urgency = state_evaluation['goal_features']['goal_urgency']

        # Discretize values
        sensory_bin = int(sensory_intensity * 5)  # 0-4
        emotional_bin = int((emotional_valence + 1) * 2.5)  # 0-4
        goal_bin = int(goal_urgency * 5)  # 0-4

        return f"{sensory_bin}_{emotional_bin}_{goal_bin}"

    def _get_q_value(self, state_key: str, action_id: str) -> float:
        """Get Q-value for state-action pair"""
        return self.q_table.get(state_key, {}).get(action_id, 0.0)

    def get_status(self) -> Dict[str, Any]:
        """Get reinforcement learning status"""
        return {
            'q_table_size': len(self.q_table),
            'total_state_action_pairs': sum(len(actions) for actions in self.q_table.values()),
            'learning_rate': self.learning_rate,
            'exploration_rate': self.exploration_rate,
            'discount_factor': self.discount_factor
        }


class DopamineSystem:
    """Dopamine system for reward signaling"""

    def __init__(self):
        self.baseline_levels = 0.5
        self.current_levels = 0.5
        self.reward_sensitivity = 0.8
        self.dopamine_decay = 0.05

    def update_dopamine(self, execution_result: Dict[str, Any],
                       state_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Update dopamine levels based on outcome"""

        reward = execution_result.get('reward_value', 0)

        # Calculate dopamine response
        dopamine_response = reward * self.reward_sensitivity

        # Apply prediction error (simplified)
        expected_reward = state_evaluation.get('state_value', 0.5)
        prediction_error = reward - expected_reward
        dopamine_response += prediction_error * 0.3

        # Update current levels
        self.current_levels += dopamine_response
        self.current_levels = max(0.0, min(1.0, self.current_levels))

        return {
            'dopamine_response': dopamine_response,
            'current_levels': self.current_levels,
            'reward_sensitivity': self.reward_sensitivity,
            'prediction_error': prediction_error
        }

    def get_current_levels(self) -> float:
        """Get current dopamine levels"""
        # Apply natural decay
        decay_amount = (self.current_levels - self.baseline_levels) * self.dopamine_decay
        self.current_levels -= decay_amount

        return max(0.0, min(1.0, self.current_levels))

    def get_levels(self) -> Dict[str, float]:
        """Get dopamine system levels"""
        return {
            'baseline': self.baseline_levels,
            'current': self.current_levels,
            'sensitivity': self.reward_sensitivity,
            'decay_rate': self.dopamine_decay
        }


class RewardPrediction:
    """Reward prediction system"""

    def __init__(self):
        self.prediction_model = {}
        self.prediction_accuracy = 0.7
        self.learning_rate = 0.1

    def predict_reward(self, action: Dict[str, Any],
                      state: Dict[str, Any]) -> float:
        """Predict reward for action in state"""
        action_id = action['action']['id']
        state_key = f"{state['sensory_features']['overall_intensity']:.1f}_{state['emotional_features']['valence']:.1f}"

        # Get prediction from model
        prediction_key = f"{state_key}_{action_id}"
        predicted_reward = self.prediction_model.get(prediction_key, 0.0)

        return predicted_reward

    def update_prediction(self, selected_action: Dict[str, Any],
                         execution_result: Dict[str, Any],
                         state_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Update reward prediction based on actual outcome"""

        action_id = selected_action['action']['id']
        actual_reward = execution_result['reward_value']
        state_key = f"{state_evaluation['sensory_features']['overall_intensity']:.1f}_{state_evaluation['emotional_features']['valence']:.1f}"

        prediction_key = f"{state_key}_{action_id}"
        current_prediction = self.prediction_model.get(prediction_key, 0.0)

        # Update prediction
        prediction_error = actual_reward - current_prediction
        new_prediction = current_prediction + self.learning_rate * prediction_error

        self.prediction_model[prediction_key] = new_prediction

        return {
            'prediction_key': prediction_key,
            'actual_reward': actual_reward,
            'predicted_reward': current_prediction,
            'prediction_error': prediction_error,
            'new_prediction': new_prediction
        }


class MotorLearning:
    """Motor learning system"""

    def __init__(self):
        self.motor_patterns = {}
        self.learning_efficiency = 0.8
        self.skill_levels = {}

    def update_motor_learning(self, action: Dict[str, Any],
                            execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update motor learning based on action execution"""

        action_id = action['id']

        if action_id not in self.motor_patterns:
            self.motor_patterns[action_id] = {
                'execution_count': 0,
                'success_rate': 0.0,
                'average_time': 0.0,
                'skill_level': 0.0
            }

        pattern = self.motor_patterns[action_id]
        pattern['execution_count'] += 1

        # Update success rate
        if execution_result['success']:
            pattern['success_rate'] = ((pattern['success_rate'] * (pattern['execution_count'] - 1)) + 1) / pattern['execution_count']
        else:
            pattern['success_rate'] = (pattern['success_rate'] * (pattern['execution_count'] - 1)) / pattern['execution_count']

        # Update execution time
        current_time = execution_result['execution_time']
        pattern['average_time'] = ((pattern['average_time'] * (pattern['execution_count'] - 1)) + current_time) / pattern['execution_count']

        # Update skill level based on success rate and efficiency
        if pattern['execution_count'] > 5:
            skill_improvement = (pattern['success_rate'] - 0.5) * 0.1
            pattern['skill_level'] = min(1.0, max(0.0, pattern['skill_level'] + skill_improvement))

        return {
            'action_id': action_id,
            'skill_level': pattern['skill_level'],
            'success_rate': pattern['success_rate'],
            'average_time': pattern['average_time'],
            'execution_count': pattern['execution_count']
        }

    def get_skill_level(self, action_id: str) -> float:
        """Get skill level for action"""
        return self.motor_patterns.get(action_id, {}).get('skill_level', 0.0)

    def get_motor_status(self) -> Dict[str, Any]:
        """Get motor learning status"""
        return {
            'learned_patterns': len(self.motor_patterns),
            'average_skill_level': sum(p['skill_level'] for p in self.motor_patterns.values()) / max(len(self.motor_patterns), 1),
            'total_executions': sum(p['execution_count'] for p in self.motor_patterns.values()),
            'learning_efficiency': self.learning_efficiency
        }
