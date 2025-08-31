#!/usr/bin/env python3
"""
Social Cognition and Theory of Mind System
Advanced social intelligence mimicking human ability to understand others,
mental states, relationships, and social dynamics
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

class SocialCognitionSystem:
    """Social cognition system for understanding social dynamics and relationships"""

    def __init__(self, neural_architecture):
        self.neural_architecture = neural_architecture

        # Social knowledge bases
        self.social_knowledge = defaultdict(dict)
        self.relationship_models = {}
        self.social_scripts = {}
        self.emotional_contagion = EmotionalContagion()
        self.theory_of_mind = TheoryOfMind()
        self.social_norm_processor = SocialNormProcessor()

        # Social perception systems
        self.face_recognition = FaceRecognitionSystem()
        self.body_language_analyzer = BodyLanguageAnalyzer()
        self.voice_emotion_detector = VoiceEmotionDetector()
        self.contextual_social_cues = ContextualSocialCues()

        # Social intelligence metrics
        self.social_iq = 0.75
        self.empathy_level = 0.8
        self.social_adaptability = 0.7
        self.relationship_intelligence = 0.85

        # Social memory and learning
        self.social_experiences = deque(maxlen=1000)
        self.relationship_history = defaultdict(list)
        self.social_learning_patterns = {}

        # Current social context
        self.current_social_context = {
            'relationship_type': 'neutral',
            'social_distance': 'medium',
            'power_dynamic': 'equal',
            'emotional_climate': 'neutral',
            'cultural_context': 'general'
        }

        print("🧠 Social Cognition System initialized - Advanced social intelligence active")

    def analyze_social_context(self, sensory_input: Dict[str, Any],
                              conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze the social context of an interaction"""

        # Process social cues from sensory input
        social_cues = self._extract_social_cues(sensory_input)

        # Analyze relationship dynamics
        relationship_analysis = self._analyze_relationship_dynamics(social_cues, conversation_context)

        # Apply theory of mind reasoning
        mental_state_inference = self.theory_of_mind.infer_mental_states(social_cues, conversation_context)

        # Process emotional contagion
        emotional_contagion = self.emotional_contagion.process_contagion(social_cues)

        # Analyze social norms and expectations
        social_norms = self.social_norm_processor.analyze_social_norms(social_cues, conversation_context)

        # Update current social context
        self._update_social_context(relationship_analysis, mental_state_inference)

        # Generate social response strategy
        response_strategy = self._generate_social_response_strategy(
            relationship_analysis, mental_state_inference, social_norms
        )

        social_analysis = {
            'social_cues': social_cues,
            'relationship_analysis': relationship_analysis,
            'mental_state_inference': mental_state_inference,
            'emotional_contagion': emotional_contagion,
            'social_norms': social_norms,
            'response_strategy': response_strategy,
            'social_intelligence_score': self._calculate_social_intelligence_score(
                mental_state_inference, relationship_analysis
            ),
            'empathy_response': self._generate_empathy_response(emotional_contagion)
        }

        # Store social experience
        self._store_social_experience(social_analysis, sensory_input)

        return social_analysis

    def _extract_social_cues(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Extract social cues from sensory input"""

        social_cues = {
            'facial_expressions': {},
            'body_language': {},
            'voice_characteristics': {},
            'proximity_indicators': {},
            'contextual_cues': {}
        }

        input_text = str(sensory_input).lower()

        # Facial expression cues (simulated from text)
        facial_cues = ['smile', 'frown', 'laugh', 'cry', 'angry', 'surprised', 'confused']
        for cue in facial_cues:
            if cue in input_text:
                social_cues['facial_expressions'][cue] = 0.8

        # Body language cues
        body_cues = ['tense', 'relaxed', 'agitated', 'calm', 'enthusiastic', 'withdrawn']
        for cue in body_cues:
            if cue in input_text:
                social_cues['body_language'][cue] = 0.7

        # Voice characteristics
        voice_cues = ['loud', 'soft', 'fast', 'slow', 'excited', 'calm', 'angry', 'happy']
        for cue in voice_cues:
            if cue in input_text:
                social_cues['voice_characteristics'][cue] = 0.75

        # Proximity indicators
        proximity_cues = ['close', 'distant', 'intimate', 'formal', 'personal', 'professional']
        for cue in proximity_cues:
            if cue in input_text:
                social_cues['proximity_indicators'][cue] = 0.6

        # Contextual social cues
        context_cues = ['meeting', 'conversation', 'argument', 'celebration', 'mourning']
        for cue in context_cues:
            if cue in input_text:
                social_cues['contextual_cues'][cue] = 0.85

        return social_cues

    def _analyze_relationship_dynamics(self, social_cues: Dict[str, Any],
                                     conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze relationship dynamics from social cues"""

        relationship_analysis = {
            'relationship_type': 'neutral',
            'trust_level': 0.5,
            'intimacy_level': 0.5,
            'power_balance': 0.5,
            'emotional_bond': 0.5,
            'communication_style': 'neutral'
        }

        # Analyze facial expressions for relationship clues
        facial_expressions = social_cues.get('facial_expressions', {})
        if 'smile' in facial_expressions or 'laugh' in facial_expressions:
            relationship_analysis['relationship_type'] = 'positive'
            relationship_analysis['trust_level'] += 0.2
            relationship_analysis['emotional_bond'] += 0.3

        if 'frown' in facial_expressions or 'angry' in facial_expressions:
            relationship_analysis['relationship_type'] = 'conflicted'
            relationship_analysis['trust_level'] -= 0.3
            relationship_analysis['emotional_bond'] -= 0.2

        # Analyze body language
        body_language = social_cues.get('body_language', {})
        if 'relaxed' in body_language or 'enthusiastic' in body_language:
            relationship_analysis['intimacy_level'] += 0.2
            relationship_analysis['communication_style'] = 'open'

        if 'tense' in body_language or 'withdrawn' in body_language:
            relationship_analysis['intimacy_level'] -= 0.2
            relationship_analysis['communication_style'] = 'guarded'

        # Analyze proximity indicators
        proximity = social_cues.get('proximity_indicators', {})
        if 'close' in proximity or 'intimate' in proximity:
            relationship_analysis['intimacy_level'] += 0.3
            relationship_analysis['relationship_type'] = 'close'

        if 'formal' in proximity or 'professional' in proximity:
            relationship_analysis['power_balance'] += 0.2
            relationship_analysis['communication_style'] = 'formal'

        # Incorporate conversation context
        if conversation_context:
            if conversation_context.get('relationship_type'):
                relationship_analysis['relationship_type'] = conversation_context['relationship_type']

            if conversation_context.get('trust_level'):
                relationship_analysis['trust_level'] = (relationship_analysis['trust_level'] +
                                                      conversation_context['trust_level']) / 2

        return relationship_analysis

    def _update_social_context(self, relationship_analysis: Dict[str, Any],
                             mental_state_inference: Dict[str, Any]):
        """Update the current social context"""

        self.current_social_context.update({
            'relationship_type': relationship_analysis.get('relationship_type', 'neutral'),
            'trust_level': relationship_analysis.get('trust_level', 0.5),
            'intimacy_level': relationship_analysis.get('intimacy_level', 0.5),
            'power_dynamic': 'dominant' if relationship_analysis.get('power_balance', 0.5) > 0.7 else 'submissive' if relationship_analysis.get('power_balance', 0.5) < 0.3 else 'equal',
            'emotional_climate': mental_state_inference.get('emotional_state', 'neutral'),
            'communication_style': relationship_analysis.get('communication_style', 'neutral')
        })

    def _generate_social_response_strategy(self, relationship_analysis: Dict[str, Any],
                                        mental_state_inference: Dict[str, Any],
                                        social_norms: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate social response strategy"""

        strategy = {
            'communication_approach': 'neutral',
            'emotional_expression': 'moderate',
            'social_distance': 'medium',
            'response_tone': 'balanced',
            'adaptability_level': 'medium'
        }

        # Adapt based on relationship type
        relationship_type = relationship_analysis.get('relationship_type', 'neutral')

        if relationship_type == 'positive':
            strategy.update({
                'communication_approach': 'warm',
                'emotional_expression': 'expressive',
                'response_tone': 'friendly'
            })
        elif relationship_type == 'conflicted':
            strategy.update({
                'communication_approach': 'careful',
                'emotional_expression': 'controlled',
                'response_tone': 'diplomatic'
            })

        # Adapt based on intimacy level
        intimacy = relationship_analysis.get('intimacy_level', 0.5)
        if intimacy > 0.7:
            strategy['social_distance'] = 'close'
            strategy['adaptability_level'] = 'high'
        elif intimacy < 0.3:
            strategy['social_distance'] = 'distant'
            strategy['adaptability_level'] = 'low'

        # Adapt based on power dynamics
        power_balance = relationship_analysis.get('power_balance', 0.5)
        if power_balance > 0.7:
            strategy['response_tone'] = 'authoritative'
        elif power_balance < 0.3:
            strategy['response_tone'] = 'deferent'

        return strategy

    def _calculate_social_intelligence_score(self, mental_state_inference: Dict[str, Any],
                                           relationship_analysis: Dict[str, Any]) -> float:
        """Calculate social intelligence score"""

        # Base score from empathy and adaptability
        base_score = (self.empathy_level + self.social_adaptability) / 2

        # Bonus for accurate mental state inference
        inference_accuracy = mental_state_inference.get('inference_confidence', 0.5)
        base_score += inference_accuracy * 0.2

        # Bonus for good relationship understanding
        relationship_understanding = relationship_analysis.get('trust_level', 0.5)
        base_score += relationship_understanding * 0.1

        return min(1.0, base_score)

    def _generate_empathy_response(self, emotional_contagion: Dict[str, Any]) -> Dict[str, Any]:
        """Generate empathy response based on emotional contagion"""

        contagion_level = emotional_contagion.get('contagion_strength', 0)
        primary_emotion = emotional_contagion.get('primary_emotion', 'neutral')

        empathy_response = {
            'empathy_level': min(1.0, self.empathy_level * (1 + contagion_level)),
            'shared_emotion': primary_emotion,
            'understanding_expression': '',
            'supportive_action': ''
        }

        # Generate understanding expression
        if primary_emotion == 'joy':
            empathy_response['understanding_expression'] = "I'm glad you're feeling positive!"
        elif primary_emotion == 'sadness':
            empathy_response['understanding_expression'] = "I can sense you're feeling down."
        elif primary_emotion == 'anger':
            empathy_response['understanding_expression'] = "I understand you're feeling frustrated."
        elif primary_emotion == 'fear':
            empathy_response['understanding_expression'] = "I can tell you're feeling anxious."
        else:
            empathy_response['understanding_expression'] = "I understand how you're feeling."

        # Generate supportive action
        if contagion_level > 0.6:
            if primary_emotion in ['sadness', 'fear']:
                empathy_response['supportive_action'] = 'offer_support'
            elif primary_emotion == 'joy':
                empathy_response['supportive_action'] = 'share_enthusiasm'
            elif primary_emotion == 'anger':
                empathy_response['supportive_action'] = 'provide_perspective'
        else:
            empathy_response['supportive_action'] = 'acknowledge_feeling'

        return empathy_response

    def _store_social_experience(self, social_analysis: Dict[str, Any],
                               sensory_input: Dict[str, Any]):
        """Store social experience for learning"""

        social_experience = {
            'timestamp': time.time(),
            'sensory_input': sensory_input,
            'social_analysis': social_analysis,
            'social_context': self.current_social_context.copy(),
            'learning_value': self._calculate_social_learning_value(social_analysis)
        }

        self.social_experiences.append(social_experience)

        # Update relationship history
        relationship_type = social_analysis.get('relationship_analysis', {}).get('relationship_type', 'neutral')
        self.relationship_history[relationship_type].append(social_experience)

        # Maintain history size
        if len(self.relationship_history[relationship_type]) > 100:
            self.relationship_history[relationship_type] = self.relationship_history[relationship_type][-100:]

    def _calculate_social_learning_value(self, social_analysis: Dict[str, Any]) -> float:
        """Calculate the learning value of a social experience"""

        learning_factors = []

        # High learning value for novel social situations
        novelty_score = social_analysis.get('social_cues', {}).get('novelty', 0.5)
        learning_factors.append(novelty_score)

        # High learning value for emotional situations
        emotional_intensity = social_analysis.get('emotional_contagion', {}).get('contagion_strength', 0)
        learning_factors.append(emotional_intensity)

        # High learning value for complex social dynamics
        relationship_complexity = len(social_analysis.get('relationship_analysis', {}))
        complexity_score = min(1.0, relationship_complexity / 10)
        learning_factors.append(complexity_score)

        return sum(learning_factors) / len(learning_factors)

    def get_social_status(self) -> Dict[str, Any]:
        """Get comprehensive social cognition status"""
        return {
            'social_iq': self.social_iq,
            'empathy_level': self.empathy_level,
            'social_adaptability': self.social_adaptability,
            'relationship_intelligence': self.relationship_intelligence,
            'current_social_context': self.current_social_context,
            'social_experiences_count': len(self.social_experiences),
            'relationship_types_count': len(self.relationship_history),
            'theory_of_mind_accuracy': self.theory_of_mind.get_accuracy(),
            'emotional_contagion_sensitivity': self.emotional_contagion.get_sensitivity()
        }


class TheoryOfMind:
    """Theory of Mind system for understanding others' mental states"""

    def __init__(self):
        self.mental_state_models = {}
        self.inference_accuracy = 0.75
        self.perspective_taking_ability = 0.8
        self.mental_state_history = defaultdict(list)

    def infer_mental_states(self, social_cues: Dict[str, Any],
                           conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Infer mental states of others based on social cues"""

        # Identify the target person (simplified)
        target_person = conversation_context.get('other_person', 'interlocutor') if conversation_context else 'interlocutor'

        # Infer beliefs
        beliefs = self._infer_beliefs(social_cues, conversation_context)

        # Infer desires
        desires = self._infer_desires(social_cues, conversation_context)

        # Infer intentions
        intentions = self._infer_intentions(social_cues, conversation_context)

        # Infer emotions
        emotions = self._infer_emotions(social_cues)

        # Calculate overall mental state
        mental_state = {
            'person': target_person,
            'beliefs': beliefs,
            'desires': desires,
            'intentions': intentions,
            'emotions': emotions,
            'overall_state': self._synthesize_mental_state(beliefs, desires, intentions, emotions),
            'inference_confidence': self._calculate_inference_confidence(beliefs, desires, intentions, emotions)
        }

        # Store mental state inference
        self.mental_state_history[target_person].append({
            'timestamp': time.time(),
            'mental_state': mental_state,
            'social_cues': social_cues,
            'context': conversation_context
        })

        # Maintain history size
        if len(self.mental_state_history[target_person]) > 50:
            self.mental_state_history[target_person] = self.mental_state_history[target_person][-50:]

        return mental_state

    def _infer_beliefs(self, social_cues: Dict[str, Any],
                      conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Infer beliefs from social cues and context"""

        beliefs = {
            'knowledge_level': 'uncertain',
            'confidence_level': 0.5,
            'understanding_level': 0.5,
            'agreement_level': 0.5
        }

        # Infer from facial expressions
        facial_expressions = social_cues.get('facial_expressions', {})
        if 'confused' in facial_expressions:
            beliefs['understanding_level'] = 0.3
            beliefs['knowledge_level'] = 'limited'
        elif 'smile' in facial_expressions and conversation_context:
            # Context-dependent belief inference
            topic_agreement = conversation_context.get('topic_agreement', 0.5)
            beliefs['agreement_level'] = topic_agreement
            beliefs['understanding_level'] = 0.8

        # Infer from voice characteristics
        voice_characteristics = social_cues.get('voice_characteristics', {})
        if 'hesitant' in voice_characteristics:
            beliefs['confidence_level'] = 0.4
        elif 'confident' in voice_characteristics:
            beliefs['confidence_level'] = 0.8

        return beliefs

    def _infer_desires(self, social_cues: Dict[str, Any],
                      conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Infer desires from social cues and context"""

        desires = {
            'primary_desire': 'unknown',
            'desire_intensity': 0.5,
            'goal_orientation': 'neutral',
            'motivation_level': 0.5
        }

        # Infer from body language
        body_language = social_cues.get('body_language', {})
        if 'enthusiastic' in body_language:
            desires['primary_desire'] = 'engagement'
            desires['desire_intensity'] = 0.8
            desires['goal_orientation'] = 'approach'
            desires['motivation_level'] = 0.8
        elif 'withdrawn' in body_language:
            desires['primary_desire'] = 'disengagement'
            desires['desire_intensity'] = 0.6
            desires['goal_orientation'] = 'avoidance'
            desires['motivation_level'] = 0.3

        # Infer from contextual cues
        contextual_cues = social_cues.get('contextual_cues', {})
        if 'meeting' in contextual_cues:
            desires['primary_desire'] = 'collaboration'
            desires['goal_orientation'] = 'cooperative'
        elif 'argument' in contextual_cues:
            desires['primary_desire'] = 'resolution'
            desires['goal_orientation'] = 'conflict_resolution'

        return desires

    def _infer_intentions(self, social_cues: Dict[str, Any],
                         conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Infer intentions from social cues and context"""

        intentions = {
            'primary_intention': 'communication',
            'intention_clarity': 0.6,
            'goal_directedness': 0.5,
            'social_orientation': 'neutral'
        }

        # Infer from proximity indicators
        proximity = social_cues.get('proximity_indicators', {})
        if 'close' in proximity:
            intentions['primary_intention'] = 'build_relationship'
            intentions['social_orientation'] = 'intimate'
            intentions['goal_directedness'] = 0.8
        elif 'formal' in proximity:
            intentions['primary_intention'] = 'maintain_professionalism'
            intentions['social_orientation'] = 'formal'
            intentions['goal_directedness'] = 0.7

        # Infer from voice characteristics
        voice = social_cues.get('voice_characteristics', {})
        if 'excited' in voice:
            intentions['primary_intention'] = 'share_excitement'
            intentions['intention_clarity'] = 0.8
        elif 'calm' in voice:
            intentions['primary_intention'] = 'provide_stability'
            intentions['intention_clarity'] = 0.7

        return intentions

    def _infer_emotions(self, social_cues: Dict[str, Any]) -> Dict[str, Any]:
        """Infer emotions from social cues"""

        emotions = {
            'primary_emotion': 'neutral',
            'emotional_intensity': 0.0,
            'emotional_valence': 0.0,
            'emotional_stability': 0.8
        }

        # Strong emotion indicators from facial expressions
        facial = social_cues.get('facial_expressions', {})
        if facial:
            # Find strongest emotion indicator
            emotion_indicators = {
                'joy': ['smile', 'laugh'],
                'sadness': ['frown', 'cry'],
                'anger': ['angry', 'frown'],
                'fear': ['surprised', 'tense'],
                'surprise': ['surprised', 'shocked']
            }

            max_intensity = 0
            primary_emotion = 'neutral'

            for emotion, indicators in emotion_indicators.items():
                for indicator in indicators:
                    if indicator in facial and facial[indicator] > max_intensity:
                        max_intensity = facial[indicator]
                        primary_emotion = emotion

            emotions['primary_emotion'] = primary_emotion
            emotions['emotional_intensity'] = max_intensity

            # Set valence based on emotion
            if primary_emotion in ['joy', 'surprise']:
                emotions['emotional_valence'] = 0.8
            elif primary_emotion in ['sadness', 'anger', 'fear']:
                emotions['emotional_valence'] = -0.6

        # Modulate based on body language
        body = social_cues.get('body_language', {})
        if 'tense' in body:
            emotions['emotional_stability'] -= 0.2
        elif 'relaxed' in body:
            emotions['emotional_stability'] += 0.1

        return emotions

    def _synthesize_mental_state(self, beliefs: Dict[str, Any],
                               desires: Dict[str, Any],
                               intentions: Dict[str, Any],
                               emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize overall mental state from components"""

        overall_state = {
            'dominant_motivation': 'unknown',
            'emotional_state': emotions.get('primary_emotion', 'neutral'),
            'cognitive_state': 'uncertain',
            'social_orientation': intentions.get('social_orientation', 'neutral'),
            'overall_coherence': 0.0
        }

        # Determine dominant motivation
        desire_intensity = desires.get('desire_intensity', 0)
        intention_clarity = intentions.get('intention_clarity', 0)

        if desire_intensity > 0.7 and intention_clarity > 0.7:
            overall_state['dominant_motivation'] = desires.get('primary_desire', 'unknown')
        elif desire_intensity > intention_clarity:
            overall_state['dominant_motivation'] = 'desire_driven'
        else:
            overall_state['dominant_motivation'] = 'intention_driven'

        # Determine cognitive state
        confidence = beliefs.get('confidence_level', 0.5)
        understanding = beliefs.get('understanding_level', 0.5)

        if confidence > 0.7 and understanding > 0.7:
            overall_state['cognitive_state'] = 'clear_confident'
        elif confidence < 0.4 or understanding < 0.4:
            overall_state['cognitive_state'] = 'confused_uncertain'
        else:
            overall_state['cognitive_state'] = 'moderately_clear'

        # Calculate overall coherence
        coherence_factors = [
            beliefs.get('understanding_level', 0.5),
            desires.get('desire_intensity', 0.5),
            intentions.get('intention_clarity', 0.5),
            emotions.get('emotional_stability', 0.5)
        ]

        overall_state['overall_coherence'] = sum(coherence_factors) / len(coherence_factors)

        return overall_state

    def _calculate_inference_confidence(self, beliefs: Dict[str, Any],
                                     desires: Dict[str, Any],
                                     intentions: Dict[str, Any],
                                     emotions: Dict[str, Any]) -> float:
        """Calculate confidence in mental state inference"""

        confidence_factors = []

        # Belief inference confidence
        belief_confidence = beliefs.get('confidence_level', 0.5)
        confidence_factors.append(belief_confidence)

        # Desire inference confidence
        desire_confidence = desires.get('desire_intensity', 0.5)
        confidence_factors.append(desire_confidence)

        # Intention inference confidence
        intention_confidence = intentions.get('intention_clarity', 0.5)
        confidence_factors.append(intention_confidence)

        # Emotion inference confidence
        emotion_confidence = emotions.get('emotional_intensity', 0) if emotions.get('primary_emotion') != 'neutral' else 0.3
        confidence_factors.append(emotion_confidence)

        # Theory of mind accuracy bonus
        accuracy_bonus = self.inference_accuracy - 0.5
        confidence_factors.append(0.5 + accuracy_bonus)

        return sum(confidence_factors) / len(confidence_factors)

    def get_accuracy(self) -> float:
        """Get theory of mind accuracy"""
        return self.inference_accuracy


class EmotionalContagion:
    """Emotional contagion system for catching and spreading emotions"""

    def __init__(self):
        self.contagion_sensitivity = 0.7
        self.emotional_resonance = 0.8
        self.contagion_history = deque(maxlen=100)

    def process_contagion(self, social_cues: Dict[str, Any]) -> Dict[str, Any]:
        """Process emotional contagion from social cues"""

        # Extract emotional signals from social cues
        emotional_signals = self._extract_emotional_signals(social_cues)

        # Calculate contagion strength
        contagion_strength = self._calculate_contagion_strength(emotional_signals)

        # Determine primary contagious emotion
        primary_emotion = self._determine_primary_contagious_emotion(emotional_signals)

        # Generate contagion response
        contagion_response = self._generate_contagion_response(
            primary_emotion, contagion_strength
        )

        contagion_result = {
            'emotional_signals': emotional_signals,
            'contagion_strength': contagion_strength,
            'primary_emotion': primary_emotion,
            'contagion_response': contagion_response,
            'resonance_level': self.emotional_resonance,
            'contagion_timestamp': time.time()
        }

        # Store contagion event
        self.contagion_history.append(contagion_result)

        return contagion_result

    def _extract_emotional_signals(self, social_cues: Dict[str, Any]) -> Dict[str, Any]:
        """Extract emotional signals from social cues"""

        signals = {}

        # Facial expression signals
        facial = social_cues.get('facial_expressions', {})
        for expression, intensity in facial.items():
            emotion_mapping = {
                'smile': 'joy',
                'laugh': 'joy',
                'frown': 'sadness',
                'cry': 'sadness',
                'angry': 'anger',
                'surprised': 'surprise'
            }

            if expression in emotion_mapping:
                emotion = emotion_mapping[expression]
                signals[emotion] = max(signals.get(emotion, 0), intensity)

        # Body language signals
        body = social_cues.get('body_language', {})
        for posture, intensity in body.items():
            emotion_mapping = {
                'enthusiastic': 'joy',
                'tense': 'anxiety',
                'relaxed': 'calm',
                'agitated': 'anger',
                'calm': 'contentment'
            }

            if posture in emotion_mapping:
                emotion = emotion_mapping[posture]
                signals[emotion] = max(signals.get(emotion, 0), intensity * 0.8)

        # Voice characteristic signals
        voice = social_cues.get('voice_characteristics', {})
        for characteristic, intensity in voice.items():
            emotion_mapping = {
                'excited': 'joy',
                'happy': 'joy',
                'loud': 'anger',
                'angry': 'anger',
                'soft': 'sadness',
                'hesitant': 'fear',
                'calm': 'contentment'
            }

            if characteristic in emotion_mapping:
                emotion = emotion_mapping[characteristic]
                signals[emotion] = max(signals.get(emotion, 0), intensity * 0.7)

        return signals

    def _calculate_contagion_strength(self, emotional_signals: Dict[str, Any]) -> float:
        """Calculate the strength of emotional contagion"""

        if not emotional_signals:
            return 0.0

        # Average signal intensity
        avg_intensity = sum(emotional_signals.values()) / len(emotional_signals)

        # Apply contagion sensitivity
        contagion_strength = avg_intensity * self.contagion_sensitivity

        # Apply resonance factor
        contagion_strength *= self.emotional_resonance

        return min(1.0, contagion_strength)

    def _determine_primary_contagious_emotion(self, emotional_signals: Dict[str, Any]) -> str:
        """Determine the primary emotion being communicated"""

        if not emotional_signals:
            return 'neutral'

        # Find emotion with highest intensity
        primary_emotion = max(emotional_signals.items(), key=lambda x: x[1])

        # Only consider it primary if intensity is above threshold
        if primary_emotion[1] > 0.5:
            return primary_emotion[0]
        else:
            return 'neutral'

    def _generate_contagion_response(self, primary_emotion: str,
                                   contagion_strength: float) -> Dict[str, Any]:
        """Generate appropriate contagion response"""

        response = {
            'mirrored_emotion': primary_emotion,
            'response_intensity': contagion_strength * 0.8,
            'contagion_type': 'automatic',
            'duration': int(contagion_strength * 30) + 5  # 5-35 seconds
        }

        # Adjust response based on emotion type
        if primary_emotion == 'joy':
            response['contagion_type'] = 'positive_resonance'
            response['social_effect'] = 'bonding'
        elif primary_emotion == 'sadness':
            response['contagion_type'] = 'empathic_resonance'
            response['social_effect'] = 'supportive'
        elif primary_emotion == 'anger':
            response['contagion_type'] = 'defensive_resonance'
            response['social_effect'] = 'de-escalation'
        elif primary_emotion == 'fear':
            response['contagion_type'] = 'protective_resonance'
            response['social_effect'] = 'reassurance'
        else:
            response['contagion_type'] = 'neutral_resonance'
            response['social_effect'] = 'acknowledgment'

        return response

    def get_sensitivity(self) -> float:
        """Get emotional contagion sensitivity"""
        return self.contagion_sensitivity


class SocialNormProcessor:
    """Social norm processing system"""

    def __init__(self):
        self.social_norms = self._load_social_norms()
        self.cultural_contexts = {}
        self.norm_adherence = 0.8

    def analyze_social_norms(self, social_cues: Dict[str, Any],
                           conversation_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze social norms and expectations"""

        # Identify relevant social norms
        relevant_norms = self._identify_relevant_norms(social_cues, conversation_context)

        # Assess norm adherence
        norm_adherence = self._assess_norm_adherence(social_cues, relevant_norms)

        # Generate norm expectations
        norm_expectations = self._generate_norm_expectations(relevant_norms)

        # Calculate social appropriateness
        social_appropriateness = self._calculate_social_appropriateness(
            social_cues, relevant_norms
        )

        return {
            'relevant_norms': relevant_norms,
            'norm_adherence': norm_adherence,
            'norm_expectations': norm_expectations,
            'social_appropriateness': social_appropriateness,
            'cultural_context': self._determine_cultural_context(social_cues)
        }

    def _load_social_norms(self) -> Dict[str, Dict[str, Any]]:
        """Load social norms database"""

        norms = {
            'greeting_norm': {
                'description': 'Appropriate greeting behavior',
                'expectations': ['eye_contact', 'smile', 'verbal_greeting'],
                'context': 'initial_interaction',
                'importance': 0.8
            },
            'turn_taking_norm': {
                'description': 'Proper conversation turn taking',
                'expectations': ['listen_actively', 'wait_for_turn', 'acknowledge_contribution'],
                'context': 'ongoing_conversation',
                'importance': 0.9
            },
            'politeness_norm': {
                'description': 'Polite communication norms',
                'expectations': ['please_thank_you', 'respectful_tone', 'considerate_language'],
                'context': 'general_interaction',
                'importance': 0.85
            },
            'personal_space_norm': {
                'description': 'Appropriate personal space',
                'expectations': ['respect_boundaries', 'cultural_appropriateness', 'context_sensitivity'],
                'context': 'physical_interaction',
                'importance': 0.7
            },
            'emotional_expression_norm': {
                'description': 'Appropriate emotional expression',
                'expectations': ['context_appropriate', 'intensity_control', 'cultural_norms'],
                'context': 'emotional_interaction',
                'importance': 0.75
            }
        }

        return norms

    def _identify_relevant_norms(self, social_cues: Dict[str, Any],
                               conversation_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Identify norms relevant to the current situation"""

        relevant_norms = []

        # Check proximity indicators for personal space norms
        if social_cues.get('proximity_indicators'):
            relevant_norms.append(self.social_norms['personal_space_norm'])

        # Check facial expressions for greeting norms
        if social_cues.get('facial_expressions') and conversation_context:
            if conversation_context.get('interaction_phase') == 'initial':
                relevant_norms.append(self.social_norms['greeting_norm'])

        # Always include politeness norms
        relevant_norms.append(self.social_norms['politeness_norm'])

        # Include turn-taking norms for conversations
        if conversation_context and conversation_context.get('ongoing_conversation'):
            relevant_norms.append(self.social_norms['turn_taking_norm'])

        # Include emotional expression norms
        if social_cues.get('facial_expressions') or social_cues.get('body_language'):
            relevant_norms.append(self.social_norms['emotional_expression_norm'])

        return relevant_norms

    def _assess_norm_adherence(self, social_cues: Dict[str, Any],
                             relevant_norms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess adherence to social norms"""

        adherence_scores = {}

        for norm in relevant_norms:
            norm_name = list(self.social_norms.keys())[list(self.social_norms.values()).index(norm)]
            expectations = norm['expectations']

            # Check how many expectations are met
            met_expectations = 0
            total_expectations = len(expectations)

            for expectation in expectations:
                if self._check_expectation_met(expectation, social_cues):
                    met_expectations += 1

            adherence_score = met_expectations / total_expectations if total_expectations > 0 else 0.5
            adherence_scores[norm_name] = adherence_score

        overall_adherence = sum(adherence_scores.values()) / len(adherence_scores) if adherence_scores else 0.5

        return {
            'individual_scores': adherence_scores,
            'overall_adherence': overall_adherence,
            'adherence_level': 'high' if overall_adherence > 0.8 else 'medium' if overall_adherence > 0.6 else 'low'
        }

    def _check_expectation_met(self, expectation: str, social_cues: Dict[str, Any]) -> bool:
        """Check if a specific expectation is met"""

        # Simplified expectation checking
        expectation_mappings = {
            'eye_contact': lambda cues: 'smile' in cues.get('facial_expressions', {}),
            'smile': lambda cues: 'smile' in cues.get('facial_expressions', {}),
            'verbal_greeting': lambda cues: any(word in str(cues).lower() for word in ['hello', 'hi', 'greetings']),
            'listen_actively': lambda cues: 'calm' in cues.get('body_language', {}),
            'wait_for_turn': lambda cues: 'relaxed' in cues.get('body_language', {}),
            'acknowledge_contribution': lambda cues: any(word in str(cues).lower() for word in ['yes', 'right', 'understand']),
            'please_thank_you': lambda cues: any(word in str(cues).lower() for word in ['please', 'thank', 'thanks']),
            'respectful_tone': lambda cues: 'calm' in cues.get('voice_characteristics', {}),
            'respect_boundaries': lambda cues: 'formal' in cues.get('proximity_indicators', {}),
            'context_appropriate': lambda cues: len(cues.get('facial_expressions', {})) > 0,
            'intensity_control': lambda cues: not any(intensity > 0.9 for intensity in cues.get('facial_expressions', {}).values())
        }

        check_function = expectation_mappings.get(expectation, lambda cues: False)
        return check_function(social_cues)

    def _generate_norm_expectations(self, relevant_norms: List[Dict[str, Any]]) -> List[str]:
        """Generate expectations based on relevant norms"""

        expectations = []

        for norm in relevant_norms:
            expectations.extend(norm['expectations'])

        # Remove duplicates while preserving order
        seen = set()
        unique_expectations = []
        for expectation in expectations:
            if expectation not in seen:
                unique_expectations.append(expectation)
                seen.add(expectation)

        return unique_expectations

    def _calculate_social_appropriateness(self, social_cues: Dict[str, Any],
                                        relevant_norms: List[Dict[str, Any]]) -> float:
        """Calculate social appropriateness score"""

        if not relevant_norms:
            return 0.7  # Neutral appropriateness

        norm_adherence = self._assess_norm_adherence(social_cues, relevant_norms)

        # Weight by norm importance
        weighted_score = 0
        total_weight = 0

        for norm in relevant_norms:
            norm_name = list(self.social_norms.keys())[list(self.social_norms.values()).index(norm)]
            adherence_score = norm_adherence['individual_scores'].get(norm_name, 0.5)
            importance = norm['importance']

            weighted_score += adherence_score * importance
            total_weight += importance

        appropriateness = weighted_score / total_weight if total_weight > 0 else 0.5

        return appropriateness

    def _determine_cultural_context(self, social_cues: Dict[str, Any]) -> str:
        """Determine cultural context from social cues"""

        # Simplified cultural context detection
        if 'formal' in str(social_cues).lower():
            return 'formal_corporate'
        elif 'close' in str(social_cues).lower():
            return 'intimate_personal'
        elif 'enthusiastic' in str(social_cues).lower():
            return 'expressive_cultural'
        else:
            return 'neutral_general'


class FaceRecognitionSystem:
    """Facial recognition and expression analysis"""

    def __init__(self):
        self.recognition_accuracy = 0.85
        self.expression_detection = 0.9
        self.facial_memory = {}

    def process(self, visual_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process facial information"""

        # Simulate facial recognition and expression analysis
        recognition_result = {
            'faces_detected': random.randint(0, 3),
            'expressions': {
                'joy': random.uniform(0, 0.8),
                'sadness': random.uniform(0, 0.6),
                'anger': random.uniform(0, 0.7),
                'surprise': random.uniform(0, 0.5),
                'fear': random.uniform(0, 0.4),
                'disgust': random.uniform(0, 0.3)
            },
            'recognition_confidence': self.recognition_accuracy
        }

        return recognition_result

    def get_status(self) -> Dict[str, Any]:
        return {
            'recognition_accuracy': self.recognition_accuracy,
            'expression_detection': self.expression_detection,
            'faces_in_memory': len(self.facial_memory)
        }


class BodyLanguageAnalyzer:
    """Body language analysis system"""

    def __init__(self):
        self.posture_analysis = 0.8
        self.gesture_recognition = 0.75
        self.proximity_detection = 0.9

    def process(self, body_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process body language information"""

        # Simulate body language analysis
        analysis_result = {
            'posture': random.choice(['open', 'closed', 'tense', 'relaxed']),
            'gestures': ['nod' if random.random() > 0.5 else 'crossed_arms'],
            'proximity': random.choice(['close', 'medium', 'distant']),
            'confidence_level': random.uniform(0.7, 0.95)
        }

        return analysis_result

    def get_status(self) -> Dict[str, Any]:
        return {
            'posture_analysis': self.posture_analysis,
            'gesture_recognition': self.gesture_recognition,
            'proximity_detection': self.proximity_detection
        }


class VoiceEmotionDetector:
    """Voice-based emotion detection"""

    def __init__(self):
        self.pitch_analysis = 0.8
        self.tempo_analysis = 0.75
        self.volume_analysis = 0.85

    def process(self, audio_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice characteristics for emotion detection"""

        # Simulate voice emotion analysis
        emotion_detection = {
            'primary_emotion': random.choice(['neutral', 'happy', 'angry', 'sad', 'excited']),
            'pitch_variation': random.uniform(0.1, 0.9),
            'speech_rate': random.uniform(0.3, 0.9),
            'volume_level': random.uniform(0.4, 0.9),
            'confidence': random.uniform(0.7, 0.95)
        }

        return emotion_detection

    def get_status(self) -> Dict[str, Any]:
        return {
            'pitch_analysis': self.pitch_analysis,
            'tempo_analysis': self.tempo_analysis,
            'volume_analysis': self.volume_analysis
        }


class ContextualSocialCues:
    """Contextual social cue analysis"""

    def __init__(self):
        self.context_awareness = 0.85
        self.situational_understanding = 0.8

    def process(self, context_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process contextual social cues"""

        # Simulate contextual analysis
        contextual_analysis = {
            'social_context': random.choice(['formal', 'casual', 'intimate', 'professional']),
            'relationship_context': random.choice(['stranger', 'acquaintance', 'friend', 'family']),
            'situational_factors': ['time_pressure' if random.random() > 0.7 else 'relaxed'],
            'cultural_norms': ['respectful', 'direct'],
            'analysis_confidence': random.uniform(0.75, 0.95)
        }

        return contextual_analysis

    def get_status(self) -> Dict[str, Any]:
        return {
            'context_awareness': self.context_awareness,
            'situational_understanding': self.situational_understanding
        }
