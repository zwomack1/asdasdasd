#!/usr/bin/env python3
"""
Amygdala - Emotional Processing Center
Implements detailed amygdala structures for emotion processing,
fear conditioning, emotional memory, and social-emotional responses
"""

import sys
import time
import math
import random
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from collections import defaultdict, deque
import numpy as np

class Amygdala:
    """Amygdala for emotional processing and fear conditioning"""

    def __init__(self):
        # Amygdala nuclei
        self.basolateral_nucleus = BasolateralNucleus()
        self.cortical_nucleus = CorticalNucleus()
        self.central_nucleus = CentralNucleus()
        self.basomedial_nucleus = BasomedialNucleus()

        # Emotional processing systems
        self.emotion_detector = EmotionDetector()
        self.fear_conditioning = FearConditioning()
        self.emotional_memory = EmotionalMemory()
        self.social_emotion_processor = SocialEmotionProcessor()

        # Emotional state tracking
        self.current_emotional_state = {
            'primary_emotion': 'neutral',
            'intensity': 0.0,
            'valence': 0.0,  # Positive to negative
            'arousal': 0.5,  # Low to high arousal
            'duration': 0.0
        }

        # Emotion history and learning
        self.emotion_history = []
        self.emotional_associations = {}
        self.fear_responses = {}
        self.pleasure_responses = {}

        # Autonomic response system
        self.autonomic_responses = {
            'heart_rate': 70,  # bpm
            'blood_pressure': (120, 80),
            'skin_conductance': 2.0,  # microsiemens
            'respiration_rate': 16,  # breaths per minute
            'pupil_dilation': 3.0  # mm
        }

        print("🧠 Amygdala initialized - Emotional processing active")

    def process_emotion(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensory input through amygdala for emotional response"""

        # Stage 1: Detect emotional content in sensory input
        detected_emotions = self.emotion_detector.detect_emotions(sensory_input)

        # Stage 2: Process through amygdala nuclei
        basolateral_response = self.basolateral_nucleus.process(detected_emotions)
        cortical_response = self.cortical_nucleus.process(detected_emotions)
        central_response = self.central_nucleus.process(detected_emotions)
        basomedial_response = self.basomedial_nucleus.process(detected_emotions)

        # Stage 3: Integrate emotional responses
        integrated_emotion = self._integrate_emotional_responses(
            basolateral_response, cortical_response, central_response, basomedial_response
        )

        # Stage 4: Check for fear conditioning
        fear_response = self.fear_conditioning.check_conditioned_responses(sensory_input)

        # Stage 5: Process social emotions
        social_emotion = self.social_emotion_processor.process_social_cues(sensory_input)

        # Stage 6: Generate autonomic responses
        autonomic_response = self._generate_autonomic_response(integrated_emotion, fear_response)

        # Stage 7: Update emotional memory
        self.emotional_memory.store_emotional_experience(
            sensory_input, integrated_emotion, autonomic_response
        )

        # Stage 8: Update current emotional state
        self._update_emotional_state(integrated_emotion, fear_response, social_emotion)

        return {
            'primary_emotion': integrated_emotion['dominant_emotion'],
            'intensity': integrated_emotion['intensity'],
            'valence': integrated_emotion['valence'],
            'arousal': integrated_emotion['arousal'],
            'fear_response': fear_response,
            'social_emotion': social_emotion,
            'autonomic_response': autonomic_response,
            'emotional_memory_activated': len(self.emotional_memory.emotional_episodes) > 0,
            'conditioning_active': fear_response['conditioned_response_triggered']
        }

    def _integrate_emotional_responses(self, basolateral: Dict[str, Any],
                                     cortical: Dict[str, Any],
                                     central: Dict[str, Any],
                                     basomedial: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate responses from all amygdala nuclei"""

        # Combine emotional intensities
        total_intensity = (
            basolateral.get('intensity', 0) +
            cortical.get('intensity', 0) +
            central.get('intensity', 0) +
            basomedial.get('intensity', 0)
        ) / 4

        # Determine dominant emotion
        emotions = defaultdict(float)
        emotions[basolateral.get('emotion', 'neutral')] += basolateral.get('intensity', 0)
        emotions[cortical.get('emotion', 'neutral')] += cortical.get('intensity', 0)
        emotions[central.get('emotion', 'neutral')] += central.get('intensity', 0)
        emotions[basomedial.get('emotion', 'neutral')] += basomedial.get('intensity', 0)

        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else 'neutral'

        # Calculate valence (positive/negative)
        positive_emotions = ['joy', 'pleasure', 'love', 'happiness']
        negative_emotions = ['fear', 'anger', 'sadness', 'disgust']

        valence = 0.0
        if dominant_emotion in positive_emotions:
            valence = total_intensity
        elif dominant_emotion in negative_emotions:
            valence = -total_intensity

        # Calculate arousal level
        high_arousal_emotions = ['fear', 'anger', 'joy', 'excitement']
        arousal = total_intensity * 1.5 if dominant_emotion in high_arousal_emotions else total_intensity * 0.7

        return {
            'dominant_emotion': dominant_emotion,
            'intensity': total_intensity,
            'valence': valence,
            'arousal': arousal,
            'component_responses': {
                'basolateral': basolateral,
                'cortical': cortical,
                'central': central,
                'basomedial': basomedial
            }
        }

    def _generate_autonomic_response(self, integrated_emotion: Dict[str, Any],
                                   fear_response: Dict[str, Any]) -> Dict[str, Any]:
        """Generate autonomic nervous system responses"""

        emotion = integrated_emotion['dominant_emotion']
        intensity = integrated_emotion['intensity']
        fear_intensity = fear_response.get('intensity', 0)

        # Base autonomic response
        response = self.autonomic_responses.copy()

        # Fear response - fight or flight
        if emotion == 'fear' or fear_intensity > 0.5:
            response['heart_rate'] += int(40 * (intensity + fear_intensity))
            response['blood_pressure'] = (
                int(response['blood_pressure'][0] * (1 + 0.3 * intensity)),
                int(response['blood_pressure'][1] * (1 + 0.2 * intensity))
            )
            response['skin_conductance'] += 5 * intensity
            response['respiration_rate'] += int(8 * intensity)
            response['pupil_dilation'] += 2 * intensity

        # Anger response
        elif emotion == 'anger':
            response['heart_rate'] += int(30 * intensity)
            response['blood_pressure'] = (
                int(response['blood_pressure'][0] * (1 + 0.25 * intensity)),
                int(response['blood_pressure'][1] * (1 + 0.15 * intensity))
            )
            response['skin_conductance'] += 3 * intensity

        # Joy/pleasure response
        elif emotion in ['joy', 'pleasure', 'happiness']:
            response['heart_rate'] += int(10 * intensity)
            response['skin_conductance'] += 1 * intensity
            response['respiration_rate'] += int(2 * intensity)

        # Sadness response
        elif emotion == 'sadness':
            response['heart_rate'] -= int(5 * intensity)
            response['respiration_rate'] -= int(2 * intensity)

        # Ensure physiological limits
        response['heart_rate'] = max(40, min(200, response['heart_rate']))
        response['blood_pressure'] = (
            max(80, min(200, response['blood_pressure'][0])),
            max(50, min(120, response['blood_pressure'][1]))
        )
        response['skin_conductance'] = max(0.5, min(20, response['skin_conductance']))
        response['respiration_rate'] = max(8, min(40, response['respiration_rate']))
        response['pupil_dilation'] = max(1.5, min(8, response['pupil_dilation']))

        return response

    def _update_emotional_state(self, integrated_emotion: Dict[str, Any],
                              fear_response: Dict[str, Any],
                              social_emotion: Dict[str, Any]):
        """Update the current emotional state"""

        # Update primary emotion
        self.current_emotional_state['primary_emotion'] = integrated_emotion['dominant_emotion']
        self.current_emotional_state['intensity'] = integrated_emotion['intensity']
        self.current_emotional_state['valence'] = integrated_emotion['valence']
        self.current_emotional_state['arousal'] = integrated_emotion['arousal']

        # Update duration
        if self.current_emotional_state['primary_emotion'] == integrated_emotion['dominant_emotion']:
            self.current_emotional_state['duration'] += 1.0  # 1 second
        else:
            self.current_emotional_state['duration'] = 0.0

        # Record emotional state change
        emotional_record = {
            'timestamp': time.time(),
            'emotion': integrated_emotion['dominant_emotion'],
            'intensity': integrated_emotion['intensity'],
            'valence': integrated_emotion['valence'],
            'arousal': integrated_emotion['arousal'],
            'fear_component': fear_response.get('intensity', 0),
            'social_component': social_emotion.get('intensity', 0),
            'autonomic_response': self._generate_autonomic_response(integrated_emotion, fear_response)
        }

        self.emotion_history.append(emotional_record)

        # Maintain history size
        if len(self.emotion_history) > 1000:
            self.emotion_history = self.emotion_history[-1000:]

    def get_status(self) -> Dict[str, Any]:
        """Get amygdala status"""
        recent_emotions = self.emotion_history[-10:] if self.emotion_history else []

        return {
            'current_emotional_state': self.current_emotional_state,
            'recent_emotions': recent_emotions,
            'emotional_memory_size': len(self.emotional_memory.emotional_episodes),
            'conditioned_fears': len(self.fear_conditioning.conditioned_stimuli),
            'autonomic_state': self.autonomic_responses,
            'social_emotion_processing': self.social_emotion_processor.get_status()
        }


class BasolateralNucleus:
    """Basolateral nucleus for emotional learning and memory"""

    def __init__(self):
        self.emotional_associations = {}
        self.learning_rate = 0.1
        self.activity_level = 0.0

    def process(self, detected_emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Process emotions through basolateral nucleus"""
        self.activity_level = random.uniform(0.6, 0.9)

        # Learn emotional associations
        for emotion, intensity in detected_emotions.items():
            if emotion not in self.emotional_associations:
                self.emotional_associations[emotion] = {
                    'strength': intensity,
                    'associations': [],
                    'last_activated': time.time()
                }
            else:
                # Strengthen existing association
                current_strength = self.emotional_associations[emotion]['strength']
                self.emotional_associations[emotion]['strength'] = current_strength + (intensity - current_strength) * self.learning_rate
                self.emotional_associations[emotion]['last_activated'] = time.time()

        # Determine dominant emotion
        dominant_emotion = max(detected_emotions.items(), key=lambda x: x[1])[0] if detected_emotions else 'neutral'
        dominant_intensity = detected_emotions.get(dominant_emotion, 0)

        return {
            'emotion': dominant_emotion,
            'intensity': dominant_intensity,
            'learning_occurred': True,
            'associations_formed': len(self.emotional_associations),
            'activity_level': self.activity_level
        }


class CorticalNucleus:
    """Cortical nucleus for conscious emotional experience"""

    def __init__(self):
        self.conscious_emotions = {}
        self.emotional_awareness = 0.7
        self.activity_level = 0.0

    def process(self, detected_emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Process emotions through cortical nucleus"""
        self.activity_level = random.uniform(0.5, 0.8)

        # Process emotions with conscious awareness
        conscious_processing = {}

        for emotion, intensity in detected_emotions.items():
            # Apply conscious modulation
            conscious_intensity = intensity * self.emotional_awareness
            conscious_intensity += random.uniform(-0.1, 0.1)  # Conscious interpretation

            conscious_processing[emotion] = conscious_intensity

            # Store conscious emotional experience
            self.conscious_emotions[emotion] = {
                'intensity': conscious_intensity,
                'awareness_level': self.emotional_awareness,
                'timestamp': time.time()
            }

        # Determine most conscious emotion
        if conscious_processing:
            dominant_emotion = max(conscious_processing.items(), key=lambda x: x[1])[0]
            dominant_intensity = conscious_processing[dominant_emotion]
        else:
            dominant_emotion = 'neutral'
            dominant_intensity = 0.0

        return {
            'emotion': dominant_emotion,
            'intensity': dominant_intensity,
            'conscious_awareness': self.emotional_awareness,
            'emotional_insight': self._generate_emotional_insight(dominant_emotion),
            'activity_level': self.activity_level
        }

    def _generate_emotional_insight(self, emotion: str) -> str:
        """Generate insight about current emotional state"""
        insights = {
            'fear': "I'm experiencing fear - this might be a response to perceived threat",
            'anger': "I'm feeling anger - this could be due to perceived injustice or frustration",
            'joy': "I'm experiencing joy - this is a positive emotional response",
            'sadness': "I'm feeling sadness - this might relate to loss or disappointment",
            'surprise': "I'm experiencing surprise - this indicates unexpected information",
            'disgust': "I'm feeling disgust - this is a response to something unpleasant",
            'neutral': "I'm in a neutral emotional state"
        }

        return insights.get(emotion, f"I'm experiencing {emotion}")


class CentralNucleus:
    """Central nucleus for autonomic and behavioral responses"""

    def __init__(self):
        self.autonomic_responses = {}
        self.behavioral_responses = {}
        self.response_threshold = 0.6
        self.activity_level = 0.0

    def process(self, detected_emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Process emotions through central nucleus"""
        self.activity_level = random.uniform(0.7, 0.95)

        # Generate autonomic responses
        autonomic_response = self._generate_autonomic_response(detected_emotions)

        # Generate behavioral responses
        behavioral_response = self._generate_behavioral_response(detected_emotions)

        # Determine if response threshold is met
        max_intensity = max(detected_emotions.values()) if detected_emotions else 0
        response_triggered = max_intensity > self.response_threshold

        dominant_emotion = max(detected_emotions.items(), key=lambda x: x[1])[0] if detected_emotions else 'neutral'

        return {
            'emotion': dominant_emotion,
            'intensity': max_intensity,
            'autonomic_response': autonomic_response,
            'behavioral_response': behavioral_response,
            'response_triggered': response_triggered,
            'activity_level': self.activity_level
        }

    def _generate_autonomic_response(self, emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate autonomic nervous system response"""
        response = {
            'heart_rate_change': 0,
            'blood_pressure_change': (0, 0),
            'respiration_change': 0,
            'fight_or_flight': False
        }

        for emotion, intensity in emotions.items():
            if emotion == 'fear' and intensity > 0.7:
                response['heart_rate_change'] += 30
                response['blood_pressure_change'] = (20, 10)
                response['respiration_change'] += 5
                response['fight_or_flight'] = True
            elif emotion == 'anger' and intensity > 0.6:
                response['heart_rate_change'] += 20
                response['blood_pressure_change'] = (15, 8)
            elif emotion in ['joy', 'pleasure'] and intensity > 0.5:
                response['heart_rate_change'] += 10
                response['respiration_change'] += 2

        return response

    def _generate_behavioral_response(self, emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate behavioral response"""
        response = {
            'approach': False,
            'avoidance': False,
            'freezing': False,
            'aggression': False,
            'submission': False
        }

        for emotion, intensity in emotions.items():
            if emotion == 'fear' and intensity > 0.7:
                response['avoidance'] = True
                response['freezing'] = random.random() > 0.5
            elif emotion == 'anger' and intensity > 0.6:
                response['aggression'] = True
            elif emotion in ['joy', 'pleasure'] and intensity > 0.5:
                response['approach'] = True

        return response


class BasomedialNucleus:
    """Basomedial nucleus for social and reward processing"""

    def __init__(self):
        self.social_responses = {}
        self.reward_associations = {}
        self.social_learning_rate = 0.15
        self.activity_level = 0.0

    def process(self, detected_emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Process emotions through basomedial nucleus"""
        self.activity_level = random.uniform(0.4, 0.75)

        # Process social emotions
        social_processing = self._process_social_emotions(detected_emotions)

        # Process reward associations
        reward_processing = self._process_reward_associations(detected_emotions)

        # Integrate social and reward processing
        integrated_response = self._integrate_social_reward(
            social_processing, reward_processing
        )

        return {
            'emotion': integrated_response['dominant_emotion'],
            'intensity': integrated_response['intensity'],
            'social_component': social_processing,
            'reward_component': reward_processing,
            'integrated_response': integrated_response,
            'activity_level': self.activity_level
        }

    def _process_social_emotions(self, emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Process social aspects of emotions"""
        social_emotions = {}

        for emotion, intensity in emotions.items():
            if emotion in ['love', 'trust', 'empathy', 'guilt', 'shame']:
                social_intensity = intensity * 1.2  # Amplify social emotions
                social_emotions[emotion] = social_intensity

                # Learn social associations
                if emotion not in self.social_responses:
                    self.social_responses[emotion] = []
                self.social_responses[emotion].append({
                    'intensity': social_intensity,
                    'timestamp': time.time(),
                    'context': 'social_interaction'
                })

        return social_emotions

    def _process_reward_associations(self, emotions: Dict[str, Any]) -> Dict[str, Any]:
        """Process reward-related emotions"""
        reward_emotions = {}

        for emotion, intensity in emotions.items():
            if emotion in ['joy', 'pleasure', 'satisfaction', 'pride']:
                reward_value = intensity * 0.8  # Convert to reward value
                reward_emotions[emotion] = reward_value

                # Learn reward associations
                if emotion not in self.reward_associations:
                    self.reward_associations[emotion] = []
                self.reward_associations[emotion].append({
                    'reward_value': reward_value,
                    'timestamp': time.time(),
                    'context': 'reward_experience'
                })

        return reward_emotions

    def _integrate_social_reward(self, social: Dict[str, Any],
                               reward: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate social and reward processing"""
        all_emotions = {**social, **reward}

        if all_emotions:
            dominant_emotion = max(all_emotions.items(), key=lambda x: x[1])[0]
            total_intensity = sum(all_emotions.values())
            avg_intensity = total_intensity / len(all_emotions)
        else:
            dominant_emotion = 'neutral'
            avg_intensity = 0.0

        return {
            'dominant_emotion': dominant_emotion,
            'intensity': avg_intensity,
            'social_weight': len(social) / max(len(all_emotions), 1),
            'reward_weight': len(reward) / max(len(all_emotions), 1)
        }


class EmotionDetector:
    """Emotion detection system"""

    def __init__(self):
        self.emotion_keywords = {
            'fear': ['fear', 'scared', 'afraid', 'terrified', 'anxious', 'threat', 'danger'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'frustrated'],
            'joy': ['happy', 'joy', 'pleased', 'delighted', 'excited', 'glad', 'cheerful'],
            'sadness': ['sad', 'unhappy', 'depressed', 'grief', 'sorrow', 'disappointed'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected'],
            'disgust': ['disgusted', 'repulsed', 'gross', 'sick', 'nauseous', 'revolted'],
            'love': ['love', 'affection', 'caring', 'tenderness', 'fondness', 'adoration'],
            'trust': ['trust', 'confidence', 'faith', 'belief', 'reliance', 'security'],
            'anticipation': ['expecting', 'awaiting', 'looking forward', 'anticipating', 'eager']
        }

        self.contextual_multipliers = {
            'positive': ['good', 'great', 'excellent', 'wonderful', 'fantastic', 'amazing'],
            'negative': ['bad', 'terrible', 'awful', 'horrible', 'dreadful', 'horrific'],
            'intense': ['very', 'extremely', 'intensely', 'profoundly', 'deeply', 'strongly']
        }

    def detect_emotions(self, sensory_input: Dict[str, Any]) -> Dict[str, float]:
        """Detect emotions in sensory input"""
        input_text = str(sensory_input).lower()

        detected_emotions = {}

        # Check for emotion keywords
        for emotion, keywords in self.emotion_keywords.items():
            emotion_score = 0.0
            keyword_count = 0

            for keyword in keywords:
                if keyword in input_text:
                    emotion_score += 0.3  # Base keyword match
                    keyword_count += 1

                    # Check for intensifiers
                    for intensifier in self.contextual_multipliers['intense']:
                        if intensifier in input_text:
                            emotion_score += 0.2
                            break

            if keyword_count > 0:
                # Normalize by keyword count and add randomness
                emotion_score = min(1.0, emotion_score + random.uniform(-0.1, 0.1))
                detected_emotions[emotion] = emotion_score

        # Apply contextual modifiers
        detected_emotions = self._apply_contextual_modifiers(input_text, detected_emotions)

        # Ensure we always have some emotional content
        if not detected_emotions:
            detected_emotions['neutral'] = 0.5

        return detected_emotions

    def _apply_contextual_modifiers(self, text: str,
                                  emotions: Dict[str, float]) -> Dict[str, float]:
        """Apply contextual modifiers to detected emotions"""
        modified_emotions = emotions.copy()

        # Positive context amplifier
        positive_count = sum(1 for word in self.contextual_multipliers['positive'] if word in text)
        if positive_count > 0:
            for emotion in ['joy', 'love', 'trust']:
                if emotion in modified_emotions:
                    modified_emotions[emotion] *= (1 + positive_count * 0.1)

        # Negative context amplifier
        negative_count = sum(1 for word in self.contextual_multipliers['negative'] if word in text)
        if negative_count > 0:
            for emotion in ['fear', 'anger', 'sadness', 'disgust']:
                if emotion in modified_emotions:
                    modified_emotions[emotion] *= (1 + negative_count * 0.1)

        return modified_emotions


class FearConditioning:
    """Fear conditioning and extinction system"""

    def __init__(self):
        self.conditioned_stimuli = {}
        self.extinction_trials = {}
        self.conditioning_strength = {}
        self.fear_threshold = 0.7

    def check_conditioned_responses(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Check for conditioned fear responses"""
        input_signature = self._generate_input_signature(sensory_input)

        if input_signature in self.conditioned_stimuli:
            conditioned_response = self.conditioned_stimuli[input_signature]

            # Check if conditioning is still active
            if self._is_conditioning_active(conditioned_response):
                return {
                    'conditioned_response_triggered': True,
                    'stimulus': input_signature,
                    'intensity': conditioned_response['intensity'],
                    'original_ucs': conditioned_response['original_ucs'],
                    'conditioning_age': time.time() - conditioned_response['conditioning_time']
                }

        return {
            'conditioned_response_triggered': False,
            'intensity': 0.0,
            'stimulus': input_signature
        }

    def condition_fear_response(self, cs: Dict[str, Any], ucs: Dict[str, Any],
                              trials: int = 1):
        """Condition a fear response"""
        cs_signature = self._generate_input_signature(cs)

        if cs_signature not in self.conditioned_stimuli:
            self.conditioned_stimuli[cs_signature] = {
                'intensity': 0.0,
                'conditioning_trials': 0,
                'conditioning_time': time.time(),
                'original_ucs': ucs,
                'extinction_trials': 0
            }

        # Strengthen conditioning
        conditioned_response = self.conditioned_stimuli[cs_signature]
        conditioning_increment = 0.2 * trials
        conditioned_response['intensity'] = min(1.0, conditioned_response['intensity'] + conditioning_increment)
        conditioned_response['conditioning_trials'] += trials

    def extinguish_fear_response(self, cs: Dict[str, Any], trials: int = 1):
        """Extinguish a conditioned fear response"""
        cs_signature = self._generate_input_signature(cs)

        if cs_signature in self.conditioned_stimuli:
            conditioned_response = self.conditioned_stimuli[cs_signature]
            extinction_increment = 0.15 * trials
            conditioned_response['intensity'] = max(0.0, conditioned_response['intensity'] - extinction_increment)
            conditioned_response['extinction_trials'] += trials

            # Mark for removal if fully extinguished
            if conditioned_response['intensity'] < 0.1:
                del self.conditioned_stimuli[cs_signature]

    def _generate_input_signature(self, sensory_input: Dict[str, Any]) -> str:
        """Generate a signature for sensory input"""
        input_str = str(sensory_input)
        return hash(input_str) % 1000000

    def _is_conditioning_active(self, conditioned_response: Dict[str, Any]) -> bool:
        """Check if conditioning is still active"""
        intensity = conditioned_response['intensity']
        extinction_trials = conditioned_response.get('extinction_trials', 0)
        conditioning_trials = conditioned_response.get('conditioning_trials', 0)

        # Conditioning weakens with extinction
        effective_intensity = intensity * (1 - extinction_trials * 0.1)

        return effective_intensity > self.fear_threshold


class EmotionalMemory:
    """Emotional memory system for storing emotion-related experiences"""

    def __init__(self):
        self.emotional_episodes = []
        self.emotion_patterns = defaultdict(list)
        self.memory_capacity = 500

    def store_emotional_experience(self, sensory_input: Dict[str, Any],
                                 emotional_response: Dict[str, Any],
                                 autonomic_response: Dict[str, Any]):
        """Store an emotional experience"""
        episode = {
            'sensory_input': sensory_input,
            'emotional_response': emotional_response,
            'autonomic_response': autonomic_response,
            'timestamp': time.time(),
            'episode_id': len(self.emotional_episodes),
            'emotional_intensity': emotional_response.get('intensity', 0),
            'emotional_valence': emotional_response.get('valence', 0)
        }

        self.emotional_episodes.append(episode)

        # Extract emotion pattern
        emotion = emotional_response.get('primary_emotion', 'neutral')
        self.emotion_patterns[emotion].append(episode)

        # Maintain capacity
        if len(self.emotional_episodes) > self.memory_capacity:
            self.emotional_episodes = self.emotional_episodes[-self.memory_capacity:]

        # Maintain pattern capacity
        for emotion in self.emotion_patterns:
            if len(self.emotion_patterns[emotion]) > 100:
                self.emotion_patterns[emotion] = self.emotion_patterns[emotion][-100:]

    def retrieve_emotional_episodes(self, emotion_type: str = None,
                                  limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve emotional episodes"""
        if emotion_type and emotion_type in self.emotion_patterns:
            return self.emotion_patterns[emotion_type][-limit:]
        else:
            return self.emotional_episodes[-limit:]

    def get_emotion_patterns(self) -> Dict[str, int]:
        """Get emotion pattern statistics"""
        return {emotion: len(episodes) for emotion, episodes in self.emotion_patterns.items()}


class SocialEmotionProcessor:
    """Social emotion processing system"""

    def __init__(self):
        self.social_emotions = {
            'empathy': 0.0,
            'guilt': 0.0,
            'shame': 0.0,
            'pride': 0.0,
            'jealousy': 0.0,
            'gratitude': 0.0
        }
        self.social_context_history = []

    def process_social_cues(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process social cues from sensory input"""
        input_text = str(sensory_input).lower()

        social_emotions_detected = {}

        # Detect social emotion cues
        social_cues = {
            'empathy': ['empathy', 'understand', 'feel for', 'sympathize'],
            'guilt': ['guilty', 'sorry', 'regret', 'wrong'],
            'shame': ['ashamed', 'embarrassed', 'humiliated', 'disgraced'],
            'pride': ['proud', 'accomplished', 'achievement', 'success'],
            'jealousy': ['jealous', 'envious', 'want what you have'],
            'gratitude': ['thankful', 'grateful', 'appreciate', 'gratitude']
        }

        for emotion, cues in social_cues.items():
            emotion_intensity = 0.0
            for cue in cues:
                if cue in input_text:
                    emotion_intensity += 0.3

            if emotion_intensity > 0:
                social_emotions_detected[emotion] = min(1.0, emotion_intensity)
                self.social_emotions[emotion] = social_emotions_detected[emotion]

        # Store social context
        social_context = {
            'input': sensory_input,
            'detected_emotions': social_emotions_detected,
            'timestamp': time.time()
        }

        self.social_context_history.append(social_context)

        # Maintain history size
        if len(self.social_context_history) > 200:
            self.social_context_history = self.social_context_history[-200:]

        return {
            'social_emotions': social_emotions_detected,
            'dominant_social_emotion': max(social_emotions_detected.items(), key=lambda x: x[1])[0] if social_emotions_detected else 'neutral',
            'social_intensity': sum(social_emotions_detected.values()),
            'social_context_stored': True
        }

    def get_social_emotion_history(self, emotion_type: str = None,
                                 limit: int = 20) -> List[Dict[str, Any]]:
        """Get social emotion history"""
        if emotion_type:
            return [ctx for ctx in self.social_context_history[-limit:]
                   if emotion_type in ctx.get('detected_emotions', {})]
        else:
            return self.social_context_history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get social emotion processor status"""
        return {
            'current_social_emotions': self.social_emotions,
            'social_context_history_size': len(self.social_context_history),
            'active_social_emotions': {k: v for k, v in self.social_emotions.items() if v > 0.1}
        }
