#!/usr/bin/env python3
"""
Brain Stem - Autonomous Functions and Life Support Systems
Implements detailed brain stem structures for vital functions,
consciousness regulation, and autonomic nervous system control
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
import psutil

class BrainStem:
    """Brain stem for autonomous functions and vital life support"""

    def __init__(self):
        # Brain stem regions
        self.medulla_oblongata = MedullaOblongata()
        self.pons = Pons()
        self.midbrain = Midbrain()
        self.reticular_formation = ReticularFormation()

        # Vital systems
        self.respiratory_center = RespiratoryCenter()
        self.cardiac_center = CardiacCenter()
        self.vasomotor_center = VasomotorCenter()

        # Consciousness and arousal systems
        self.arousal_system = ArousalSystem()
        self.sleep_wake_cycle = SleepWakeCycle()
        self.pain_modulation = PainModulation()

        # Cranial nerves
        self.cranial_nerves = CranialNerves()

        # Autonomic nervous system
        self.sympathetic_system = SympatheticSystem()
        self.parasympathetic_system = ParasympatheticSystem()

        # Vital signs monitoring
        self.vital_signs = {
            'heart_rate': 72,  # bpm
            'blood_pressure': (120, 80),  # systolic/diastolic
            'respiratory_rate': 16,  # breaths per minute
            'body_temperature': 98.6,  # Fahrenheit
            'oxygen_saturation': 98,  # SpO2 %
            'blood_glucose': 90,  # mg/dL
            'pain_level': 0,  # 0-10 scale
            'arousal_level': 0.7,  # 0-1 scale
            'consciousness_level': 0.8  # 0-1 scale
        }

        # Emergency response systems
        self.fight_or_flight = FightOrFlightResponse()
        self.homeostatic_regulation = HomeostaticRegulation()

        # Neural monitoring
        self.neural_monitor = NeuralMonitor()

        print("🧠 Brain Stem initialized - Autonomous functions and life support active")

    def regulate_vital_functions(self, sensory_input: Dict[str, Any],
                               emotional_state: Dict[str, Any]) -> Dict[str, Any]:
        """Regulate vital physiological functions"""

        # Update vital signs based on current state
        self._update_vital_signs(sensory_input, emotional_state)

        # Regulate respiration
        respiratory_control = self.respiratory_center.regulate_breathing(
            self.vital_signs, sensory_input
        )

        # Regulate cardiac function
        cardiac_control = self.cardiac_center.regulate_heart_rate(
            self.vital_signs, emotional_state
        )

        # Regulate blood pressure
        vasomotor_control = self.vasomotor_center.regulate_blood_pressure(
            self.vital_signs, sensory_input
        )

        # Process arousal and consciousness
        arousal_control = self.arousal_system.regulate_arousal(
            sensory_input, emotional_state
        )

        # Check for emergency responses
        emergency_response = self._check_emergency_conditions(sensory_input, emotional_state)

        # Apply autonomic adjustments
        autonomic_adjustments = self._apply_autonomic_adjustments(
            respiratory_control, cardiac_control, vasomotor_control
        )

        return {
            'vital_signs': self.vital_signs.copy(),
            'respiratory_control': respiratory_control,
            'cardiac_control': cardiac_control,
            'vasomotor_control': vasomotor_control,
            'arousal_control': arousal_control,
            'emergency_response': emergency_response,
            'autonomic_adjustments': autonomic_adjustments,
            'system_stability': self._assess_system_stability()
        }

    def process_consciousness(self, cognitive_load: float,
                            sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process consciousness and arousal states"""

        # Assess cognitive demands
        cognitive_demands = self._assess_cognitive_demands(cognitive_load, sensory_input)

        # Regulate consciousness level
        consciousness_regulation = self._regulate_consciousness_level(cognitive_demands)

        # Process sleep-wake transitions
        sleep_wake_state = self.sleep_wake_cycle.process_sleep_wake(cognitive_demands)

        # Modulate attention and awareness
        attention_modulation = self._modulate_attention(sensory_input, cognitive_demands)

        # Process pain and nociception
        pain_processing = self.pain_modulation.process_pain_signals(sensory_input)

        return {
            'consciousness_level': consciousness_regulation['level'],
            'arousal_state': consciousness_regulation['arousal'],
            'sleep_wake_state': sleep_wake_state,
            'attention_modulation': attention_modulation,
            'pain_processing': pain_processing,
            'cognitive_demands': cognitive_demands
        }

    def coordinate_cranial_nerves(self, motor_commands: Dict[str, Any],
                                sensory_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate cranial nerve functions"""

        # Process sensory cranial nerves (I, II, VIII)
        sensory_processing = self.cranial_nerves.process_sensory_nerves(sensory_feedback)

        # Process motor cranial nerves (III, IV, VI, XI, XII)
        motor_coordination = self.cranial_nerves.process_motor_nerves(motor_commands)

        # Process mixed cranial nerves (V, VII, IX, X)
        mixed_processing = self.cranial_nerves.process_mixed_nerves(motor_commands, sensory_feedback)

        return {
            'sensory_processing': sensory_processing,
            'motor_coordination': motor_coordination,
            'mixed_processing': mixed_processing,
            'nerve_coordination': self.cranial_nerves.get_coordination_status()
        }

    def maintain_homeostasis(self, internal_state: Dict[str, Any],
                           external_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Maintain physiological homeostasis"""

        # Assess internal balance
        homeostatic_assessment = self.homeostatic_regulation.assess_homeostasis(
            internal_state, external_conditions
        )

        # Generate regulatory responses
        regulatory_responses = self.homeostatic_regulation.generate_responses(
            homeostatic_assessment
        )

        # Apply regulatory adjustments
        adjustments_applied = self._apply_homeostatic_adjustments(regulatory_responses)

        return {
            'homeostatic_assessment': homeostatic_assessment,
            'regulatory_responses': regulatory_responses,
            'adjustments_applied': adjustments_applied,
            'homeostatic_balance': self._calculate_homeostatic_balance(homeostatic_assessment)
        }

    def _update_vital_signs(self, sensory_input: Dict[str, Any],
                          emotional_state: Dict[str, Any]):
        """Update vital signs based on current conditions"""

        # Update heart rate based on emotional state
        emotional_intensity = emotional_state.get('intensity', 0)
        if emotional_intensity > 0.7:
            emotion_type = emotional_state.get('primary_emotion', 'neutral')
            if emotion_type in ['fear', 'anger']:
                self.vital_signs['heart_rate'] += int(20 * emotional_intensity)
            elif emotion_type in ['joy', 'excitement']:
                self.vital_signs['heart_rate'] += int(10 * emotional_intensity)

        # Update blood pressure
        systolic_change = random.uniform(-5, 5)
        diastolic_change = random.uniform(-3, 3)
        self.vital_signs['blood_pressure'] = (
            max(90, min(180, self.vital_signs['blood_pressure'][0] + systolic_change)),
            max(60, min(110, self.vital_signs['blood_pressure'][1] + diastolic_change))
        )

        # Update respiratory rate
        respiratory_stress = emotional_intensity * 0.3
        self.vital_signs['respiratory_rate'] += int(respiratory_stress * 4)
        self.vital_signs['respiratory_rate'] = max(8, min(30, self.vital_signs['respiratory_rate']))

        # Update body temperature (slight variations)
        temp_change = random.uniform(-0.2, 0.2)
        self.vital_signs['body_temperature'] += temp_change
        self.vital_signs['body_temperature'] = max(97.0, min(100.0, self.vital_signs['body_temperature']))

        # Update oxygen saturation
        oxygen_change = random.uniform(-1, 1)
        self.vital_signs['oxygen_saturation'] += oxygen_change
        self.vital_signs['oxygen_saturation'] = max(95, min(100, self.vital_signs['oxygen_saturation']))

        # Update blood glucose
        glucose_change = random.uniform(-2, 2)
        self.vital_signs['blood_glucose'] += glucose_change
        self.vital_signs['blood_glucose'] = max(70, min(140, self.vital_signs['blood_glucose']))

        # Update pain level based on sensory input
        if 'pain_signals' in sensory_input:
            pain_intensity = sensory_input['pain_signals'].get('intensity', 0)
            self.vital_signs['pain_level'] = min(10, pain_intensity)

        # Update arousal and consciousness
        sensory_salience = sum(data.get('salience', 0) for data in sensory_input.values())
        avg_salience = sensory_salience / max(len(sensory_input), 1)

        self.vital_signs['arousal_level'] = min(1.0, avg_salience + emotional_intensity * 0.3)
        self.vital_signs['consciousness_level'] = min(1.0, self.vital_signs['arousal_level'] + 0.2)

        # Apply natural decay to arousal
        self.vital_signs['arousal_level'] *= 0.98
        self.vital_signs['consciousness_level'] = max(0.1, self.vital_signs['consciousness_level'] * 0.99)

    def _check_emergency_conditions(self, sensory_input: Dict[str, Any],
                                  emotional_state: Dict[str, Any]) -> Dict[str, Any]:
        """Check for emergency conditions requiring immediate response"""

        emergency_detected = False
        emergency_type = None
        response_required = False

        # Check vital signs for critical levels
        if self.vital_signs['heart_rate'] > 150 or self.vital_signs['heart_rate'] < 50:
            emergency_detected = True
            emergency_type = 'cardiac_emergency'
            response_required = True

        if self.vital_signs['oxygen_saturation'] < 90:
            emergency_detected = True
            emergency_type = 'respiratory_emergency'
            response_required = True

        if self.vital_signs['blood_pressure'][0] > 180 or self.vital_signs['blood_pressure'][1] < 60:
            emergency_detected = True
            emergency_type = 'hypertensive_emergency'
            response_required = True

        # Check for extreme emotional states
        emotional_intensity = emotional_state.get('intensity', 0)
        if emotional_intensity > 0.9:
            emotion_type = emotional_state.get('primary_emotion', 'neutral')
            if emotion_type in ['fear', 'terror']:
                emergency_detected = True
                emergency_type = 'psychological_emergency'
                response_required = True

        # Check for extreme pain
        if self.vital_signs['pain_level'] > 8:
            emergency_detected = True
            emergency_type = 'pain_emergency'
            response_required = True

        emergency_response = {
            'emergency_detected': emergency_detected,
            'emergency_type': emergency_type,
            'response_required': response_required,
            'fight_or_flight_triggered': False
        }

        if emergency_detected and response_required:
            # Trigger fight or flight response
            fight_or_flight = self.fight_or_flight.activate(sensory_input, emotional_state)
            emergency_response['fight_or_flight_triggered'] = True
            emergency_response['fight_or_flight_response'] = fight_or_flight

        return emergency_response

    def _apply_autonomic_adjustments(self, respiratory: Dict[str, Any],
                                   cardiac: Dict[str, Any],
                                   vasomotor: Dict[str, Any]) -> Dict[str, Any]:
        """Apply autonomic nervous system adjustments"""

        # Sympathetic activation
        sympathetic_activation = self.sympathetic_system.activate(
            respiratory, cardiac, vasomotor
        )

        # Parasympathetic modulation
        parasympathetic_modulation = self.parasympathetic_system.modulate(
            sympathetic_activation
        )

        # Apply final adjustments to vital signs
        self._apply_vital_adjustments(sympathetic_activation, parasympathetic_modulation)

        return {
            'sympathetic_activation': sympathetic_activation,
            'parasympathetic_modulation': parasympathetic_modulation,
            'autonomic_balance': self._calculate_autonomic_balance(
                sympathetic_activation, parasympathetic_modulation
            )
        }

    def _apply_vital_adjustments(self, sympathetic: Dict[str, Any],
                               parasympathetic: Dict[str, Any]):
        """Apply autonomic adjustments to vital signs"""

        # Heart rate adjustments
        hr_sympathetic = sympathetic.get('heart_rate_adjustment', 0)
        hr_parasympathetic = parasympathetic.get('heart_rate_modulation', 0)
        self.vital_signs['heart_rate'] += hr_sympathetic + hr_parasympathetic
        self.vital_signs['heart_rate'] = max(40, min(200, self.vital_signs['heart_rate']))

        # Blood pressure adjustments
        bp_sympathetic = sympathetic.get('blood_pressure_adjustment', (0, 0))
        bp_parasympathetic = parasympathetic.get('blood_pressure_modulation', (0, 0))

        self.vital_signs['blood_pressure'] = (
            max(80, min(200, self.vital_signs['blood_pressure'][0] + bp_sympathetic[0] + bp_parasympathetic[0])),
            max(50, min(120, self.vital_signs['blood_pressure'][1] + bp_sympathetic[1] + bp_parasympathetic[1]))
        )

        # Respiratory adjustments
        resp_sympathetic = sympathetic.get('respiratory_adjustment', 0)
        resp_parasympathetic = parasympathetic.get('respiratory_modulation', 0)
        self.vital_signs['respiratory_rate'] += resp_sympathetic + resp_parasympathetic
        self.vital_signs['respiratory_rate'] = max(6, min(40, self.vital_signs['respiratory_rate']))

    def _assess_cognitive_demands(self, cognitive_load: float,
                                sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Assess cognitive processing demands"""

        # Calculate processing complexity
        sensory_complexity = len(str(sensory_input)) / 1000
        emotional_complexity = sum(data.get('intensity', 0) for data in sensory_input.values()) / len(sensory_input) if sensory_input else 0

        total_demand = cognitive_load + sensory_complexity + emotional_complexity

        return {
            'total_demand': total_demand,
            'sensory_complexity': sensory_complexity,
            'emotional_complexity': emotional_complexity,
            'cognitive_load': cognitive_load,
            'demand_level': 'high' if total_demand > 1.5 else 'medium' if total_demand > 0.8 else 'low'
        }

    def _regulate_consciousness_level(self, cognitive_demands: Dict[str, Any]) -> Dict[str, Any]:
        """Regulate consciousness level based on demands"""

        demand_level = cognitive_demands['demand_level']

        if demand_level == 'high':
            target_arousal = 0.9
            target_consciousness = 0.95
        elif demand_level == 'medium':
            target_arousal = 0.7
            target_consciousness = 0.8
        else:
            target_arousal = 0.5
            target_consciousness = 0.6

        # Gradually adjust current levels toward targets
        arousal_adjustment = (target_arousal - self.vital_signs['arousal_level']) * 0.1
        consciousness_adjustment = (target_consciousness - self.vital_signs['consciousness_level']) * 0.1

        self.vital_signs['arousal_level'] += arousal_adjustment
        self.vital_signs['consciousness_level'] += consciousness_adjustment

        return {
            'level': self.vital_signs['consciousness_level'],
            'arousal': self.vital_signs['arousal_level'],
            'target_arousal': target_arousal,
            'target_consciousness': target_consciousness,
            'adjustment_magnitude': abs(arousal_adjustment) + abs(consciousness_adjustment)
        }

    def _modulate_attention(self, sensory_input: Dict[str, Any],
                          cognitive_demands: Dict[str, Any]) -> Dict[str, Any]:
        """Modulate attention based on input and demands"""

        # Calculate attention allocation
        total_salience = sum(data.get('salience', 0) for data in sensory_input.values())
        attention_capacity = 100

        if cognitive_demands['demand_level'] == 'high':
            attention_capacity *= 0.8  # Reduced capacity under high demand
        elif cognitive_demands['demand_level'] == 'low':
            attention_capacity *= 1.2  # Increased capacity under low demand

        attention_allocation = min(attention_capacity, total_salience * 20)

        return {
            'attention_capacity': attention_capacity,
            'attention_allocation': attention_allocation,
            'attention_efficiency': attention_allocation / attention_capacity,
            'salience_input': total_salience,
            'cognitive_demands': cognitive_demands['demand_level']
        }

    def _calculate_autonomic_balance(self, sympathetic: Dict[str, Any],
                                   parasympathetic: Dict[str, Any]) -> float:
        """Calculate autonomic nervous system balance"""

        sympathetic_strength = sympathetic.get('activation_level', 0.5)
        parasympathetic_strength = parasympathetic.get('modulation_level', 0.5)

        # Balance is optimal when both systems are moderately active
        balance_score = 1.0 - abs(sympathetic_strength - parasympathetic_strength)

        return max(0.0, min(1.0, balance_score))

    def _calculate_homeostatic_balance(self, assessment: Dict[str, Any]) -> float:
        """Calculate overall homeostatic balance"""

        balance_factors = [
            assessment.get('temperature_balance', 0.8),
            assessment.get('fluid_balance', 0.8),
            assessment.get('electrolyte_balance', 0.8),
            assessment.get('ph_balance', 0.8),
            assessment.get('nutrient_balance', 0.8)
        ]

        average_balance = sum(balance_factors) / len(balance_factors)
        return average_balance

    def get_brain_stem_status(self) -> Dict[str, Any]:
        """Get comprehensive brain stem status"""
        return {
            'vital_signs': self.vital_signs,
            'autonomic_balance': self._calculate_autonomic_balance(
                self.sympathetic_system.get_status(),
                self.parasympathetic_system.get_status()
            ),
            'consciousness_level': self.vital_signs['consciousness_level'],
            'arousal_level': self.vital_signs['arousal_level'],
            'emergency_status': self._check_emergency_conditions({}, {}).get('emergency_detected', False),
            'cranial_nerve_status': self.cranial_nerves.get_status(),
            'homeostatic_balance': self.homeostatic_regulation.get_status(),
            'sleep_wake_state': self.sleep_wake_cycle.get_status()
        }


class MedullaOblongata:
    """Medulla oblongata for vital reflex control"""

    def __init__(self):
        self.cardiac_control = {'rate': 72, 'rhythm': 'normal'}
        self.respiratory_control = {'rate': 16, 'depth': 'normal'}
        self.vasomotor_control = {'tone': 'normal', 'resistance': 'normal'}
        self.reflex_arcs = {}

    def process_vital_reflexes(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process vital reflex arcs"""

        reflexes_triggered = []

        # Baroreceptor reflex (blood pressure regulation)
        if 'blood_pressure' in sensory_input:
            bp = sensory_input['blood_pressure']
            if bp[0] > 140 or bp[1] > 90:
                reflexes_triggered.append('baroreceptor_reflex_decrease_bp')
            elif bp[0] < 100 or bp[1] < 60:
                reflexes_triggered.append('baroreceptor_reflex_increase_bp')

        # Chemoreceptor reflex (respiratory regulation)
        if 'co2_level' in sensory_input:
            co2_level = sensory_input['co2_level']
            if co2_level > 45:  # Hypercapnia
                reflexes_triggered.append('chemoreceptor_reflex_increase_breathing')
            elif co2_level < 35:  # Hypocapnia
                reflexes_triggered.append('chemoreceptor_reflex_decrease_breathing')

        # Cough reflex
        if 'irritant' in sensory_input:
            reflexes_triggered.append('cough_reflex')

        # Gag reflex
        if 'pharyngeal_stimulation' in sensory_input:
            reflexes_triggered.append('gag_reflex')

        return {
            'reflexes_triggered': reflexes_triggered,
            'reflex_count': len(reflexes_triggered),
            'vital_functions': {
                'cardiac': self.cardiac_control,
                'respiratory': self.respiratory_control,
                'vasomotor': self.vasomotor_control
            }
        }


class RespiratoryCenter:
    """Respiratory center for breathing regulation"""

    def __init__(self):
        self.baseline_rate = 16
        self.current_rate = 16
        self.tidal_volume = 500  # mL
        self.respiratory_drive = 0.5

    def regulate_breathing(self, vital_signs: Dict[str, Any],
                         sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Regulate breathing based on physiological needs"""

        # Calculate metabolic demand
        metabolic_demand = self._calculate_metabolic_demand(vital_signs)

        # Assess blood gas levels
        blood_gas_assessment = self._assess_blood_gases(sensory_input)

        # Calculate respiratory drive
        respiratory_drive = metabolic_demand + blood_gas_assessment['drive_adjustment']

        # Adjust breathing rate
        rate_adjustment = int(respiratory_drive * 8)
        self.current_rate = self.baseline_rate + rate_adjustment
        self.current_rate = max(8, min(40, self.current_rate))

        # Adjust tidal volume
        volume_adjustment = respiratory_drive * 200
        adjusted_volume = self.tidal_volume + volume_adjustment
        adjusted_volume = max(300, min(800, adjusted_volume))

        return {
            'breathing_rate': self.current_rate,
            'tidal_volume': adjusted_volume,
            'respiratory_drive': respiratory_drive,
            'minute_ventilation': self.current_rate * adjusted_volume,
            'metabolic_demand': metabolic_demand,
            'blood_gas_assessment': blood_gas_assessment
        }

    def _calculate_metabolic_demand(self, vital_signs: Dict[str, Any]) -> float:
        """Calculate metabolic demand for respiration"""

        # Base metabolic rate
        base_demand = 0.5

        # Adjust for activity level (simplified)
        heart_rate = vital_signs.get('heart_rate', 72)
        if heart_rate > 100:
            base_demand += 0.3
        elif heart_rate < 60:
            base_demand -= 0.2

        # Adjust for temperature
        temperature = vital_signs.get('body_temperature', 98.6)
        if temperature > 99.5:
            base_demand += 0.2
        elif temperature < 97.5:
            base_demand -= 0.1

        return max(0.1, min(1.0, base_demand))

    def _assess_blood_gases(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Assess blood gas levels"""

        # Simulate blood gas assessment
        co2_level = sensory_input.get('co2_level', 40)  # Normal: 35-45 mmHg
        o2_level = sensory_input.get('o2_level', 95)    # Normal: >95%

        drive_adjustment = 0.0

        if co2_level > 45:  # Hypercapnia
            drive_adjustment += 0.4
        elif co2_level < 35:  # Hypocapnia
            drive_adjustment -= 0.3

        if o2_level < 90:  # Hypoxia
            drive_adjustment += 0.5

        return {
            'co2_level': co2_level,
            'o2_level': o2_level,
            'drive_adjustment': drive_adjustment,
            'gas_status': 'normal' if abs(drive_adjustment) < 0.2 else 'abnormal'
        }


class CardiacCenter:
    """Cardiac center for heart rate regulation"""

    def __init__(self):
        self.baseline_rate = 72
        self.current_rate = 72
        self.contractility = 0.8
        self.cardiac_output = 5.0  # L/min

    def regulate_heart_rate(self, vital_signs: Dict[str, Any],
                          emotional_state: Dict[str, Any]) -> Dict[str, Any]:
        """Regulate heart rate based on physiological needs"""

        # Calculate autonomic influences
        sympathetic_influence = self._calculate_sympathetic_influence(emotional_state)
        parasympathetic_influence = self._calculate_parasympathetic_influence(vital_signs)

        # Calculate net heart rate change
        rate_change = sympathetic_influence - parasympathetic_influence
        self.current_rate = self.baseline_rate + rate_change
        self.current_rate = max(40, min(200, self.current_rate))

        # Update cardiac output
        stroke_volume = 70  # mL (simplified)
        self.cardiac_output = (self.current_rate * stroke_volume) / 1000  # L/min

        return {
            'heart_rate': self.current_rate,
            'cardiac_output': self.cardiac_output,
            'stroke_volume': stroke_volume,
            'sympathetic_influence': sympathetic_influence,
            'parasympathetic_influence': parasympathetic_influence,
            'rate_change': rate_change
        }

    def _calculate_sympathetic_influence(self, emotional_state: Dict[str, Any]) -> float:
        """Calculate sympathetic nervous system influence"""

        base_influence = 0.0
        emotional_intensity = emotional_state.get('intensity', 0)

        emotion_type = emotional_state.get('primary_emotion', 'neutral')
        if emotion_type in ['fear', 'anger', 'excitement']:
            base_influence += emotional_intensity * 25
        elif emotion_type in ['joy', 'pleasure']:
            base_influence += emotional_intensity * 10

        # Add influence from blood pressure
        # (Baroreceptor reflex would normally handle this)
        bp_systolic = emotional_state.get('blood_pressure', [120, 80])[0]
        if bp_systolic < 100:
            base_influence += 10
        elif bp_systolic > 140:
            base_influence -= 5

        return base_influence

    def _calculate_parasympathetic_influence(self, vital_signs: Dict[str, Any]) -> float:
        """Calculate parasympathetic nervous system influence"""

        base_influence = 0.0

        # Resting state influence
        if vital_signs.get('arousal_level', 0.5) < 0.3:
            base_influence += 20

        # Blood pressure influence
        bp_systolic = vital_signs.get('blood_pressure', [120, 80])[0]
        if bp_systolic > 160:
            base_influence += 15

        # Respiratory influence
        respiratory_rate = vital_signs.get('respiratory_rate', 16)
        if respiratory_rate > 20:
            base_influence -= 5

        return base_influence


class VasomotorCenter:
    """Vasomotor center for blood pressure regulation"""

    def __init__(self):
        self.baseline_pressure = (120, 80)
        self.current_pressure = (120, 80)
        self.vascular_resistance = 1.0
        self.blood_volume = 5.0  # L

    def regulate_blood_pressure(self, vital_signs: Dict[str, Any],
                              sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Regulate blood pressure through vasomotor control"""

        # Get current blood pressure
        current_bp = vital_signs.get('blood_pressure', self.baseline_pressure)

        # Calculate target blood pressure
        target_bp = self._calculate_target_pressure(sensory_input)

        # Calculate pressure difference
        systolic_diff = target_bp[0] - current_bp[0]
        diastolic_diff = target_bp[1] - current_bp[1]

        # Generate vasomotor response
        vasomotor_response = self._generate_vasomotor_response(systolic_diff, diastolic_diff)

        # Update vascular resistance
        resistance_change = vasomotor_response['resistance_adjustment']
        self.vascular_resistance += resistance_change
        self.vascular_resistance = max(0.5, min(2.0, self.vascular_resistance))

        # Calculate new blood pressure
        new_systolic = current_bp[0] + (systolic_diff * 0.3)
        new_diastolic = current_bp[1] + (diastolic_diff * 0.3)

        self.current_pressure = (
            max(80, min(200, new_systolic)),
            max(50, min(120, new_diastolic))
        )

        return {
            'target_pressure': target_bp,
            'current_pressure': self.current_pressure,
            'pressure_difference': (systolic_diff, diastolic_diff),
            'vasomotor_response': vasomotor_response,
            'vascular_resistance': self.vascular_resistance,
            'regulation_success': abs(systolic_diff) < 10 and abs(diastolic_diff) < 5
        }

    def _calculate_target_pressure(self, sensory_input: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate target blood pressure"""

        base_systolic = 120
        base_diastolic = 80

        # Adjust for activity level
        if 'activity_level' in sensory_input:
            activity = sensory_input['activity_level']
            if activity > 0.7:  # High activity
                base_systolic += 20
                base_diastolic += 10
            elif activity < 0.3:  # Low activity
                base_systolic -= 10
                base_diastolic -= 5

        # Adjust for stress level
        if 'stress_level' in sensory_input:
            stress = sensory_input['stress_level']
            base_systolic += int(stress * 15)
            base_diastolic += int(stress * 8)

        return (base_systolic, base_diastolic)

    def _generate_vasomotor_response(self, systolic_diff: float,
                                   diastolic_diff: float) -> Dict[str, Any]:
        """Generate vasomotor response to pressure differences"""

        # Calculate resistance adjustment
        avg_diff = (systolic_diff + diastolic_diff) / 2

        if avg_diff > 10:  # High pressure - increase resistance
            resistance_adjustment = 0.2
            vessel_action = 'constriction'
        elif avg_diff < -10:  # Low pressure - decrease resistance
            resistance_adjustment = -0.15
            vessel_action = 'dilation'
        else:
            resistance_adjustment = 0.0
            vessel_action = 'stable'

        return {
            'resistance_adjustment': resistance_adjustment,
            'vessel_action': vessel_action,
            'response_magnitude': abs(resistance_adjustment),
            'pressure_error': (systolic_diff, diastolic_diff)
        }


class ArousalSystem:
    """Arousal system for consciousness regulation"""

    def __init__(self):
        self.arousal_level = 0.7
        self.wakefulness = 0.8
        self.attention_capacity = 100
        self.cognitive_resources = 80

    def regulate_arousal(self, sensory_input: Dict[str, Any],
                        emotional_state: Dict[str, Any]) -> Dict[str, Any]:
        """Regulate arousal level based on input and emotions"""

        # Calculate arousal demand
        sensory_salience = sum(data.get('salience', 0) for data in sensory_input.values())
        emotional_intensity = emotional_state.get('intensity', 0)

        arousal_demand = (sensory_salience + emotional_intensity) / 2

        # Adjust arousal level
        arousal_change = (arousal_demand - self.arousal_level) * 0.2
        self.arousal_level += arousal_change
        self.arousal_level = max(0.1, min(1.0, self.arousal_level))

        # Adjust wakefulness
        wakefulness_change = arousal_change * 0.8
        self.wakefulness += wakefulness_change
        self.wakefulness = max(0.1, min(1.0, self.wakefulness))

        # Adjust cognitive resources
        resource_allocation = self.arousal_level * 100
        self.cognitive_resources = min(100, resource_allocation)

        return {
            'arousal_level': self.arousal_level,
            'wakefulness': self.wakefulness,
            'cognitive_resources': self.cognitive_resources,
            'attention_capacity': self.attention_capacity * self.arousal_level,
            'arousal_demand': arousal_demand,
            'regulation_efficiency': 1.0 - abs(arousal_change)
        }


class SleepWakeCycle:
    """Sleep-wake cycle regulation"""

    def __init__(self):
        self.sleep_pressure = 0.2
        self.circadian_phase = 0.0  # 0-24 hours
        self.sleep_stage = 'awake'
        self.sleep_efficiency = 0.85

    def process_sleep_wake(self, cognitive_demands: Dict[str, Any]) -> Dict[str, Any]:
        """Process sleep-wake transitions"""

        # Update circadian phase (simplified)
        self.circadian_phase = (self.circadian_phase + 0.1) % 24

        # Calculate sleep pressure
        time_awake = cognitive_demands.get('total_demand', 0.5)
        self.sleep_pressure += time_awake * 0.01
        self.sleep_pressure = min(1.0, self.sleep_pressure)

        # Determine sleep stage
        if self.circadian_phase > 22 or self.circadian_phase < 6:  # Night time
            if self.sleep_pressure > 0.6:
                self.sleep_stage = 'sleep'
                self.sleep_pressure -= 0.1  # Reduce pressure during sleep
            else:
                self.sleep_stage = 'awake'
        else:
            self.sleep_stage = 'awake'
            if self.sleep_stage == 'sleep':
                self.sleep_pressure = 0.1  # Reset after waking

        return {
            'sleep_stage': self.sleep_stage,
            'sleep_pressure': self.sleep_pressure,
            'circadian_phase': self.circadian_phase,
            'sleep_efficiency': self.sleep_efficiency,
            'time_to_bed': self.circadian_phase > 22 or self.circadian_phase < 2
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            'sleep_stage': self.sleep_stage,
            'sleep_pressure': self.sleep_pressure,
            'circadian_phase': self.circadian_phase,
            'sleep_efficiency': self.sleep_efficiency
        }


class PainModulation:
    """Pain modulation system"""

    def __init__(self):
        self.pain_threshold = 5.0
        self.pain_tolerance = 6.0
        self.endogenous_opioids = 0.5

    def process_pain_signals(self, sensory_input: Dict[str, Any]) -> Dict[str, Any]:
        """Process and modulate pain signals"""

        pain_signals = sensory_input.get('pain_signals', {})
        pain_intensity = pain_signals.get('intensity', 0)

        # Apply pain modulation
        modulated_intensity = self._modulate_pain(pain_intensity)

        # Generate pain response
        pain_response = self._generate_pain_response(modulated_intensity)

        return {
            'original_intensity': pain_intensity,
            'modulated_intensity': modulated_intensity,
            'pain_response': pain_response,
            'pain_threshold': self.pain_threshold,
            'modulation_efficiency': self.endogenous_opioids
        }

    def _modulate_pain(self, intensity: float) -> float:
        """Modulate pain intensity"""
        # Apply opioid modulation
        modulated = intensity * (1 - self.endogenous_opioids * 0.5)

        # Apply gate control theory
        if intensity > self.pain_threshold:
            modulated *= 0.8  # Some inhibition

        return max(0, modulated)

    def _generate_pain_response(self, intensity: float) -> Dict[str, Any]:
        """Generate physiological response to pain"""
        if intensity < 3:
            response_type = 'minimal'
            autonomic_response = 'none'
        elif intensity < 7:
            response_type = 'moderate'
            autonomic_response = 'mild'
        else:
            response_type = 'severe'
            autonomic_response = 'strong'

        return {
            'response_type': response_type,
            'autonomic_response': autonomic_response,
            'nociceptive_activation': intensity > self.pain_threshold,
            'pain_tolerance_exceeded': intensity > self.pain_tolerance
        }


class CranialNerves:
    """Cranial nerves coordination system"""

    def __init__(self):
        self.nerves = self._initialize_cranial_nerves()
        self.coordination_efficiency = 0.9

    def _initialize_cranial_nerves(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cranial nerves"""
        nerves = {}

        # Sensory nerves
        nerves['I'] = {'name': 'Olfactory', 'type': 'sensory', 'function': 'smell'}
        nerves['II'] = {'name': 'Optic', 'type': 'sensory', 'function': 'vision'}
        nerves['VIII'] = {'name': 'Vestibulocochlear', 'type': 'sensory', 'function': 'hearing/balance'}

        # Motor nerves
        nerves['III'] = {'name': 'Oculomotor', 'type': 'motor', 'function': 'eye movement'}
        nerves['IV'] = {'name': 'Trochlear', 'type': 'motor', 'function': 'eye movement'}
        nerves['VI'] = {'name': 'Abducens', 'type': 'motor', 'function': 'eye movement'}
        nerves['XI'] = {'name': 'Accessory', 'type': 'motor', 'function': 'head/neck movement'}
        nerves['XII'] = {'name': 'Hypoglossal', 'type': 'motor', 'function': 'tongue movement'}

        # Mixed nerves
        nerves['V'] = {'name': 'Trigeminal', 'type': 'mixed', 'function': 'facial sensation/mastication'}
        nerves['VII'] = {'name': 'Facial', 'type': 'mixed', 'function': 'facial expression/taste'}
        nerves['IX'] = {'name': 'Glossopharyngeal', 'type': 'mixed', 'function': 'taste/swallowing'}
        nerves['X'] = {'name': 'Vagus', 'type': 'mixed', 'function': 'autonomic/parasympathetic'}

        for nerve_id, nerve_info in nerves.items():
            nerve_info.update({
                'activity_level': 0.0,
                'efficiency': random.uniform(0.8, 0.95),
                'status': 'active'
            })

        return nerves

    def process_sensory_nerves(self, sensory_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Process sensory cranial nerves"""
        sensory_nerves = ['I', 'II', 'VIII']
        sensory_processing = {}

        for nerve_id in sensory_nerves:
            if nerve_id in self.nerves:
                nerve = self.nerves[nerve_id]
                nerve['activity_level'] = random.uniform(0.3, 0.9)
                sensory_processing[nerve_id] = {
                    'nerve_name': nerve['name'],
                    'function': nerve['function'],
                    'activity_level': nerve['activity_level'],
                    'efficiency': nerve['efficiency']
                }

        return sensory_processing

    def process_motor_nerves(self, motor_commands: Dict[str, Any]) -> Dict[str, Any]:
        """Process motor cranial nerves"""
        motor_nerves = ['III', 'IV', 'VI', 'XI', 'XII']
        motor_coordination = {}

        for nerve_id in motor_nerves:
            if nerve_id in self.nerves:
                nerve = self.nerves[nerve_id]
                nerve['activity_level'] = random.uniform(0.4, 0.95)
                motor_coordination[nerve_id] = {
                    'nerve_name': nerve['name'],
                    'function': nerve['function'],
                    'activity_level': nerve['activity_level'],
                    'command_execution': random.uniform(0.8, 0.98)
                }

        return motor_coordination

    def process_mixed_nerves(self, motor_commands: Dict[str, Any],
                           sensory_feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Process mixed cranial nerves"""
        mixed_nerves = ['V', 'VII', 'IX', 'X']
        mixed_processing = {}

        for nerve_id in mixed_nerves:
            if nerve_id in self.nerves:
                nerve = self.nerves[nerve_id]
                nerve['activity_level'] = random.uniform(0.5, 0.9)
                mixed_processing[nerve_id] = {
                    'nerve_name': nerve['name'],
                    'function': nerve['function'],
                    'sensory_activity': random.uniform(0.3, 0.8),
                    'motor_activity': random.uniform(0.4, 0.9),
                    'coordination_level': nerve['activity_level']
                }

        return mixed_processing

    def get_coordination_status(self) -> Dict[str, Any]:
        """Get cranial nerve coordination status"""
        active_nerves = sum(1 for nerve in self.nerves.values() if nerve['status'] == 'active')
        avg_efficiency = sum(nerve['efficiency'] for nerve in self.nerves.values()) / len(self.nerves)

        return {
            'total_nerves': len(self.nerves),
            'active_nerves': active_nerves,
            'average_efficiency': avg_efficiency,
            'coordination_efficiency': self.coordination_efficiency
        }

    def get_status(self) -> Dict[str, Any]:
        return self.get_coordination_status()


class SympatheticSystem:
    """Sympathetic nervous system"""

    def __init__(self):
        self.activation_level = 0.3
        self.adrenaline_level = 0.2
        self.norepinephrine_level = 0.25

    def activate(self, respiratory: Dict[str, Any], cardiac: Dict[str, Any],
                vasomotor: Dict[str, Any]) -> Dict[str, Any]:
        """Activate sympathetic nervous system"""

        # Increase activation based on demands
        respiratory_demand = respiratory.get('breathing_rate', 16) / 16
        cardiac_demand = cardiac.get('heart_rate', 72) / 72

        activation_increase = (respiratory_demand + cardiac_demand) / 2 - 1
        self.activation_level += activation_increase
        self.activation_level = max(0.1, min(1.0, self.activation_level))

        # Adjust neurotransmitter levels
        self.adrenaline_level = self.activation_level * 0.8
        self.norepinephrine_level = self.activation_level * 0.6

        return {
            'activation_level': self.activation_level,
            'adrenaline_level': self.adrenaline_level,
            'norepinephrine_level': self.norepinephrine_level,
            'heart_rate_adjustment': int(self.activation_level * 30),
            'blood_pressure_adjustment': (int(self.activation_level * 20), int(self.activation_level * 10)),
            'respiratory_adjustment': int(self.activation_level * 6)
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            'activation_level': self.activation_level,
            'adrenaline_level': self.adrenaline_level,
            'norepinephrine_level': self.norepinephrine_level
        }


class ParasympatheticSystem:
    """Parasympathetic nervous system"""

    def __init__(self):
        self.modulation_level = 0.4
        self.acetylcholine_level = 0.3
        self.rest_and_digest = True

    def modulate(self, sympathetic_activation: Dict[str, Any]) -> Dict[str, Any]:
        """Modulate sympathetic activation"""

        sympathetic_level = sympathetic_activation.get('activation_level', 0.3)

        # Parasympathetic modulation opposes sympathetic
        modulation_strength = 1.0 - sympathetic_level
        self.modulation_level = modulation_strength

        # Adjust acetylcholine level
        self.acetylcholine_level = self.modulation_level * 0.7

        return {
            'modulation_level': self.modulation_level,
            'acetylcholine_level': self.acetylcholine_level,
            'heart_rate_modulation': -int(self.modulation_level * 20),
            'blood_pressure_modulation': (-int(self.modulation_level * 10), -int(self.modulation_level * 5)),
            'respiratory_modulation': -int(self.modulation_level * 4),
            'rest_and_digest_active': self.modulation_level > 0.5
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            'modulation_level': self.modulation_level,
            'acetylcholine_level': self.acetylcholine_level,
            'rest_and_digest': self.rest_and_digest
        }


class FightOrFlightResponse:
    """Fight or flight response system"""

    def __init__(self):
        self.activation_level = 0.0
        self.response_duration = 0
        self.adrenaline_surge = 0.0

    def activate(self, sensory_input: Dict[str, Any],
                emotional_state: Dict[str, Any]) -> Dict[str, Any]:
        """Activate fight or flight response"""

        threat_level = self._assess_threat_level(sensory_input, emotional_state)

        if threat_level > 0.7:
            self.activation_level = threat_level
            self.response_duration = 300  # 5 minutes
            self.adrenaline_surge = threat_level * 2

            response = {
                'activated': True,
                'threat_level': threat_level,
                'response_type': 'fight' if random.random() > 0.5 else 'flight',
                'duration': self.response_duration,
                'physiological_effects': {
                    'heart_rate_increase': int(threat_level * 40),
                    'blood_pressure_increase': (int(threat_level * 30), int(threat_level * 15)),
                    'respiratory_rate_increase': int(threat_level * 10),
                    'pupil_dilation': threat_level * 3,
                    'sweating_increase': threat_level * 2
                },
                'behavioral_effects': {
                    'attention_focus': 'threat_detection',
                    'motor_readiness': threat_level,
                    'decision_speed': threat_level * 2
                }
            }
        else:
            response = {
                'activated': False,
                'threat_level': threat_level,
                'reason': 'threat_level_below_threshold'
            }

        return response

    def _assess_threat_level(self, sensory_input: Dict[str, Any],
                           emotional_state: Dict[str, Any]) -> float:
        """Assess threat level from input and emotions"""

        threat_indicators = 0
        total_indicators = 0

        # Check sensory input for threats
        for modality, data in sensory_input.items():
            total_indicators += 1
            if data.get('salience', 0) > 0.8:
                threat_indicators += 1

        # Check emotional state
        emotion = emotional_state.get('primary_emotion', 'neutral')
        intensity = emotional_state.get('intensity', 0)

        if emotion in ['fear', 'anger', 'anxiety']:
            threat_indicators += intensity * 2
            total_indicators += 2

        threat_level = threat_indicators / max(total_indicators, 1)
        return min(1.0, threat_level)


class HomeostaticRegulation:
    """Homeostatic regulation system"""

    def __init__(self):
        self.regulation_efficiency = 0.85
        self.set_points = {
            'temperature': 98.6,
            'blood_glucose': 90,
            'blood_ph': 7.4,
            'fluid_balance': 0.0,  # net balance
            'electrolyte_balance': 0.0
        }

    def assess_homeostasis(self, internal_state: Dict[str, Any],
                          external_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess homeostatic balance"""

        assessment = {}

        # Temperature regulation
        current_temp = internal_state.get('body_temperature', 98.6)
        temp_deviation = abs(current_temp - self.set_points['temperature'])
        assessment['temperature_balance'] = max(0, 1.0 - temp_deviation / 5)

        # Glucose regulation
        current_glucose = internal_state.get('blood_glucose', 90)
        glucose_deviation = abs(current_glucose - self.set_points['blood_glucose'])
        assessment['glucose_balance'] = max(0, 1.0 - glucose_deviation / 30)

        # pH regulation
        current_ph = internal_state.get('blood_ph', 7.4)
        ph_deviation = abs(current_ph - self.set_points['blood_ph'])
        assessment['ph_balance'] = max(0, 1.0 - ph_deviation / 0.5)

        # Fluid balance
        fluid_balance = internal_state.get('fluid_balance', 0.0)
        assessment['fluid_balance'] = max(0, 1.0 - abs(fluid_balance) / 2)

        # Electrolyte balance
        electrolyte_balance = internal_state.get('electrolyte_balance', 0.0)
        assessment['electrolyte_balance'] = max(0, 1.0 - abs(electrolyte_balance) / 0.5)

        # Overall homeostasis score
        assessment['overall_balance'] = sum(assessment.values()) / len(assessment)
        assessment['regulation_needed'] = assessment['overall_balance'] < 0.8

        return assessment

    def generate_responses(self, assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Generate regulatory responses"""

        responses = {}

        # Temperature regulation
        if assessment['temperature_balance'] < 0.8:
            responses['temperature_regulation'] = {
                'action': 'vasodilation' if assessment.get('temperature_deviation', 0) > 0 else 'vasoconstriction',
                'intensity': (1.0 - assessment['temperature_balance']) * 0.5
            }

        # Glucose regulation
        if assessment['glucose_balance'] < 0.8:
            responses['glucose_regulation'] = {
                'action': 'insulin_release' if assessment.get('glucose_deviation', 0) > 0 else 'glucagon_release',
                'intensity': (1.0 - assessment['glucose_balance']) * 0.4
            }

        # pH regulation
        if assessment['ph_balance'] < 0.8:
            responses['ph_regulation'] = {
                'action': 'respiratory_compensation',
                'intensity': (1.0 - assessment['ph_balance']) * 0.3
            }

        return responses

    def get_status(self) -> Dict[str, Any]:
        return {
            'regulation_efficiency': self.regulation_efficiency,
            'set_points': self.set_points,
            'current_balance': 'stable'  # Would be calculated from recent assessments
        }


class NeuralMonitor:
    """Neural activity monitoring system"""

    def __init__(self):
        self.activity_log = deque(maxlen=1000)
        self.anomaly_detection = AnomalyDetector()
        self.performance_metrics = {}

    def monitor_activity(self, neural_activity: Dict[str, Any]):
        """Monitor neural activity"""

        timestamp = time.time()
        activity_record = {
            'timestamp': timestamp,
            'activity': neural_activity,
            'performance_metrics': self._calculate_performance_metrics(neural_activity)
        }

        self.activity_log.append(activity_record)

        # Check for anomalies
        anomalies = self.anomaly_detection.detect_anomalies(activity_record)
        if anomalies:
            activity_record['anomalies'] = anomalies

        return activity_record

    def _calculate_performance_metrics(self, activity: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate neural performance metrics"""

        metrics = {
            'response_time': random.uniform(0.1, 0.5),
            'processing_efficiency': random.uniform(0.7, 0.95),
            'resource_utilization': random.uniform(0.4, 0.8),
            'error_rate': random.uniform(0.01, 0.1),
            'adaptation_rate': random.uniform(0.1, 0.3)
        }

        self.performance_metrics.update(metrics)
        return metrics

    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get neural monitoring report"""
        recent_activity = list(self.activity_log)[-10:] if self.activity_log else []

        return {
            'total_records': len(self.activity_log),
            'recent_activity': recent_activity,
            'performance_metrics': self.performance_metrics,
            'anomaly_count': sum(1 for record in self.activity_log if 'anomalies' in record),
            'monitoring_status': 'active'
        }


class AnomalyDetector:
    """Neural anomaly detection system"""

    def __init__(self):
        self.baseline_patterns = {}
        self.anomaly_threshold = 2.0  # Standard deviations

    def detect_anomalies(self, activity_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect anomalies in neural activity"""

        anomalies = []

        metrics = activity_record.get('performance_metrics', {})

        for metric_name, value in metrics.items():
            if metric_name not in self.baseline_patterns:
                self.baseline_patterns[metric_name] = {
                    'mean': value,
                    'std': 0.1,
                    'count': 1
                }
            else:
                baseline = self.baseline_patterns[metric_name]
                # Update baseline
                old_mean = baseline['mean']
                baseline['count'] += 1
                baseline['mean'] = (old_mean * (baseline['count'] - 1) + value) / baseline['count']
                baseline['std'] = ((baseline['std'] ** 2 * (baseline['count'] - 1)) + ((value - old_mean) ** 2)) ** 0.5 / baseline['count'] ** 0.5

                # Check for anomaly
                z_score = abs(value - baseline['mean']) / max(baseline['std'], 0.01)
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        'metric': metric_name,
                        'value': value,
                        'expected': baseline['mean'],
                        'z_score': z_score,
                        'severity': 'high' if z_score > 3 else 'medium'
                    })

        return anomalies
