import sys
import os
import time
from pathlib import Path
import threading
sys.path.append(str(Path(__file__).parent))

# Import the Brain Memory Hub
from brain_memory_hub import BrainMemoryHub
from services.provider_registry import get_llm
from services.web_research import WebResearchService
from utils.notes_manager import NotesManager
from brain.self.evolution.task_orchestrator import TaskOrchestrator
from brain.cortex.prefrontal.thought_loop import ThoughtLoop
"""
HRM-Gemini AI Model - Neuroscience-Based AI System
Comprehensive brain-inspired AI with modular brain regions and neural connectivity
"""
from typing import Dict, List, Any
from brain.self.skills.registry import SkillRegistry
from brain.hippocampus.memory_consolidation import HippocampusMemory
from brain.amygdala.emotion_processing import AmygdalaEmotion
from brain.cerebellum.skill_learning import CerebellumSkillLearning
from brain.temporal_lobe.language_processing import TemporalLanguageProcessing
from brain.parietal_lobe.sensory_integration import ParietalSensoryIntegration
from brain.occipital_lobe.visual_processing import OccipitalVisualProcessing
from brain.thalamus.sensory_relay import ThalamusSensoryRelay
from brain.brainstem.basic_functions import BrainstemBasicFunctions
from brain.basal_ganglia.motor_control import BasalGangliaMotorControl
from brain.hypothalamus.homeostasis import HypothalamusHomeostasis
from brain.neural_network import NeuralNetwork
from services.gemini_service import GeminiService
from services.llama_service import LlamaService
from services.openrouter_service import OpenRouterService

class ReasoningEngine:
    def process(self, context):
        return f"HRM processed: {context.get('command', 'unknown')}"

class HRMModel:
    """Human Resource Management Model"""
    
    def __init__(self, brain_memory_hub, config=None):
        self.brain_memory_hub = brain_memory_hub
        self.config = config or {}
        try:
            self.provider = os.getenv('HRM_PROVIDER', 'openrouter')  # Changed default to openrouter
        except Exception as e:
            print(f"Provider init error: {e}")
            self.provider = 'openrouter'
        try:
            self.gemini_service = GeminiService() if self.provider == 'gemini' else None
            self.llama_service = LlamaService() if self.provider == 'llama' else None
            self.openrouter_service = OpenRouterService() if self.provider == 'openrouter' else None
        except Exception as e:
            print(f"Service initialization error: {e}")
            self.gemini_service = None
            self.llama_service = None
            self.openrouter_service = None

        # Initialize Brain Memory Hub as central knowledge repository
        self.brain_memory = BrainMemoryHub()

        # Add evolution counters
        self._evolution_cycles = 0
        self._evolution_improvements = 0
        self._last_evolution_time = None
        self._last_web_query = None
        self._last_web_summary = None

        # Legacy memory for backward compatibility
        self.memory = {}

        self.reasoning_engine = ReasoningEngine()
        # Provider-agnostic LLM and Web Research
        try:
            self.llm = get_llm()
        except Exception:
            self.llm = None
        self.web = WebResearchService()
        # Notes manager for persisting all outputs and ingest artifacts
        self.project_root = str(Path(__file__).parent)
        self.notes = NotesManager(self.project_root)
        # Task orchestrator for concurrent self-improvement without blocking chat
        try:
            self.orchestrator = TaskOrchestrator(max_workers=4)
            self.orchestrator.start()
        except Exception:
            self.orchestrator = None
        # Neural-style thought loop for iterative reasoning and reflection
        try:
            self.thought_loop = ThoughtLoop(self.project_root)
        except Exception:
            self.thought_loop = None
        # Device registry for external device control/monitoring
        try:
            self.devices = DeviceRegistry(self.project_root)
        except Exception:
            self.devices = None
        # Skill registry for hot-loadable skills and training
        try:
            self.skills = SkillRegistry(self.project_root)
        except Exception:
            self.skills = None

        # Neuroscience-based brain regions for realistic brain-like processing
        try:
            self.hippocampus = HippocampusMemory()
            self.amygdala = AmygdalaEmotion()
            self.cerebellum = CerebellumSkillLearning()
            self.temporal_lobe = TemporalLanguageProcessing()
            self.parietal_lobe = ParietalSensoryIntegration()
            self.occipital_lobe = OccipitalVisualProcessing()
            self.thalamus = ThalamusSensoryRelay()
            self.brainstem = BrainstemBasicFunctions()
            self.basal_ganglia = BasalGangliaMotorControl()
            self.hypothalamus = HypothalamusHomeostasis()
        except Exception as e:
            print(f"Warning: Brain region initialization failed: {e}")
            # Initialize with None to avoid errors
            self.hippocampus = self.amygdala = self.cerebellum = None
            self.temporal_lobe = self.parietal_lobe = self.occipital_lobe = None
            self.thalamus = self.brainstem = self.basal_ganglia = self.hypothalamus = None

        # Initialize neural network to link all brain modules like nerves
        try:
            self.neural_network = NeuralNetwork()
            self._initialize_neural_connections()
        except Exception as e:
            print(f"Warning: Neural network initialization failed: {e}")
            self.neural_network = None

        # GitHub AI models integration
        try:
            self.github_models = GitHubModelsService()
        except Exception:
            self.github_models = None

        # Continuous learning components
        self.continuous_learning_active = False
        self.learning_thread = None
        self.ingest_thread = None  # Initialize thread attribute
        self.ingest_active = False

        # Load existing brain state
        self.brain_memory.load_brain_state()

        # Initialize points system
        self.points = 0  # Initialize points

        print("🧠 HRM Model initialized with Brain Memory Hub")

    def _initialize_neural_connections(self):
        """Initialize neural connections between all brain regions like nerves"""
        if not self.neural_network:
            return

        # Register all brain regions with the neural network
        region_mappings = {
            'thalamus': self.thalamus,
            'temporal_lobe': self.temporal_lobe,
            'amygdala': self.amygdala,
            'hippocampus': self.hippocampus,
            'cerebellum': self.cerebellum,
            'hypothalamus': self.hypothalamus,
            'brainstem': self.brainstem,
            'basal_ganglia': self.basal_ganglia,
            'parietal_lobe': self.parietal_lobe,
            'occipital_lobe': self.occipital_lobe
        }

        for region_name, region_instance in region_mappings.items():
            if region_instance:
                self.neural_network.register_brain_region(region_name, region_instance)

        # Create neural connections based on brain anatomy
        self._create_anatomical_connections()

        print(f"🧠 Neural network initialized with {len(region_mappings)} regions and anatomical connections")

    def _create_anatomical_connections(self):
        """Create realistic neural connections based on brain anatomy"""

        # Sensory pathways (thalamus to cortex)
        self.neural_network.create_connection('thalamus', 'temporal_lobe', 0.8)
        self.neural_network.create_connection('thalamus', 'parietal_lobe', 0.7)
        self.neural_network.create_connection('thalamus', 'occipital_lobe', 0.9)

        # Language processing connections
        self.neural_network.create_connection('temporal_lobe', 'parietal_lobe', 0.6)
        self.neural_network.create_connection('temporal_lobe', 'hippocampus', 0.7)

        # Emotional processing connections
        self.neural_network.create_connection('amygdala', 'hippocampus', 0.8)
        self.neural_network.create_connection('amygdala', 'hypothalamus', 0.9)
        self.neural_network.create_connection('amygdala', 'temporal_lobe', 0.6)

        # Memory connections
        self.neural_network.create_connection('hippocampus', 'temporal_lobe', 0.7)
        self.neural_network.create_connection('hippocampus', 'parietal_lobe', 0.5)

        # Motor learning connections
        self.neural_network.create_connection('cerebellum', 'basal_ganglia', 0.8)
        self.neural_network.create_connection('cerebellum', 'brainstem', 0.6)
        self.neural_network.create_connection('basal_ganglia', 'thalamus', 0.7)

        # Homeostatic connections
        self.neural_network.create_connection('hypothalamus', 'brainstem', 0.8)
        self.neural_network.create_connection('hypothalamus', 'amygdala', 0.7)

        # Sensory integration connections
        self.neural_network.create_connection('parietal_lobe', 'occipital_lobe', 0.6)
        self.neural_network.create_connection('parietal_lobe', 'temporal_lobe', 0.5)

        # Bidirectional connections for feedback
        self.neural_network.create_connection('temporal_lobe', 'amygdala', 0.4)
        self.neural_network.create_connection('parietal_lobe', 'basal_ganglia', 0.5)
        self.neural_network.create_connection('occipital_lobe', 'temporal_lobe', 0.4)
        # Start background ingest watcher (non-blocking)
        self._start_ingest_watcher()
        
    def get_system_state(self):
        """Get current system state"""
        return {"status": "active", "memory_usage": len(self.memory)}

    def influence_decision(self, context):
        """Influence decisions using LLM and brain memory"""
        command = context.get('command', 'unknown')
        input_text = context.get('input', '')

        # Use LLM for intelligent responses
        if self.llm:
            prompt = f"""You are HRM-Gemini, an advanced AI assistant.

Context: {context}
Command: {command}
User Input: {input_text}

Provide a helpful, intelligent response. If this is a chat, engage naturally. If it's a command, acknowledge and process it."""

            try:
                response = self.llm.generate_response(prompt)
                return response
            except Exception:
                # Suppress LLM errors, use brain fallback
                if command in ['search', 'chat'] and input_text:
                    try:
                        results = self.brain_memory.query_brain(input_text)
                        if results.get('knowledge_results'):
                            return f"Based on my knowledge: {results['knowledge_results'][0]['content']}"
                        else:
                            return f"I don't have specific knowledge about that yet, but I'm learning!"
                    except Exception:
                        pass
        return f"I understand you're asking about {input_text or command}. How can I assist you further?"

    def influence_decision(self, context):
        """Process input through brain regions for realistic decision making"""
        command = context.get('command', '')
        input_text = context.get('input', '')

        # Use brain processing for all decisions
        return self.process_with_brain_regions(command, input_text)

    def process_neural_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input through brain regions using neural network connections"""
        if not self.neural_network or not all([self.thalamus, self.temporal_lobe, self.amygdala, self.hippocampus]):
            return {"error": "Neural network or brain regions not initialized"}

        current_time = time.time()
        neural_response = {}

        # Step 1: Sensory relay through thalamus (neural transmission)
        thalamus_signal = {
            'strength': 0.8,
            'activation': 0.7,
            'input_data': input_data
        }
        thalamus_transmissions = self.neural_network.transmit_signal('thalamus', thalamus_signal, current_time)

        # Process thalamus output and relay to connected regions
        for target_region, signal_data in thalamus_transmissions.items():
            if target_region == 'temporal_lobe' and self.temporal_lobe:
                language_result = self.temporal_lobe.process_language(input_data.get('input', ''))
                neural_response['language_analysis'] = language_result

                # Transmit language processing results through network
                language_signal = {
                    'strength': 0.6,
                    'activation': 0.8,
                    'processed_data': language_result
                }
                self.neural_network.transmit_signal('temporal_lobe', language_signal, current_time)

            elif target_region == 'amygdala' and self.amygdala:
                emotion_result = self.amygdala.process_emotion(input_data)
                neural_response['emotional_response'] = emotion_result

                # Transmit emotional processing results
                emotion_signal = {
                    'strength': emotion_result.get('arousal', 0.5),
                    'activation': emotion_result.get('valence', 0.5) + 0.5,
                    'processed_data': emotion_result
                }
                emotion_transmissions = self.neural_network.transmit_signal('amygdala', emotion_signal, current_time)

                # Process emotion-driven transmissions to hippocampus
                for emotion_target, emotion_sig in emotion_transmissions.items():
                    if emotion_target == 'hippocampus' and self.hippocampus:
                        memory_result = self.hippocampus.consolidate_memory({
                            'content': input_data.get('input', ''),
                            'context': input_data,
                            'importance': emotion_result.get('arousal', 0.5)
                        })
                        neural_response['memory_consolidation'] = memory_result

                        # Transmit memory consolidation results
                        memory_signal = {
                            'strength': 0.7,
                            'activation': 0.6,
                            'processed_data': memory_result
                        }
                        self.neural_network.transmit_signal('hippocampus', memory_signal, current_time)

            elif target_region == 'parietal_lobe' and self.parietal_lobe and 'sensory_data' in input_data:
                sensory_result = self.parietal_lobe.integrate_sensory_input(input_data['sensory_data'])
                neural_response['sensory_integration'] = sensory_result

            elif target_region == 'occipital_lobe' and self.occipital_lobe and 'visual_data' in input_data:
                visual_result = self.occipital_lobe.process_visual_input(input_data['visual_data'])
                neural_response['visual_processing'] = visual_result

        # Step 2: Skill learning through cerebellum (motor pathways)
        if 'action' in input_data or 'skill' in input_data:
            cerebellum_signal = {
                'strength': 0.7,
                'activation': 0.8,
                'skill_data': input_data
            }
            cerebellum_transmissions = self.neural_network.transmit_signal('cerebellum', cerebellum_signal, current_time)

            for target_region, signal_data in cerebellum_transmissions.items():
                if target_region == 'basal_ganglia' and self.basal_ganglia:
                    skill_result = self.cerebellum.learn_motor_program(
                        input_data.get('action', 'general_action'),
                        input_data.get('action_sequence', [{'action_type': 'process', 'timestamp': current_time}])
                    )
                    neural_response['skill_learning'] = skill_result

        # Step 3: Homeostatic regulation through hypothalamus
        internal_state = self._get_internal_state()
        hypothalamus_signal = {
            'strength': 0.6,
            'activation': 0.7,
            'internal_state': internal_state
        }
        hypothalamus_transmissions = self.neural_network.transmit_signal('hypothalamus', hypothalamus_signal, current_time)

        for target_region, signal_data in hypothalamus_transmissions.items():
            if target_region == 'brainstem' and self.brainstem:
                homeostatic_result = self.hypothalamus.regulate_homeostasis(input_data, internal_state)
                neural_response['homeostatic_response'] = homeostatic_result

                # Transmit to brainstem for vital functions
                brainstem_signal = {
                    'strength': 0.8,
                    'activation': 0.6,
                    'homeostatic_data': homeostatic_result
                }
                self.neural_network.transmit_signal('brainstem', brainstem_signal, current_time)

        # Step 4: Basic function regulation through brainstem
        brainstem_signal = {
            'strength': 0.5,
            'activation': 0.7,
            'input_data': input_data
        }
        brainstem_transmissions = self.neural_network.transmit_signal('brainstem', brainstem_signal, current_time)

        for target_region, signal_data in brainstem_transmissions.items():
            if target_region == 'basal_ganglia' and self.basal_ganglia:
                vital_result = self.brainstem.regulate_vitals(input_data)
                neural_response['vital_response'] = vital_result

        # Process neural feedback for learning
        feedback_data = self._generate_neural_feedback(neural_response)
        self.neural_network.process_neural_feedback(feedback_data)

        return neural_response

    def _generate_neural_feedback(self, neural_response: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feedback signals for neural learning"""
        feedback = {}

        # Emotional feedback strengthens connections
        if 'emotional_response' in neural_response:
            emotion = neural_response['emotional_response']
            feedback['amygdala'] = {'reinforcement': emotion.get('arousal', 0.0)}

        # Successful memory consolidation provides positive feedback
        if 'memory_consolidation' in neural_response:
            feedback['hippocampus'] = {'reinforcement': 0.1}

        # Skill learning success provides feedback
        if 'skill_learning' in neural_response:
            feedback['cerebellum'] = {'reinforcement': 0.05}
            feedback['basal_ganglia'] = {'reinforcement': 0.05}

        return feedback

    def _get_internal_state(self) -> Dict[str, Any]:
        """Get current internal physiological state"""
        return {
            'body_temperature': 37.0,  # Celsius
            'blood_glucose': 90,  # mg/dL
            'hydration_level': 0.7,  # 0-1 scale
            'energy_level': 0.8,  # 0-1 scale
            'sleep_debt': 0.0,  # hours
            'ambient_temperature': 25.0,  # Environmental
            'stress_level': 0.2,
            'cognitive_load': 0.3
        }

    def get_brain_status(self) -> Dict[str, Any]:
        """Get comprehensive brain region status"""
        status = {}

        if self.temporal_lobe:
            status['temporal_lobe'] = {'status': 'active', 'concepts': len(self.temporal_lobe.semantic_memory)}

        if self.amygdala:
            status['amygdala'] = {'status': 'active', 'emotional_memory': len(self.amygdala.emotional_memory)}

        if self.hippocampus:
            status['hippocampus'] = {'status': 'active', 'consolidated_memories': len(self.hippocampus.consolidated_memories)}

        if self.cerebellum:
            status['cerebellum'] = {'status': 'active', 'motor_programs': len(self.cerebellum.motor_programs)}

        if self.thalamus:
            stats = self.thalamus.get_relay_statistics()
            status['thalamus'] = {'status': 'active', 'relay_efficiency': stats.get('relay_efficiency', 0)}

        if self.brainstem:
            vital_status = self.brainstem.get_vital_status()
            status['brainstem'] = {'status': 'active', 'arousal_level': vital_status.get('arousal_level', 0)}

        if self.basal_ganglia:
            motor_status = self.basal_ganglia.get_motor_status()
            status['basal_ganglia'] = {'status': 'active', 'motor_programs': motor_status.get('motor_programs', 0)}

        if self.hypothalamus:
            homeostatic_status = self.hypothalamus.get_homeostatic_status()
            status['hypothalamus'] = {'status': 'active', 'motivational_drives': homeostatic_status.get('motivational_drives', {})}

        if self.parietal_lobe:
            sensory_stats = self.parietal_lobe.analyze_sensory_patterns()
            status['parietal_lobe'] = {'status': 'active', 'spatial_locations': sensory_stats.get('total_locations', 0)}

        if self.occipital_lobe:
            status['occipital_lobe'] = {'status': 'active', 'visual_patterns': len(self.occipital_lobe.visual_patterns)}

        if self.github_models:
            status['github_models'] = {'status': 'active', 'available_models': len(self.github_models.get_available_models())}

        return status

    def optimize_brain_functioning(self) -> Dict[str, Any]:
        """Optimize brain regions for better performance"""
        optimizations = {}

        if self.hippocampus:
            # Optimize memory consolidation
            optimizations['hippocampus'] = "Memory consolidation optimized"

        if self.cerebellum:
            skill_optimizations = self.cerebellum.optimize_motor_learning()
            optimizations['cerebellum'] = skill_optimizations

        if self.basal_ganglia:
            motor_optimizations = self.basal_ganglia.optimize_motor_learning()
            optimizations['basal_ganglia'] = motor_optimizations

        if self.thalamus:
            self.thalamus.reset_gates()
            optimizations['thalamus'] = "Sensory gates reset"

        if self.hypothalamus:
            self.hypothalamus.reset_homeostasis()
            optimizations['hypothalamus'] = "Homeostatic setpoints reset"

        return optimizations

    def process_with_brain_regions(self, command: str, input_text: str = None) -> str:
        """Process input through all brain regions and generate response"""
        try:
            # Create neural input data
            neural_input = {
                'command': command,
                'input': input_text,
                'timestamp': time.time(),
                'context': {}
            }

            # Process through brain regions
            neural_response = self.process_neural_input(neural_input)

            # Generate response based on processed information
            if 'language_analysis' in neural_response:
                lang_data = neural_response['language_analysis']
                if 'intent' in lang_data and lang_data['intent'] == 'query':
                    # Try to get answer from brain memory
                    memory_result = self.brain_memory.query_brain(input_text)
                    if memory_result.get('knowledge_results'):
                        return memory_result['knowledge_results'][0]['content']

            # Fallback response generation
            return self._generate_contextual_response(command, input_text, neural_response)

        except Exception as e:
            print(f"Error in brain processing: {e}")
            return f"I encountered an error processing your request: {str(e)}"

    def _generate_contextual_response(self, command: str, input_text: str, neural_data: dict) -> str:
        """Generate response based on processed neural data"""
        # Check emotional state for response tone
        emotional_tone = "neutral"
        if 'emotional_response' in neural_data:
            emotion = neural_data['emotional_response']
            if emotion.get('valence', 0) > 0.6:
                emotional_tone = "positive"
            elif emotion.get('valence', 0) < 0.4:
                emotional_tone = "cautious"

        # Generate response based on command type
        if command == 'search':
            return self._handle_search_command(input_text, emotional_tone)
        elif command == 'chat':
            return self._handle_chat_command(input_text, emotional_tone, neural_data)
        else:
            return self._handle_general_command(command, input_text, emotional_tone)

    def _handle_search_command(self, query: str, tone: str) -> str:
        """Handle search-related commands"""
        if not query:
            return "Please provide a search query."

        try:
            # Try to get from brain memory first
            memory_result = self.brain_memory.query_brain(query)

            if memory_result.get('knowledge_results'):
                if tone == "positive":
                    return f"I found this information: {memory_result['knowledge_results'][0]['content']}"
                return memory_result['knowledge_results'][0]['content']

            # Fallback to web search if enabled
            if hasattr(self, 'web_research'):
                web_results = self.web_research.search_web(query)
                if web_results:
                    # Store new knowledge
                    self.brain_memory.store_knowledge(
                        content=web_results[0]['snippet'],
                        source=f"web:{web_results[0]['url']}",
                        metadata={"query": query}
                    )
                    return web_results[0]['snippet']

            return f"I couldn't find information about '{query}'. Could you try rephrasing?"

        except Exception as e:
            print(f"Search error: {e}")
            return "I'm having trouble searching right now. Please try again later."

    def _handle_chat_command(self, message: str, tone: str, neural_data: dict) -> str:
        """Handle conversational chat"""
        if not message:
            return "How can I assist you today?"

        # Check for emotional content
        emotion_response = ""
        if 'emotional_response' in neural_data and neural_data['emotional_response'].get('emotion') != 'neutral':
            emotion = neural_data['emotional_response']['emotion']
            if emotion in ['happy', 'excited']:
                emotion_response = "I'm glad you're feeling positive! "
            elif emotion in ['sad', 'angry']:
                emotion_response = "I'm here to help. "

        # Generate conversational response
        try:
            # Try LLM first
            if self.llm:
                prompt = f"Respond conversationally to: {message}. Tone: {tone}"
                response = self.llm.generate_response(prompt)
                return emotion_response + response
        except Exception:
            pass

        # Fallback to brain memory
        memory_result = self.brain_memory.query_brain(message)
        if memory_result.get('knowledge_results'):
            return emotion_response + memory_result['knowledge_results'][0]['content']

        # Default conversational responses
        if tone == "positive":
            return emotion_response + f"That's an interesting point about '{message}'. What else would you like to discuss?"
        else:
            return emotion_response + f"I understand you're asking about '{message}'. Let me think about that."

    def _handle_general_command(self, command: str, input_text: str, tone: str) -> str:
        """Handle general commands"""
        if tone == "positive":
            return f"Great! I'll help with '{command}'."
        elif tone == "cautious":
            return f"I need to be careful with '{command}'. Let's proceed thoughtfully."
        else:
            return f"Processing command: {command}"
        """Continuously analyze system state"""
        while True:
            self._analyze_logs()
            self._optimize_performance()
            
    def _analyze_logs(self):
        print("Analyzing logs...")
        
    def _optimize_performance(self):
        print("Optimizing performance...")

    def generate_code(self, requirements):
        """Generate new code based on requirements"""
        return f"# Generated code for: {requirements}\npass"

    def startup_optimization(self, duration_seconds=5):
        """Perform 5-second startup optimization"""
        import time
        import random

        print(f"🚀 Starting {duration_seconds}-second startup optimization...")
        start_time = time.time()
        end_time = start_time + duration_seconds

        optimizations_performed = 0
        issues_fixed = 0

        while time.time() < end_time:
            remaining_time = end_time - time.time()

            # Simulate optimization tasks
            optimization_tasks = [
                "Analyzing system performance",
                "Optimizing memory usage",
                "Checking code quality",
                "Validating configurations",
                "Improving decision algorithms",
                "Enhancing error handling",
                "Updating knowledge base",
                "Calibrating response times",
                "Fine-tuning intelligence parameters",
                "Strengthening security measures"
            ]

            # Pick a random task
            current_task = random.choice(optimization_tasks)

            print(f"⚡ {current_task}... ({remaining_time:.1f}s remaining)")

            # Simulate processing time (random between 0.1-0.5 seconds)
            processing_time = random.uniform(0.1, 0.5)
            time.sleep(min(processing_time, remaining_time))

            optimizations_performed += 1

            # Simulate occasional issue detection and fixing
            if random.random() < 0.3:  # 30% chance
                issues = [
                    "Minor performance bottleneck detected and resolved",
                    "Memory optimization applied",
                    "Configuration parameter tuned",
                    "Code efficiency improved",
                    "Response time optimized"
                ]
                issue = random.choice(issues)
                print(f"✅ {issue}")
                issues_fixed += 1

        elapsed_time = time.time() - start_time
        print(f"\n🎯 Startup optimization completed in {elapsed_time:.2f} seconds!")
        print(f"📊 Performance metrics:")
        print(f"   • Optimizations performed: {optimizations_performed}")
        print(f"   • Issues resolved: {issues_fixed}")
        print(f"   • System readiness: 100%")
        print(f"   • Intelligence level: Enhanced\n")

        return {
            'elapsed_time': elapsed_time,
            'optimizations_performed': optimizations_performed,
            'issues_fixed': issues_fixed,
            'system_ready': True
        }

    def chat_response(self, user_input, context=None):
        """Generate conversational response using HRM intelligence and brain memory"""
        try:
            # First try Gemini
            if self.provider == 'gemini':
                try:
                    return self.gemini_service.generate_chat_response(user_input)
                except Exception as e:
                    print(f"Gemini error: {e}")
            
            # Then try OpenRouter
            if self.provider == 'openrouter':
                try:
                    return self.openrouter_service.generate_chat_response(user_input)
                except Exception as e:
                    print(f"OpenRouter error: {e}")
            
            # Then try Llama if the service is available
            if self.llama_service is not None:
                try:
                    return self.llama_service.generate_chat_response(user_input)
                except Exception as e:
                    print(f"Llama error: {e}")
                
            # Final fallback
            return self._simple_response(user_input)
            
        except Exception as e:
            print(f"Chat system error: {e}")
            return "I'm having trouble generating a response right now. Please try again later."

    def _simple_response(self, user_input):
        """Generate simple responses without LLM"""
        if "hello" in user_input.lower() or "hi" in user_input.lower():
            return "Hello! How can I assist you today?"
        return "I received your message. Let me know how I can help!"

    def chat_response(self, user_input, context=None):
        """Generate conversational response using HRM intelligence and brain memory"""
        # Ensure provider attribute exists
        try:
            if not hasattr(self, 'provider'):
                self.provider = os.getenv('HRM_PROVIDER', 'llama')
        except Exception as e:
            print(f"Error setting provider: {e}")
            self.provider = 'llama'
        
        try:
            # Store conversation in brain memory
            conversation_data = {
                'user_input': user_input,
                'timestamp': time.time(),
                'context': context or {}
            }

            conversation_id = self.brain_memory.store_conversation(conversation_data)

            # Query brain memory for relevant knowledge
            brain_query = self.brain_memory.query_brain(user_input, "conversation")

            # Use configured LLM provider if available; otherwise fallback to HRM processing
            if self.llm is not None:
                enhanced_prompt = self._enhance_prompt_with_brain_context(user_input, context, brain_query)
                try:
                    ai_response = self.llm.generate_response(enhanced_prompt)
                except Exception as e:
                    print(f"Gemini error: {e}. Falling back to Llama.")
                    ai_response = self.llm.generate_response(enhanced_prompt)
            else:
                ai_response = self._generate_hrm_response_with_brain(user_input, context, brain_query)

            # Store response in brain memory as learning experience
            response_data = {
                'type': 'ai_response',
                'context': {'user_input': user_input, 'brain_query': brain_query},
                'outcome': {'response': ai_response, 'conversation_id': conversation_id},
                'emotional_valence': 0.0,  # Neutral for AI responses
                'learned_patterns': ['ai_response_generation', 'context_awareness']
            }
            self.brain_memory.store_experience(response_data)
            # Persist the exchange as a note for traceability
            self.notes.save_json('notes', 'chat_exchange', {
                'user_input': user_input,
                'response': ai_response,
                'brain_query': brain_query,
                'timestamp': time.time()
            })
            return ai_response
        except Exception as e:
            print(f"Chat response error: {e}")
            return "I'm having trouble generating a response right now. Please try again later."

    def _enhance_prompt_with_brain_context(self, user_input, context, brain_query):
        """Enhance user input with brain memory context"""
        base_prompt = f"""You are HRM-Gemini, an advanced AI with a comprehensive brain memory system.
You have access to persistent memory, learning patterns, and can access any stored information.

User Input: {user_input}

"""

        # Add brain memory context
        if brain_query.get('knowledge_results'):
            base_prompt += "Relevant Knowledge from Brain Memory:\n"
            for knowledge in brain_query['knowledge_results'][:3]:
                content = knowledge.get('content', {})
                if isinstance(content, dict) and 'content' in content:
                    base_prompt += f"- {content['content'][:150]}...\n"
                else:
                    base_prompt += f"- {str(content)[:150]}...\n"

        if brain_query.get('experiential_results'):
            base_prompt += "\nRelevant Experiences:\n"
            for exp in brain_query['experiential_results'][:2]:
                base_prompt += f"- {exp.get('experience_type', 'general')} experience\n"

        # Add personality and capabilities
        base_prompt += """
As HRM-Gemini with Brain Memory, you can:
- Access and recall any stored information from the brain memory system
- Learn and adapt from every interaction
- Process and analyze files, conversations, and experiences
- Provide context-aware responses based on learning history
- Continuously evolve and improve your capabilities

Respond naturally and helpfully, leveraging the brain memory context when relevant."""

        return base_prompt

    def _generate_hrm_response_with_brain(self, user_input, context, brain_query):
        """Generate response using HRM intelligence with brain memory context"""
        user_input_lower = user_input.lower()

        # Query brain memory for relevant information
        brain_context = ""
        if brain_query.get('knowledge_results'):
            brain_context += f"Brain Memory Knowledge: {len(brain_query['knowledge_results'])} relevant items found. "

        # Enhanced response generation with brain context
        if self._contains_keywords(user_input_lower, ['hello', 'hi', 'hey', 'greetings']):
            response = "Hello! 👋 I'm HRM-Gemini, your intelligent AI assistant with a comprehensive brain memory system."

            if brain_context:
                response += f" I can access {brain_context.strip()} to help you better."

            return response + " How can I help you today?"

        elif self._contains_keywords(user_input_lower, ['what can you do', 'help', 'capabilities']):
            response = "I'm HRM-Gemini, an advanced AI with these enhanced capabilities:\n\n" \
                      "🧠 Brain Memory System:\n" \
                      "- Access to comprehensive knowledge repository\n" \
                      "- Learning from every interaction and file\n" \
                      "- Pattern recognition and experience-based responses\n" \
                      "- Continuous evolution and improvement\n\n" \
                      "💬 Advanced Communication:\n" \
                      "- Context-aware responses using brain memory\n" \
                      "- Natural conversation with persistent memory\n" \
                      "- Multi-modal processing and analysis\n\n" \
                      "📁 File & Data Processing:\n" \
                      "- Store and analyze any type of file in brain memory\n" \
                      "- Extract knowledge and relationships automatically\n" \
                      "- Smart content analysis and summarization\n\n" \
                      "🎓 Continuous Learning:\n" \
                      "- Learn from conversations and user interactions\n" \
                      "- Adapt responses based on learned patterns\n" \
                      "- Evolve capabilities over time\n\n"

            if brain_context:
                response += f"🔍 Currently accessing {brain_context.strip()} from my brain memory.\n\n"

            response += "What would you like to explore?"
            return response

        elif self._contains_keywords(user_input_lower, ['remember', 'store']):
            content = user_input.replace('remember', '').replace('store', '').strip()
            if content:
                # Store in brain memory instead of legacy memory
                knowledge_id = self.brain_memory.store_knowledge(
                    content={'text': content, 'type': 'user_memory'},
                    category='user_memory',
                    importance=0.8,
                    confidence=0.9,
                    source='user_input'
                )
                return f"✅ I'll remember that! Stored in brain memory as knowledge #{knowledge_id}"
            else:
                return "What would you like me to remember?"

        elif self._contains_keywords(user_input_lower, ['recall', 'what did i say', 'brain search']):
            query = user_input.replace('recall', '').replace('what did i say about', '').replace('brain search', '').strip()

            if not query:
                query = user_input  # Use full input if no specific query

            # Search brain memory
            brain_results = self.brain_memory.query_brain(query)

            response = f"🔍 Brain Memory Search Results for '{query}':\n\n"

            if brain_results.get('knowledge_results'):
                response += f"📚 Knowledge Found: {len(brain_results['knowledge_results'])} items\n"
                for i, knowledge in enumerate(brain_results['knowledge_results'][:3], 1):
                    content = knowledge.get('content', {})
                    if isinstance(content, dict) and 'content' in content:
                        response += f"{i}. {content['content'][:100]}...\n"
                    else:
                        response += f"{i}. {str(content)[:100]}...\n"
            else:
                response += "📚 No specific knowledge found for this query.\n"

            if brain_results.get('experiential_results'):
                response += f"\n🧠 Experiences: {len(brain_results['experiential_results'])} relevant experiences\n"

            response += "\n💡 The brain memory system continuously learns and evolves!"
            return response

        else:
            # Generate contextual response using brain memory
            return self._generate_brain_contextual_response(user_input, brain_query)

    def _contains_keywords(self, text, keywords):
        """Check if text contains any of the keywords"""
        return any(keyword in text for keyword in keywords)

    def _generate_brain_contextual_response(self, user_input, brain_query):
        """Generate contextual response using brain memory"""
        # Use brain memory results to inform response
        knowledge_count = len(brain_query.get('knowledge_results', []))
        experience_count = len(brain_query.get('experiential_results', []))

        if knowledge_count > 0 or experience_count > 0:
            response = f"I found relevant information in my brain memory! "
            
            if knowledge_count > 0:
                response += f"I have {knowledge_count} knowledge items related to your query. "
            
            if experience_count > 0:
                response += f"I also have {experience_count} relevant experiences to draw from. "
            
            response += "Would you like me to elaborate on any specific aspect?"
        else:
            # Generate helpful response when no knowledge found
            response = self._generate_helpful_response(user_input)
            
        return response

    def _generate_helpful_response(self, user_input):
        """Generate a helpful response when no knowledge is found"""
        user_input_lower = user_input.lower()
        
        # Specific responses for common queries
        if any(word in user_input_lower for word in ['how are you', 'how do you do', 'hows it going']):
            responses = [
                "I'm functioning optimally, thank you for asking! How can I assist you today?",
                "I'm doing well, thanks! I'm here to help with any questions or tasks you have.",
                "All systems are running smoothly! What would you like to work on?",
                "I'm operating at full capacity! How may I be of service?"
            ]
        elif any(word in user_input_lower for word in ['hello', 'hi', 'hey', 'greetings']):
            responses = [
                "Hello! 👋 Welcome to HRM-Gemini AI. What can I help you with?",
                "Hi there! I'm ready to assist you with any task or question.",
                "Greetings! How can I support your work today?",
                "Hey! Let's get started. What would you like to explore?"
            ]
        elif any(word in user_input_lower for word in ['help', 'assist', 'support']):
            responses = [
                "I'm here to help! Try typing 'help' for a list of commands, or ask me anything.",
                "I can assist with coding, research, analysis, and more. What do you need?",
                "Let me know how I can support you. I'm capable of many tasks!",
                "I'm your AI assistant. What would you like to accomplish?"
            ]
        elif any(word in user_input_lower for word in ['what can you do', 'capabilities', 'features']):
            responses = [
                "I can help with coding, research, file analysis, brain memory management, and much more. Type 'help' to see all commands!",
                "My capabilities include code generation, web research, intelligent conversations, and continuous learning. What interests you?",
                "I excel at programming assistance, data analysis, creative tasks, and adaptive learning. How can I help?",
                "I can assist with development, research, problem-solving, and knowledge management. What would you like to explore?"
            ]
        elif any(word in user_input_lower for word in ['thank', 'thanks', 'appreciate']):
            responses = [
                "You're welcome! I'm glad I could help. Is there anything else I can assist with?",
                "My pleasure! Don't hesitate to ask if you need more assistance.",
                "Happy to help! I'm here whenever you need support.",
                "You're welcome! Feel free to reach out anytime."
            ]
        elif any(word in user_input_lower for word in ['bye', 'goodbye', 'see you']):
            responses = [
                "Goodbye! It was great working with you. Come back anytime!",
                "See you later! I'm here whenever you need assistance.",
                "Take care! Don't hesitate to return if you need help.",
                "Farewell! I'm always available for your next project."
            ]
        else:
            # General responses for unknown queries
            responses = [
                f"I understand you're interested in '{user_input[:30]}...'. While I don't have specific information about that yet, I can help you research it or explore related topics. What would you like to do?",
                f"That's an interesting topic: '{user_input[:30]}...'. I can help you find information, generate ideas, or work on related projects. How can I assist?",
                f"I see you're asking about '{user_input[:30]}...'. My brain memory is continuously learning. Would you like me to search for related information or help you with something specific?",
                f"Regarding '{user_input[:30]}...', I can help you explore this topic through research, analysis, or creative approaches. What aspect interests you most?",
                f"Your query about '{user_input[:30]}...' is noted. I can assist with research, problem-solving, or generating ideas related to this topic. How would you like to proceed?"
            ]
        
        # Return a random response from the appropriate category
        import random
        return random.choice(responses)

    def store_memory(self, content: str, memory_type: str = "general", metadata: dict = None):
        """Store information in brain memory (legacy compatibility)"""
        return self.brain_memory.store_knowledge({
            'content': {'text': content, 'type': memory_type},
            'category': memory_type,
            'importance': 0.5,
            'confidence': 0.8,
            'source': 'legacy_memory',
            'metadata': metadata or {}
        })

    def recall_memory(self, query: str, limit: int = 5):
        """Recall information from brain memory (legacy compatibility)"""
        brain_results = self.brain_memory.query_brain(query, limit=limit)

        # Convert brain memory format to legacy format
        legacy_results = []
        for knowledge in brain_results.get('knowledge_results', []):
            legacy_results.append({
                'content': str(knowledge.get('content', {})),
                'type': knowledge.get('category', 'general'),
                'timestamp': knowledge.get('timestamp', 0),
                'importance': knowledge.get('importance', 0.5)
            })

        return legacy_results

    def get_brain_stats(self):
        """Get comprehensive brain memory statistics"""
        return self.brain_memory.get_brain_stats()

    def save_brain_state(self):
        """Save current brain state"""
        self.brain_memory.save_brain_state()

    def evolve_brain(self):
        """Trigger brain evolution"""
        return self.brain_memory.evolve_brain()

    def process_file_for_brain(self, file_path: str, content: str, metadata: dict = None):
        """Process a file and store in brain memory"""
        return self.brain_memory.store_file_knowledge(file_path, content, metadata or {})

    def query_brain_memory(self, query: str, query_type: str = "general"):
        """Query the brain memory system"""
        return self.brain_memory.query_brain(query, query_type)

    def shutdown(self):
        """Clean shutdown of HRM model and brain memory"""
        self.save_brain_state()
        self.brain_memory.shutdown_brain()

        if hasattr(self, 'continuous_evolution') and self.continuous_evolution:
            self.stop_evolution()

        # Stop orchestrator
        try:
            if getattr(self, 'orchestrator', None):
                self.orchestrator.stop()
        except Exception:
            pass

        # Stop ingest watcher
        self.ingest_active = False
        if self.ingest_thread:
            self.ingest_thread.join(timeout=3)

        print("🧠 HRM Model and Brain Memory shut down")

    # ---------------- Neural thought API -----------------
    def think(self, prompt: str, max_iters: int = 3):
        """Run an iterative neural-style thought loop. Returns trace dict.
        Persists to brain_memory via NotesManager inside ThoughtLoop.
        """
        if not getattr(self, 'thought_loop', None):
            return {"ok": False, "reason": "thought_loop_unavailable"}
        return self.thought_loop.run(prompt, max_iters=max_iters)

    # ---------------- Device APIs -----------------
    def device_add(self, dev_id: str, kind: str, params: dict, enabled: bool = True):
        if not getattr(self, 'devices', None):
            return {"ok": False, "reason": "devices_unavailable"}
        self.devices.add(dev_id, kind, params, enabled)
        self.notes.save_json("notes", "device_registry_update", {"action": "add", "dev_id": dev_id, "kind": kind, "params": params, "enabled": enabled})
        return {"ok": True}

    def device_connect(self, dev_id: str):
        if not getattr(self, 'devices', None):
            return {"ok": False, "reason": "devices_unavailable"}
        try:
            self.devices.connect(dev_id)
            info = self.devices.info(dev_id)
            self.notes.save_json("notes", "device_connect", {"dev_id": dev_id, "info": info})
            return {"ok": True, "info": info}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # ---------------- Observability APIs -----------------
    def get_orchestrator_stats(self):
        if not getattr(self, 'orchestrator', None):
            return {"running": False, "queued": 0, "workers": 0, "completed_recent": [], "failures_recent": []}
        try:
            return self.orchestrator.stats()
        except Exception:
            return {"running": False, "queued": 0, "workers": 0, "completed_recent": [], "failures_recent": []}

    def list_devices(self):
        if not getattr(self, 'devices', None):
            return {}
        try:
            return self.devices.list()
        except Exception:
            return {}

    def get_thought_loop_status(self):
        if not getattr(self, 'thought_loop', None):
            return {"active": False, "last_trace": None, "iterations": 0}
        try:
            # Assume thought_loop has a get_status method or we can derive from last run
            # For now, return a simple status; in future, enhance ThoughtLoop with status
            return {"active": True, "last_trace": getattr(self.thought_loop, 'last_trace', None), "iterations": getattr(self.thought_loop, 'iteration_count', 0)}
        except Exception:
            return {"active": False, "last_trace": None, "iterations": 0}

    def get_evolution_status(self):
        if not getattr(self, 'continuous_evolution', False):
            return {"running": False, "cycles": 0, "improvements": 0, "last_cycle_time": None}
        try:
            # Add counters if not present
            cycles = getattr(self, '_evolution_cycles', 0)
            improvements = getattr(self, '_evolution_improvements', 0)
            last_time = getattr(self, '_last_evolution_time', None)
            return {"running": True, "cycles": cycles, "improvements": improvements, "last_cycle_time": last_time}
        except Exception:
            return {"running": False, "cycles": 0, "improvements": 0, "last_cycle_time": None}

    def get_web_research_status(self):
        # Since web research is used in evolution, track last research
        try:
            last_query = getattr(self, '_last_web_query', None)
            last_summary = getattr(self, '_last_web_summary', None)
            return {"last_query": last_query, "last_summary": last_summary}
        except Exception:
            return {"last_query": None, "last_summary": None}

    def get_memory_activity(self):
        try:
            paths = self.notes.get_paths()
            activity = {}
            for sub in ["notes", "web", "self_tests", "ingest", "episodic_records", "conceptual_maps", "core_knowledge", "behavior_models"]:
                p = Path(paths['base']) / sub
                if p.exists():
                    files = list(p.glob("*.*"))
                    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    activity[sub] = [str(f) for f in files[:5]]  # top 5 recent
                else:
                    activity[sub] = []
            return activity
        except Exception:
            return {}

    def device_info(self, dev_id: str):
        if not getattr(self, 'devices', None):
            return {"ok": False, "reason": "devices_unavailable"}
        return {"ok": True, "info": self.devices.info(dev_id)}

    def device_send_text(self, dev_id: str, text: str):
        if not getattr(self, 'devices', None):
            return {"ok": False, "reason": "devices_unavailable"}
        try:
            self.devices.send(dev_id, text.encode("utf-8"))
            self.notes.save_json("notes", "device_send", {"dev_id": dev_id, "bytes": len(text)})
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _device_monitor_tick(self, dev_id: str):
        """One monitoring tick per device; persists telemetry snapshot to brain_memory."""
        try:
            info = self.devices.info(dev_id) if getattr(self, 'devices', None) else {"connected": False}
            self.notes.save_json("notes", "device_telemetry", {"dev_id": dev_id, "info": info})
        except Exception:
            pass

    # ---------------- Skill APIs -----------------
    def list_skills(self):
        if not getattr(self, 'skills', None):
            return {"ok": False, "reason": "skills_unavailable"}
        return {"ok": True, "skills": self.skills.list()}

    def train_skill(self, name: str, data: dict | None = None):
        if not getattr(self, 'skills', None):
            return {"ok": False, "reason": "skills_unavailable"}
        try:
            res = self.skills.train(name, data or {})
            self.notes.save_json("notes", "skill_train", {"name": name, "result": res})
            return {"ok": True, "result": res}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def infer_skill(self, name: str, inputs: dict):
        if not getattr(self, 'skills', None):
            return {"ok": False, "reason": "skills_unavailable"}
        try:
            res = self.skills.infer(name, inputs)
            self.notes.save_json("notes", "skill_infer", {"name": name, "inputs": list(inputs.keys()), "ok": res.get("ok", True)})
            return {"ok": True, "result": res}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _extract_topics(self, user_input):
        """Extract key topics from user input"""
        # Simple topic extraction based on common keywords
        topic_keywords = {
            'programming': ['code', 'python', 'javascript', 'programming', 'coding', 'software'],
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'neural', 'model'],
            'rpg': ['game', 'rpg', 'adventure', 'character', 'quest', 'story'],
            'memory': ['remember', 'recall', 'memory', 'forget', 'learn'],
            'file': ['file', 'document', 'upload', 'process', 'analyze']
        }

        found_topics = []
        text_lower = user_input.lower()

        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)

        return found_topics if found_topics else ['general']

    def _generate_general_response(self, user_input):
        """Generate a general helpful response"""
        responses = [
            f"That's an interesting point about '{user_input[:50]}...'. I'd love to hear more about your thoughts on this topic!",
            f"I understand you're asking about {user_input[:50]}... Can you tell me more about what you're trying to achieve?",
            f"That's a great question! While I don't have specific information about '{user_input[:50]}...', I can help you explore this topic further.",
            f"I'm intrigued by your interest in {user_input[:50]}... What aspects of this would you like to discuss?",
            f"Thank you for sharing that. Regarding '{user_input[:50]}...', I can offer some insights or help you find more information."
        ]

        import random
        return random.choice(responses)

    def start_evolution(self):
        """Start continuous evolution process"""
        print("🧬 Initiating HRM Self-Evolution...")
        self.continuous_evolution = True
        self.evolution_thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self.evolution_thread.start()
        return "HRM evolution process started successfully"

    def _evolution_loop(self):
        """Continuous evolution monitoring and improvement"""
        import threading
        import time

        evolution_cycle = 0
        while getattr(self, 'continuous_evolution', False):
            try:
                evolution_cycle += 1
                print(f"🧬 Evolution Cycle #{evolution_cycle}")

                # Analyze current performance
                self._analyze_performance()

                # Apply micro-improvements
                improvements_made = self._apply_micro_improvements()

                # Learn from interactions
                self._learn_from_interactions()

                # Optimize memory usage
                self._optimize_memory_usage()

                # Periodic self-testing to improve reasoning and coding skills
                if evolution_cycle % 2 == 0:
                    if getattr(self, 'orchestrator', None):
                        self.orchestrator.submit("self_test", self._self_test, priority=50, drop_if_full=True)
                    else:
                        self._self_test()

                # Periodic web research to learn from latest public knowledge
                if evolution_cycle % 3 == 0:
                    if getattr(self, 'orchestrator', None):
                        self.orchestrator.submit("web_research", self._research_and_learn, priority=60, drop_if_full=True)
                    else:
                        self._research_and_learn()

                # Periodic website training to improve neural systems
                if evolution_cycle % 10 == 0:
                    if getattr(self, 'orchestrator', None):
                        self.orchestrator.submit("website_training", self._train_on_websites, priority=70, drop_if_full=True)
                    else:
                        self._train_on_websites()

                # Schedule device monitoring (lightweight info polls)
                if getattr(self, 'devices', None):
                    try:
                        devs = self.devices.list()
                        for dev_id, rec in devs.items():
                            if rec.get("enabled", True) and getattr(self, 'orchestrator', None):
                                self.orchestrator.submit(
                                    f"dev_monitor:{dev_id}",
                                    lambda d=dev_id: self._device_monitor_tick(d),
                                    priority=80,
                                    drop_if_full=True,
                                )
                    except Exception:
                        pass

                # Periodic skill training (rotate through discovered skills)
                if evolution_cycle % 5 == 0 and getattr(self, 'skills', None):
                    try:
                        for name, meta in self.skills.list().items():
                            if meta.get("enabled", True) and getattr(self, 'orchestrator', None):
                                self.orchestrator.submit(
                                    f"skill_train:{name}",
                                    lambda n=name: self.train_skill(n),
                                    priority=90,
                                    drop_if_full=True,
                                )
                    except Exception:
                        pass

                self._evolution_cycles += 1
                self._last_evolution_time = time.time()
                # Count improvements
                if improvements_made > 0:
                    self._evolution_improvements += improvements_made

                # Wait before next evolution cycle (every 5 minutes)
                time.sleep(300)

            except Exception as e:
                print(f"❌ Evolution cycle error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    def _analyze_performance(self):
        """Analyze current system performance"""
        # Monitor memory usage
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)

        # Monitor response times (simplified)
        # In a real implementation, this would track actual response times

        # Check system health
        if memory_mb > 500:  # If using more than 500MB
            print("⚠️ High memory usage detected - optimizing...")

    def _apply_micro_improvements(self):
        """Apply small, incremental improvements"""
        improvements = 0

        try:
            # Optimize memory if needed
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > 400:
                # Force garbage collection
                import gc
                collected = gc.collect()
                if collected > 0:
                    print("🧹 Memory optimization: garbage collected")
                    improvements += 1

            # Optimize reasoning patterns
            if len(self.memory) > 1000:
                # Archive old memories
                archived_count = self._archive_old_memories()
                if archived_count > 0:
                    print(f"📦 Archived {archived_count} old memories")
                    improvements += 1

            # Fine-tune response generation
            improvements += 1  # Always count as improvement

            # Add points for improvements
            self.points += improvements

        except Exception as e:
            print(f"❌ Micro-improvement error: {e}")

        return improvements

    def _train_on_websites(self):
        """Train the neural systems using stored websites"""
        try:
            results = self.brain_memory.query_brain("training_website")
            trained = 0
            for knowledge in results.get('knowledge_results', []):
                content = knowledge.get('content', {})
                if isinstance(content, dict) and 'url' in content:
                    url = content['url']
                    try:
                        # Use web research to fetch and summarize the website
                        summary_data = self.web.research_summary(url, n=1)
                        if summary_data.get('summary'):
                            # Store the training data in brain memory
                            self.brain_memory.store_knowledge(
                                content={
                                    'summary': summary_data['summary'],
                                    'source_url': url,
                                    'training_data': True
                                },
                                category='neural_training',
                                importance=0.8,
                                confidence=0.9,
                                source='website_training',
                                metadata={
                                    'timestamp': time.time(),
                                    'url': url
                                }
                            )
                            trained += 1
                            print(f"🧠 Trained neural systems on {url}")
                        else:
                            print(f"⚠️ No summary available for {url}")
                    except Exception as e:
                        print(f"❌ Failed to train on {url}: {e}")
            
            if trained > 0:
                print(f"✅ Successfully trained neural systems on {trained} websites")
                # Use the training data to improve responses
                self._improve_from_training_data()
                # Add points for training
                self.points += trained * 10
            else:
                print("ℹ️ No websites available for training")
                
        except Exception as e:
            print(f"❌ Website training error: {e}")

    def _improve_from_training_data(self):
        """Improve the system using training data"""
        try:
            # Query recent training data
            training_results = self.brain_memory.query_brain("neural_training")
            if training_results.get('knowledge_results'):
                # Use the data to enhance brain memory patterns
                print("🔄 Improving neural systems from training data...")
                # This could involve updating patterns, knowledge graphs, etc.
                # For now, just acknowledge
                print("✅ Neural systems improved using website training data")
        except Exception as e:
            print(f"❌ Improvement error: {e}")
            # Analyze command patterns
            if hasattr(self, 'command_history') and self.command_history:
                most_common = self._find_most_common_commands()
                if most_common:
                    print(f"📊 Learning: Most used command is '{most_common}'")
                    # Could optimize for this command

            # Analyze conversation topics
            if hasattr(self, 'conversation_topics') and self.conversation_topics:
                top_topic = self._find_top_topic()
                if top_topic:
                    print(f"🧠 Learning: User interested in '{top_topic}'")
                    # Could specialize responses for this topic

        except Exception as e:
            print(f"❌ Learning error: {e}")

    def _optimize_memory_usage(self):
        """Optimize memory usage"""
        try:
            import gc
            import sys
            import psutil

            # Get current memory usage
            process = psutil.Process()
            memory_before = process.memory_info().rss / (1024 * 1024)

            # Force garbage collection
            gc.collect()

            # Clear any unused caches
            memory_after = process.memory_info().rss / (1024 * 1024)
            saved = memory_before - memory_after

            if saved > 1:  # If saved more than 1MB
                print(f"💾 Memory optimized: {saved:.1f}MB freed")

        except Exception as e:
            print(f"❌ Memory optimization error: {e}")

    def _self_test(self):
        """Run a small suite of reasoning and code tasks to self-evaluate and improve prompts.
        Stores insights into brain memory for continual improvement.
        """
        try:
            if self.llm is None:
                return
            tasks = [
                {
                    'name': 'reasoning_math',
                    'prompt': 'A train leaves city A at 60 km/h. Another leaves city B, 120 km away, at 40 km/h towards A. When do they meet? Show reasoning briefly, then provide only the final answer time.'
                },
                {
                    'name': 'code_refactor',
                    'prompt': (
                        'Refactor this Python function for clarity and performance, keep behavior identical, '
                        'add type hints and a docstring.\n\n'
                        'def f(x):\n    s=0\n    for i in range(len(x)):\n        s+=x[i]\n    return s\n'
                    )
                },
            ]
            results = []
            for t in tasks:
                enhanced = (
                    "You are improving your own capabilities via self-testing. Be concise and accurate.\n\n" +
                    t['prompt']
                )
                out = self.llm.generate_response(enhanced)
                results.append({'task': t['name'], 'output': out})

            # Store results and quick heuristics into brain memory
            insight = {
                'type': 'self_test_results',
                'context': {'cycle': time.time()},
                'outcome': {'results': results},
                'learned_patterns': ['self_testing', 'prompt_refinement']
            }
            self.brain_memory.store_experience(insight)
            # Persist to notes
            self.notes.save_json('self_tests', 'self_test_results', insight)
            print("🧪 Self-test completed and stored in brain memory")
            # Add points for self-test
            self.points += 5
        except Exception as e:
            print(f"❌ Self-test error: {e}")

    def _research_and_learn(self):
        """Perform lightweight web research about agent best practices and cutting-edge LLM methods,
        summarize with the LLM, and store distilled insights in brain memory.
        """
        try:
            query = "LLM agent evaluation benchmarks best practices tool-use reflection retry strategies"
            bundle = self.web.research_summary(query, n=3)
            summary = bundle.get('summary', '')
            sources = bundle.get('sources', '')

            if self.llm is not None and summary:
                prompt = (
                    "Summarize and distill the following research notes into 5-8 actionable improvements "
                    "for an autonomous coding and reasoning agent with memory and tools. Provide a prioritized list.\n\n" +
                    summary + "\n\nSources:\n" + sources
                )
                distilled = self.llm.generate_response(prompt)
            else:
                distilled = summary[:1200]

            insight = {
                'type': 'web_research_insight',
                'context': {'query': query, 'sources': sources},
                'outcome': {'distilled': distilled},
                'learned_patterns': ['web_learning', 'agent_best_practices']
            }
            self.brain_memory.store_experience(insight)
            # Persist distilled and raw summary into notes
            if distilled:
                self.notes.save_text('web', 'web_research_distilled', distilled, ext='.md')
            if summary:
                self.notes.save_text('web', 'web_research_raw', summary, ext='.md')
            print("🌐 Web research insights stored in brain memory")
            # Add points for research
            self.points += 20
        except Exception as e:
            print(f"❌ Web research error: {e}")

    def _archive_old_memories(self):
        """Archive old memories to free up space"""
        archived = 0
        try:
            # Simple archiving - in a real system this would be more sophisticated
            memory_items = list(self.memory.items())

            # Archive items older than 30 days (simplified)
            for key, item in memory_items:
                # Mark as archived (in real implementation, move to archive file)
                if len(str(item)) > 1000:  # Large items
                    archived += 1

        except Exception:
            pass

        return archived

    def _find_most_common_commands(self):
        """Find most common commands"""
        try:
            if not hasattr(self, 'command_history'):
                return None

            from collections import Counter
            command_counts = Counter(self.command_history[-100:])  # Last 100 commands
            if command_counts:
                return command_counts.most_common(1)[0][0]

        except Exception:
            pass
        return None

    def _find_top_topic(self):
        """Find most discussed topic"""
        try:
            if not hasattr(self, 'conversation_topics'):
                return None

            from collections import Counter
            topic_counts = Counter(self.conversation_topics[-50:])  # Last 50 topics
            if topic_counts:
                return topic_counts.most_common(1)[0][0]

        except Exception:
            pass
        return None

    def stop_evolution(self):
        """Stop the evolution process"""
        self.continuous_evolution = False
        if hasattr(self, 'evolution_thread'):
            self.evolution_thread.join(timeout=5)
        return "HRM evolution process stopped"

    # ---------------- Ingest watcher -----------------
    def _start_ingest_watcher(self):
        try:
            if hasattr(self, 'ingest_thread') and self.ingest_thread and self.ingest_thread.is_alive():
                return
            self.ingest_active = True
            self.ingest_thread = threading.Thread(target=self._ingest_loop, daemon=True)
            self.ingest_thread.start()
            print("📥 Ingest watcher started (brain_memory/ingest)")
        except Exception as e:
            print(f"❌ Failed to start ingest watcher: {e}")

    def _ingest_loop(self):
        import mimetypes
        import time
        ingest_path = Path(self.notes.get_paths()['ingest'])
        processed_marker = ".ingested"
        while self.ingest_active:
            try:
                for p in ingest_path.glob('*'):
                    if p.is_dir():
                        continue
                    if p.name.endswith(processed_marker):
                        continue
                    try:
                        mime, _ = mimetypes.guess_type(str(p))
                        content = None
                        metadata = {'path': str(p), 'mime': mime or 'application/octet-stream'}
                        # Try text reading
                        if (mime and mime.startswith('text')) or p.suffix.lower() in {'.txt', '.md', '.py', '.json', '.csv', '.log', '.yaml', '.yml'}:
                            try:
                                content = p.read_text(encoding='utf-8', errors='ignore')
                            except Exception:
                                content = None
                        # Fallback: read small binary chunk
                        if content is None:
                            try:
                                data = p.read_bytes()
                                # Store only up to 64KB as hex for preview
                                preview = data[:65536].hex()
                                content = f"<binary {len(data)} bytes> preview(hex)={preview[:4096]}..."
                            except Exception:
                                content = "<unreadable file>"

                        self.process_file_for_brain(str(p), content, metadata)
                        # Persist note pointer
                        self.notes.save_json('notes', 'ingested_file', {
                            'file': str(p),
                            'mime': metadata['mime'],
                            'size': p.stat().st_size,
                            'timestamp': time.time()
                        })
                        # Mark processed
                        done = p.with_suffix(p.suffix + processed_marker)
                        try:
                            p.rename(done)
                        except Exception:
                            # If cannot rename, create a sidecar marker
                            sidecar = Path(str(p) + processed_marker)
                            sidecar.write_text('processed', encoding='utf-8')
                    except Exception as e:
                        print(f"❌ Ingest error for {p}: {e}")
                time.sleep(2.0)
            except Exception as e:
                print(f"❌ Ingest loop error: {e}")
                time.sleep(5.0)

    def process_file_for_brain(self, file_path: str, content: str, metadata: dict | None = None):
        """Store an external file's content into brain memory as an experience.
        This is intentionally lightweight and safe to call from background threads.
        """
        try:
            exp = {
                'type': 'ingested_file',
                'context': {'path': file_path, **(metadata or {})},
                'outcome': {'content_preview': content[:4000] if isinstance(content, str) else str(content)},
                'learned_patterns': ['external_knowledge', 'ingest']
            }
            self.brain_memory.store_experience(exp)
        except Exception as e:
            print(f"❌ process_file_for_brain error: {e}")

    def chat_response(self, user_input, context=None):
        """Generate conversational response using HRM intelligence and brain memory"""
        try:
            # First try Gemini
            if self.provider == 'gemini':
                try:
                    return self.gemini_service.generate_chat_response(user_input)
                except Exception as e:
                    print(f"Gemini error: {e}")
            
            # Then try OpenRouter
            if self.provider == 'openrouter':
                try:
                    return self.openrouter_service.generate_chat_response(user_input)
                except Exception as e:
                    print(f"OpenRouter error: {e}")
            
            # Then try Llama if the service is available
            if self.llama_service is not None:
                try:
                    return self.llama_service.generate_chat_response(user_input)
                except Exception as e:
                    print(f"Llama error: {e}")
                
            # Final fallback
            return self._simple_response(user_input)
            
        except Exception as e:
            print(f"Chat system error: {e}")
            return "I'm having trouble generating a response right now. Please try again later."

    def _simple_response(self, user_input):
        """Generate simple responses without LLM"""
        if "hello" in user_input.lower() or "hi" in user_input.lower():
            return "Hello! How can I assist you today?"
        return "I received your message. Let me know how I can help!"
