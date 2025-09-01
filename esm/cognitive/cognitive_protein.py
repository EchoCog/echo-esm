"""
Cognitive Protein representation for ESM3 Cognitive Framework

Enhanced protein representation with cognitive capabilities:
- Multiple cognitive account types (Traditional, Adaptive, Predictive, etc.)
- Attention-driven processing prioritization
- Learning and adaptation mechanisms
- Multi-modal interaction support

This integrates ESMProtein with cognitive computing capabilities.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, Flag
import time

from esm.sdk.api import ESMProtein, GenerationConfig
from .atomspace import AtomHandle, TruthValue, ProteinAtomSpace
from .ecan import AttentionValue, ProteinActivity


class CognitiveAccountType(Flag):
    """Cognitive account type flags"""
    TRADITIONAL = 1          # Standard accounting behavior
    ADAPTIVE = 2             # Learning-enabled accounts
    PREDICTIVE = 4           # Accounts with forecasting capabilities
    MULTIMODAL = 8           # Support for complex transaction types
    ATTENTION_DRIVEN = 16    # Dynamically prioritized accounts


class CognitiveState(Enum):
    """Cognitive processing state"""
    INACTIVE = "inactive"
    LEARNING = "learning"
    PREDICTING = "predicting"
    OPTIMIZING = "optimizing"
    CONVERGED = "converged"
    ERROR = "error"


@dataclass
class CognitiveCapabilities:
    """Cognitive capabilities of a protein"""
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.7
    prediction_horizon: int = 10
    attention_decay: float = 0.05
    memory_capacity: int = 1000
    pattern_recognition_enabled: bool = True
    evolutionary_optimization: bool = False


@dataclass
class LearningHistory:
    """History of learning and adaptation"""
    iterations: int = 0
    performance_scores: List[float] = field(default_factory=list)
    adaptation_events: List[Dict[str, Any]] = field(default_factory=list)
    convergence_reached: bool = False
    last_update: float = field(default_factory=time.time)


class CognitiveProtein:
    """
    Enhanced protein with cognitive computing capabilities
    
    Combines ESMProtein with cognitive features:
    - AtomSpace representation
    - Attention allocation
    - Learning mechanisms
    - Predictive capabilities
    - Multi-modal processing
    """
    
    def __init__(self, esm_protein: ESMProtein, atomspace: ProteinAtomSpace,
                 cognitive_type: CognitiveAccountType = CognitiveAccountType.TRADITIONAL,
                 capabilities: Optional[CognitiveCapabilities] = None):
        
        self.esm_protein = esm_protein
        self.atomspace = atomspace
        self.cognitive_type = cognitive_type
        self.capabilities = capabilities or CognitiveCapabilities()
        
        # Cognitive state
        self.cognitive_state = CognitiveState.INACTIVE
        self.learning_history = LearningHistory()
        self.attention_value: Optional[AttentionValue] = None
        
        # Atom representation
        self.protein_atom: Optional[AtomHandle] = None
        self._create_atom_representation()
        
        # Cognitive memory
        self.working_memory: Dict[str, Any] = {}
        self.episodic_memory: List[Dict[str, Any]] = []
        self.semantic_memory: Dict[str, TruthValue] = {}
        
        # Performance tracking
        self.performance_metrics: Dict[str, float] = {}
        self.prediction_accuracy: Dict[str, List[float]] = {}
        
    def _create_atom_representation(self):
        """Create AtomSpace representation of the protein"""
        if self.esm_protein:
            self.protein_atom = self.atomspace.protein_to_atomspace(self.esm_protein)
            
            # Add cognitive type annotation
            cognitive_type_predicate = self.atomspace.create_predicate_node("hasCognitiveType")
            type_value = float(self.cognitive_type.value)
            
            self.atomspace.create_evaluation_link(
                cognitive_type_predicate,
                self.protein_atom,
                type_value,
                TruthValue(1.0, 0.9)  # High confidence in cognitive type
            )
    
    def set_cognitive_type(self, cognitive_type: CognitiveAccountType):
        """Update cognitive account type"""
        old_type = self.cognitive_type
        self.cognitive_type = cognitive_type
        
        # Record adaptation event
        adaptation_event = {
            'type': 'cognitive_type_change',
            'old_type': old_type.name,
            'new_type': cognitive_type.name,
            'timestamp': time.time(),
            'reason': 'manual_update'
        }
        self.learning_history.adaptation_events.append(adaptation_event)
        
        # Update atom representation
        if self.protein_atom:
            cognitive_type_predicate = self.atomspace.create_predicate_node("hasCognitiveType")
            type_value = float(cognitive_type.value)
            
            self.atomspace.create_evaluation_link(
                cognitive_type_predicate,
                self.protein_atom,
                type_value,
                TruthValue(1.0, 0.9)
            )
    
    def enable_learning(self):
        """Enable adaptive learning capabilities"""
        if CognitiveAccountType.ADAPTIVE not in self.cognitive_type:
            self.set_cognitive_type(self.cognitive_type | CognitiveAccountType.ADAPTIVE)
        
        self.cognitive_state = CognitiveState.LEARNING
        self.capabilities.pattern_recognition_enabled = True
        
        # Initialize learning structures
        if not hasattr(self, 'learned_patterns'):
            self.learned_patterns: Dict[str, TruthValue] = {}
        
        adaptation_event = {
            'type': 'learning_enabled',
            'timestamp': time.time(),
            'parameters': {
                'learning_rate': self.capabilities.learning_rate,
                'adaptation_threshold': self.capabilities.adaptation_threshold
            }
        }
        self.learning_history.adaptation_events.append(adaptation_event)
    
    def enable_prediction(self):
        """Enable predictive capabilities"""
        if CognitiveAccountType.PREDICTIVE not in self.cognitive_type:
            self.set_cognitive_type(self.cognitive_type | CognitiveAccountType.PREDICTIVE)
        
        self.cognitive_state = CognitiveState.PREDICTING
        
        # Initialize prediction structures
        if not hasattr(self, 'prediction_models'):
            self.prediction_models: Dict[str, Any] = {}
        
        adaptation_event = {
            'type': 'prediction_enabled',
            'timestamp': time.time(),
            'horizon': self.capabilities.prediction_horizon
        }
        self.learning_history.adaptation_events.append(adaptation_event)
    
    def enable_multimodal_processing(self):
        """Enable multimodal capabilities"""
        if CognitiveAccountType.MULTIMODAL not in self.cognitive_type:
            self.set_cognitive_type(self.cognitive_type | CognitiveAccountType.MULTIMODAL)
        
        # Initialize multimodal structures
        if not hasattr(self, 'modality_handlers'):
            self.modality_handlers: Dict[str, Callable] = {}
        
        adaptation_event = {
            'type': 'multimodal_enabled',
            'timestamp': time.time(),
            'supported_modalities': ['sequence', 'structure', 'function']
        }
        self.learning_history.adaptation_events.append(adaptation_event)
    
    def enable_attention_driven_processing(self, ecan_controller):
        """Enable attention-driven processing"""
        if CognitiveAccountType.ATTENTION_DRIVEN not in self.cognitive_type:
            self.set_cognitive_type(self.cognitive_type | CognitiveAccountType.ATTENTION_DRIVEN)
        
        # Register with ECAN controller
        if self.protein_atom:
            initial_attention = AttentionValue(sti=100.0, lti=50.0, vlti=0.0)
            ecan_controller.set_attention_value(self.protein_atom, initial_attention)
        
        adaptation_event = {
            'type': 'attention_driven_enabled',
            'timestamp': time.time(),
            'initial_attention': {
                'sti': 100.0,
                'lti': 50.0,
                'vlti': 0.0
            }
        }
        self.learning_history.adaptation_events.append(adaptation_event)
    
    def learn_from_experience(self, experience: Dict[str, Any], performance: float):
        """Learn from experience and update internal models"""
        if CognitiveAccountType.ADAPTIVE not in self.cognitive_type:
            return False
        
        self.learning_history.iterations += 1
        self.learning_history.performance_scores.append(performance)
        self.learning_history.last_update = time.time()
        
        # Extract patterns from experience
        patterns = self._extract_patterns(experience)
        
        # Initialize learned_patterns if not exists
        if not hasattr(self, 'learned_patterns'):
            self.learned_patterns: Dict[str, TruthValue] = {}
        
        # Update learned patterns with reinforcement learning
        for pattern_name, pattern_strength in patterns.items():
            if pattern_name in self.learned_patterns:
                # Update existing pattern
                old_tv = self.learned_patterns[pattern_name]
                new_strength = old_tv.strength + self.capabilities.learning_rate * (pattern_strength - old_tv.strength)
                new_confidence = min(1.0, old_tv.confidence + 0.1)
                self.learned_patterns[pattern_name] = TruthValue(new_strength, new_confidence)
            else:
                # Learn new pattern
                self.learned_patterns[pattern_name] = TruthValue(pattern_strength, 0.5)
        
        # Check for convergence
        if len(self.learning_history.performance_scores) >= 10:
            recent_scores = self.learning_history.performance_scores[-10:]
            if np.std(recent_scores) < 0.05 and np.mean(recent_scores) > self.capabilities.adaptation_threshold:
                self.learning_history.convergence_reached = True
                self.cognitive_state = CognitiveState.CONVERGED
        
        # Record learning event
        learning_event = {
            'type': 'learning_iteration',
            'iteration': self.learning_history.iterations,
            'performance': performance,
            'patterns_learned': len(patterns),
            'timestamp': time.time()
        }
        self.learning_history.adaptation_events.append(learning_event)
        
        return True
    
    def _extract_patterns(self, experience: Dict[str, Any]) -> Dict[str, float]:
        """Extract learnable patterns from experience"""
        patterns = {}
        
        # Analyze sequence patterns
        if 'sequence_activity' in experience:
            sequence_score = experience['sequence_activity']
            patterns['sequence_optimization'] = sequence_score
        
        # Analyze structure patterns
        if 'structure_stability' in experience:
            stability_score = experience['structure_stability']
            patterns['structure_stability'] = stability_score
        
        # Analyze function patterns
        if 'function_performance' in experience:
            function_score = experience['function_performance']
            patterns['function_optimization'] = function_score
        
        # Meta-learning patterns
        if len(self.learning_history.performance_scores) > 5:
            recent_trend = np.mean(self.learning_history.performance_scores[-5:]) - np.mean(self.learning_history.performance_scores[-10:-5]) if len(self.learning_history.performance_scores) >= 10 else 0.1
            patterns['learning_trend'] = max(0.0, min(1.0, 0.5 + recent_trend))
        
        return patterns
    
    def predict_future_state(self, steps_ahead: int = 1) -> Dict[str, Any]:
        """Predict future protein state"""
        if CognitiveAccountType.PREDICTIVE not in self.cognitive_type:
            return {'error': 'Predictive capabilities not enabled'}
        
        predictions = {}
        
        # Predict performance based on learning trend
        if len(self.learning_history.performance_scores) >= 3:
            recent_scores = self.learning_history.performance_scores[-3:]
            trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]
            
            future_performance = recent_scores[-1] + trend * steps_ahead
            predictions['performance'] = max(0.0, min(1.0, future_performance))
        
        # Predict attention decay
        if self.attention_value:
            decay_factor = (1.0 - self.capabilities.attention_decay) ** steps_ahead
            predictions['attention_sti'] = self.attention_value.sti * decay_factor
            predictions['attention_lti'] = self.attention_value.lti * decay_factor
        
        # Predict convergence
        if not self.learning_history.convergence_reached and len(self.learning_history.performance_scores) >= 5:
            performance_variance = np.var(self.learning_history.performance_scores[-5:])
            convergence_probability = 1.0 - min(1.0, performance_variance * 10)
            predictions['convergence_probability'] = convergence_probability
        
        predictions['prediction_timestamp'] = time.time()
        predictions['steps_ahead'] = steps_ahead
        
        return predictions
    
    def process_multimodal_input(self, modality: str, input_data: Any) -> Dict[str, Any]:
        """Process input from different modalities"""
        if CognitiveAccountType.MULTIMODAL not in self.cognitive_type:
            return {'error': 'Multimodal capabilities not enabled'}
        
        processing_result = {
            'modality': modality,
            'timestamp': time.time(),
            'status': 'processed'
        }
        
        if modality == 'sequence':
            # Process sequence information
            if isinstance(input_data, str):
                processing_result['sequence_length'] = len(input_data)
                processing_result['amino_acid_composition'] = self._analyze_sequence_composition(input_data)
        
        elif modality == 'structure':
            # Process structural information
            if hasattr(input_data, 'coordinates'):
                processing_result['structure_quality'] = self._assess_structure_quality(input_data)
        
        elif modality == 'function':
            # Process functional information
            if isinstance(input_data, list):
                processing_result['function_keywords'] = len(input_data)
                processing_result['functional_diversity'] = self._assess_functional_diversity(input_data)
        
        else:
            processing_result['status'] = 'unknown_modality'
        
        # Store in episodic memory
        self.episodic_memory.append(processing_result)
        
        # Limit memory size
        if len(self.episodic_memory) > self.capabilities.memory_capacity:
            self.episodic_memory = self.episodic_memory[-self.capabilities.memory_capacity:]
        
        return processing_result
    
    def _analyze_sequence_composition(self, sequence: str) -> Dict[str, float]:
        """Analyze amino acid composition"""
        composition = {}
        total = len(sequence)
        
        for aa in sequence:
            composition[aa] = composition.get(aa, 0) + 1
        
        # Convert to frequencies
        for aa in composition:
            composition[aa] /= total
        
        return composition
    
    def _assess_structure_quality(self, structure_data: Any) -> float:
        """Assess structural quality"""
        # Simplified structure quality assessment
        return 0.8  # Placeholder
    
    def _assess_functional_diversity(self, function_keywords: List[str]) -> float:
        """Assess functional diversity"""
        if not function_keywords:
            return 0.0
        
        # Simple diversity measure based on unique keywords
        unique_keywords = set(function_keywords)
        diversity = len(unique_keywords) / max(1, len(function_keywords))
        return diversity
    
    def update_performance_metrics(self, metrics: Dict[str, float]):
        """Update performance tracking"""
        self.performance_metrics.update(metrics)
        
        for metric_name, value in metrics.items():
            if metric_name not in self.prediction_accuracy:
                self.prediction_accuracy[metric_name] = []
            
            self.prediction_accuracy[metric_name].append(value)
            
            # Keep only recent predictions
            if len(self.prediction_accuracy[metric_name]) > 100:
                self.prediction_accuracy[metric_name] = self.prediction_accuracy[metric_name][-100:]
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Get comprehensive cognitive state summary"""
        summary = {
            'protein_id': self.protein_atom.name if self.protein_atom else 'unknown',
            'cognitive_type': {
                'value': self.cognitive_type.value,
                'flags': [flag.name for flag in CognitiveAccountType if flag in self.cognitive_type]
            },
            'cognitive_state': self.cognitive_state.value,
            'capabilities': {
                'learning_rate': self.capabilities.learning_rate,
                'adaptation_threshold': self.capabilities.adaptation_threshold,
                'prediction_horizon': self.capabilities.prediction_horizon,
                'memory_capacity': self.capabilities.memory_capacity,
                'pattern_recognition': self.capabilities.pattern_recognition_enabled,
                'evolutionary_optimization': self.capabilities.evolutionary_optimization
            },
            'learning_history': {
                'iterations': self.learning_history.iterations,
                'convergence_reached': self.learning_history.convergence_reached,
                'avg_performance': np.mean(self.learning_history.performance_scores) if self.learning_history.performance_scores else 0.0,
                'adaptation_events': len(self.learning_history.adaptation_events)
            },
            'memory_state': {
                'working_memory_size': len(self.working_memory),
                'episodic_memory_size': len(self.episodic_memory),
                'semantic_memory_size': len(self.semantic_memory),
                'learned_patterns': len(getattr(self, 'learned_patterns', {}))
            },
            'performance_metrics': self.performance_metrics.copy(),
            'timestamp': time.time()
        }
        
        return summary
    
    def reset_cognitive_state(self):
        """Reset cognitive state to initial conditions"""
        self.cognitive_state = CognitiveState.INACTIVE
        self.learning_history = LearningHistory()
        self.working_memory.clear()
        self.episodic_memory.clear()
        self.semantic_memory.clear()
        self.performance_metrics.clear()
        self.prediction_accuracy.clear()
        
        if hasattr(self, 'learned_patterns'):
            self.learned_patterns.clear()
        
        adaptation_event = {
            'type': 'cognitive_reset',
            'timestamp': time.time(),
            'reason': 'manual_reset'
        }
        self.learning_history.adaptation_events.append(adaptation_event)