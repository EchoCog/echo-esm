"""
Main Cognitive Accounting Framework for ESM3

This module integrates all cognitive components into a unified framework:
- AtomSpace knowledge representation
- PLN probabilistic reasoning
- ECAN attention allocation
- MOSES evolutionary optimization
- URE uncertainty reasoning
- Inter-module communication
- Emergent behavior detection

The framework transforms ESM3 protein modeling into a cognitive system.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
import time
import threading
from enum import Enum

from esm.sdk.api import ESMProtein, GenerationConfig

from .atomspace import ProteinAtomSpace, AtomHandle, TruthValue
from .pln import ProteinPLN, PLNProof
from .ecan import ProteinECAN, AttentionValue, ECANParams, ProteinActivity
from .moses import ProteinMOSES, EvolutionaryStrategy, FitnessType
from .ure import ProteinURE, PredictionResult, UncertaintyFactor
from .cognitive_protein import CognitiveProtein, CognitiveAccountType


class CognitiveMessage:
    """Message for inter-module communication"""
    def __init__(self, sender: str, receiver: str, message_type: str, 
                 payload: Any, priority: float = 0.5, timestamp: float = None):
        self.sender = sender
        self.receiver = receiver
        self.message_type = message_type
        self.payload = payload
        self.priority = priority
        self.timestamp = timestamp or time.time()
        self.message_id = f"{sender}_{receiver}_{int(self.timestamp)}_{id(self)}"


@dataclass
class EmergenceParams:
    """Parameters for emergent behavior detection"""
    complexity_threshold: float = 0.7
    coherence_measure: float = 0.6
    novelty_score: float = 0.5
    frequency_threshold: int = 3


class CognitiveAccountingFramework:
    """
    Main cognitive accounting framework integrating all components
    
    This framework orchestrates the interaction between:
    - AtomSpace: Knowledge representation
    - PLN: Probabilistic reasoning
    - ECAN: Attention allocation
    - MOSES: Evolutionary optimization
    - URE: Uncertainty quantification
    """
    
    def __init__(self, ecan_params: Optional[ECANParams] = None):
        # Initialize core components
        self.atomspace = ProteinAtomSpace()
        self.pln = ProteinPLN(self.atomspace)
        self.ecan = ProteinECAN(self.atomspace, ecan_params)
        self.moses = ProteinMOSES(self.atomspace)
        self.ure = ProteinURE(self.atomspace)
        
        # Cognitive proteins registry
        self.cognitive_proteins: Dict[str, CognitiveProtein] = {}
        
        # Inter-module communication
        self.message_handlers: Dict[str, Callable] = {}
        self.message_queue: List[CognitiveMessage] = []
        self.message_lock = threading.Lock()
        
        # System state
        self.framework_initialized = False
        self.processing_active = False
        self.cycle_count = 0
        self.last_cycle_time = time.time()
        
        # Performance tracking
        self.performance_metrics: Dict[str, List[float]] = {}
        self.emergent_patterns: List[Dict[str, Any]] = []
        
        # Initialize framework
        self._initialize_framework()
    
    def _initialize_framework(self):
        """Initialize the cognitive framework"""
        # Register message handlers for each module
        self._register_message_handlers()
        
        # Set up initial cognitive protocols
        self._setup_cognitive_protocols()
        
        self.framework_initialized = True
        
        # Log initialization
        init_message = CognitiveMessage(
            sender="Framework",
            receiver="All",
            message_type="Initialization",
            payload={'timestamp': time.time(), 'status': 'initialized'},
            priority=1.0
        )
        self._send_message(init_message)
    
    def _register_message_handlers(self):
        """Register message handlers for inter-module communication"""
        self.message_handlers = {
            'AtomSpace': self._handle_atomspace_message,
            'PLN': self._handle_pln_message,
            'ECAN': self._handle_ecan_message,
            'MOSES': self._handle_moses_message,
            'URE': self._handle_ure_message,
            'Framework': self._handle_framework_message
        }
    
    def _setup_cognitive_protocols(self):
        """Set up cognitive communication protocols"""
        # Define standard message types and priorities
        self.message_priorities = {
            'Attention_Update': 0.9,
            'Validation_Request': 0.8,
            'Optimization_Task': 0.7,
            'Uncertainty_Query': 0.6,
            'Pattern_Detection': 0.5,
            'Status_Report': 0.3
        }
    
    def add_protein(self, esm_protein: ESMProtein, 
                   cognitive_type: CognitiveAccountType = CognitiveAccountType.TRADITIONAL) -> CognitiveProtein:
        """Add protein to cognitive framework"""
        # Create cognitive protein
        cognitive_protein = CognitiveProtein(
            esm_protein=esm_protein,
            atomspace=self.atomspace,
            cognitive_type=cognitive_type
        )
        
        protein_id = cognitive_protein.protein_atom.atom_id if cognitive_protein.protein_atom else f"protein_{len(self.cognitive_proteins)}"
        self.cognitive_proteins[protein_id] = cognitive_protein
        
        # Initialize attention for the protein
        if cognitive_protein.protein_atom:
            initial_attention = AttentionValue(sti=50.0, lti=25.0, vlti=0.0)
            self.ecan.set_attention_value(cognitive_protein.protein_atom, initial_attention)
        
        # Send notification
        add_message = CognitiveMessage(
            sender="Framework",
            receiver="All",
            message_type="Protein_Added",
            payload={
                'protein_id': protein_id,
                'cognitive_type': cognitive_type.name,
                'atom_id': cognitive_protein.protein_atom.atom_id if cognitive_protein.protein_atom else None
            },
            priority=0.7
        )
        self._send_message(add_message)
        
        return cognitive_protein
    
    def validate_protein_set(self, protein_ids: List[str]) -> PLNProof:
        """Validate a set of proteins using PLN reasoning"""
        protein_atoms = []
        
        for protein_id in protein_ids:
            if protein_id in self.cognitive_proteins:
                cognitive_protein = self.cognitive_proteins[protein_id]
                if cognitive_protein.protein_atom:
                    protein_atoms.append(cognitive_protein.protein_atom)
        
        if not protein_atoms:
            raise ValueError("No valid proteins found for validation")
        
        # Generate trial balance proof
        proof = self.pln.generate_trial_balance_proof(protein_atoms)
        
        # Send validation result
        validation_message = CognitiveMessage(
            sender="PLN",
            receiver="Framework",
            message_type="Validation_Complete",
            payload={
                'proof': proof,
                'protein_count': len(protein_atoms),
                'confidence': proof.confidence
            },
            priority=0.8
        )
        self._send_message(validation_message)
        
        return proof
    
    def optimize_protein_design(self, protein_id: str, 
                              fitness_type: FitnessType = FitnessType.MULTI_OBJECTIVE,
                              generations: int = 10) -> Dict[str, Any]:
        """Optimize protein design using MOSES"""
        if protein_id not in self.cognitive_proteins:
            raise ValueError(f"Protein {protein_id} not found")
        
        cognitive_protein = self.cognitive_proteins[protein_id]
        if not cognitive_protein.protein_atom:
            raise ValueError(f"Protein {protein_id} has no atom representation")
        
        # Run MOSES optimization
        best_variant = self.moses.optimize_protein(
            cognitive_protein.protein_atom,
            fitness_type,
            generations
        )
        
        # Update attention based on optimization success
        self.ecan.record_activity(
            cognitive_protein.protein_atom,
            "optimization",
            best_variant.fitness_scores[fitness_type]
        )
        
        optimization_result = {
            'original_protein': protein_id,
            'best_variant': best_variant.variant_id,
            'fitness_improvement': best_variant.fitness_scores[fitness_type],
            'generations': generations,
            'optimization_summary': self.moses.get_optimization_summary()
        }
        
        # Send optimization result
        opt_message = CognitiveMessage(
            sender="MOSES",
            receiver="Framework",
            message_type="Optimization_Complete",
            payload=optimization_result,
            priority=0.7
        )
        self._send_message(opt_message)
        
        return optimization_result
    
    def predict_with_uncertainty(self, protein_id: str, property_name: str) -> PredictionResult:
        """Make prediction with uncertainty quantification"""
        if protein_id not in self.cognitive_proteins:
            raise ValueError(f"Protein {protein_id} not found")
        
        cognitive_protein = self.cognitive_proteins[protein_id]
        if not cognitive_protein.protein_atom:
            raise ValueError(f"Protein {protein_id} has no atom representation")
        
        # Get base prediction from PLN validation
        validation = self.pln.validate_protein(cognitive_protein.protein_atom)
        base_prediction = validation.strength
        
        # Get uncertainty analysis from URE
        prediction_result = self.ure.predict_property_with_uncertainty(
            cognitive_protein.protein_atom,
            property_name,
            base_prediction
        )
        
        # Update attention based on prediction request
        self.ecan.record_activity(
            cognitive_protein.protein_atom,
            "prediction_request",
            1.0
        )
        
        # Send prediction result
        pred_message = CognitiveMessage(
            sender="URE",
            receiver="Framework",
            message_type="Prediction_Complete",
            payload={
                'protein_id': protein_id,
                'property': property_name,
                'prediction': prediction_result.prediction,
                'uncertainty': prediction_result.total_uncertainty
            },
            priority=0.6
        )
        self._send_message(pred_message)
        
        return prediction_result
    
    def run_cognitive_cycle(self):
        """Run one cycle of cognitive processing"""
        self.cycle_count += 1
        cycle_start = time.time()
        
        # Process message queue
        self._process_messages()
        
        # Run ECAN attention allocation
        self.ecan.allocate_attention_cycle()
        
        # Detect emergent patterns
        if self.cycle_count % 5 == 0:  # Every 5 cycles
            self._detect_emergent_patterns()
        
        # Update cognitive proteins
        self._update_cognitive_proteins()
        
        # Performance tracking
        cycle_time = time.time() - cycle_start
        self._record_performance('cycle_time', cycle_time)
        
        self.last_cycle_time = time.time()
        
        # Send cycle completion message
        cycle_message = CognitiveMessage(
            sender="Framework",
            receiver="All",
            message_type="Cycle_Complete",
            payload={
                'cycle_number': self.cycle_count,
                'cycle_time': cycle_time,
                'messages_processed': len(self.message_queue)
            },
            priority=0.2
        )
        self._send_message(cycle_message)
    
    def _process_messages(self):
        """Process inter-module messages"""
        with self.message_lock:
            # Sort messages by priority (highest first)
            self.message_queue.sort(key=lambda m: m.priority, reverse=True)
            
            # Process up to 10 messages per cycle
            messages_to_process = self.message_queue[:10]
            self.message_queue = self.message_queue[10:]
        
        for message in messages_to_process:
            try:
                if message.receiver in self.message_handlers:
                    self.message_handlers[message.receiver](message)
                elif message.receiver == "All":
                    # Broadcast to all handlers
                    for handler in self.message_handlers.values():
                        handler(message)
            except Exception as e:
                # Log error and continue
                error_message = CognitiveMessage(
                    sender="Framework",
                    receiver="Framework",
                    message_type="Message_Error",
                    payload={'error': str(e), 'message': message.message_id},
                    priority=0.9
                )
                self._send_message(error_message)
    
    def _detect_emergent_patterns(self):
        """Detect emergent behavior patterns"""
        emergence_params = EmergenceParams()
        
        # Analyze attention patterns
        attention_patterns = self._analyze_attention_patterns(emergence_params)
        
        # Analyze validation patterns  
        validation_patterns = self._analyze_validation_patterns(emergence_params)
        
        # Analyze optimization patterns
        optimization_patterns = self._analyze_optimization_patterns(emergence_params)
        
        # Combine patterns
        all_patterns = attention_patterns + validation_patterns + optimization_patterns
        
        # Filter for emergent patterns
        emergent_patterns = [p for p in all_patterns 
                           if p.get('complexity', 0) > emergence_params.complexity_threshold
                           and p.get('novelty', 0) > emergence_params.novelty_score]
        
        if emergent_patterns:
            self.emergent_patterns.extend(emergent_patterns)
            
            # Send emergence notification
            emergence_message = CognitiveMessage(
                sender="Framework",
                receiver="All",
                message_type="Emergence_Detected",
                payload={
                    'patterns': emergent_patterns,
                    'cycle': self.cycle_count
                },
                priority=0.5
            )
            self._send_message(emergence_message)
    
    def _analyze_attention_patterns(self, params: EmergenceParams) -> List[Dict[str, Any]]:
        """Analyze attention allocation patterns"""
        patterns = []
        
        # Get top attention atoms
        top_sti = self.ecan.get_top_attention_atoms(10)
        top_lti = self.ecan.get_top_attention_atoms(10, attention_type="LTI")
        
        if len(top_sti) >= 5:
            # Analyze STI distribution
            sti_scores = [score for _, score in top_sti]
            sti_variance = np.var(sti_scores)
            
            if sti_variance > 1000:  # High variance indicates complex attention patterns
                patterns.append({
                    'type': 'attention_complexity',
                    'complexity': min(1.0, sti_variance / 5000),
                    'novelty': 0.7,
                    'description': 'Complex STI attention distribution detected',
                    'data': {'variance': sti_variance, 'top_atoms': len(top_sti)}
                })
        
        return patterns
    
    def _analyze_validation_patterns(self, params: EmergenceParams) -> List[Dict[str, Any]]:
        """Analyze PLN validation patterns"""
        patterns = []
        
        # Get validation statistics from recent proofs
        # This would analyze validation trends and complex reasoning patterns
        
        # Placeholder for validation pattern analysis
        if len(self.cognitive_proteins) > 5:
            patterns.append({
                'type': 'validation_coherence',
                'complexity': 0.6,
                'novelty': 0.5,
                'description': 'Multiple protein validation coherence',
                'data': {'protein_count': len(self.cognitive_proteins)}
            })
        
        return patterns
    
    def _analyze_optimization_patterns(self, params: EmergenceParams) -> List[Dict[str, Any]]:
        """Analyze MOSES optimization patterns"""
        patterns = []
        
        optimization_summary = self.moses.get_optimization_summary()
        
        if optimization_summary.get('discovered_strategies', 0) > 2:
            patterns.append({
                'type': 'optimization_emergence',
                'complexity': 0.8,
                'novelty': 0.6,
                'description': 'Multiple optimization strategies discovered',
                'data': optimization_summary
            })
        
        return patterns
    
    def _update_cognitive_proteins(self):
        """Update all cognitive proteins"""
        for protein_id, cognitive_protein in self.cognitive_proteins.items():
            if cognitive_protein.protein_atom:
                # Update attention value
                attention = self.ecan.get_attention_value(cognitive_protein.protein_atom)
                cognitive_protein.attention_value = attention
                
                # Update performance metrics if learning enabled
                if CognitiveAccountType.ADAPTIVE in cognitive_protein.cognitive_type:
                    # Simulate learning experience
                    experience = {
                        'sequence_activity': attention.sti / 1000.0,
                        'structure_stability': 0.7,  # Placeholder
                        'function_performance': 0.8   # Placeholder
                    }
                    performance = (attention.sti + attention.lti) / 2000.0
                    
                    # Only learn if learning has been enabled
                    if hasattr(cognitive_protein, 'learned_patterns'):
                        cognitive_protein.learn_from_experience(experience, performance)
    
    def _send_message(self, message: CognitiveMessage):
        """Send message to the queue"""
        with self.message_lock:
            self.message_queue.append(message)
    
    def _handle_atomspace_message(self, message: CognitiveMessage):
        """Handle AtomSpace messages"""
        if message.message_type == "Atom_Created":
            # Handle new atom creation
            pass
        elif message.message_type == "Truth_Value_Updated":
            # Handle truth value updates
            pass
    
    def _handle_pln_message(self, message: CognitiveMessage):
        """Handle PLN messages"""
        if message.message_type == "Validation_Request":
            # Handle validation requests
            pass
        elif message.message_type == "Proof_Generated":
            # Handle proof generation
            pass
    
    def _handle_ecan_message(self, message: CognitiveMessage):
        """Handle ECAN messages"""
        if message.message_type == "Attention_Update":
            # Handle attention updates
            pass
        elif message.message_type == "Focus_Changed":
            # Handle focus changes
            pass
    
    def _handle_moses_message(self, message: CognitiveMessage):
        """Handle MOSES messages"""
        if message.message_type == "Optimization_Task":
            # Handle optimization tasks
            pass
        elif message.message_type == "Strategy_Discovered":
            # Handle strategy discoveries
            pass
    
    def _handle_ure_message(self, message: CognitiveMessage):
        """Handle URE messages"""
        if message.message_type == "Uncertainty_Query":
            # Handle uncertainty queries
            pass
        elif message.message_type == "Prediction_Request":
            # Handle prediction requests
            pass
    
    def _handle_framework_message(self, message: CognitiveMessage):
        """Handle framework messages"""
        if message.message_type == "Status_Request":
            # Handle status requests
            pass
    
    def _record_performance(self, metric_name: str, value: float):
        """Record performance metric"""
        if metric_name not in self.performance_metrics:
            self.performance_metrics[metric_name] = []
        
        self.performance_metrics[metric_name].append(value)
        
        # Keep only recent values
        if len(self.performance_metrics[metric_name]) > 1000:
            self.performance_metrics[metric_name] = self.performance_metrics[metric_name][-1000:]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        attention_summary = {}
        if self.cognitive_proteins:
            focused_atoms = self.ecan.get_attentional_focus()
            attention_summary = {
                'focused_atoms': len(focused_atoms),
                'total_proteins': len(self.cognitive_proteins),
                'attention_distribution': self.ecan._compute_attention_distribution()
            }
        
        return {
            'framework_initialized': self.framework_initialized,
            'processing_active': self.processing_active,
            'cycle_count': self.cycle_count,
            'last_cycle_time': self.last_cycle_time,
            'cognitive_proteins': len(self.cognitive_proteins),
            'message_queue_size': len(self.message_queue),
            'emergent_patterns': len(self.emergent_patterns),
            'atomspace_atoms': self.atomspace.atom_count(),
            'attention_summary': attention_summary,
            'performance_metrics': {k: len(v) for k, v in self.performance_metrics.items()},
            'timestamp': time.time()
        }
    
    def shutdown(self):
        """Shutdown the cognitive framework"""
        self.processing_active = False
        
        # Send shutdown message
        shutdown_message = CognitiveMessage(
            sender="Framework",
            receiver="All",
            message_type="Shutdown",
            payload={'timestamp': time.time()},
            priority=1.0
        )
        self._send_message(shutdown_message)
        
        # Clear all data structures
        self.cognitive_proteins.clear()
        self.message_queue.clear()
        self.performance_metrics.clear()
        self.emergent_patterns.clear()
        
        self.framework_initialized = False