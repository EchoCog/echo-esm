"""
Uncertain Reasoning Engine (URE) for ESM3 Cognitive Framework

Implements uncertain reasoning for protein predictions:
- Multi-factor uncertainty quantification
- Probabilistic predictions with confidence intervals
- Evidence integration from multiple sources
- Temporal uncertainty modeling

URE handles incomplete and uncertain biological information gracefully.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import math
from enum import Enum
import time

from .atomspace import AtomHandle, TruthValue, ProteinAtomSpace


class UncertaintyType(Enum):
    TEMPORAL = "temporal"
    COMPLEXITY = "complexity"
    EVIDENCE = "evidence"
    MODEL = "model"
    MEASUREMENT = "measurement"


@dataclass
class UncertaintyFactor:
    """Individual uncertainty factor"""
    factor_type: UncertaintyType
    magnitude: float  # 0.0 to 1.0
    confidence: float  # Confidence in uncertainty estimate
    description: str
    source: str


@dataclass
class PredictionResult:
    """Prediction with uncertainty quantification"""
    prediction: float
    confidence_interval: Tuple[float, float]
    uncertainty_factors: List[UncertaintyFactor]
    total_uncertainty: float
    prediction_type: str
    timestamp: float


@dataclass
class EvidenceSource:
    """Source of evidence for reasoning"""
    source_id: str
    reliability: float  # 0.0 to 1.0
    evidence_atoms: List[AtomHandle]
    weight: float
    timestamp: float


class ProteinURE:
    """
    Uncertain Reasoning Engine for protein analysis
    
    Performs uncertain reasoning to:
    - Predict protein properties with confidence intervals
    - Quantify multiple sources of uncertainty
    - Integrate conflicting evidence
    - Model temporal uncertainty decay
    """
    
    def __init__(self, atomspace: ProteinAtomSpace):
        self.atomspace = atomspace
        self.evidence_sources: Dict[str, EvidenceSource] = {}
        self.prediction_history: List[PredictionResult] = []
        self.uncertainty_models: Dict[str, Any] = {}
        self._initialize_uncertainty_models()
    
    def _initialize_uncertainty_models(self):
        """Initialize uncertainty quantification models"""
        self.uncertainty_models = {
            'temporal_decay': self._model_temporal_uncertainty,
            'complexity_scaling': self._model_complexity_uncertainty,
            'evidence_conflicts': self._model_evidence_uncertainty,
            'model_limitations': self._model_model_uncertainty,
            'measurement_noise': self._model_measurement_uncertainty
        }
    
    def add_evidence_source(self, source_id: str, reliability: float, 
                           evidence_atoms: List[AtomHandle], weight: float = 1.0):
        """Add evidence source for reasoning"""
        source = EvidenceSource(
            source_id=source_id,
            reliability=reliability,
            evidence_atoms=evidence_atoms,
            weight=weight,
            timestamp=time.time()
        )
        self.evidence_sources[source_id] = source
    
    def _model_temporal_uncertainty(self, protein_atom: AtomHandle, age_seconds: float) -> UncertaintyFactor:
        """Model uncertainty that increases over time"""
        # Uncertainty grows with time due to changing conditions
        decay_rate = 0.1  # Per hour
        hours = age_seconds / 3600.0
        
        temporal_uncertainty = 1.0 - np.exp(-decay_rate * hours)
        temporal_uncertainty = min(0.9, temporal_uncertainty)  # Cap at 90%
        
        return UncertaintyFactor(
            factor_type=UncertaintyType.TEMPORAL,
            magnitude=temporal_uncertainty,
            confidence=0.8,
            description=f"Temporal uncertainty after {hours:.1f} hours",
            source="temporal_model"
        )
    
    def _model_complexity_uncertainty(self, protein_atom: AtomHandle) -> UncertaintyFactor:
        """Model uncertainty based on protein complexity"""
        # Get protein complexity indicators
        sequence_length = self._extract_sequence_length(protein_atom)
        structure_complexity = self._estimate_structure_complexity(protein_atom)
        
        # Longer/more complex proteins have higher uncertainty
        length_factor = min(0.5, sequence_length / 1000.0)  # Normalize to max 50%
        structure_factor = structure_complexity * 0.3
        
        complexity_uncertainty = (length_factor + structure_factor) / 2
        
        return UncertaintyFactor(
            factor_type=UncertaintyType.COMPLEXITY,
            magnitude=complexity_uncertainty,
            confidence=0.7,
            description=f"Complexity uncertainty (length={sequence_length})",
            source="complexity_model"
        )
    
    def _model_evidence_uncertainty(self, evidence_atoms: List[AtomHandle]) -> UncertaintyFactor:
        """Model uncertainty from conflicting evidence"""
        if len(evidence_atoms) < 2:
            return UncertaintyFactor(
                factor_type=UncertaintyType.EVIDENCE,
                magnitude=0.3,  # Default uncertainty for single evidence
                confidence=0.6,
                description="Single evidence source",
                source="evidence_model"
            )
        
        # Calculate evidence consistency
        truth_values = [atom.truth_value for atom in evidence_atoms if atom.truth_value]
        if not truth_values:
            return UncertaintyFactor(
                factor_type=UncertaintyType.EVIDENCE,
                magnitude=0.5,
                confidence=0.4,
                description="No truth values available",
                source="evidence_model"
            )
        
        # Measure variance in evidence strength
        strengths = [tv.strength for tv in truth_values]
        confidences = [tv.confidence for tv in truth_values]
        
        strength_variance = np.var(strengths) if len(strengths) > 1 else 0.0
        confidence_variance = np.var(confidences) if len(confidences) > 1 else 0.0
        
        # Higher variance = higher uncertainty
        evidence_uncertainty = (strength_variance + confidence_variance) / 2
        evidence_uncertainty = min(0.8, evidence_uncertainty)
        
        return UncertaintyFactor(
            factor_type=UncertaintyType.EVIDENCE,
            magnitude=evidence_uncertainty,
            confidence=0.8,
            description=f"Evidence conflicts (variance={evidence_uncertainty:.3f})",
            source="evidence_model"
        )
    
    def _model_model_uncertainty(self, protein_atom: AtomHandle) -> UncertaintyFactor:
        """Model inherent model limitations uncertainty"""
        # Base model uncertainty for biological predictions
        base_uncertainty = 0.2  # 20% base uncertainty
        
        # Increase uncertainty for less well-studied protein types
        novelty_factor = self._estimate_protein_novelty(protein_atom)
        model_uncertainty = base_uncertainty + (novelty_factor * 0.3)
        
        return UncertaintyFactor(
            factor_type=UncertaintyType.MODEL,
            magnitude=min(0.7, model_uncertainty),
            confidence=0.9,  # We're confident about our model limitations
            description="Model limitation uncertainty",
            source="model_uncertainty"
        )
    
    def _model_measurement_uncertainty(self, protein_atom: AtomHandle) -> UncertaintyFactor:
        """Model measurement and experimental uncertainty"""
        # Estimate measurement uncertainty based on data source quality
        measurement_quality = self._estimate_measurement_quality(protein_atom)
        
        # Higher quality = lower uncertainty
        measurement_uncertainty = 1.0 - measurement_quality
        
        return UncertaintyFactor(
            factor_type=UncertaintyType.MEASUREMENT,
            magnitude=measurement_uncertainty,
            confidence=0.7,
            description=f"Measurement uncertainty (quality={measurement_quality:.2f})",
            source="measurement_model"
        )
    
    def _extract_sequence_length(self, protein_atom: AtomHandle) -> int:
        """Extract sequence length"""
        for link in self.atomspace.get_incoming_links(protein_atom):
            if "sequenceLength" in link.name:
                try:
                    length_str = link.name.split('=')[1]
                    return int(float(length_str))
                except:
                    continue
        return 100  # Default assumption
    
    def _estimate_structure_complexity(self, protein_atom: AtomHandle) -> float:
        """Estimate structural complexity (0.0 to 1.0)"""
        # Simplified complexity estimation
        structure_links = [link for link in self.atomspace.get_incoming_links(protein_atom)
                          if "structure" in link.name.lower()]
        
        if not structure_links:
            return 0.5  # Unknown complexity
        
        # More structure-related atoms = higher complexity
        complexity = min(1.0, len(structure_links) / 10.0)
        return complexity
    
    def _estimate_protein_novelty(self, protein_atom: AtomHandle) -> float:
        """Estimate protein novelty/uniqueness (0.0 to 1.0)"""
        # Check for function annotations - more annotations = less novel
        function_links = [link for link in self.atomspace.get_incoming_links(protein_atom)
                         if "function" in link.name.lower()]
        
        if len(function_links) > 3:
            return 0.2  # Well-studied
        elif len(function_links) > 1:
            return 0.5  # Moderately studied
        else:
            return 0.8  # Novel/poorly studied
    
    def _estimate_measurement_quality(self, protein_atom: AtomHandle) -> float:
        """Estimate quality of experimental measurements (0.0 to 1.0)"""
        # Higher truth value confidence suggests better measurements
        incoming = self.atomspace.get_incoming_links(protein_atom)
        
        if not incoming:
            return 0.5
        
        confidences = [link.truth_value.confidence for link in incoming 
                      if link.truth_value]
        
        if confidences:
            return np.mean(confidences)
        else:
            return 0.5
    
    def predict_property_with_uncertainty(self, protein_atom: AtomHandle, 
                                        property_name: str,
                                        prediction_base: float) -> PredictionResult:
        """Make prediction with full uncertainty quantification"""
        current_time = time.time()
        
        # Collect all uncertainty factors
        uncertainty_factors = []
        
        # Temporal uncertainty
        protein_age = self._estimate_protein_data_age(protein_atom)
        temporal_factor = self._model_temporal_uncertainty(protein_atom, protein_age)
        uncertainty_factors.append(temporal_factor)
        
        # Complexity uncertainty
        complexity_factor = self._model_complexity_uncertainty(protein_atom)
        uncertainty_factors.append(complexity_factor)
        
        # Evidence uncertainty
        evidence_atoms = self.atomspace.get_incoming_links(protein_atom)
        evidence_factor = self._model_evidence_uncertainty(evidence_atoms)
        uncertainty_factors.append(evidence_factor)
        
        # Model uncertainty
        model_factor = self._model_model_uncertainty(protein_atom)
        uncertainty_factors.append(model_factor)
        
        # Measurement uncertainty
        measurement_factor = self._model_measurement_uncertainty(protein_atom)
        uncertainty_factors.append(measurement_factor)
        
        # Combine uncertainty factors
        total_uncertainty = self._combine_uncertainties(uncertainty_factors)
        
        # Calculate confidence interval
        uncertainty_magnitude = total_uncertainty.magnitude
        confidence_radius = prediction_base * uncertainty_magnitude
        
        confidence_interval = (
            prediction_base - confidence_radius,
            prediction_base + confidence_radius
        )
        
        # Create prediction result
        result = PredictionResult(
            prediction=prediction_base,
            confidence_interval=confidence_interval,
            uncertainty_factors=uncertainty_factors,
            total_uncertainty=uncertainty_magnitude,
            prediction_type=property_name,
            timestamp=current_time
        )
        
        self.prediction_history.append(result)
        return result
    
    def _estimate_protein_data_age(self, protein_atom: AtomHandle) -> float:
        """Estimate age of protein data in seconds"""
        # Check for timestamp information in atoms
        incoming = self.atomspace.get_incoming_links(protein_atom)
        
        # For now, assume recent data (would be based on actual metadata)
        return 3600.0  # 1 hour default
    
    def _combine_uncertainties(self, uncertainty_factors: List[UncertaintyFactor]) -> UncertaintyFactor:
        """Combine multiple uncertainty factors"""
        if not uncertainty_factors:
            return UncertaintyFactor(
                factor_type=UncertaintyType.MODEL,
                magnitude=0.5,
                confidence=0.3,
                description="No uncertainty factors",
                source="default"
            )
        
        # Weight uncertainties by their confidence
        weighted_sum = 0.0
        total_weight = 0.0
        
        for factor in uncertainty_factors:
            weight = factor.confidence
            weighted_sum += factor.magnitude * weight
            total_weight += weight
        
        if total_weight == 0:
            combined_magnitude = np.mean([f.magnitude for f in uncertainty_factors])
        else:
            combined_magnitude = weighted_sum / total_weight
        
        # Confidence is the minimum of all factor confidences
        combined_confidence = min(f.confidence for f in uncertainty_factors)
        
        factor_descriptions = [f"{f.factor_type.value}={f.magnitude:.2f}" for f in uncertainty_factors]
        
        return UncertaintyFactor(
            factor_type=UncertaintyType.MODEL,  # Combined type
            magnitude=combined_magnitude,
            confidence=combined_confidence,
            description=f"Combined: {', '.join(factor_descriptions)}",
            source="uncertainty_combination"
        )
    
    def predict_protein_stability(self, protein_atom: AtomHandle) -> PredictionResult:
        """Predict protein stability with uncertainty"""
        # Base stability prediction (simplified)
        base_stability = self._calculate_base_stability(protein_atom)
        
        return self.predict_property_with_uncertainty(
            protein_atom, 
            "stability",
            base_stability
        )
    
    def predict_binding_affinity(self, protein_atom: AtomHandle, 
                               ligand_info: Optional[str] = None) -> PredictionResult:
        """Predict binding affinity with uncertainty"""
        # Base binding prediction (simplified)
        base_affinity = self._calculate_base_binding_affinity(protein_atom, ligand_info)
        
        return self.predict_property_with_uncertainty(
            protein_atom,
            "binding_affinity", 
            base_affinity
        )
    
    def predict_function_probability(self, protein_atom: AtomHandle, 
                                   function_name: str) -> PredictionResult:
        """Predict probability of protein having specific function"""
        # Base function probability (simplified)
        base_probability = self._calculate_function_probability(protein_atom, function_name)
        
        return self.predict_property_with_uncertainty(
            protein_atom,
            f"function_{function_name}",
            base_probability
        )
    
    def _calculate_base_stability(self, protein_atom: AtomHandle) -> float:
        """Calculate base stability prediction"""
        # Simplified stability calculation
        factors = []
        
        # Hydrophobic core strength
        hydrophobic_score = self._evaluate_hydrophobic_interactions(protein_atom)
        factors.append(hydrophobic_score)
        
        # Disulfide bonds
        disulfide_score = self._evaluate_disulfide_bonds(protein_atom)
        factors.append(disulfide_score)
        
        # Charge interactions
        charge_score = self._evaluate_charge_interactions(protein_atom)
        factors.append(charge_score)
        
        return np.mean(factors) if factors else 0.5
    
    def _calculate_base_binding_affinity(self, protein_atom: AtomHandle, 
                                       ligand_info: Optional[str]) -> float:
        """Calculate base binding affinity prediction"""
        # Simplified binding affinity calculation
        base_affinity = 0.5  # Default
        
        # Adjust based on binding site features
        if ligand_info:
            # Would analyze binding site complementarity
            base_affinity += 0.2
        
        # Check for known binding domains
        function_links = [link for link in self.atomspace.get_incoming_links(protein_atom)
                         if "function" in link.name.lower()]
        
        if any("bind" in link.name.lower() for link in function_links):
            base_affinity += 0.3
        
        return min(1.0, base_affinity)
    
    def _calculate_function_probability(self, protein_atom: AtomHandle, 
                                      function_name: str) -> float:
        """Calculate probability of protein having function"""
        function_links = [link for link in self.atomspace.get_incoming_links(protein_atom)
                         if "function" in link.name.lower()]
        
        # Check for matching function annotations
        matching_functions = [link for link in function_links 
                            if function_name.lower() in link.name.lower()]
        
        if matching_functions:
            # Average truth values of matching functions
            truth_values = [link.truth_value.strength for link in matching_functions
                          if link.truth_value]
            if truth_values:
                return np.mean(truth_values)
        
        # Default based on total function annotations
        if len(function_links) > 0:
            return 0.3  # Some functional information available
        else:
            return 0.1  # No functional information
    
    def _evaluate_hydrophobic_interactions(self, protein_atom: AtomHandle) -> float:
        """Evaluate hydrophobic interaction strength"""
        hydrophobic_aas = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
        
        aa_atoms = [atom for atom in self.atomspace.get_atoms_by_name_pattern("AminoAcid_")]
        if not aa_atoms:
            return 0.5
        
        hydrophobic_count = 0
        for aa_atom in aa_atoms:
            if aa_atom.name.startswith("AminoAcid_"):
                aa_type = aa_atom.name.split("_")[1] 
                if aa_type in hydrophobic_aas:
                    hydrophobic_count += 1
        
        hydrophobic_fraction = hydrophobic_count / len(aa_atoms)
        
        # Optimal hydrophobic fraction for stability
        if 0.3 <= hydrophobic_fraction <= 0.5:
            return 0.9
        elif 0.2 <= hydrophobic_fraction <= 0.6:
            return 0.7
        else:
            return 0.5
    
    def _evaluate_disulfide_bonds(self, protein_atom: AtomHandle) -> float:
        """Evaluate disulfide bond contribution to stability"""
        cysteine_atoms = [atom for atom in self.atomspace.get_atoms_by_name_pattern("AminoAcid_C")]
        cysteine_count = len(cysteine_atoms)
        
        if cysteine_count == 0:
            return 0.6  # No disulfide bonds, moderate stability
        elif cysteine_count % 2 == 0 and cysteine_count >= 2:
            return 0.9  # Even number, likely forms bonds
        else:
            return 0.4  # Odd number, unpaired cysteines
    
    def _evaluate_charge_interactions(self, protein_atom: AtomHandle) -> float:
        """Evaluate electrostatic interaction contribution"""
        positive_aas = {'R', 'H', 'K'}
        negative_aas = {'D', 'E'}
        
        positive_count = negative_count = 0
        
        aa_atoms = [atom for atom in self.atomspace.get_atoms_by_name_pattern("AminoAcid_")]
        for aa_atom in aa_atoms:
            if aa_atom.name.startswith("AminoAcid_"):
                aa_type = aa_atom.name.split("_")[1]
                if aa_type in positive_aas:
                    positive_count += 1
                elif aa_type in negative_aas:
                    negative_count += 1
        
        total_charged = positive_count + negative_count
        if total_charged == 0:
            return 0.7  # Neutral proteins can be stable
        
        # Balanced charges are stabilizing
        charge_balance = 1.0 - abs(positive_count - negative_count) / total_charged
        return charge_balance
    
    def get_uncertainty_analysis(self, protein_atom: AtomHandle) -> Dict[str, Any]:
        """Get comprehensive uncertainty analysis"""
        # Get predictions for multiple properties
        predictions = {}
        
        stability_pred = self.predict_protein_stability(protein_atom)
        predictions['stability'] = {
            'prediction': stability_pred.prediction,
            'confidence_interval': stability_pred.confidence_interval,
            'uncertainty': stability_pred.total_uncertainty
        }
        
        binding_pred = self.predict_binding_affinity(protein_atom)
        predictions['binding_affinity'] = {
            'prediction': binding_pred.prediction,
            'confidence_interval': binding_pred.confidence_interval,
            'uncertainty': binding_pred.total_uncertainty
        }
        
        # Analyze uncertainty sources
        all_factors = stability_pred.uncertainty_factors + binding_pred.uncertainty_factors
        uncertainty_by_type = {}
        
        for factor in all_factors:
            factor_type = factor.factor_type.value
            if factor_type not in uncertainty_by_type:
                uncertainty_by_type[factor_type] = []
            uncertainty_by_type[factor_type].append(factor.magnitude)
        
        # Average uncertainty by type
        uncertainty_summary = {}
        for factor_type, magnitudes in uncertainty_by_type.items():
            uncertainty_summary[factor_type] = {
                'mean': np.mean(magnitudes),
                'max': np.max(magnitudes),
                'count': len(magnitudes)
            }
        
        return {
            'protein': protein_atom.name,
            'predictions': predictions,
            'uncertainty_sources': uncertainty_summary,
            'total_predictions': len(self.prediction_history),
            'analysis_timestamp': time.time()
        }