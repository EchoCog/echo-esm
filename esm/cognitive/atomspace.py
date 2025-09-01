"""
AtomSpace implementation for ESM3 Cognitive Framework

Maps protein structures to hypergraph representations using AtomSpace concepts:
- ConceptNodes: Amino acid and protein concepts
- PredicateNodes: Structural and functional properties
- InheritanceLinks: Protein taxonomies and relationships
- EvaluationLinks: Property assertions with truth values

This provides the foundational knowledge representation for the cognitive framework.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from esm.sdk.api import ESMProtein


@dataclass
class TruthValue:
    """OpenCog-style truth value with strength and confidence"""
    strength: float  # Probability/belief strength (0.0 to 1.0)
    confidence: float  # Evidence confidence (0.0 to 1.0)
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass 
class AtomHandle:
    """Handle to an atom in the AtomSpace"""
    atom_id: str
    atom_type: str
    name: str
    truth_value: Optional[TruthValue] = None
    
    def __hash__(self):
        return hash(self.atom_id)


class ProteinAtomSpace:
    """
    AtomSpace implementation for protein cognitive representation
    
    Creates hypergraph structures from protein data where:
    - Amino acids become ConceptNodes
    - Properties become PredicateNodes  
    - Relationships become Links
    - Assertions get truth values
    """
    
    def __init__(self):
        self.atoms: Dict[str, AtomHandle] = {}
        self.links: Dict[str, List[AtomHandle]] = {}
        self.next_id = 0
    
    def _generate_id(self) -> str:
        """Generate unique atom ID"""
        self.next_id += 1
        return f"atom_{self.next_id}"
    
    def create_concept_node(self, name: str, truth_value: Optional[TruthValue] = None) -> AtomHandle:
        """Create a ConceptNode representing a protein or amino acid concept"""
        atom_id = self._generate_id()
        if truth_value is None:
            truth_value = TruthValue(0.8, 0.6)  # Default moderate confidence
            
        atom = AtomHandle(atom_id, "ConceptNode", name, truth_value)
        self.atoms[atom_id] = atom
        return atom
    
    def create_predicate_node(self, name: str, truth_value: Optional[TruthValue] = None) -> AtomHandle:
        """Create a PredicateNode for protein properties"""
        atom_id = self._generate_id()
        if truth_value is None:
            truth_value = TruthValue(0.7, 0.5)
            
        atom = AtomHandle(atom_id, "PredicateNode", name, truth_value)
        self.atoms[atom_id] = atom
        return atom
    
    def create_inheritance_link(self, child: AtomHandle, parent: AtomHandle, 
                              truth_value: Optional[TruthValue] = None) -> AtomHandle:
        """Create inheritance relationship between concepts"""
        atom_id = self._generate_id()
        if truth_value is None:
            truth_value = TruthValue(0.9, 0.8)
            
        link = AtomHandle(atom_id, "InheritanceLink", f"{child.name}→{parent.name}", truth_value)
        self.atoms[atom_id] = link
        self.links[atom_id] = [child, parent]
        return link
    
    def create_evaluation_link(self, predicate: AtomHandle, concept: AtomHandle,
                             value: float, truth_value: Optional[TruthValue] = None) -> AtomHandle:
        """Create property evaluation with numeric value"""
        atom_id = self._generate_id()
        if truth_value is None:
            # Adjust truth value based on property value confidence
            strength = min(0.95, 0.5 + abs(value) * 0.4)
            confidence = 0.7
            truth_value = TruthValue(strength, confidence)
            
        name = f"{predicate.name}({concept.name})={value:.3f}"
        link = AtomHandle(atom_id, "EvaluationLink", name, truth_value)
        self.atoms[atom_id] = link
        self.links[atom_id] = [predicate, concept]
        return link
    
    def protein_to_atomspace(self, protein: ESMProtein) -> AtomHandle:
        """Convert ESMProtein to AtomSpace representation"""
        # Create main protein concept
        protein_name = f"Protein_{hash(str(protein.sequence))}"
        protein_concept = self.create_concept_node(protein_name, TruthValue(0.95, 0.9))
        
        # Add sequence information
        if protein.sequence:
            sequence_predicate = self.create_predicate_node("hasSequence")
            seq_length = len(protein.sequence)
            length_eval = self.create_evaluation_link(
                self.create_predicate_node("sequenceLength"),
                protein_concept, 
                float(seq_length),
                TruthValue(1.0, 1.0)  # Length is certain
            )
            
            # Create amino acid nodes and relationships
            for i, aa in enumerate(protein.sequence):
                aa_concept = self.create_concept_node(f"AminoAcid_{aa}")
                position_eval = self.create_evaluation_link(
                    self.create_predicate_node("atPosition"),
                    aa_concept,
                    float(i),
                    TruthValue(1.0, 1.0)
                )
                
                # Link amino acid to protein
                self.create_inheritance_link(aa_concept, protein_concept)
        
        # Add structure information if available
        if hasattr(protein, 'coordinates') and protein.coordinates is not None:
            structure_predicate = self.create_predicate_node("hasStructure")
            structure_eval = self.create_evaluation_link(
                structure_predicate,
                protein_concept,
                1.0,  # Has structure
                TruthValue(0.9, 0.8)
            )
        
        # Add function keywords if available  
        if hasattr(protein, 'function_keywords') and protein.function_keywords:
            for keyword in protein.function_keywords:
                if keyword.strip():
                    function_concept = self.create_concept_node(f"Function_{keyword}")
                    function_link = self.create_inheritance_link(
                        protein_concept, 
                        function_concept,
                        TruthValue(0.8, 0.7)
                    )
        
        return protein_concept
    
    def get_atom(self, atom_id: str) -> Optional[AtomHandle]:
        """Retrieve atom by ID"""
        return self.atoms.get(atom_id)
    
    def get_atoms_by_type(self, atom_type: str) -> List[AtomHandle]:
        """Get all atoms of specified type"""
        return [atom for atom in self.atoms.values() if atom.atom_type == atom_type]
    
    def get_atoms_by_name_pattern(self, pattern: str) -> List[AtomHandle]:
        """Get atoms matching name pattern"""
        return [atom for atom in self.atoms.values() if pattern in atom.name]
    
    def set_truth_value(self, atom_id: str, truth_value: TruthValue) -> bool:
        """Update truth value for an atom"""
        if atom_id in self.atoms:
            self.atoms[atom_id].truth_value = truth_value
            return True
        return False
    
    def get_incoming_links(self, atom: AtomHandle) -> List[AtomHandle]:
        """Get links that include this atom"""
        incoming = []
        for link_id, linked_atoms in self.links.items():
            if atom in linked_atoms:
                incoming.append(self.atoms[link_id])
        return incoming
    
    def get_outgoing_atoms(self, link: AtomHandle) -> List[AtomHandle]:
        """Get atoms connected by this link"""
        return self.links.get(link.atom_id, [])
    
    def atom_count(self) -> int:
        """Total number of atoms in the space"""
        return len(self.atoms)
    
    def to_scheme_representation(self, atom: AtomHandle) -> str:
        """Generate Scheme representation of atom for cognitive processing"""
        if atom.atom_type == "ConceptNode":
            tv_str = f" (stv {atom.truth_value.strength:.3f} {atom.truth_value.confidence:.3f})"
            return f"(ConceptNode \"{atom.name}\"{tv_str})"
        elif atom.atom_type == "PredicateNode":
            tv_str = f" (stv {atom.truth_value.strength:.3f} {atom.truth_value.confidence:.3f})"
            return f"(PredicateNode \"{atom.name}\"{tv_str})"
        elif atom.atom_type == "InheritanceLink":
            outgoing = self.get_outgoing_atoms(atom)
            if len(outgoing) == 2:
                child_scheme = self.to_scheme_representation(outgoing[0])
                parent_scheme = self.to_scheme_representation(outgoing[1])
                tv_str = f" (stv {atom.truth_value.strength:.3f} {atom.truth_value.confidence:.3f})"
                return f"(InheritanceLink{tv_str}\n  {child_scheme}\n  {parent_scheme})"
        elif atom.atom_type == "EvaluationLink":
            outgoing = self.get_outgoing_atoms(atom)
            if len(outgoing) == 2:
                pred_scheme = self.to_scheme_representation(outgoing[0])
                concept_scheme = self.to_scheme_representation(outgoing[1])
                tv_str = f" (stv {atom.truth_value.strength:.3f} {atom.truth_value.confidence:.3f})"
                return f"(EvaluationLink{tv_str}\n  {pred_scheme}\n  {concept_scheme})"
        
        return f"({atom.atom_type} \"{atom.name}\")"
    
    def create_hypergraph_pattern_encoding(self, root_concept: AtomHandle) -> str:
        """Generate hypergraph pattern for the protein structure"""
        patterns = []
        patterns.append(f";; Hypergraph Pattern for {root_concept.name}")
        patterns.append(self.to_scheme_representation(root_concept))
        
        # Add related structures
        incoming = self.get_incoming_links(root_concept)
        for link in incoming:
            patterns.append(self.to_scheme_representation(link))
        
        return "\n".join(patterns)