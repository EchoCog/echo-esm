"""
Meta-Optimizing Semantic Evolutionary Search (MOSES) for ESM3 Cognitive Framework

Implements evolutionary optimization for protein design and analysis:
- Pattern recognition in protein sequences
- Strategy evolution for protein optimization
- Fitness-based selection of protein variants
- Semantic evolutionary search

MOSES discovers optimal protein configurations through evolutionary computation.
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
import random
from enum import Enum

from .atomspace import AtomHandle, TruthValue, ProteinAtomSpace
from esm.sdk.api import ESMProtein


class FitnessType(Enum):
    SEQUENCE_QUALITY = "sequence_quality"
    STRUCTURE_STABILITY = "structure_stability" 
    FUNCTION_OPTIMIZATION = "function_optimization"
    MULTI_OBJECTIVE = "multi_objective"


@dataclass
class EvolutionaryStrategy:
    """Evolutionary strategy discovered by MOSES"""
    strategy_id: str
    description: str
    fitness_score: float
    parameters: Dict[str, Any]
    success_rate: float
    generations_discovered: int


@dataclass
class ProteinVariant:
    """Protein variant in evolutionary population"""
    variant_id: str
    protein_atom: AtomHandle
    fitness_scores: Dict[FitnessType, float]
    generation: int
    parent_ids: List[str]
    mutations: List[str]


class ProteinMOSES:
    """
    Meta-Optimizing Semantic Evolutionary Search for proteins
    
    Performs evolutionary optimization to:
    - Discover optimal protein sequences
    - Evolve protein design strategies
    - Optimize structure-function relationships
    - Learn from evolutionary patterns
    """
    
    def __init__(self, atomspace: ProteinAtomSpace):
        self.atomspace = atomspace
        self.population: List[ProteinVariant] = []
        self.strategies: Dict[str, EvolutionaryStrategy] = {}
        self.generation = 0
        self.fitness_functions: Dict[FitnessType, Callable] = {}
        self._initialize_fitness_functions()
        
    def _initialize_fitness_functions(self):
        """Initialize fitness evaluation functions"""
        self.fitness_functions = {
            FitnessType.SEQUENCE_QUALITY: self._evaluate_sequence_quality,
            FitnessType.STRUCTURE_STABILITY: self._evaluate_structure_stability,
            FitnessType.FUNCTION_OPTIMIZATION: self._evaluate_function_optimization,
            FitnessType.MULTI_OBJECTIVE: self._evaluate_multi_objective
        }
    
    def _evaluate_sequence_quality(self, protein_atom: AtomHandle) -> float:
        """Evaluate protein sequence quality"""
        # Get sequence-related atoms
        sequence_atoms = []
        for link in self.atomspace.get_incoming_links(protein_atom):
            if "sequence" in link.name.lower():
                sequence_atoms.append(link)
        
        if not sequence_atoms:
            return 0.0
        
        # Factors for sequence quality
        factors = []
        
        # Length appropriateness
        length_eval = self._extract_sequence_length(protein_atom)
        if 50 <= length_eval <= 500:  # Reasonable protein length
            factors.append(1.0)
        elif 30 <= length_eval <= 1000:
            factors.append(0.7)
        else:
            factors.append(0.3)
        
        # Amino acid composition diversity
        composition_score = self._evaluate_composition_diversity(protein_atom)
        factors.append(composition_score)
        
        # Secondary structure propensity
        ss_score = self._evaluate_secondary_structure_propensity(protein_atom)
        factors.append(ss_score)
        
        return np.mean(factors) if factors else 0.0
    
    def _evaluate_structure_stability(self, protein_atom: AtomHandle) -> float:
        """Evaluate predicted structural stability"""
        structure_links = [link for link in self.atomspace.get_incoming_links(protein_atom)
                          if "structure" in link.name.lower()]
        
        if not structure_links:
            return 0.5  # Unknown structure
        
        # Stability factors
        factors = []
        
        # Hydrophobic core formation potential
        hydrophobic_score = self._evaluate_hydrophobic_core(protein_atom)
        factors.append(hydrophobic_score)
        
        # Disulfide bond potential
        cysteine_score = self._evaluate_disulfide_potential(protein_atom)
        factors.append(cysteine_score)
        
        # Charge distribution
        charge_score = self._evaluate_charge_distribution(protein_atom)
        factors.append(charge_score)
        
        return np.mean(factors) if factors else 0.5
    
    def _evaluate_function_optimization(self, protein_atom: AtomHandle) -> float:
        """Evaluate functional optimization potential"""
        function_links = [link for link in self.atomspace.get_incoming_links(protein_atom)
                         if "function" in link.name.lower()]
        
        if not function_links:
            return 0.5  # Unknown function
        
        # Function-related scoring
        factors = []
        
        # Active site conservation
        active_site_score = self._evaluate_active_site_conservation(protein_atom)
        factors.append(active_site_score)
        
        # Functional domain integrity  
        domain_score = self._evaluate_domain_integrity(protein_atom)
        factors.append(domain_score)
        
        # Allosteric potential
        allosteric_score = self._evaluate_allosteric_potential(protein_atom)
        factors.append(allosteric_score)
        
        return np.mean(factors) if factors else 0.5
    
    def _evaluate_multi_objective(self, protein_atom: AtomHandle) -> float:
        """Multi-objective fitness combining all factors"""
        seq_score = self._evaluate_sequence_quality(protein_atom)
        struct_score = self._evaluate_structure_stability(protein_atom)
        func_score = self._evaluate_function_optimization(protein_atom)
        
        # Weighted combination
        weights = [0.3, 0.4, 0.3]  # Sequence, Structure, Function
        return np.average([seq_score, struct_score, func_score], weights=weights)
    
    def _extract_sequence_length(self, protein_atom: AtomHandle) -> int:
        """Extract sequence length from protein atom"""
        for link in self.atomspace.get_incoming_links(protein_atom):
            if "sequenceLength" in link.name:
                try:
                    length_str = link.name.split('=')[1]
                    return int(float(length_str))
                except:
                    continue
        return 0
    
    def _evaluate_composition_diversity(self, protein_atom: AtomHandle) -> float:
        """Evaluate amino acid composition diversity"""
        aa_atoms = [atom for atom in self.atomspace.get_atoms_by_name_pattern("AminoAcid_")]
        if len(aa_atoms) < 5:
            return 0.5
        
        # Count amino acid types
        aa_counts = {}
        for aa_atom in aa_atoms:
            if aa_atom.name.startswith("AminoAcid_"):
                aa_type = aa_atom.name.split("_")[1]
                aa_counts[aa_type] = aa_counts.get(aa_type, 0) + 1
        
        if not aa_counts:
            return 0.0
        
        # Shannon entropy for diversity
        total = sum(aa_counts.values())
        entropy = 0.0
        for count in aa_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        # Normalize by maximum possible entropy (log2(20) for 20 amino acids)
        max_entropy = np.log2(20)
        return entropy / max_entropy
    
    def _evaluate_secondary_structure_propensity(self, protein_atom: AtomHandle) -> float:
        """Evaluate secondary structure forming potential"""
        # Simplified SS propensity based on amino acid preferences
        helix_favorable = {'A', 'E', 'L', 'M'}
        sheet_favorable = {'V', 'I', 'Y', 'F'}
        turn_favorable = {'G', 'N', 'P', 'S', 'D'}
        
        aa_atoms = [atom for atom in self.atomspace.get_atoms_by_name_pattern("AminoAcid_")]
        if not aa_atoms:
            return 0.5
        
        helix_count = sheet_count = turn_count = 0
        
        for aa_atom in aa_atoms:
            if aa_atom.name.startswith("AminoAcid_"):
                aa_type = aa_atom.name.split("_")[1]
                if aa_type in helix_favorable:
                    helix_count += 1
                elif aa_type in sheet_favorable:
                    sheet_count += 1
                elif aa_type in turn_favorable:
                    turn_count += 1
        
        total = len(aa_atoms)
        if total == 0:
            return 0.5
        
        # Good balance of secondary structure elements
        helix_ratio = helix_count / total
        sheet_ratio = sheet_count / total
        turn_ratio = turn_count / total
        
        # Prefer moderate ratios for each SS type
        balance_score = 1.0 - abs(helix_ratio - 0.3) - abs(sheet_ratio - 0.3) - abs(turn_ratio - 0.2)
        return max(0.0, min(1.0, balance_score))
    
    def _evaluate_hydrophobic_core(self, protein_atom: AtomHandle) -> float:
        """Evaluate hydrophobic core formation potential"""
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
        
        hydrophobic_ratio = hydrophobic_count / len(aa_atoms)
        
        # Optimal hydrophobic ratio for stable proteins
        if 0.3 <= hydrophobic_ratio <= 0.6:
            return 1.0
        elif 0.2 <= hydrophobic_ratio <= 0.7:
            return 0.8
        else:
            return 0.4
    
    def _evaluate_disulfide_potential(self, protein_atom: AtomHandle) -> float:
        """Evaluate disulfide bond formation potential"""
        cysteine_count = 0
        aa_atoms = [atom for atom in self.atomspace.get_atoms_by_name_pattern("AminoAcid_C")]
        cysteine_count = len(aa_atoms)
        
        # Even number of cysteines is favorable for disulfide bonds
        if cysteine_count % 2 == 0 and cysteine_count >= 2:
            return 0.9
        elif cysteine_count == 0:
            return 0.7  # No cysteines is also okay
        else:
            return 0.5  # Odd number might leave unpaired cysteines
    
    def _evaluate_charge_distribution(self, protein_atom: AtomHandle) -> float:
        """Evaluate charge distribution for stability"""
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
        
        if positive_count + negative_count == 0:
            return 0.7  # Neutral is okay
        
        # Reasonable charge balance
        charge_imbalance = abs(positive_count - negative_count) / (positive_count + negative_count)
        
        if charge_imbalance <= 0.2:
            return 1.0
        elif charge_imbalance <= 0.5:
            return 0.8
        else:
            return 0.4
    
    def _evaluate_active_site_conservation(self, protein_atom: AtomHandle) -> float:
        """Evaluate conservation of putative active site residues"""
        # Placeholder for active site analysis
        # Would need specific functional information
        return 0.75
    
    def _evaluate_domain_integrity(self, protein_atom: AtomHandle) -> float:
        """Evaluate functional domain integrity"""
        # Placeholder for domain analysis
        return 0.7
    
    def _evaluate_allosteric_potential(self, protein_atom: AtomHandle) -> float:
        """Evaluate allosteric regulation potential"""
        # Placeholder for allosteric site analysis
        return 0.65
    
    def initialize_population(self, protein_atoms: List[AtomHandle], population_size: int = 50):
        """Initialize evolutionary population"""
        self.population = []
        
        # Add seed proteins to population
        for i, protein_atom in enumerate(protein_atoms[:population_size]):
            fitness_scores = {}
            for fitness_type in FitnessType:
                fitness_scores[fitness_type] = self.fitness_functions[fitness_type](protein_atom)
            
            variant = ProteinVariant(
                variant_id=f"seed_{i}",
                protein_atom=protein_atom,
                fitness_scores=fitness_scores,
                generation=0,
                parent_ids=[],
                mutations=[]
            )
            
            self.population.append(variant)
        
        # Fill remaining population with variants if needed
        while len(self.population) < population_size:
            # Create variant by mutating existing protein
            parent = random.choice(self.population)
            variant = self._create_variant(parent)
            self.population.append(variant)
    
    def _create_variant(self, parent: ProteinVariant) -> ProteinVariant:
        """Create variant through mutation"""
        # Simplified mutation - would need actual protein mutation logic
        variant_id = f"variant_{self.generation}_{len(self.population)}"
        
        # For now, copy parent atom (in real implementation, would mutate sequence)
        variant_atom = parent.protein_atom
        
        # Evaluate fitness
        fitness_scores = {}
        for fitness_type in FitnessType:
            fitness_scores[fitness_type] = self.fitness_functions[fitness_type](variant_atom)
        
        return ProteinVariant(
            variant_id=variant_id,
            protein_atom=variant_atom,
            fitness_scores=fitness_scores,
            generation=self.generation,
            parent_ids=[parent.variant_id],
            mutations=["point_mutation"]  # Placeholder
        )
    
    def evolve_generation(self, fitness_type: FitnessType = FitnessType.MULTI_OBJECTIVE) -> List[ProteinVariant]:
        """Evolve one generation"""
        self.generation += 1
        
        # Selection - choose top performers
        sorted_population = sorted(self.population, 
                                 key=lambda v: v.fitness_scores[fitness_type], 
                                 reverse=True)
        
        elite_size = len(self.population) // 4  # Top 25%
        elite = sorted_population[:elite_size]
        
        # Create new generation
        new_population = elite.copy()  # Keep elite
        
        # Generate offspring
        while len(new_population) < len(self.population):
            # Tournament selection for parents
            parent1 = self._tournament_selection(fitness_type)
            parent2 = self._tournament_selection(fitness_type)
            
            # Create offspring (mutation for now, crossover could be added)
            offspring = self._create_variant(parent1)
            new_population.append(offspring)
        
        self.population = new_population
        return new_population
    
    def _tournament_selection(self, fitness_type: FitnessType, tournament_size: int = 3) -> ProteinVariant:
        """Tournament selection for parent choice"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda v: v.fitness_scores[fitness_type])
    
    def discover_strategies(self, generations: int = 20) -> List[EvolutionaryStrategy]:
        """Discover optimization strategies through evolution"""
        strategies = []
        
        for gen in range(generations):
            # Evolve population
            new_pop = self.evolve_generation()
            
            # Analyze successful patterns every 5 generations
            if gen % 5 == 0:
                strategy = self._analyze_successful_patterns(gen)
                if strategy:
                    strategies.append(strategy)
                    self.strategies[strategy.strategy_id] = strategy
        
        return strategies
    
    def _analyze_successful_patterns(self, generation: int) -> Optional[EvolutionaryStrategy]:
        """Analyze patterns in successful variants"""
        # Get top performers
        top_performers = sorted(self.population, 
                              key=lambda v: v.fitness_scores[FitnessType.MULTI_OBJECTIVE],
                              reverse=True)[:5]
        
        if not top_performers:
            return None
        
        # Analyze common patterns (simplified)
        avg_fitness = np.mean([v.fitness_scores[FitnessType.MULTI_OBJECTIVE] for v in top_performers])
        
        if avg_fitness > 0.7:  # Good performance threshold
            strategy_id = f"strategy_gen_{generation}"
            
            # Extract parameters from successful variants
            parameters = {
                'avg_fitness': avg_fitness,
                'population_diversity': len(set(v.variant_id[:10] for v in top_performers)),  # Simplified diversity
                'successful_mutations': [mut for v in top_performers for mut in v.mutations]
            }
            
            strategy = EvolutionaryStrategy(
                strategy_id=strategy_id,
                description=f"High-performance strategy discovered at generation {generation}",
                fitness_score=avg_fitness,
                parameters=parameters,
                success_rate=len(top_performers) / len(self.population),
                generations_discovered=generation
            )
            
            return strategy
        
        return None
    
    def optimize_protein(self, protein_atom: AtomHandle, 
                        fitness_type: FitnessType = FitnessType.MULTI_OBJECTIVE,
                        generations: int = 10) -> ProteinVariant:
        """Optimize a specific protein through evolution"""
        # Initialize population with variants of the target protein
        self.population = []
        
        # Create population of variants
        for i in range(20):  # Small population for single protein optimization
            if i == 0:
                # Include original protein
                fitness_scores = {}
                for ft in FitnessType:
                    fitness_scores[ft] = self.fitness_functions[ft](protein_atom)
                
                variant = ProteinVariant(
                    variant_id=f"original",
                    protein_atom=protein_atom,
                    fitness_scores=fitness_scores,
                    generation=0,
                    parent_ids=[],
                    mutations=[]
                )
            else:
                # Create variants
                variant = self._create_variant(ProteinVariant(
                    variant_id="template",
                    protein_atom=protein_atom,
                    fitness_scores={ft: 0.5 for ft in FitnessType},
                    generation=0,
                    parent_ids=[],
                    mutations=[]
                ))
            
            self.population.append(variant)
        
        # Evolve for specified generations
        for _ in range(generations):
            self.evolve_generation(fitness_type)
        
        # Return best variant
        best_variant = max(self.population, key=lambda v: v.fitness_scores[fitness_type])
        return best_variant
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not self.population:
            return {'status': 'no_population'}
        
        fitness_stats = {}
        for fitness_type in FitnessType:
            scores = [v.fitness_scores[fitness_type] for v in self.population]
            fitness_stats[fitness_type.value] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'max': np.max(scores),
                'min': np.min(scores)
            }
        
        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'fitness_statistics': fitness_stats,
            'discovered_strategies': len(self.strategies),
            'strategy_list': list(self.strategies.keys())
        }