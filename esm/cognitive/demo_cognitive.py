"""
Example usage and demonstration of ESM3 Cognitive Accounting Framework

This script demonstrates the key features and capabilities of the framework:
- Creating cognitive proteins with different account types
- AtomSpace knowledge representation  
- PLN reasoning and validation
- ECAN attention allocation
- MOSES evolutionary optimization
- URE uncertainty quantification
- Inter-module communication
- Emergent behavior detection
"""

import time
import numpy as np
from typing import List, Dict, Any

# Import ESM3 SDK
from esm.sdk.api import ESMProtein

# Import cognitive framework components
from esm.cognitive import (
    CognitiveAccountingFramework,
    CognitiveAccountType, 
    CognitiveProtein,
    FitnessType,
    TruthValue
)


def create_sample_proteins() -> List[ESMProtein]:
    """Create sample proteins for demonstration"""
    
    # Sample protein sequences (shortened for demo)
    sequences = [
        # Small enzyme-like protein
        "MKLLVLLAIVCFGAAALAQQGTSDPEKSKMNQIIQRPVFNGQHFNEHHHFFGSAAHLHYGRPQCCSYGFGGLYVYNQRVSGGL",
        
        # Structural protein fragment  
        "MSPVLVQMSPKGSPQVVQVVQVMQDQVLVQMSPKGSPQVVQVVQVMQDQVLVQMSPKGSPQVVQVVQVMQDQVLVQMSPKGSP",
        
        # Membrane protein-like
        "MGAAALLLWLWLWLWGGGPPPAAAVVVIIIFFFLLLAAALLLPPPGGGAAAVVVIIIFFFLLLAAALLLPPPGGGAAAVVVIII",
        
        # Catalytic domain
        "MHDEYGRATGGQRKEEALEKVARLKQLHPDIIFTAFDQFTKQGVYLIGTVSKPDHVLLGTQAQDLLKIAAGYSDHQVR",
        
        # Binding domain
        "MTGRKVYSGKGGVQQFQITPTQVLQVQPMLQVMPQVQVQMQVQVQVQVQVQVQVQVQVQVQVQVQVQVQVQVQVQ"
    ]
    
    proteins = []
    for i, seq in enumerate(sequences):
        protein = ESMProtein(sequence=seq)
        
        # Add mock function keywords
        if i == 0:
            protein.function_keywords = ["enzyme", "catalysis", "metabolic"]
        elif i == 1:
            protein.function_keywords = ["structural", "scaffold", "support"] 
        elif i == 2:
            protein.function_keywords = ["transport", "membrane", "channel"]
        elif i == 3:
            protein.function_keywords = ["binding", "catalytic", "active_site"]
        else:
            protein.function_keywords = ["regulatory", "binding", "control"]
            
        proteins.append(protein)
    
    return proteins


def demonstrate_atomspace_representation(framework: CognitiveAccountingFramework, 
                                       cognitive_proteins: List[CognitiveProtein]):
    """Demonstrate AtomSpace knowledge representation"""
    print("\n🧠 AtomSpace Knowledge Representation")
    print("=" * 50)
    
    # Show AtomSpace statistics
    atom_count = framework.atomspace.atom_count()
    concept_nodes = len(framework.atomspace.get_atoms_by_type("ConceptNode"))
    predicate_nodes = len(framework.atomspace.get_atoms_by_type("PredicateNode"))
    
    print(f"Total atoms in AtomSpace: {atom_count}")
    print(f"ConceptNodes: {concept_nodes}")
    print(f"PredicateNodes: {predicate_nodes}")
    
    # Show Scheme representation for first protein
    if cognitive_proteins:
        first_protein = cognitive_proteins[0]
        if first_protein.protein_atom:
            scheme_repr = framework.atomspace.to_scheme_representation(first_protein.protein_atom)
            print(f"\nScheme representation of first protein:")
            print(scheme_repr)
            
            # Show hypergraph pattern
            pattern = framework.atomspace.create_hypergraph_pattern_encoding(first_protein.protein_atom)
            print(f"\nHypergraph pattern (first 200 chars):")
            print(pattern[:200] + "..." if len(pattern) > 200 else pattern)


def demonstrate_pln_reasoning(framework: CognitiveAccountingFramework,
                            protein_ids: List[str]):
    """Demonstrate PLN probabilistic reasoning"""
    print("\n⚖️ PLN Probabilistic Logic Networks")
    print("=" * 50)
    
    # Validate individual proteins
    print("Individual protein validations:")
    for i, protein_id in enumerate(protein_ids[:3]):
        cognitive_protein = framework.cognitive_proteins[protein_id]
        if cognitive_protein.protein_atom:
            validation = framework.pln.validate_protein(cognitive_protein.protein_atom)
            print(f"Protein {i+1}: strength={validation.strength:.3f}, confidence={validation.confidence:.3f}")
    
    # Generate trial balance proof
    print(f"\nGenerating trial balance proof for {len(protein_ids)} proteins...")
    trial_proof = framework.validate_protein_set(protein_ids)
    
    print(f"Trial Balance Proof:")
    print(f"  Theorem: {trial_proof.theorem}")
    print(f"  Confidence: {trial_proof.confidence:.3f}")
    print(f"  Premises: {len(trial_proof.premises)} evidence atoms")
    print(f"  Proof steps: {len(trial_proof.steps)}")
    
    # Show first few proof steps
    print(f"\nProof steps (first 3):")
    for i, step in enumerate(trial_proof.steps[:3]):
        print(f"  {i+1}. {step}")
    
    # Function-structure proof for first protein
    if protein_ids:
        first_id = protein_ids[0]
        first_protein = framework.cognitive_proteins[first_id]
        if first_protein.protein_atom:
            func_struct_proof = framework.pln.generate_function_structure_proof(first_protein.protein_atom)
            print(f"\nFunction-Structure Consistency Proof:")
            print(f"  Confidence: {func_struct_proof.confidence:.3f}")


def demonstrate_ecan_attention(framework: CognitiveAccountingFramework,
                             cognitive_proteins: List[CognitiveProtein]):
    """Demonstrate ECAN attention allocation"""
    print("\n👁️ ECAN Economic Attention Allocation")
    print("=" * 50)
    
    # Show initial attention distribution
    print("Initial attention values:")
    for i, cp in enumerate(cognitive_proteins):
        if cp.protein_atom:
            attention = framework.ecan.get_attention_value(cp.protein_atom)
            print(f"Protein {i+1}: STI={attention.sti:.1f}, LTI={attention.lti:.1f}, VLTI={attention.vlti:.1f}")
    
    # Simulate activities to trigger attention updates
    print(f"\nSimulating protein activities...")
    activities = ["validation", "optimization", "prediction", "binding", "catalysis"]
    
    for i, cp in enumerate(cognitive_proteins):
        if cp.protein_atom:
            activity = activities[i % len(activities)]
            intensity = np.random.uniform(0.5, 2.0)
            framework.ecan.record_activity(cp.protein_atom, activity, intensity)
            print(f"Protein {i+1}: {activity} (intensity={intensity:.2f})")
    
    # Run attention allocation cycles
    print(f"\nRunning 3 ECAN attention cycles...")
    for cycle in range(3):
        framework.ecan.allocate_attention_cycle()
    
    # Show updated attention values
    print(f"\nAttention after cycles:")
    for i, cp in enumerate(cognitive_proteins):
        if cp.protein_atom:
            attention = framework.ecan.get_attention_value(cp.protein_atom)
            print(f"Protein {i+1}: STI={attention.sti:.1f}, LTI={attention.lti:.1f}, VLTI={attention.vlti:.1f}")
    
    # Show attention focus
    focused_atoms = framework.ecan.get_attentional_focus()
    print(f"\nProteins in attentional focus: {len(focused_atoms)}")
    
    # Top attention atoms
    top_sti = framework.ecan.get_top_attention_atoms(3)
    print(f"\nTop 3 STI atoms:")
    for atom, score in top_sti:
        print(f"  {atom.name}: {score:.1f}")


def demonstrate_moses_optimization(framework: CognitiveAccountingFramework,
                                 cognitive_proteins: List[CognitiveProtein]):
    """Demonstrate MOSES evolutionary optimization"""
    print("\n🧬 MOSES Evolutionary Optimization")
    print("=" * 50)
    
    # Initialize population with current proteins
    protein_atoms = [cp.protein_atom for cp in cognitive_proteins if cp.protein_atom]
    if not protein_atoms:
        print("No protein atoms available for optimization")
        return
    
    print(f"Initializing population with {len(protein_atoms)} proteins...")
    framework.moses.initialize_population(protein_atoms, population_size=min(20, len(protein_atoms) * 4))
    
    # Show initial fitness distribution
    print("\nInitial fitness scores (Multi-objective):")
    for i, variant in enumerate(framework.moses.population[:5]):
        fitness = variant.fitness_scores[FitnessType.MULTI_OBJECTIVE]
        print(f"Variant {i+1}: {fitness:.3f}")
    
    # Run evolutionary cycles
    print(f"\nRunning 5 evolutionary generations...")
    strategies = framework.moses.discover_strategies(generations=5)
    
    # Show optimization results
    optimization_summary = framework.moses.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"  Generation: {optimization_summary['generation']}")
    print(f"  Population size: {optimization_summary['population_size']}")
    print(f"  Discovered strategies: {optimization_summary['discovered_strategies']}")
    
    # Show fitness statistics
    fitness_stats = optimization_summary['fitness_statistics'][FitnessType.MULTI_OBJECTIVE.value]
    print(f"\nFitness Statistics (Multi-objective):")
    print(f"  Mean: {fitness_stats['mean']:.3f}")
    print(f"  Max: {fitness_stats['max']:.3f}")
    print(f"  Std Dev: {fitness_stats['std']:.3f}")
    
    # Show discovered strategies
    if strategies:
        print(f"\nDiscovered Strategies:")
        for strategy in strategies[:2]:  # Show first 2
            print(f"  {strategy.strategy_id}: {strategy.description}")
            print(f"    Fitness: {strategy.fitness_score:.3f}")
            print(f"    Success rate: {strategy.success_rate:.3f}")


def demonstrate_ure_uncertainty(framework: CognitiveAccountingFramework,
                              cognitive_proteins: List[CognitiveProtein]):
    """Demonstrate URE uncertainty reasoning"""
    print("\n🎯 URE Uncertain Reasoning Engine")
    print("=" * 50)
    
    # Make predictions for first few proteins
    predictions = []
    properties = ["stability", "binding_affinity", "function_enzyme"]
    
    for i, cp in enumerate(cognitive_proteins[:3]):
        if cp.protein_atom:
            protein_id = list(framework.cognitive_proteins.keys())[i]
            property_name = properties[i % len(properties)]
            
            prediction = framework.predict_with_uncertainty(protein_id, property_name)
            predictions.append((protein_id, property_name, prediction))
            
            print(f"\nProtein {i+1} - {property_name}:")
            print(f"  Prediction: {prediction.prediction:.3f}")
            print(f"  Confidence interval: [{prediction.confidence_interval[0]:.3f}, {prediction.confidence_interval[1]:.3f}]")
            print(f"  Total uncertainty: {prediction.total_uncertainty:.3f}")
            
            # Show uncertainty factors
            print(f"  Uncertainty sources:")
            for factor in prediction.uncertainty_factors:
                print(f"    {factor.factor_type.value}: {factor.magnitude:.3f} (conf: {factor.confidence:.3f})")
    
    # Comprehensive uncertainty analysis
    if cognitive_proteins and cognitive_proteins[0].protein_atom:
        protein_id = list(framework.cognitive_proteins.keys())[0]
        cp = framework.cognitive_proteins[protein_id]
        
        uncertainty_analysis = framework.ure.get_uncertainty_analysis(cp.protein_atom)
        print(f"\nComprehensive Uncertainty Analysis for Protein 1:")
        print(f"  Stability prediction: {uncertainty_analysis['predictions']['stability']['prediction']:.3f}")
        print(f"  Binding affinity prediction: {uncertainty_analysis['predictions']['binding_affinity']['prediction']:.3f}")
        
        print(f"\n  Uncertainty sources summary:")
        for source, stats in uncertainty_analysis['uncertainty_sources'].items():
            print(f"    {source}: mean={stats['mean']:.3f}, max={stats['max']:.3f}")


def demonstrate_cognitive_features(framework: CognitiveAccountingFramework,
                                 cognitive_proteins: List[CognitiveProtein]):
    """Demonstrate cognitive protein features"""
    print("\n🤖 Cognitive Protein Features")  
    print("=" * 50)
    
    # Enable different cognitive features on different proteins
    features = [
        (CognitiveAccountType.ADAPTIVE, "Learning"),
        (CognitiveAccountType.PREDICTIVE, "Prediction"), 
        (CognitiveAccountType.MULTIMODAL, "Multi-modal"),
        (CognitiveAccountType.ATTENTION_DRIVEN, "Attention-driven")
    ]
    
    for i, cp in enumerate(cognitive_proteins[:4]):
        feature_type, feature_name = features[i % len(features)]
        
        print(f"\nProtein {i+1} - Enabling {feature_name}:")
        
        if feature_type == CognitiveAccountType.ADAPTIVE:
            cp.enable_learning()
            
            # Simulate learning experience
            for j in range(3):
                experience = {
                    'sequence_activity': np.random.uniform(0.6, 0.9),
                    'structure_stability': np.random.uniform(0.5, 0.8),
                    'function_performance': np.random.uniform(0.7, 0.95)
                }
                performance = np.mean(list(experience.values()))
                cp.learn_from_experience(experience, performance)
            
            print(f"  Learning iterations: {cp.learning_history.iterations}")
            print(f"  Learned patterns: {len(getattr(cp, 'learned_patterns', {}))}")
            print(f"  Average performance: {np.mean(cp.learning_history.performance_scores):.3f}")
            
        elif feature_type == CognitiveAccountType.PREDICTIVE:
            cp.enable_prediction()
            
            predictions = cp.predict_future_state(steps_ahead=5)
            print(f"  Prediction horizon: {cp.capabilities.prediction_horizon}")
            print(f"  Future predictions available: {len([k for k in predictions.keys() if 'prediction_timestamp' not in k])}")
            
        elif feature_type == CognitiveAccountType.MULTIMODAL:
            cp.enable_multimodal_processing()
            
            # Test different modalities
            seq_result = cp.process_multimodal_input('sequence', 'ACDEFGHIKLMNPQRSTVWY')
            struct_result = cp.process_multimodal_input('structure', {'coordinates': 'mock'})
            func_result = cp.process_multimodal_input('function', ['enzyme', 'catalysis'])
            
            print(f"  Processed modalities: sequence, structure, function")
            print(f"  Episodic memory size: {len(cp.episodic_memory)}")
            
        elif feature_type == CognitiveAccountType.ATTENTION_DRIVEN:
            cp.enable_attention_driven_processing(framework.ecan)
            
            if cp.attention_value:
                print(f"  Attention STI: {cp.attention_value.sti:.1f}")
                print(f"  Attention LTI: {cp.attention_value.lti:.1f}")
    
    # Show cognitive summaries
    print(f"\nCognitive State Summaries:")
    for i, cp in enumerate(cognitive_proteins[:3]):
        summary = cp.get_cognitive_summary()
        print(f"\nProtein {i+1}:")
        print(f"  Cognitive type: {', '.join(summary['cognitive_type']['flags'])}")
        print(f"  State: {summary['cognitive_state']}")
        print(f"  Learning iterations: {summary['learning_history']['iterations']}")
        print(f"  Memory usage: working={summary['memory_state']['working_memory_size']}, episodic={summary['memory_state']['episodic_memory_size']}")


def demonstrate_emergent_behavior(framework: CognitiveAccountingFramework):
    """Demonstrate emergent behavior detection"""
    print("\n✨ Emergent Behavior Detection")
    print("=" * 50)
    
    # Run multiple cognitive cycles to generate activity
    print("Running 10 cognitive cycles to generate emergent patterns...")
    
    for cycle in range(10):
        framework.run_cognitive_cycle()
        time.sleep(0.1)  # Brief pause between cycles
    
    # Show emergent patterns
    patterns = framework.emergent_patterns
    print(f"\nDetected emergent patterns: {len(patterns)}")
    
    if patterns:
        for i, pattern in enumerate(patterns[-3:]):  # Show last 3
            print(f"\nPattern {i+1}:")
            print(f"  Type: {pattern['type']}")
            print(f"  Complexity: {pattern['complexity']:.3f}")
            print(f"  Novelty: {pattern['novelty']:.3f}") 
            print(f"  Description: {pattern['description']}")
    else:
        print("No emergent patterns detected yet. This is normal for short demonstrations.")
    
    # Show system performance metrics
    print(f"\nSystem Performance Metrics:")
    for metric, values in framework.performance_metrics.items():
        if values:
            print(f"  {metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")


def main():
    """Main demonstration function"""
    print("🧠 ESM3 Cognitive Accounting Framework Demonstration")
    print("=" * 60)
    
    print("Initializing cognitive framework...")
    framework = CognitiveAccountingFramework()
    
    print("Creating sample proteins...")
    sample_proteins = create_sample_proteins()
    
    print("Adding proteins to framework...")
    cognitive_proteins = []
    protein_ids = []
    
    cognitive_types = [
        CognitiveAccountType.TRADITIONAL,
        CognitiveAccountType.ADAPTIVE, 
        CognitiveAccountType.PREDICTIVE,
        CognitiveAccountType.MULTIMODAL,
        CognitiveAccountType.ATTENTION_DRIVEN
    ]
    
    for i, esm_protein in enumerate(sample_proteins):
        cognitive_type = cognitive_types[i % len(cognitive_types)]
        cp = framework.add_protein(esm_protein, cognitive_type)
        cognitive_proteins.append(cp)
        
        # Get protein ID (last added)
        protein_id = list(framework.cognitive_proteins.keys())[-1]
        protein_ids.append(protein_id)
    
    print(f"Added {len(cognitive_proteins)} proteins to framework")
    
    # Run demonstrations
    try:
        demonstrate_atomspace_representation(framework, cognitive_proteins)
        demonstrate_pln_reasoning(framework, protein_ids)
        demonstrate_ecan_attention(framework, cognitive_proteins) 
        demonstrate_moses_optimization(framework, cognitive_proteins)
        demonstrate_ure_uncertainty(framework, cognitive_proteins)
        demonstrate_cognitive_features(framework, cognitive_proteins)
        demonstrate_emergent_behavior(framework)
        
        # Final system status
        print("\n📊 Final System Status")
        print("=" * 50)
        status = framework.get_system_status()
        
        print(f"Framework initialized: {status['framework_initialized']}")
        print(f"Cognitive cycles completed: {status['cycle_count']}")
        print(f"Total proteins: {status['cognitive_proteins']}")
        print(f"AtomSpace atoms: {status['atomspace_atoms']}")
        print(f"Message queue size: {status['message_queue_size']}")
        print(f"Emergent patterns: {status['emergent_patterns']}")
        print(f"Performance metrics tracked: {len(status['performance_metrics'])}")
        
        if status['attention_summary']:
            att_sum = status['attention_summary']
            print(f"Attention focus atoms: {att_sum['focused_atoms']}")
            print(f"Total attention atoms: {att_sum['total_proteins']}")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\nShutting down framework...")
        framework.shutdown()
        print(f"Framework shutdown complete.")
    
    print(f"\n🎉 Demonstration completed successfully!")
    print(f"\nThe ESM3 Cognitive Accounting Framework successfully transformed")
    print(f"traditional protein modeling into a neural-symbolic cognitive system!")


if __name__ == "__main__":
    main()