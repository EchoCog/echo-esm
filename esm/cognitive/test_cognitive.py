"""
Test suite for ESM3 Cognitive Accounting Framework

Comprehensive tests for all cognitive components:
- AtomSpace functionality
- PLN reasoning
- ECAN attention allocation
- MOSES optimization
- URE uncertainty reasoning
- Framework integration
- Cognitive protein features
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

from esm.sdk.api import ESMProtein
from esm.cognitive import (
    ProteinAtomSpace, TruthValue, AtomHandle,
    ProteinPLN, PLNProof,
    ProteinECAN, AttentionValue, ECANParams,
    ProteinMOSES, FitnessType,
    ProteinURE, UncertaintyType,
    CognitiveProtein, CognitiveAccountType,
    CognitiveAccountingFramework
)


class TestProteinAtomSpace:
    """Test AtomSpace functionality"""
    
    def setup_method(self):
        self.atomspace = ProteinAtomSpace()
    
    def test_truth_value_creation(self):
        tv = TruthValue(0.8, 0.6)
        assert tv.strength == 0.8
        assert tv.confidence == 0.6
        
        # Test clamping
        tv_clamped = TruthValue(1.5, -0.1)
        assert tv_clamped.strength == 1.0
        assert tv_clamped.confidence == 0.0
    
    def test_concept_node_creation(self):
        node = self.atomspace.create_concept_node("TestProtein")
        assert node.atom_type == "ConceptNode"
        assert node.name == "TestProtein"
        assert node.truth_value is not None
        assert 0 <= node.truth_value.strength <= 1
        assert 0 <= node.truth_value.confidence <= 1
    
    def test_predicate_node_creation(self):
        pred = self.atomspace.create_predicate_node("hasSequence")
        assert pred.atom_type == "PredicateNode"
        assert pred.name == "hasSequence"
    
    def test_inheritance_link_creation(self):
        child = self.atomspace.create_concept_node("AminoAcid_A")
        parent = self.atomspace.create_concept_node("Protein_Test")
        
        link = self.atomspace.create_inheritance_link(child, parent)
        assert link.atom_type == "InheritanceLink"
        assert child in self.atomspace.get_outgoing_atoms(link)
        assert parent in self.atomspace.get_outgoing_atoms(link)
    
    def test_evaluation_link_creation(self):
        predicate = self.atomspace.create_predicate_node("hasLength")
        concept = self.atomspace.create_concept_node("TestSequence")
        
        eval_link = self.atomspace.create_evaluation_link(predicate, concept, 150.0)
        assert eval_link.atom_type == "EvaluationLink"
        assert "150.000" in eval_link.name
    
    def test_protein_to_atomspace(self):
        # Create mock ESMProtein
        mock_protein = Mock(spec=ESMProtein)
        mock_protein.sequence = "MKLLVLLAIVCFGAAALAQGTSDPEKSKMNQIIQRPVFNGQHFNEHHHFFGSAAHLHYGRPQCCSYGFGGLYVYNQRVSGGLSYAMLSLRHSVCFLPKGNHGAHMSFNPSRVSPEKNIGQYVGIRVRRRVTVQPEEYMVSTPEKKLPHSSQDLITRGSFDSSEDFDKRGKSYMGVYKCMDQSQRHAFKQEEHLLQSLPCKITQALIQGMLEDGNALFDIDFKTDGQMLRYFLPSGGSLVLWPHPEGKLNKYFDDEDLFLVQHFIDKSQFHFPAFMLNMVNRQRDKDNLLLSSSDSTEEEIINEKPTGFKSGKLVNYSHYFYGQHVQDGCPEFIINYLMPLIIPAPNPTLKQFVLLSDDVNGDFILLALSHPLLRGGPTDDILQNYCYCDFSGGSLSGRLQCDYMVTGSGWDQVLAFGPFAKQVGPFVIHPQGAAHKVADAVFHTMAHIHKGDRFGTLGSRRLSPYAGRNNVQLLMISLDGDQKRLRCGFAQLMEHYKGKLPKAYTAKVIALLIRQMKDGKASIDAAGKYDQSLMQNGHQSTDQASFKMVPQPRDMPNGLEHSLTKVFDFLLFHLAYNQEGGFGFRAEHPELGGQDQSQPGMGYLHVNVQIGYIKDGTGAYLFFDSPGSADGRRLLIPQVTPALLKDKTNSIYQGKSGAKMKMITLRGMIKFPIQLLSCLAKGVDQMNTLSGFTPIQLLRGAILVIKPLTKSGDKKLPKAKQGMGIDNLNQYTLQIGALYVAGQASSSITPQFLEEGYTLRVQEILLMHSKHVKMLLSNVNTLKGGNLVGVATVPAFQNGLGGDVTLLDFQEYDGQVNFLLLPNRTTLGRPYDSKQSTFQLEQFAKVGKLVAKLLKPNYTNARALFGAQAAQAAKGHTSSTSGGVLGNLVHGKAFLAFDTMYIVEGFMNLLADTQAQAKMTYRQFQHQVLLTRKNNTGNLGFYVDAIYLQIKFQKDGHKLRGTGKLTDFQKFLQTKSQFGFLQFSGRQKSKLVYPPMPFGDADKQAQMMLANQGGQKEVKRTIQAFVSLILQADFPQAQLGEKAVGQFKTLEAFQAAQESAAQEPELGTGKKEGKSLQKQPVKAQVQRRQIDWLKNSGKQWSPLRPVLGQGEGVALLRGAAESIVTQKAKRVLAGATMGGALVVQNYQGWQIKYDGYVEIVPPGAFVAGQTLKEGTRRVITARLQHAENSGAQLNYQGKRLQNMSLVPDLDGLQAGDWVQMVPVAFLRQGSLVQVWDLTLPDGRLTLQKQIQALDEQMQQRQVFGAGDIAGITSSEHFLVAALTFNGQHQVVVPKTEGSLIVKLDHPEAIGSAIGAARTLTRDAPRILRGTIEIGFLKQHLQGQKLQQLQLALQGLVEQSVRAAAGMAQQGLHKRRSIAGQYVQLPNPRNSPYLNSMKQGCQEDEEVNFLFQNLNQLDAGQAAEQFAAVKAGDVNLTPFMALLTSRSDGSNFAFRVVDGQIHDQQDVITGTGVVTIDLQQLGLKFVDQTTAGAIQNQEKMAILKHYRQSGKQVIVGAGQAAGAGGGMGQAGSLVAGIIQQNEQLDGGMQGTSNQMTAATQAMAGNMVRGKSGDAVTGGIKSTGTLTTKSQEFAGKVTGGIDNNTFQVSQEQGIQKQLKQDKSQTQAKTQTAAQGAAEVAGQGGDQAFLSGRQKWGQVRAGAAAGAAGTGQGTGQGQTGQAAAMGIAGQVADQRTGTGQGTGQSGGTQSMAGQDQMMHNGQQMAGLNQMGGAIVAFQRNKQGGGQNVGKLAALVDTRNQLAQAAQQGQSLQRNLQQQMQGQAALQETGQSLVDSKAGAIALYTEKALLRSQGQGMGGVLNTAGQAIQEKATKQGQTGLQVAQAILADSKSNQTLQAAGAAQQLLAAELQGQATQALQEAGQGLQAQAANLQNALGQQLQRLKQQMQAQAQAQGQAGQAAQQLQQGQAAGQEQVAAKFQNQQGQQKAQAGQQHQGQGLVSLAQQGQLSSLQQQGQALGQAALQAQAQAATAGQATAGQAAGQQSAAQQGQSLQTTKSKWGQVRAGTAGAAGTGQGTGQGQTGQAAAMGIAGQVADQRTGTGQGTGQSGGTQSMAGQDQMMHNGQQMAGLNQMGGAIVAFQRNKQGGGQNVGKLAALVDTRNQLAQAAQQGQSLQRNLQQQMQGQAALQETGQSLVDSKAGAIALYTEKALLRSQGQGMGGVLNTAGQAIQEKATKQGQTGLQVAQAILADSKSNQTLQAAGAAQQLLAAELQGQATQALQEAGQGLQAQAANLQNALGQQLQRLKQQMQAQAQAQGQAGQAAQQLQQGQAAGQEQVAAKFQNQQGQQKAQAGQQHQGQGLVSLAQQGQLSSLQQQGQALGQAALQAQAQAATAGQATAGQAAGQQSAAQQGQSLQTTKS"
        
        protein_atom = self.atomspace.protein_to_atomspace(mock_protein)
        
        assert protein_atom is not None
        assert protein_atom.atom_type == "ConceptNode"
        assert "Protein_" in protein_atom.name
        
        # Check that amino acid atoms were created
        aa_atoms = self.atomspace.get_atoms_by_name_pattern("AminoAcid_")
        assert len(aa_atoms) > 0
    
    def test_scheme_representation(self):
        concept = self.atomspace.create_concept_node("TestAtom", TruthValue(0.8, 0.6))
        scheme_repr = self.atomspace.to_scheme_representation(concept)
        
        assert "(ConceptNode" in scheme_repr
        assert "TestAtom" in scheme_repr
        assert "0.800" in scheme_repr
        assert "0.600" in scheme_repr
    
    def test_atom_count(self):
        initial_count = self.atomspace.atom_count()
        
        self.atomspace.create_concept_node("Test1")
        self.atomspace.create_concept_node("Test2")
        
        assert self.atomspace.atom_count() == initial_count + 2


class TestProteinPLN:
    """Test PLN reasoning functionality"""
    
    def setup_method(self):
        self.atomspace = ProteinAtomSpace()
        self.pln = ProteinPLN(self.atomspace)
    
    def test_pln_initialization(self):
        assert len(self.pln.validation_rules) > 0
        assert 'sequence_validity' in self.pln.validation_rules
        assert 'structure_coherence' in self.pln.validation_rules
    
    def test_protein_validation(self):
        # Create test protein atom
        protein_atom = self.atomspace.create_concept_node("TestProtein")
        
        # Add some sequence information
        length_pred = self.atomspace.create_predicate_node("sequenceLength")
        self.atomspace.create_evaluation_link(length_pred, protein_atom, 200.0)
        
        # Validate protein
        validation_result = self.pln.validate_protein(protein_atom)
        
        assert isinstance(validation_result, TruthValue)
        assert 0.0 <= validation_result.strength <= 1.0
        assert 0.0 <= validation_result.confidence <= 1.0
    
    def test_trial_balance_proof(self):
        # Create multiple protein atoms
        proteins = []
        for i in range(3):
            protein = self.atomspace.create_concept_node(f"Protein_{i}")
            proteins.append(protein)
        
        proof = self.pln.generate_trial_balance_proof(proteins)
        
        assert isinstance(proof, PLNProof)
        assert proof.theorem == "TrialBalance: All proteins maintain structural-functional consistency"
        assert len(proof.premises) > 0
        assert len(proof.steps) > 0
        assert 0.0 <= proof.confidence <= 1.0


class TestProteinECAN:
    """Test ECAN attention allocation"""
    
    def setup_method(self):
        self.atomspace = ProteinAtomSpace()
        self.ecan = ProteinECAN(self.atomspace)
    
    def test_attention_value_creation(self):
        av = AttentionValue(100.0, 50.0, 25.0)
        assert av.sti == 100.0
        assert av.lti == 50.0
        assert av.vlti == 25.0
        
        # Test clamping
        av_clamped = AttentionValue(2000.0, -100.0, 1500.0)
        assert av_clamped.sti == 1000.0
        assert av_clamped.lti == 0.0
        assert av_clamped.vlti == 1000.0
    
    def test_attention_allocation(self):
        protein_atom = self.atomspace.create_concept_node("TestProtein")
        
        # Set initial attention
        initial_attention = AttentionValue(100.0, 50.0, 0.0)
        self.ecan.set_attention_value(protein_atom, initial_attention)
        
        # Verify attention was set
        retrieved_attention = self.ecan.get_attention_value(protein_atom)
        assert retrieved_attention.sti == 100.0
        assert retrieved_attention.lti == 50.0
        assert retrieved_attention.vlti == 0.0
    
    def test_activity_recording(self):
        protein_atom = self.atomspace.create_concept_node("ActiveProtein")
        
        # Record activity
        self.ecan.record_activity(protein_atom, "validation", 2.0)
        
        # Check that attention was increased
        attention = self.ecan.get_attention_value(protein_atom)
        assert attention.sti > 0
        assert attention.lti > 0
    
    def test_attention_cycle(self):
        protein_atom = self.atomspace.create_concept_node("CycleTestProtein")
        
        # Set initial attention
        self.ecan.set_attention_value(protein_atom, AttentionValue(200.0, 100.0, 0.0))
        
        # Run attention cycle
        self.ecan.allocate_attention_cycle()
        
        # Verify system processed the cycle
        assert self.ecan.cycle_count > 0


class TestProteinMOSES:
    """Test MOSES evolutionary optimization"""
    
    def setup_method(self):
        self.atomspace = ProteinAtomSpace()
        self.moses = ProteinMOSES(self.atomspace)
    
    def test_fitness_evaluation(self):
        protein_atom = self.atomspace.create_concept_node("FitnessTestProtein")
        
        # Add sequence length for testing
        length_pred = self.atomspace.create_predicate_node("sequenceLength")
        self.atomspace.create_evaluation_link(length_pred, protein_atom, 150.0)
        
        # Evaluate fitness
        fitness_score = self.moses._evaluate_sequence_quality(protein_atom)
        
        assert isinstance(fitness_score, float)
        assert 0.0 <= fitness_score <= 1.0
    
    def test_population_initialization(self):
        # Create test proteins
        proteins = []
        for i in range(5):
            protein = self.atomspace.create_concept_node(f"TestProtein_{i}")
            proteins.append(protein)
        
        # Initialize population
        self.moses.initialize_population(proteins, population_size=10)
        
        assert len(self.moses.population) == 10
        assert all(variant.generation == 0 for variant in self.moses.population[:5])
    
    def test_evolution_generation(self):
        # Create and initialize population
        proteins = [self.atomspace.create_concept_node(f"EvolveTest_{i}") for i in range(3)]
        self.moses.initialize_population(proteins, population_size=10)
        
        # Evolve one generation
        new_population = self.moses.evolve_generation(FitnessType.MULTI_OBJECTIVE)
        
        assert len(new_population) == 10
        assert all(variant.generation <= 1 for variant in new_population)


class TestProteinURE:
    """Test URE uncertainty reasoning"""
    
    def setup_method(self):
        self.atomspace = ProteinAtomSpace()
        self.ure = ProteinURE(self.atomspace)
    
    def test_uncertainty_modeling(self):
        protein_atom = self.atomspace.create_concept_node("UncertaintyTestProtein")
        
        # Test temporal uncertainty
        temporal_factor = self.ure._model_temporal_uncertainty(protein_atom, 7200.0)  # 2 hours
        
        assert temporal_factor.factor_type == UncertaintyType.TEMPORAL
        assert 0.0 <= temporal_factor.magnitude <= 1.0
        assert 0.0 <= temporal_factor.confidence <= 1.0
    
    def test_prediction_with_uncertainty(self):
        protein_atom = self.atomspace.create_concept_node("PredictionTestProtein")
        
        # Make prediction
        prediction_result = self.ure.predict_property_with_uncertainty(
            protein_atom, "stability", 0.8
        )
        
        assert prediction_result.prediction == 0.8
        assert len(prediction_result.uncertainty_factors) > 0
        assert 0.0 <= prediction_result.total_uncertainty <= 1.0
        assert len(prediction_result.confidence_interval) == 2
    
    def test_stability_prediction(self):
        protein_atom = self.atomspace.create_concept_node("StabilityTestProtein")
        
        stability_result = self.ure.predict_protein_stability(protein_atom)
        
        assert 0.0 <= stability_result.prediction <= 1.0
        assert stability_result.prediction_type == "stability"


class TestCognitiveProtein:
    """Test cognitive protein functionality"""
    
    def setup_method(self):
        self.atomspace = ProteinAtomSpace()
        
        # Create mock ESMProtein
        self.mock_protein = Mock(spec=ESMProtein)
        self.mock_protein.sequence = "MKLLVLLAIVCFGAA"
        
        self.cognitive_protein = CognitiveProtein(
            self.mock_protein, 
            self.atomspace,
            CognitiveAccountType.TRADITIONAL
        )
    
    def test_cognitive_protein_creation(self):
        assert self.cognitive_protein.esm_protein == self.mock_protein
        assert self.cognitive_protein.cognitive_type == CognitiveAccountType.TRADITIONAL
        assert self.cognitive_protein.protein_atom is not None
    
    def test_cognitive_type_change(self):
        initial_events = len(self.cognitive_protein.learning_history.adaptation_events)
        
        self.cognitive_protein.set_cognitive_type(CognitiveAccountType.ADAPTIVE)
        
        assert self.cognitive_protein.cognitive_type == CognitiveAccountType.ADAPTIVE
        assert len(self.cognitive_protein.learning_history.adaptation_events) > initial_events
    
    def test_learning_enablement(self):
        self.cognitive_protein.enable_learning()
        
        assert CognitiveAccountType.ADAPTIVE in self.cognitive_protein.cognitive_type
        assert hasattr(self.cognitive_protein, 'learned_patterns')
    
    def test_learning_from_experience(self):
        self.cognitive_protein.enable_learning()
        
        experience = {
            'sequence_activity': 0.8,
            'structure_stability': 0.7,
            'function_performance': 0.9
        }
        
        success = self.cognitive_protein.learn_from_experience(experience, 0.8)
        
        assert success
        assert self.cognitive_protein.learning_history.iterations > 0
        assert len(self.cognitive_protein.learned_patterns) > 0
    
    def test_prediction_capabilities(self):
        self.cognitive_protein.enable_prediction()
        
        assert CognitiveAccountType.PREDICTIVE in self.cognitive_protein.cognitive_type
        
        predictions = self.cognitive_protein.predict_future_state(steps_ahead=3)
        assert 'prediction_timestamp' in predictions
        assert 'steps_ahead' in predictions
    
    def test_multimodal_processing(self):
        self.cognitive_protein.enable_multimodal_processing()
        
        assert CognitiveAccountType.MULTIMODAL in self.cognitive_protein.cognitive_type
        
        # Test sequence modality
        result = self.cognitive_protein.process_multimodal_input('sequence', 'ACDEFGH')
        assert result['modality'] == 'sequence'
        assert result['status'] == 'processed'


class TestCognitiveAccountingFramework:
    """Test the main framework integration"""
    
    def setup_method(self):
        self.framework = CognitiveAccountingFramework()
    
    def test_framework_initialization(self):
        assert self.framework.framework_initialized
        assert self.framework.atomspace is not None
        assert self.framework.pln is not None
        assert self.framework.ecan is not None
        assert self.framework.moses is not None
        assert self.framework.ure is not None
    
    def test_protein_addition(self):
        mock_protein = Mock(spec=ESMProtein)
        mock_protein.sequence = "MKLLVLLAIVCFGAA"
        
        cognitive_protein = self.framework.add_protein(
            mock_protein, 
            CognitiveAccountType.ADAPTIVE
        )
        
        assert len(self.framework.cognitive_proteins) == 1
        assert cognitive_protein.cognitive_type == CognitiveAccountType.ADAPTIVE
    
    def test_protein_validation(self):
        # Add test proteins
        proteins = []
        for i in range(3):
            mock_protein = Mock(spec=ESMProtein)
            mock_protein.sequence = f"MKLL{'A'*i}"
            
            cognitive_protein = self.framework.add_protein(mock_protein)
            proteins.append(list(self.framework.cognitive_proteins.keys())[-1])
        
        # Validate protein set
        proof = self.framework.validate_protein_set(proteins)
        
        assert isinstance(proof, PLNProof)
        assert len(proof.premises) > 0
    
    def test_cognitive_cycle(self):
        initial_cycle = self.framework.cycle_count
        
        self.framework.run_cognitive_cycle()
        
        assert self.framework.cycle_count > initial_cycle
        assert self.framework.last_cycle_time > 0
    
    def test_system_status(self):
        status = self.framework.get_system_status()
        
        assert 'framework_initialized' in status
        assert 'cycle_count' in status
        assert 'cognitive_proteins' in status
        assert 'atomspace_atoms' in status
        assert status['framework_initialized'] == True
    
    def test_framework_shutdown(self):
        self.framework.shutdown()
        
        assert not self.framework.framework_initialized
        assert not self.framework.processing_active
        assert len(self.framework.cognitive_proteins) == 0


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_workflow(self):
        """Test complete cognitive workflow"""
        framework = CognitiveAccountingFramework()
        
        # Create test protein
        mock_protein = Mock(spec=ESMProtein)
        mock_protein.sequence = "MKLLVLLAIVCFGAAALA"
        
        # Add to framework
        cognitive_protein = framework.add_protein(
            mock_protein,
            CognitiveAccountType.ADAPTIVE | CognitiveAccountType.PREDICTIVE
        )
        
        # Enable cognitive features
        cognitive_protein.enable_learning()
        cognitive_protein.enable_prediction()
        
        # Run some cognitive cycles
        for _ in range(3):
            framework.run_cognitive_cycle()
        
        # Test validation
        protein_id = list(framework.cognitive_proteins.keys())[0]
        proof = framework.validate_protein_set([protein_id])
        
        # Test prediction
        prediction = framework.predict_with_uncertainty(protein_id, "stability")
        
        # Verify results
        assert proof is not None
        assert prediction is not None
        assert framework.cycle_count >= 3
        
        # Get final status
        status = framework.get_system_status()
        assert status['cognitive_proteins'] == 1
        assert status['cycle_count'] >= 3
        
        framework.shutdown()


if __name__ == "__main__":
    # Run basic tests
    print("Running ESM3 Cognitive Accounting Framework Tests...")
    
    # Test AtomSpace
    test_atomspace = TestProteinAtomSpace()
    test_atomspace.setup_method()
    test_atomspace.test_concept_node_creation()
    test_atomspace.test_truth_value_creation()
    print("✓ AtomSpace tests passed")
    
    # Test PLN
    test_pln = TestProteinPLN()
    test_pln.setup_method()
    test_pln.test_protein_validation()
    print("✓ PLN tests passed")
    
    # Test ECAN
    test_ecan = TestProteinECAN()
    test_ecan.setup_method()
    test_ecan.test_attention_allocation()
    print("✓ ECAN tests passed")
    
    # Test Framework
    test_framework = TestCognitiveAccountingFramework()
    test_framework.setup_method()
    test_framework.test_framework_initialization()
    print("✓ Framework tests passed")
    
    # Integration test
    test_integration = TestIntegration()
    test_integration.test_end_to_end_workflow()
    print("✓ Integration test passed")
    
    print("\nAll tests completed successfully! 🧠✨")