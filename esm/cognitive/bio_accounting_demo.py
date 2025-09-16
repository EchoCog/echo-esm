"""
Bio-Cognitive Accounting Demo Implementation

This module provides a working demonstration of the bio-cognitive accounting framework 
that can run independently of the full ESM3 stack for testing and educational purposes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Flag, Enum
import time
import uuid
import json
from collections import defaultdict

class CognitiveAccountType(Flag):
    """Cognitive account type flags for protein accounts"""
    TRADITIONAL = 1          # Standard accounting behavior
    ADAPTIVE = 2             # Learning-enabled accounts  
    PREDICTIVE = 4           # Accounts with forecasting capabilities
    MULTIMODAL = 8           # Support for complex transaction types
    ATTENTION_DRIVEN = 16    # Dynamically prioritized accounts
    PLN_REASONING = 32       # Probabilistic logic reasoning
    UNCERTAINTY_AWARE = 64   # Uncertainty quantification
    EVOLUTIONARY_OPTIMIZED = 128  # MOSES evolutionary optimization

@dataclass
class TruthValue:
    """Truth value for cognitive reasoning"""
    strength: float
    confidence: float
    
    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))

@dataclass
class AttentionValue:
    """Attention value for ECAN processing"""
    sti: float  # Short-term importance
    lti: float  # Long-term importance
    vlti: float # Very long-term importance
    
    def __post_init__(self):
        self.sti = max(0.0, min(1000.0, self.sti))
        self.lti = max(0.0, min(1000.0, self.lti)) 
        self.vlti = max(0.0, min(1000.0, self.vlti))

@dataclass
class PLNProof:
    """Probabilistic Logic Networks proof"""
    theorem: str
    premises: List[str]
    steps: List[Dict[str, Any]]
    confidence: float
    strength: float
    validation_time: float

@dataclass
class UncertaintyFactor:
    """Uncertainty factor for URE reasoning"""
    factor_type: str
    magnitude: float
    confidence: float
    source: str

@dataclass
class PredictionResult:
    """Result of uncertain prediction"""
    prediction: float
    uncertainty_factors: List[UncertaintyFactor]
    total_uncertainty: float
    confidence_interval: Tuple[float, float]
    prediction_type: str

class DemoProtein:
    """Demo protein for testing bio-cognitive accounting"""
    def __init__(self, sequence: str, name: Optional[str] = None):
        self.sequence = sequence
        self.name = name or f"Protein_{str(uuid.uuid4())[:8]}"
        self.coordinates = None  # Placeholder for 3D structure
        self.function_keywords = []

class BioCognitiveAccount:
    """Bio-cognitive account representing a protein with accounting metaphors"""
    
    def __init__(self, protein: DemoProtein, cognitive_type: CognitiveAccountType = CognitiveAccountType.TRADITIONAL):
        self.protein = protein
        self.cognitive_type = cognitive_type
        self.account_id = f"ACC_{protein.name}_{str(uuid.uuid4())[:8]}"
        
        # Accounting components
        self.transactions = []  # Amino acid transactions
        self.ledger = self._create_transaction_ledger()
        self.balance_state = self._calculate_balance_state()
        self.performance_metrics = self._calculate_performance_metrics()
        
        # Cognitive components
        self.attention_value = AttentionValue(100.0, 50.0, 0.0)
        self.truth_value = TruthValue(0.8, 0.7)
        self.uncertainty_factors = []
        self.learned_patterns = {} if CognitiveAccountType.ADAPTIVE in cognitive_type else None
        
        # Process amino acids as transactions
        self._process_amino_acid_transactions()
    
    def _create_transaction_ledger(self) -> Dict[str, Any]:
        """Create transaction ledger from amino acid sequence"""
        ledger = {
            'protein_id': self.protein.name,
            'account_id': self.account_id,
            'opening_date': time.time(),
            'transactions': [],
            'running_balance': 0.0,
            'transaction_count': 0
        }
        return ledger
    
    def _process_amino_acid_transactions(self):
        """Process each amino acid as a cognitive transaction"""
        aa_properties = {
            'A': {'hydrophobicity': 1.8, 'volume': 88.6, 'charge': 0, 'debit': 1.8, 'credit': 0.0},
            'R': {'hydrophobicity': -4.5, 'volume': 173.4, 'charge': 1, 'debit': 0.0, 'credit': 4.5},
            'N': {'hydrophobicity': -3.5, 'volume': 114.1, 'charge': 0, 'debit': 0.0, 'credit': 3.5},
            'D': {'hydrophobicity': -3.5, 'volume': 111.1, 'charge': -1, 'debit': 0.0, 'credit': 3.5},
            'C': {'hydrophobicity': 2.5, 'volume': 108.5, 'charge': 0, 'debit': 2.5, 'credit': 0.0},
            'Q': {'hydrophobicity': -3.5, 'volume': 143.8, 'charge': 0, 'debit': 0.0, 'credit': 3.5},
            'E': {'hydrophobicity': -3.5, 'volume': 138.4, 'charge': -1, 'debit': 0.0, 'credit': 3.5},
            'G': {'hydrophobicity': -0.4, 'volume': 60.1, 'charge': 0, 'debit': 0.0, 'credit': 0.4},
            'H': {'hydrophobicity': -3.2, 'volume': 153.2, 'charge': 0.5, 'debit': 0.0, 'credit': 3.2},
            'I': {'hydrophobicity': 4.5, 'volume': 166.7, 'charge': 0, 'debit': 4.5, 'credit': 0.0},
            'L': {'hydrophobicity': 3.8, 'volume': 166.7, 'charge': 0, 'debit': 3.8, 'credit': 0.0},
            'K': {'hydrophobicity': -3.9, 'volume': 168.6, 'charge': 1, 'debit': 0.0, 'credit': 3.9},
            'M': {'hydrophobicity': 1.9, 'volume': 162.9, 'charge': 0, 'debit': 1.9, 'credit': 0.0},
            'F': {'hydrophobicity': 2.8, 'volume': 189.9, 'charge': 0, 'debit': 2.8, 'credit': 0.0},
            'P': {'hydrophobicity': -1.6, 'volume': 112.7, 'charge': 0, 'debit': 0.0, 'credit': 1.6},
            'S': {'hydrophobicity': -0.8, 'volume': 89.0, 'charge': 0, 'debit': 0.0, 'credit': 0.8},
            'T': {'hydrophobicity': -0.7, 'volume': 116.1, 'charge': 0, 'debit': 0.0, 'credit': 0.7},
            'W': {'hydrophobicity': -0.9, 'volume': 227.8, 'charge': 0, 'debit': 0.0, 'credit': 0.9},
            'Y': {'hydrophobicity': -1.3, 'volume': 193.6, 'charge': 0, 'debit': 0.0, 'credit': 1.3},
            'V': {'hydrophobicity': 4.2, 'volume': 140.0, 'charge': 0, 'debit': 4.2, 'credit': 0.0}
        }
        
        running_balance = 0.0
        
        for position, amino_acid in enumerate(self.protein.sequence):
            if amino_acid in aa_properties:
                props = aa_properties[amino_acid]
                
                # Create double-entry transaction
                debit_amount = props['debit']
                credit_amount = props['credit']
                net_amount = debit_amount - credit_amount
                running_balance += net_amount
                
                transaction = {
                    'transaction_id': f"TXN_{self.account_id}_{position:04d}",
                    'position': position,
                    'amino_acid': amino_acid,
                    'timestamp': time.time() + position * 0.001,  # Simulate processing time
                    'debit_amount': debit_amount,
                    'credit_amount': credit_amount,
                    'net_amount': net_amount,
                    'running_balance': running_balance,
                    'properties': props,
                    'truth_value': TruthValue(0.9, 0.8),
                    'attention_weight': 1.0 / len(self.protein.sequence),
                    'description': f"Amino acid {amino_acid} at position {position}"
                }
                
                self.transactions.append(transaction)
                self.ledger['transactions'].append(transaction)
        
        self.ledger['transaction_count'] = len(self.transactions)
        self.ledger['final_balance'] = running_balance
    
    def _calculate_balance_state(self) -> Dict[str, float]:
        """Calculate current balance state (protein structure representation)"""
        sequence = self.protein.sequence
        
        if not sequence:
            return {'total_balance': 0.0, 'hydrophobic_balance': 0.0, 'hydrophilic_balance': 0.0}
        
        # Calculate various balance measures
        hydrophobic_residues = sum(1 for aa in sequence if aa in 'AILMFPWV')
        hydrophilic_residues = sum(1 for aa in sequence if aa in 'RNDQEHKSTY')
        charged_residues = sum(1 for aa in sequence if aa in 'RKDE')
        aromatic_residues = sum(1 for aa in sequence if aa in 'FYW')
        
        total_length = len(sequence)
        
        balance_state = {
            'total_balance': self.ledger.get('final_balance', 0.0),
            'hydrophobic_balance': hydrophobic_residues / total_length,
            'hydrophilic_balance': hydrophilic_residues / total_length,
            'charge_balance': charged_residues / total_length,
            'aromatic_balance': aromatic_residues / total_length,
            'length_normalized_balance': self.ledger.get('final_balance', 0.0) / total_length,
            'balance_variance': np.var([t['net_amount'] for t in self.transactions]) if self.transactions else 0.0
        }
        
        return balance_state
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics (protein function representation)"""
        sequence = self.protein.sequence
        
        if not sequence:
            return {}
        
        # Simplified performance metrics based on sequence properties
        metrics = {
            'stability_score': self._estimate_stability(),
            'solubility_score': self._estimate_solubility(),
            'binding_potential': self._estimate_binding_potential(),
            'catalytic_potential': self._estimate_catalytic_potential(),
            'structural_complexity': self._estimate_structural_complexity(),
            'evolutionary_conservation': np.random.uniform(0.5, 0.9)  # Placeholder
        }
        
        return metrics
    
    def _estimate_stability(self) -> float:
        """Estimate protein stability"""
        sequence = self.protein.sequence
        hydrophobic_core = sum(1 for aa in sequence if aa in 'AILMFPWV')
        disulfide_potential = sequence.count('C') // 2
        proline_rigidity = sequence.count('P')
        
        stability = (hydrophobic_core / len(sequence)) * 0.6 + \
                   (disulfide_potential / len(sequence)) * 0.3 + \
                   (proline_rigidity / len(sequence)) * 0.1
        
        return min(1.0, stability)
    
    def _estimate_solubility(self) -> float:
        """Estimate protein solubility"""
        sequence = self.protein.sequence
        polar_residues = sum(1 for aa in sequence if aa in 'RNDQEHKSTY')
        charged_residues = sum(1 for aa in sequence if aa in 'RKDE')
        
        solubility = (polar_residues / len(sequence)) * 0.7 + \
                    (charged_residues / len(sequence)) * 0.3
        
        return min(1.0, solubility)
    
    def _estimate_binding_potential(self) -> float:
        """Estimate binding potential"""
        sequence = self.protein.sequence
        aromatic_residues = sum(1 for aa in sequence if aa in 'FYW')
        charged_residues = sum(1 for aa in sequence if aa in 'RKDE')
        
        binding = (aromatic_residues / len(sequence)) * 0.5 + \
                 (charged_residues / len(sequence)) * 0.5
        
        return min(1.0, binding)
    
    def _estimate_catalytic_potential(self) -> float:
        """Estimate catalytic potential"""
        sequence = self.protein.sequence
        catalytic_residues = sum(1 for aa in sequence if aa in 'HDSTNC')
        
        return min(1.0, catalytic_residues / len(sequence))
    
    def _estimate_structural_complexity(self) -> float:
        """Estimate structural complexity"""
        sequence = self.protein.sequence
        aa_diversity = len(set(sequence)) / 20.0  # Normalized by max possible diversity
        length_factor = min(1.0, len(sequence) / 300.0)  # Normalized by typical protein length
        
        return (aa_diversity + length_factor) / 2.0

    def predict_with_uncertainty(self, property_name: str, horizon: int = 1) -> PredictionResult:
        """Make prediction with uncertainty quantification"""
        if CognitiveAccountType.UNCERTAINTY_AWARE not in self.cognitive_type:
            raise ValueError("Uncertainty-aware capabilities not enabled")
        
        # Get base prediction from performance metrics
        base_prediction = self.performance_metrics.get(property_name, 0.5)
        
        # Add uncertainty factors
        uncertainty_factors = [
            UncertaintyFactor("temporal", 0.05 * horizon, 0.8, "time_decay"),
            UncertaintyFactor("epistemic", 0.1, 0.7, "model_uncertainty"),
            UncertaintyFactor("aleatoric", 0.08, 0.9, "inherent_randomness")
        ]
        
        total_uncertainty = sum(uf.magnitude for uf in uncertainty_factors)
        
        # Calculate confidence interval
        lower_bound = max(0.0, base_prediction - total_uncertainty)
        upper_bound = min(1.0, base_prediction + total_uncertainty)
        
        return PredictionResult(
            prediction=base_prediction,
            uncertainty_factors=uncertainty_factors,
            total_uncertainty=total_uncertainty,
            confidence_interval=(lower_bound, upper_bound),
            prediction_type=property_name
        )

    def learn_from_experience(self, experience: Dict[str, Any], performance: float) -> bool:
        """Learn from experience and update patterns"""
        if CognitiveAccountType.ADAPTIVE not in self.cognitive_type:
            return False
        
        if self.learned_patterns is None:
            self.learned_patterns = {}
        
        # Extract patterns from experience
        pattern_name = f"experience_{len(self.learned_patterns)}"
        self.learned_patterns[pattern_name] = {
            'experience': experience,
            'performance': performance,
            'timestamp': time.time(),
            'truth_value': TruthValue(performance, 0.7)
        }
        
        # Update attention based on performance
        if performance > 0.7:
            self.attention_value.sti += 10.0
            self.attention_value.lti += 5.0
        
        return True

    def generate_trial_balance_proof(self, other_accounts: List['BioCognitiveAccount']) -> PLNProof:
        """Generate PLN proof for trial balance validation"""
        if CognitiveAccountType.PLN_REASONING not in self.cognitive_type:
            raise ValueError("PLN reasoning capabilities not enabled")
        
        start_time = time.time()
        
        # Calculate total debits and credits across all accounts
        total_debits = sum(sum(t['debit_amount'] for t in acc.transactions) for acc in [self] + other_accounts)
        total_credits = sum(sum(t['credit_amount'] for t in acc.transactions) for acc in [self] + other_accounts)
        
        # Check balance
        balance_difference = abs(total_debits - total_credits)
        is_balanced = balance_difference < 0.01  # Small tolerance for floating point
        
        premises = [
            f"Total debits across {len(other_accounts) + 1} protein accounts: {total_debits:.3f}",
            f"Total credits across {len(other_accounts) + 1} protein accounts: {total_credits:.3f}",
            f"Balance difference: {balance_difference:.6f}",
            "Accounting principle: Debits must equal credits in a balanced system"
        ]
        
        steps = [
            {
                "step": 1,
                "rule": "debit_credit_equality",  
                "operation": "sum_all_debits",
                "result": total_debits
            },
            {
                "step": 2,
                "rule": "debit_credit_equality",
                "operation": "sum_all_credits", 
                "result": total_credits
            },
            {
                "step": 3,
                "rule": "balance_validation",
                "operation": "calculate_difference",
                "result": balance_difference
            },
            {
                "step": 4,
                "rule": "trial_balance_theorem",
                "operation": "evaluate_balance",
                "result": is_balanced
            }
        ]
        
        confidence = 0.95 if is_balanced else 0.3
        strength = 1.0 - min(1.0, balance_difference)
        
        theorem = f"Trial Balance Validation: System is {'BALANCED' if is_balanced else 'UNBALANCED'}"
        
        return PLNProof(
            theorem=theorem,
            premises=premises,
            steps=steps,
            confidence=confidence,
            strength=strength,
            validation_time=time.time() - start_time
        )

    def get_account_summary(self) -> Dict[str, Any]:
        """Get comprehensive account summary"""
        return {
            'account_id': self.account_id,
            'protein_name': self.protein.name,
            'cognitive_type': {
                'value': self.cognitive_type.value,
                'flags': [flag.name for flag in CognitiveAccountType if flag in self.cognitive_type]
            },
            'sequence_info': {
                'length': len(self.protein.sequence),
                'sequence': self.protein.sequence[:50] + '...' if len(self.protein.sequence) > 50 else self.protein.sequence
            },
            'transaction_summary': {
                'total_transactions': len(self.transactions),
                'final_balance': self.ledger.get('final_balance', 0.0),
                'avg_transaction_value': np.mean([t['net_amount'] for t in self.transactions]) if self.transactions else 0.0
            },
            'balance_state': self.balance_state,
            'performance_metrics': self.performance_metrics,
            'cognitive_state': {
                'attention_value': {
                    'sti': self.attention_value.sti,
                    'lti': self.attention_value.lti,
                    'vlti': self.attention_value.vlti
                },
                'truth_value': {
                    'strength': self.truth_value.strength,
                    'confidence': self.truth_value.confidence
                },
                'learned_patterns_count': len(self.learned_patterns) if self.learned_patterns else 0
            },
            'capabilities': {
                'learning_enabled': CognitiveAccountType.ADAPTIVE in self.cognitive_type,
                'prediction_enabled': CognitiveAccountType.PREDICTIVE in self.cognitive_type,
                'multimodal_processing': CognitiveAccountType.MULTIMODAL in self.cognitive_type,
                'attention_driven': CognitiveAccountType.ATTENTION_DRIVEN in self.cognitive_type,
                'pln_reasoning': CognitiveAccountType.PLN_REASONING in self.cognitive_type,
                'uncertainty_aware': CognitiveAccountType.UNCERTAINTY_AWARE in self.cognitive_type
            },
            'timestamp': time.time()
        }

class BioCognitiveAccountingFramework:
    """Main framework for bio-cognitive accounting"""
    
    def __init__(self):
        self.accounts: Dict[str, BioCognitiveAccount] = {}
        self.framework_initialized = True
        self.cycle_count = 0
        self.last_cycle_time = time.time()
        
    def create_account(self, protein: DemoProtein, cognitive_type: CognitiveAccountType = CognitiveAccountType.TRADITIONAL) -> BioCognitiveAccount:
        """Create a new bio-cognitive account"""
        account = BioCognitiveAccount(protein, cognitive_type)
        self.accounts[account.account_id] = account
        return account
    
    def validate_trial_balance(self, account_ids: Optional[List[str]] = None) -> PLNProof:
        """Validate trial balance across specified accounts"""
        if account_ids is None:
            account_ids = list(self.accounts.keys())
        
        if not account_ids:
            raise ValueError("No accounts specified for trial balance")
        
        main_account = self.accounts[account_ids[0]]
        other_accounts = [self.accounts[aid] for aid in account_ids[1:]]
        
        return main_account.generate_trial_balance_proof(other_accounts)
    
    def run_attention_cycle(self):
        """Run one attention allocation cycle"""
        self.cycle_count += 1
        
        # Simple attention decay and redistribution
        total_sti = sum(acc.attention_value.sti for acc in self.accounts.values())
        
        for account in self.accounts.values():
            # Attention decay
            account.attention_value.sti *= 0.98
            account.attention_value.lti *= 0.995
            
            # Activity-based attention allocation
            if account.performance_metrics.get('stability_score', 0) > 0.7:
                account.attention_value.sti += 5.0
            
            # Transfer some STI to LTI for high-performing accounts
            if account.attention_value.sti > 150:
                transfer = min(10.0, account.attention_value.sti * 0.1)
                account.attention_value.sti -= transfer
                account.attention_value.lti += transfer
        
        self.last_cycle_time = time.time()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_accounts = len(self.accounts)
        total_transactions = sum(len(acc.transactions) for acc in self.accounts.values())
        total_balance = sum(acc.balance_state.get('total_balance', 0.0) for acc in self.accounts.values())
        
        return {
            'framework_initialized': self.framework_initialized,
            'total_accounts': total_accounts,
            'total_transactions': total_transactions,
            'total_system_balance': total_balance,
            'cycle_count': self.cycle_count,
            'last_cycle_time': self.last_cycle_time,
            'account_types': {
                flag.name: sum(1 for acc in self.accounts.values() if flag in acc.cognitive_type)
                for flag in CognitiveAccountType
            }
        }

def create_demo_proteins() -> List[DemoProtein]:
    """Create demonstration proteins for testing"""
    proteins = [
        DemoProtein(
            sequence="MKLLVLLAIVCFGAALATAQGTSDPEKSKMNQIIQRPVFNGQHFNEHHHFFGSAAHLHYGRPQCCSYGFGGLYVYNQRVSGGLSYAMLSLRHSVCFLPKGNHGAHMSFNPSRVSPEKNIGQYVGIRVRRRVTVQPEEYMVSTPEKKLPHSSQDLITRGSFDSSEDFDKRGKSYMGVYKCMDQSQRHAFKQEEHLLQSLPCKITQALIQGMLEDGNALFDIDFKTDGQMLRYFLPSGGSLVLWPHPEGKLNKYFDDEDLFLVQHFIDKSQFHFPAFMLNMVNRQRDKDNLLLSSSDSTEEEIINEKPTGFKSGKLVNYSHYFYGQHVQDGCPEFIINYLMPLIIPAPNPTLKQFVLLSDDVNGDFILLALSHPLLRGGPTDDILQNYCYCDFSGGSLSGRLQCDYMVTGSGWDQVLAFGPFAKQVGPFVIHPQGAAHKVADAVFHTMAHIHKGDRFGTLGSRRLSPYAGRNNVQLLMISLDGDQKRLRCGFAQLMEHYKGKLPKAYTAKVIALLIRQMKDGKASIDAAGKYDQSLMQNGHQSTDQASFKMVPQPRDMPNGLEHSLTKVFDFLLFHLAYNQEGGFGFRAEHPELGGQDQSQPGMGYLHVNVQIGYIKDGTGAYLFFDSPGSADGRRLLIPQVTPALLKDKTNSIYQGKSGAKMKMITLRGMIKFPIQLLSCLAKGVDQMNTLSGFTPIQLLRGAILVIKPLTKSGDKKLPKAKQGMGIDNLNQYTLQIGALYVAGQASSSITPQFLEEGYTLRVQEILLMHSKHVKMLLSNVNTLKGGNLVGVATVPAFQNGLGGDVTLLDFQEYDGQVNFLLLPNRTTLGRPYDSKQSTFQLEQFAKVGKLVAKLLKPNYTNARALFGAQAAQAAKGHTSSTSGGVLGNLVHGKAFLAFDTMYIVEGFMNLLADTQAQAKMTYRQFQHQVLLTRKNNTGNLGFYVDAIYLQIKFQKDGHKLRGTGKLTDFQKFLQTKSQFGFLQFSGRQKSKLVYPPMPFGDADKQAQMMLANQGGQKEVKRTIQAFVSLILQADFPQAQLGEKAVGQFKTLEAFQAAQESAAQEPELGTGKKEGKSLQKQPVKAQVQRRQIDWLKNSGKQWSPLRPVLGQGEGVALLRGAAESIVTQKAKRVLAGATMGGALVVQNYQGWQIKYDGYVEIVPPGAFVAGQTLKEGTRRVITARLQHAENSGAQLNYQGKRLQNMSLVPDLDGLQAGDWVQMVPVAFLRQGSLVQVWDLTLPDGRLTLQKQIQALDEQMQQRQVFGAGDIAGITSSEHFLVAALTFNGQHQVVVPKTEGSLIVKLDHPEAIGSAIGAARTLTRDAPRILRGTIEIGFLKQHLQGQKLQQLQLALQGLVEQSVRAAAGMAQQGLHKRRSIAGQYVQLPNPRNSPYLNSMKQGCQEDEEVNFLFQNLNQLDAGQAAEQFAAVKAGDVNLTPFMALLTSRSDGSNFAFRVVDGQIHDQQDVITGTGVVTIDLQQLGLKFVDQTTAGAIQNQEKMAILKHYRQSGKQVIVGAGQAAGAGGGMGQAGSLVAGIIQQNEQLDGGMQGTSNQMTAATQAMAGNMVRGKSGDAVTGGIKSTGTLTTKSQEFAGKVTGGIDNNTFQVSQEQGIQKQLKQDKSQTQAKTQTAAQGAAEVAGQGGDQAFLSGRQKWGQVRAGAAAGAAGTGQGTGQGQTGQAAAMGIAGQVADQRTGTGQGTGQSGGTQSMAGQDQMMHNGQQMAGLNQMGGAIVAFQRNKQGGGQNVGKLAALVDTRNQLAQAAQQGQSLQRNLQQQMQGQAALQETGQSLVDSKAGAIALYTEKALLRSQGQGMGGVLNTAGQAIQEKATKQGQTGLQVAQAILADSKSNQTLQAAGAAQQLLAAELQGQATQALQEAGQGLQAQAANLQNALGQQLQRLKQQMQAQAQAQGQAGQAAQQLQQGQAAGQEQVAAKFQNQQGQQKAQAGQQHQGQGLVSLAQQGQLSSLQQQGQALGQAALQAQAQAATAGQATAGQAAGQQSAAQQGQSLQTTKSKWGQVRAGTAGAAGTGQGTGQGQTGQAAAMGIAGQVADQRTGTGQGTGQSGGTQSMAGQDQMMHNGQQMAGLNQMGGAIVAFQRNKQGGGQNVGKLAALVDTRNQLAQAAQQGQSLQRNLQQQMQGQAALQETGQSLVDSKAGAIALYTEKALLRSQGQGMGGVLNTAGQAIQEKATKQGQTGLQVAQAILADSKSNQTLQAAGAAQQLLAAELQGQATQALQEAGQGLQAQAANLQNALGQQLQRLKQQMQAQAQAQGQAGQAAQQLQQGQAAGQEQVAAKFQNQQGQQKAQAGQQHQGQGLVSLAQQGQLSSLQQQGQALGQAALQAQAQAATAGQATAGQAAGQQSAAQQGQSLQTTKS",
            name="CarbonicanhydraseII"
        ),
        DemoProtein(
            sequence="MSPVLVQMSPKGSPQAAGIFALLLWVWLWWGPGPGPGAPDAPDAPDAPVPVPVPVPGQGQGQGQHQHQHQHQ",
            name="DemoEnzyme"
        ),
        DemoProtein(
            sequence="MGAAALLLWLWLWLWGGGGGGCCCCCCDDDDDDRRRRRKKKKKKEEEEEEHHHHHHSSSSSSTTTTTT",
            name="StructuralProtein"
        ),
        DemoProtein(
            sequence="MKKKKKRRRRRHHHHHEEEEEEDDDDDLLLLLIIIIIVVVVVFFFFWWWWWYYYYYY",
            name="RegulatoryProtein"
        )
    ]
    
    return proteins

if __name__ == "__main__":
    # Demo usage
    print("🧬 Bio-Cognitive Accounting Framework Demo")
    print("=" * 50)
    
    # Create framework
    framework = BioCognitiveAccountingFramework()
    
    # Create demo proteins
    proteins = create_demo_proteins()
    
    # Create accounts with different cognitive types
    account1 = framework.create_account(
        proteins[0], 
        CognitiveAccountType.TRADITIONAL | CognitiveAccountType.ADAPTIVE | CognitiveAccountType.PLN_REASONING
    )
    
    account2 = framework.create_account(
        proteins[1], 
        CognitiveAccountType.PREDICTIVE | CognitiveAccountType.UNCERTAINTY_AWARE
    )
    
    account3 = framework.create_account(
        proteins[2],
        CognitiveAccountType.PLN_REASONING | CognitiveAccountType.ATTENTION_DRIVEN
    )
    
    # Print account summaries
    for i, account in enumerate([account1, account2, account3], 1):
        summary = account.get_account_summary()
        print(f"\n📊 Account {i} Summary:")
        print(f"  ID: {summary['account_id']}")
        print(f"  Protein: {summary['protein_name']}")
        print(f"  Cognitive Type: {', '.join(summary['cognitive_type']['flags'])}")
        print(f"  Transactions: {summary['transaction_summary']['total_transactions']}")
        print(f"  Final Balance: {summary['transaction_summary']['final_balance']:.3f}")
        print(f"  Stability Score: {summary['performance_metrics']['stability_score']:.3f}")
    
    # Run trial balance validation
    print(f"\n⚖️  Trial Balance Validation:")
    trial_balance = framework.validate_trial_balance()
    print(f"  Theorem: {trial_balance.theorem}")
    print(f"  Confidence: {trial_balance.confidence:.3f}")
    print(f"  Validation Time: {trial_balance.validation_time:.6f}s")
    
    # Run attention cycles
    print(f"\n🎯 Running Attention Cycles:")
    for i in range(5):
        framework.run_attention_cycle()
        if i % 2 == 0:
            status = framework.get_system_status()
            print(f"  Cycle {status['cycle_count']}: {status['total_accounts']} accounts, {status['total_transactions']} transactions")
    
    print(f"\n✅ Demo completed successfully!")