"""
ESM3 Cognitive Accounting Framework

This module implements a neural-symbolic cognitive architecture that adapts
OpenCog-style cognitive computing concepts to ESM3 protein modeling tasks.

The framework maps traditional accounting concepts to protein analysis:
- Proteins → Cognitive Accounts
- Amino Acids → Cognitive Transactions  
- Structures → Balance States
- Functions → Performance Metrics

Core Components:
- AtomSpace: Hypergraph knowledge representation
- PLN: Probabilistic Logic Networks for validation
- ECAN: Economic Attention Allocation
- MOSES: Meta-Optimizing Semantic Evolutionary Search
- URE: Uncertain Reasoning Engine
"""

from .atomspace import ProteinAtomSpace, TruthValue, AtomHandle
from .pln import ProteinPLN, PLNProof
from .ecan import ProteinECAN, AttentionValue, ECANParams, ProteinActivity
from .moses import ProteinMOSES, FitnessType, EvolutionaryStrategy
from .ure import ProteinURE, UncertaintyType, PredictionResult
from .cognitive_protein import CognitiveProtein, CognitiveAccountType
from .framework import CognitiveAccountingFramework

__all__ = [
    "ProteinAtomSpace", "TruthValue", "AtomHandle",
    "ProteinPLN", "PLNProof",
    "ProteinECAN", "AttentionValue", "ECANParams", "ProteinActivity",
    "ProteinMOSES", "FitnessType", "EvolutionaryStrategy", 
    "ProteinURE", "UncertaintyType", "PredictionResult",
    "CognitiveProtein", "CognitiveAccountType",
    "CognitiveAccountingFramework"
]

__version__ = "1.0.0"