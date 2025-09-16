# Bio-Cognitive Accounting Examples

This directory contains comprehensive examples and tutorials for the **Bio-Cognitive Accounting Framework** - a revolutionary approach that maps traditional accounting concepts to protein analysis and cognitive computing.

## 🧬 Framework Overview

The Bio-Cognitive Accounting Framework transforms protein analysis by creating intuitive mappings:

| Accounting Concept | Biological Translation | Cognitive Implementation |
|-------------------|------------------------|--------------------------|
| **Accounts** → | **Proteins** | Cognitive accounts with learning capabilities |
| **Transactions** → | **Amino Acids** | Cognitive transactions with metadata |  
| **Balances** → | **Structures** | Balance states reflecting protein conformations |
| **Ledgers** → | **Sequences** | Transaction histories with full audit trails |
| **Performance** → | **Functions** | Protein function metrics and KPIs |
| **Trial Balance** → | **Validation** | PLN-based consistency proofs |

## 📚 Available Notebooks

### 1. [Bio-Cognitive Accounting Tutorial](bio_cognitive_accounting_tutorial.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EchoCog/echo-esm/blob/main/examples/bio_cognitive_accounting_tutorial.ipynb)

**Comprehensive introduction to bio-cognitive accounting**

- **Duration**: 60-90 minutes
- **Level**: Beginner to Intermediate
- **Topics Covered**:
  - Core concept mapping (Proteins → Accounts)
  - Cognitive account types and capabilities
  - Amino acid transaction processing
  - Balance state calculations
  - Attention-driven resource allocation
  - Multi-protein portfolio management
  - Trial balance validation
  - Interactive visualizations

**Key Learning Outcomes**:
- Understand the protein-to-accounting metaphor
- Create and manage cognitive protein accounts
- Implement attention-based resource economics
- Generate trial balance proofs using PLN
- Visualize protein networks with cognitive overlays

### 2. [Advanced Cognitive Protein Analysis](advanced_cognitive_protein_analysis.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EchoCog/echo-esm/blob/main/examples/advanced_cognitive_protein_analysis.ipynb)

**Deep dive into sophisticated cognitive capabilities**

- **Duration**: 90-120 minutes  
- **Level**: Intermediate to Advanced
- **Topics Covered**:
  - AtomSpace hypergraph representations
  - PLN probabilistic reasoning
  - URE uncertainty quantification
  - Multi-modal cognitive processing
  - Emergent behavior detection
  - Advanced attention economics
  - Cognitive complexity metrics

**Key Learning Outcomes**:
- Implement hypergraph protein representations
- Apply probabilistic logic to protein validation
- Quantify uncertainty in protein predictions
- Design multi-modal cognitive systems
- Detect emergent properties in protein networks

### 3. [Evolutionary Protein Optimization](evolutionary_protein_optimization.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EchoCog/echo-esm/blob/main/examples/evolutionary_protein_optimization.ipynb)

**MOSES-based evolutionary protein design**

- **Duration**: 90-120 minutes
- **Level**: Advanced
- **Topics Covered**:
  - Multi-objective fitness optimization
  - Pareto-optimal protein variants
  - Evolutionary search strategies
  - Population dynamics and convergence
  - Fitness landscape exploration
  - Mutation strategies and crossover
  - Performance monitoring and analytics

**Key Learning Outcomes**:
- Design multi-objective fitness functions
- Implement evolutionary search algorithms
- Optimize protein properties using MOSES
- Analyze evolutionary convergence patterns
- Visualize fitness landscapes and Pareto fronts

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
Click any of the Colab badges above to run notebooks directly in your browser with pre-configured environments.

### Option 2: Local Setup
```bash
# Clone the repository
git clone https://github.com/EchoCog/echo-esm.git
cd echo-esm

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook examples/
```

### Option 3: Demo Mode
For a quick demonstration without full ESM3 dependencies:

```python
# Run the standalone demo
python esm/cognitive/bio_accounting_demo.py
```

## 🔧 Framework Architecture

### Core Components

1. **AtomSpace**: Hypergraph knowledge representation
   - Protein nodes and amino acid atoms
   - Structural and functional relationships  
   - Truth values and attention weights

2. **PLN (Probabilistic Logic Networks)**:
   - Uncertain reasoning and inference
   - Trial balance proof generation
   - Confidence-weighted conclusions

3. **ECAN (Economic Attention Networks)**:
   - Attention-based resource allocation
   - Economic principles for cognitive processing
   - Dynamic prioritization of important proteins

4. **MOSES (Meta-Optimizing Semantic Evolutionary Search)**:
   - Multi-objective evolutionary optimization
   - Protein variant generation and selection
   - Fitness landscape exploration

5. **URE (Uncertain Reasoning Engine)**:
   - Uncertainty quantification and propagation  
   - Confidence interval calculations
   - Risk assessment for protein modifications

### Cognitive Account Types

The framework supports multiple levels of cognitive sophistication:

```python
from esm.cognitive import CognitiveAccountType

# Basic accounting representation
TRADITIONAL = 1

# Learning-enabled proteins
ADAPTIVE = 2          

# Forward-looking analysis  
PREDICTIVE = 4        

# Multi-input processing
MULTIMODAL = 8        

# Economic attention allocation
ATTENTION_DRIVEN = 16 

# Probabilistic reasoning
PLN_REASONING = 32    

# Uncertainty quantification
UNCERTAINTY_AWARE = 64

# Evolutionary optimization
EVOLUTIONARY_OPTIMIZED = 128

# Combine capabilities with bitwise OR
advanced_type = (ADAPTIVE | PREDICTIVE | 
                ATTENTION_DRIVEN | UNCERTAINTY_AWARE)
```

## 📊 Example Applications

### Portfolio Optimization
```python
# Create protein portfolio
portfolio = BioCognitivePortfolio()

# Add proteins with different roles
portfolio.add_protein(enzyme_protein, role="catalyst")
portfolio.add_protein(structural_protein, role="scaffold") 
portfolio.add_protein(regulatory_protein, role="control")

# Optimize for multiple objectives
optimized = portfolio.optimize(
    objectives=["stability", "solubility", "binding_affinity"],
    constraints={"immunogenicity": "minimize"},
    method="pareto_front"
)
```

### Uncertainty-Aware Predictions
```python
# Create uncertainty-aware account
account = framework.create_account(
    protein, 
    CognitiveAccountType.PREDICTIVE | CognitiveAccountType.UNCERTAINTY_AWARE
)

# Make prediction with confidence intervals
prediction = account.predict_with_uncertainty(
    property="stability",
    horizon=10,
    confidence_level=0.95
)

print(f"Stability: {prediction.prediction:.3f} ± {prediction.uncertainty:.3f}")
print(f"95% CI: [{prediction.confidence_interval[0]:.3f}, {prediction.confidence_interval[1]:.3f}]")
```

### Attention Economics
```python
# Initialize attention economy
economy = AttentionEconomy(initial_funds=1000.0)

# Run economic cycles
for cycle in range(100):
    economy.allocate_attention(proteins)
    
    # Proteins compete for computational resources
    # High-performing proteins receive more attention
    # Attention decays over time requiring continuous performance
```

## 🎯 Learning Path Recommendations

### For Beginners
1. Start with **Bio-Cognitive Accounting Tutorial**
2. Focus on basic concepts and visualizations
3. Experiment with different cognitive account types
4. Try the standalone demo mode

### For Protein Scientists
1. **Bio-Cognitive Accounting Tutorial** (concepts)
2. **Advanced Cognitive Protein Analysis** (applications)
3. **Evolutionary Protein Optimization** (design)
4. Focus on practical protein analysis examples

### For AI/ML Researchers  
1. **Advanced Cognitive Protein Analysis** (cognitive architectures)
2. **Evolutionary Protein Optimization** (optimization methods)
3. **Bio-Cognitive Accounting Tutorial** (domain application)
4. Focus on cognitive computing and OpenCog integration

### For Software Engineers
1. Review framework architecture documentation
2. **Bio-Cognitive Accounting Tutorial** (API usage)
3. **Advanced Cognitive Protein Analysis** (implementation details)
4. Contribute to open-source development

## 🔬 Research Applications

The Bio-Cognitive Accounting Framework enables novel research directions:

### Drug Discovery
- **Risk-Adjusted Portfolio**: Model drug candidates as cognitive accounts with uncertainty quantification
- **Attention-Driven Screening**: Prioritize promising compounds using economic attention allocation
- **Evolutionary Optimization**: Design improved variants using MOSES with multi-objective fitness

### Protein Engineering
- **Stability-Function Trade-offs**: Use Pareto optimization for balanced protein design
- **Uncertainty-Aware Design**: Quantify confidence in engineered protein properties  
- **Cognitive Trial Balance**: Validate engineered protein sets for consistency

### Systems Biology
- **Network-Level Accounting**: Model entire protein networks as accounting systems
- **Emergent Behavior Detection**: Identify unexpected system-level properties
- **Dynamic Resource Allocation**: Optimize cellular resource distribution

### Biotechnology
- **Production Optimization**: Optimize protein expression systems using attention economics
- **Quality Control**: Use trial balance validation for batch consistency
- **Process Monitoring**: Track production KPIs using performance metrics

## 📖 Additional Resources

### Documentation
- [Bio-Cognitive Accounting Specification](../docs/BIO_COGNITIVE_ACCOUNTING.md)
- [Technical Architecture Guide](../docs/TECHNICAL_ARCHITECTURE.md)  
- [API Reference](../docs/api_reference.md)

### Research Papers
- *ESM3: Simulating 500 Million Years of Evolution with a Language Model*
- *OpenCog: A Software Framework for Integrative Artificial General Intelligence*
- *Probabilistic Logic Networks for Uncertain Reasoning*

### Community
- [GitHub Discussions](https://github.com/EchoCog/echo-esm/discussions)
- [Issues and Bug Reports](https://github.com/EchoCog/echo-esm/issues)
- [Contributing Guidelines](../CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions to the Bio-Cognitive Accounting Framework:

1. **Notebook Improvements**: Enhanced examples and tutorials
2. **New Applications**: Novel use cases and domain applications  
3. **Framework Extensions**: Additional cognitive capabilities
4. **Documentation**: Clearer explanations and guides
5. **Bug Fixes**: Issue resolution and testing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the same terms as the ESM3 framework. See [LICENSE.md](../LICENSE.md) for details.

---

**Happy Bio-Cognitive Computing! 🧬🧠✨**

*Transforming protein analysis through the elegant marriage of accounting principles and cognitive computing.*