"""
Economic Attention Allocation Network (ECAN) for ESM3 Cognitive Framework

Implements attention economics for protein analysis:
- STI (Short-term Importance): Immediate attention allocation
- LTI (Long-term Importance): Persistent significance tracking
- VLTI (Very Long-term Importance): Historical importance
- Cognitive Economics: Resource allocation and competition
- Attention-driven processing prioritization

ECAN manages cognitive resources in protein analysis tasks.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import time
from enum import Enum

from .atomspace import AtomHandle, TruthValue, ProteinAtomSpace


class AttentionType(Enum):
    STI = "short_term_importance"
    LTI = "long_term_importance" 
    VLTI = "very_long_term_importance"


@dataclass
class AttentionValue:
    """OpenCog-style attention value with STI/LTI/VLTI"""
    sti: float  # Short-term importance (-1000 to +1000)
    lti: float  # Long-term importance (0 to +1000) 
    vlti: float  # Very long-term importance (0 to +1000)
    
    def __post_init__(self):
        self.sti = max(-1000, min(1000, self.sti))
        self.lti = max(0, min(1000, self.lti))
        self.vlti = max(0, min(1000, self.vlti))


@dataclass
class ECANParams:
    """ECAN economic parameters"""
    sti_funds: float = 10000.0  # Total STI fund pool
    lti_funds: float = 10000.0  # Total LTI fund pool
    sti_wage: float = 1.0       # STI wage per activity
    lti_wage: float = 0.1       # LTI wage per activity
    sti_rent: float = 0.01      # STI decay rate per cycle
    lti_rent: float = 0.001     # LTI decay rate per cycle
    min_sti: float = -100.0     # Minimum STI before forgetting
    attention_threshold: float = 50.0  # Threshold for attention focus


@dataclass
class ProteinActivity:
    """Protein-related activity record"""
    protein_atom: AtomHandle
    activity_type: str
    intensity: float
    timestamp: float
    duration: float = 1.0


class ProteinECAN:
    """
    Economic Attention Allocation Network for proteins
    
    Manages cognitive attention allocation across protein atoms based on:
    - Recent activity (STI)
    - Historical importance (LTI/VLTI)
    - Economic resource constraints
    - Attention competition dynamics
    """
    
    def __init__(self, atomspace: ProteinAtomSpace, params: Optional[ECANParams] = None):
        self.atomspace = atomspace
        self.params = params or ECANParams()
        self.attention_values: Dict[str, AttentionValue] = {}
        self.activity_history: List[ProteinActivity] = []
        self.last_update: float = time.time()
        self.cycle_count = 0
        
    def get_attention_value(self, atom: AtomHandle) -> AttentionValue:
        """Get current attention value for atom"""
        return self.attention_values.get(atom.atom_id, AttentionValue(0.0, 0.0, 0.0))
    
    def set_attention_value(self, atom: AtomHandle, attention: AttentionValue):
        """Set attention value for atom"""
        self.attention_values[atom.atom_id] = attention
    
    def record_activity(self, protein_atom: AtomHandle, activity_type: str, 
                       intensity: float = 1.0, duration: float = 1.0):
        """Record protein-related activity"""
        activity = ProteinActivity(
            protein_atom=protein_atom,
            activity_type=activity_type,
            intensity=intensity,
            timestamp=time.time(),
            duration=duration
        )
        self.activity_history.append(activity)
        
        # Immediate STI reward for activity
        self._reward_activity(activity)
    
    def _reward_activity(self, activity: ProteinActivity):
        """Reward atom with STI/LTI for activity"""
        current_attention = self.get_attention_value(activity.protein_atom)
        
        # Calculate wage based on activity intensity and duration
        sti_reward = self.params.sti_wage * activity.intensity * activity.duration
        lti_reward = self.params.lti_wage * activity.intensity * activity.duration
        
        # Update attention values
        new_attention = AttentionValue(
            sti=current_attention.sti + sti_reward,
            lti=current_attention.lti + lti_reward,
            vlti=current_attention.vlti
        )
        
        self.set_attention_value(activity.protein_atom, new_attention)
        
        # Update fund pools
        self.params.sti_funds -= sti_reward
        self.params.lti_funds -= lti_reward
    
    def allocate_attention_cycle(self):
        """Run one cycle of attention allocation"""
        self.cycle_count += 1
        current_time = time.time()
        
        # Decay attention values (rent collection)
        self._collect_rent()
        
        # Redistribute attention based on recent activity
        self._redistribute_attention()
        
        # Promote LTI based on sustained STI
        self._promote_to_lti()
        
        # Archive to VLTI for very important atoms
        self._promote_to_vlti()
        
        # Forget low-attention atoms
        self._forget_low_attention()
        
        # Update economic parameters
        self._update_economics()
        
        self.last_update = current_time
    
    def _collect_rent(self):
        """Collect attention rent (decay mechanism)"""
        for atom_id, attention in self.attention_values.items():
            sti_rent = attention.sti * self.params.sti_rent
            lti_rent = attention.lti * self.params.lti_rent
            
            new_attention = AttentionValue(
                sti=attention.sti - sti_rent,
                lti=attention.lti - lti_rent,
                vlti=attention.vlti  # VLTI doesn't decay
            )
            
            self.attention_values[atom_id] = new_attention
            
            # Return rent to fund pools
            self.params.sti_funds += sti_rent
            self.params.lti_funds += lti_rent
    
    def _redistribute_attention(self):
        """Redistribute attention based on activity patterns"""
        # Get recent activities (last 10 cycles)
        recent_cutoff = time.time() - 10.0
        recent_activities = [a for a in self.activity_history if a.timestamp > recent_cutoff]
        
        if not recent_activities:
            return
        
        # Calculate activity scores
        activity_scores = {}
        for activity in recent_activities:
            atom_id = activity.protein_atom.atom_id
            score = activity.intensity * np.exp(-(time.time() - activity.timestamp) / 5.0)  # Exponential decay
            activity_scores[atom_id] = activity_scores.get(atom_id, 0) + score
        
        # Distribute bonus attention to active atoms
        total_score = sum(activity_scores.values())
        if total_score > 0:
            bonus_pool = min(1000.0, self.params.sti_funds * 0.1)  # Use 10% of fund
            
            for atom_id, score in activity_scores.items():
                bonus = (score / total_score) * bonus_pool
                if atom_id in self.attention_values:
                    current = self.attention_values[atom_id]
                    self.attention_values[atom_id] = AttentionValue(
                        sti=current.sti + bonus,
                        lti=current.lti,
                        vlti=current.vlti
                    )
                    self.params.sti_funds -= bonus
    
    def _promote_to_lti(self):
        """Promote atoms with sustained STI to LTI"""
        for atom_id, attention in self.attention_values.items():
            if attention.sti > self.params.attention_threshold * 2:
                # Promote some STI to LTI for sustained importance
                promotion = attention.sti * 0.1
                
                self.attention_values[atom_id] = AttentionValue(
                    sti=attention.sti - promotion,
                    lti=attention.lti + promotion,
                    vlti=attention.vlti
                )
    
    def _promote_to_vlti(self):
        """Promote very important atoms to VLTI"""
        for atom_id, attention in self.attention_values.items():
            if attention.lti > self.params.attention_threshold * 5:
                # Very important atoms get VLTI
                vlti_promotion = attention.lti * 0.05
                
                self.attention_values[atom_id] = AttentionValue(
                    sti=attention.sti,
                    lti=attention.lti - vlti_promotion,
                    vlti=attention.vlti + vlti_promotion
                )
    
    def _forget_low_attention(self):
        """Remove atoms with very low attention"""
        to_forget = []
        
        for atom_id, attention in self.attention_values.items():
            total_attention = attention.sti + attention.lti + attention.vlti
            if total_attention < self.params.min_sti and attention.sti < 0:
                to_forget.append(atom_id)
        
        # Remove forgotten atoms
        for atom_id in to_forget:
            del self.attention_values[atom_id]
    
    def _update_economics(self):
        """Update economic parameters based on system state"""
        # Adjust fund levels based on total attention
        total_sti = sum(av.sti for av in self.attention_values.values() if av.sti > 0)
        total_lti = sum(av.lti for av in self.attention_values.values())
        
        # Maintain fund balance
        if self.params.sti_funds < 1000:
            self.params.sti_funds = min(10000, self.params.sti_funds * 1.1)
        
        if self.params.lti_funds < 1000:
            self.params.lti_funds = min(10000, self.params.lti_funds * 1.05)
    
    def get_top_attention_atoms(self, n: int = 10, attention_type: AttentionType = AttentionType.STI) -> List[Tuple[AtomHandle, float]]:
        """Get top N atoms by attention value"""
        scored_atoms = []
        
        for atom_id, attention in self.attention_values.items():
            atom = self.atomspace.get_atom(atom_id)
            if atom:
                if attention_type == AttentionType.STI:
                    score = attention.sti
                elif attention_type == AttentionType.LTI:
                    score = attention.lti
                else:  # VLTI
                    score = attention.vlti
                
                scored_atoms.append((atom, score))
        
        # Sort by score descending
        scored_atoms.sort(key=lambda x: x[1], reverse=True)
        return scored_atoms[:n]
    
    def get_attentional_focus(self) -> List[AtomHandle]:
        """Get atoms currently in attentional focus"""
        focused_atoms = []
        
        for atom_id, attention in self.attention_values.items():
            if attention.sti > self.params.attention_threshold:
                atom = self.atomspace.get_atom(atom_id)
                if atom:
                    focused_atoms.append(atom)
        
        return focused_atoms
    
    def protein_attention_summary(self, protein_atom: AtomHandle) -> Dict[str, Any]:
        """Get comprehensive attention summary for protein"""
        attention = self.get_attention_value(protein_atom)
        
        # Get related activities
        protein_activities = [a for a in self.activity_history 
                            if a.protein_atom.atom_id == protein_atom.atom_id]
        
        recent_activities = [a for a in protein_activities 
                           if a.timestamp > time.time() - 60.0]  # Last minute
        
        return {
            'attention_values': {
                'sti': attention.sti,
                'lti': attention.lti, 
                'vlti': attention.vlti,
                'total': attention.sti + attention.lti + attention.vlti
            },
            'activity_stats': {
                'total_activities': len(protein_activities),
                'recent_activities': len(recent_activities),
                'avg_intensity': np.mean([a.intensity for a in protein_activities]) if protein_activities else 0.0
            },
            'focus_status': attention.sti > self.params.attention_threshold,
            'importance_rank': self._get_importance_rank(protein_atom)
        }
    
    def _get_importance_rank(self, protein_atom: AtomHandle) -> int:
        """Get importance ranking of protein among all proteins"""
        all_proteins = self.atomspace.get_atoms_by_name_pattern("Protein_")
        
        protein_scores = []
        target_attention = self.get_attention_value(protein_atom)
        target_score = target_attention.sti + target_attention.lti + target_attention.vlti
        
        for protein in all_proteins:
            attention = self.get_attention_value(protein)
            score = attention.sti + attention.lti + attention.vlti
            protein_scores.append(score)
        
        protein_scores.sort(reverse=True)
        
        try:
            return protein_scores.index(target_score) + 1
        except ValueError:
            return len(protein_scores) + 1
    
    def update_protein_attention(self, protein_atom: AtomHandle, activity_type: str, 
                               intensity: float = 1.0):
        """Update attention for protein based on activity"""
        self.record_activity(protein_atom, activity_type, intensity)
        
        # Run mini attention cycle for immediate updates
        if self.cycle_count % 10 == 0:  # Every 10 activities
            self.allocate_attention_cycle()
    
    def optimize_attention_allocation(self, cognitive_load: float = 1.0, 
                                   available_resources: float = 1.0) -> Dict[str, Any]:
        """Optimize attention allocation based on system constraints"""
        # Adjust economic parameters based on resources
        resource_multiplier = available_resources / cognitive_load
        
        adjusted_params = ECANParams(
            sti_funds=self.params.sti_funds * resource_multiplier,
            lti_funds=self.params.lti_funds * resource_multiplier,
            sti_wage=self.params.sti_wage * resource_multiplier,
            lti_wage=self.params.lti_wage * resource_multiplier,
            attention_threshold=self.params.attention_threshold / resource_multiplier
        )
        
        # Temporarily use adjusted parameters
        old_params = self.params
        self.params = adjusted_params
        
        # Run optimization cycle
        self.allocate_attention_cycle()
        
        # Get optimization results
        focused_atoms = self.get_attentional_focus()
        top_sti = self.get_top_attention_atoms(5, AttentionType.STI)
        top_lti = self.get_top_attention_atoms(5, AttentionType.LTI)
        
        optimization_result = {
            'focused_atoms': len(focused_atoms),
            'top_sti_atoms': [(atom.name, score) for atom, score in top_sti],
            'top_lti_atoms': [(atom.name, score) for atom, score in top_lti],
            'resource_utilization': {
                'sti_fund_usage': (10000 - self.params.sti_funds) / 10000,
                'lti_fund_usage': (10000 - self.params.lti_funds) / 10000
            },
            'attention_distribution': self._compute_attention_distribution()
        }
        
        # Restore original parameters
        self.params = old_params
        
        return optimization_result
    
    def _compute_attention_distribution(self) -> Dict[str, float]:
        """Compute attention distribution statistics"""
        if not self.attention_values:
            return {'mean_sti': 0.0, 'mean_lti': 0.0, 'mean_vlti': 0.0, 'std_sti': 0.0}
        
        sti_values = [av.sti for av in self.attention_values.values()]
        lti_values = [av.lti for av in self.attention_values.values()]
        vlti_values = [av.vlti for av in self.attention_values.values()]
        
        return {
            'mean_sti': np.mean(sti_values),
            'mean_lti': np.mean(lti_values),
            'mean_vlti': np.mean(vlti_values),
            'std_sti': np.std(sti_values),
            'std_lti': np.std(lti_values),
            'total_atoms': len(self.attention_values)
        }