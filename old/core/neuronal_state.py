
from enum import Enum, auto


class NeuronalStateEnum(Enum):
    RESTING = auto()         # Baseline polarization, ready to fire
    FIRING = auto()          # Action potential, spiking
    REFRACTORY = auto()      # Temporarily unable to fire after spike
    FATIGUED = auto()        # Reduced excitability due to recent activity
    DREAM = auto()           # Offline, restoration, or concept discovery
    DIFFERENTIATED = auto()  # Mature, specialized function
    PROGENITOR = auto()      # Immature, dividing/neurogenic
    MIGRATING = auto()       # Moving to final location/layer
    MYELINATED = auto()      # With myelin, rapid conduction
    SYNAPTIC_PLASTICITY = auto() # Actively remodeling synapses
    LEARNING = auto()        # Current learning/adaptation phase
    CONSOLIDATING = auto()   # Solidifying traces, LTM formation
    SUBCONSCIOUS = auto()    # Involved in implicit/gated processes
    SHADOW = auto()          # Backup trace, not directly involved
    TRANSITORY = auto()      # Temporary or short-lived activation
    DEAD = auto()            # Cell death or pruned (for completeness)
    UNKNOWN = auto()         # Catch-all for undefined state


class NeuronalState:
    def __init__(self, kind, position, membrane_potential=0.0, 
                 gene_expression=None, cell_cycle='G1', 
                 maturation='progenitor', activity='resting',
                 connections=None, environment=None, plasticity=None):
        self.kind = kind
        self.position = position          # Spatial coordinates, e.g., (x, y, z)
        self.membrane_potential = membrane_potential
        self.gene_expression = gene_expression or {}   # {'Neurogenin': 0.8, ...}
        self.cell_cycle = cell_cycle      # 'G1', 'S', 'G2', 'M'
        self.maturation = maturation      # 'progenitor', 'migrating', etc.
        self.activity = activity          # 'resting', 'firing', etc.
        self.connections = connections or []
        self.environment = environment or {}    # {'BDNF': 0.3, 'Wnt': 0.5, ...}
        self.plasticity = plasticity or {}      # LTP/LTD/STDP traces
        self.fatigue = 0.0


    def update_fatigue(self, activity_level): 
        # Increase with activity, recover if resting
        if activity_level == 'firing':
            self.fatigue = min(1.0, self.fatigue + 0.1)
        else:
            self.fatigue = max(0.0, self.fatigue - 0.01)

    def update_potential(self, input_current):
        # Simple integrate-and-fire model as example
        self.membrane_potential += input_current
        if self.membrane_potential > 1.0:
            self.activity = 'firing'
            self.membrane_potential = 0.0
        else:
            self.activity = 'resting'
    
    def differentiate(self, signals):
        # Change neuronal state based on gene expression and signals
        if self.gene_expression.get('Neurogenin', 0) > 0.8 and signals.get('Wnt',0) > 0.7:
            self.maturation = 'differentiated'
