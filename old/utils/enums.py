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



class DNACodon(Enum):
    NEUROGENIN = auto()
    ASCL1 = auto()
    NEUROD1 = auto()
    CACNA1 = auto()    # Example: ion channel or differentiation marker
    GAD67 = auto()     # GABA synthesis
    MAP2 = auto()      # Dendritic differentiation
    SLC17A7 = auto()   # Vesicular glutamate transporter
    CHAT = auto()      # Choline acetyltransferase
    BDNF = auto()      # Growth/repair
    CUSTOM1 = auto()   # Synthetic, for programmer-defined specialization
    CUSTOM2 = auto()

