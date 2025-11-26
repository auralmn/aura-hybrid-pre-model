from enum import Enum, auto

class Specialization(Enum):
    PYRAMIDAL_CELL = auto()       # Cortex, principal excitatory
    INTERNEURON = auto()          # Inhibitory, local circuit
    PLACE_CELL = auto()           # Hippocampus, spatial memory
    GRID_CELL = auto()            # Spatial periodicity
    SPEED_CELL = auto()           # Movement encoding
    HEAD_DIRECTION_CELL = auto()  # Orientation/heading
    BETZ_CELL = auto()            # Large motor cortex neuron
    BOUNDARY_CELL = auto()        # Landmark/context boundary
    SPINDLE_CELL = auto()         # Long-range association
    MOTOR_NEURON = auto()         # Output, spinal cord, movement
    SENSORY_NEURON = auto()       # Input, various modalities
    CHOLINERGIC = auto()          # Neurotransmitter specialization
    GLUTAMATERGIC = auto()        # Excitatory transmitter
    GABAERGIC = auto()            # Inhibitory transmitter
    DOPAMINERGIC = auto()         # Reward, motivation, gating
    ASTROCYTE = auto()            # Support glial cell (optional)
    OLIGODENDROCYTE = auto()      # Myelin-forming glia (optional)
    GENERIC = auto()              # Default or unknown

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


CODON_FUNCTION_MAP = {
    frozenset({DNACodon.NEUROGENIN, DNACodon.NEUROD1}): Specialization.PYRAMIDAL_CELL,
    frozenset({DNACodon.GAD67}): Specialization.INTERNEURON,
    frozenset({DNACodon.BDNF, DNACodon.MAP2}): Specialization.PLACE_CELL,
    frozenset({DNACodon.CUSTOM1}): Specialization.GRID_CELL,
    frozenset({DNACodon.CHAT}): Specialization.CHOLINERGIC,
    # Add more as needed
}
