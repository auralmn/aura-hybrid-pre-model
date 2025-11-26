from ast import Dict
from dataclasses import dataclass, field
from typing import List

from src.base.brain_zones import BrainZoneConfig
from src.base.brain_zone_factory import BrainZoneFactory
from src.base.layers import BaseLayerContainerConfig
from src.core.layers_factory import LayersFactory
from src.core.neuron_factory import NeuronFactory

@dataclass
class Config:
    """Configuration for the Aura brain system.

    Provides default values for all fields to maintain backward compatibility.
    """
    name: str = "AURA"
    new: bool = False
    layers_config: BaseLayerContainerConfig = field(default_factory=BaseLayerContainerConfig)
    brain_zones_config: List[BrainZoneConfig] = field(default_factory=list)
    save_path: str = ""
    checkpoint_path: str = ""
    tmp_path: str = ""
    log_path: str = ""

# Alias for backward compatibility – many modules import BrainConfig
class BrainConfig(Config):
    """Alias for Config to keep existing import statements working."""
    pass

# Default configuration instance used throughout the codebase
default_config = Config(
    name="AURA",
    new=False,
    layers_config=BaseLayerContainerConfig(),
    brain_zones_config=[],
    save_path="",
    checkpoint_path="",
    tmp_path="",
    log_path="",
)