from ast import Dict
from dataclasses import dataclass
from typing import List

from base.brain_zones import BrainZoneConfig
from base.brain_zone_factory import BrainZoneFactory
from base.layers import BaseLayerContainerConfig
from core.layers_factory import LayersFactory
from core.neuron_factory import NeuronFactory

@dataclass
class Config:
    name: str
    new: bool
    layers_config: BaseLayerContainerConfig = BaseLayerContainerConfig()
    brain_zones_config: List[BrainZoneConfig]
    save_path: str
    checkpoint_path: str
    tmp_path: str
    log_path: str
    

default_config = Config(
    name="AURA",
    new=False,
    layers_config=BaseLayerContainerConfig(),
    brain_zones_config=[],
    save_path="",
    checkpoint_path="",
    tmp_path="",
    log_path=""
)