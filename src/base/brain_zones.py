
from dataclasses import dataclass
from typing import Dict, Any
from base.layers import BaseLayer, BaseLayerContainerConfig



@dataclass
class BrainZoneConfig:
    name:str = ""
    max_neurons:int = 512
    min_neurons:int = 128
    neuron_type:str = "liquid"
    gated:bool = False
    num_layers:int = 2
    base_layer_container_config:BaseLayerContainerConfig = None
    
class BrainZone:
    config:BrainZoneConfig
    layers:Dict[int,BaseLayer]
    def __init__(self, config:BrainZoneConfig, layers:Dict[int,BaseLayer]):
        self.config = config
        self.layers = layers

    def get_config(self) -> BrainZoneConfig:
        return self.config