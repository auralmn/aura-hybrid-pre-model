from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class BaseLayerConfig:
    name: str
    input_dim: int
    output_dim: int
    dt: float = 0.02
    tau_min: float = 0.02
    tau_max: float = 2.0

class BaseLayer(ABC):
    config: BaseLayerConfig
    @abstractmethod
    def forward(self, x: Any) -> Any:
        pass

    @abstractmethod
    def get_config(self) -> BaseLayerConfig:
        pass

    def __post_init__(self):
        pass

    

# similar to meningial layer in the brain
@dataclass
class BaseLayerContainerConfig:
    num_layers: int = 2
    layer_type: str = "sparse"
    layer_config: BaseLayerConfig = None


class BaseLayerImplementation(BaseLayer):
    def __init__(self, config: BaseLayerConfig):
        self.config = config
    
    def get_config(self) -> BaseLayerConfig:
        return self.config

    def forward(self, x: Any) -> Any:
        return 0
 

class BaseLayerContainer:
    config: BaseLayerContainerConfig
    layers: Dict[int,BaseLayer]

    def __init__(self, config: BaseLayerContainerConfig, layers: Dict[int,BaseLayer]):
        self.config = config
        self.layers = layers

    def get_config(self) -> BaseLayerContainerConfig:
        return self.config

    def get_layers(self) -> Dict[int,BaseLayer]:
        return self.layers


class BaseLayerFactory:
    def __init__(self):
        pass

    def create_layer(self, config: BaseLayerConfig) -> BaseLayer:
        return BaseLayerImplementation(config)