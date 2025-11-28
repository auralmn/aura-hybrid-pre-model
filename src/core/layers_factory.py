from typing import Dict, Any
from base import BaseLayerContainerConfig, BaseLayerContainer
from base.layers import BaseLayer, BaseLayerConfig, BaseLayerImplementation

class LayersFactory:
    def __init__(self, config: BaseLayerContainerConfig):
        self.config = config
        self.layers = []

    def create_layers(self, config: BaseLayerConfig) -> BaseLayerContainer:
        for i in range(self.config.num_layers):
           # print(f"Creating layer {i} with config {config}")
            self.layers.append(BaseLayerImplementation(config=config))
        return BaseLayerContainer(config=self.config, layers=self.layers)