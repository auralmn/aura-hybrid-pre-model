from dataclasses import dataclass
import json
from typing import Dict, Any


@dataclass
class BrainStats:
    num_neurons:int = 0
    num_layers:int = 0
    num_zones:int = 0
    num_experts:int = 0
    num_memories:float = 0
    num_contexts:int = 0
    num_parameters:int = 0
    
    def __repr__(self) -> object:
        return self.get_stats()
        
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'num_neurons': self.num_neurons,
            'num_layers': self.num_layers,
            'num_zones': self.num_zones,
            'num_experts': self.num_experts,
            'num_memories': self.num_memories,
            'num_contexts': self.num_contexts,
            'num_parameters': self.num_parameters
        }

    def get_stats_string(self) -> str:
        return json.dumps(self.get_stats(), indent=2)

    def save_stats(self, filename: str):
        with open(filename, 'w') as f:
            json.dump(self.get_stats(), f)

    def load_stats(self, filename: str):
        with open(filename, 'r') as f:
            stats = json.load(f)
        self.num_neurons = stats['num_neurons']
        self.num_layers = stats['num_layers']
        self.num_zones = stats['num_zones']
        self.num_experts = stats['num_experts']
        self.num_memories = stats['num_memories']
        self.num_contexts = stats['num_contexts']
        self.num_parameters = stats['num_parameters']


    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        self.save_stats(f"brain_stats_{name}.json")