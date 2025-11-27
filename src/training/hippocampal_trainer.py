"""
Hippocampal Transformer Trainer (Natural Brain Compatible)

Optimized to remove all NumPy dependencies.
- ReplayBuffer uses torch.randperm for sampling.
- Fully compatible with the GPU-native Natural Brain architecture.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Any
import time

from src.training.losses import HippocampalLoss

class ReplayBuffer:
    """
    Experience Replay Buffer (PyTorch Native).
    """
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        
    def add(self, input_ids: torch.Tensor, labels: torch.Tensor, loss: float):
        batch_size = input_ids.size(0)
        input_cpu = input_ids.detach().cpu()
        labels_cpu = labels.detach().cpu()
        
        for i in range(batch_size):
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            self.buffer.append((input_cpu[i], labels_cpu[i], loss))
            
    def sample(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        n = len(self.buffer)
        if n == 0: return []
        k = min(n, batch_size)
        indices = torch.randperm(n)[:k].tolist()
        return [self.buffer[i] for i in indices]
        
    def __len__(self):
        return len(self.buffer)

class EWCConsolidator:
    """Elastic Weight Consolidation (PyTorch Native)."""
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.optpar = {}
        
    def compute_fisher(self, dataloader, device):
        self.fisher = {}
        self.optpar = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data)
                self.optpar[name] = param.data.clone()
                
        self.model.eval()
        count = 0
        
        for inputs, labels in dataloader:
            self.model.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            output = self.model(inputs)
            # Handle tuple return (logits, info)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
                
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2
            
            count += 1
            if count >= 50: break
                
        if count > 0:
            for name in self.fisher:
                self.fisher[name] /= count
            
    def penalty(self, model: nn.Module) -> torch.Tensor:
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                fisher = self.fisher[name].to(param.device)
                optpar = self.optpar[name].to(param.device)
                loss += (fisher * (param - optpar) ** 2).sum()
        return loss * self.lambda_ewc

class HippocampalTransformerTrainer:
    """
    Main Trainer Class (Natural Brain Ready).
    """
    def __init__(self, model, config, hippocampus):
        self.model = model
        self.config = config
        self.hippocampus = hippocampus
        
        self.criterion = HippocampalLoss(
            label_smoothing=getattr(config, 'label_smoothing', 0.1),
            entropy_lambda=getattr(config, 'entropy_lambda', 0.05),
            sparsity_lambda=getattr(config, 'sparsity_lambda', 0.02),
            target_sparsity=getattr(config, 'target_sparsity', 0.03)
        )
        
        self.replay_buffer = ReplayBuffer(capacity=getattr(config, 'replay_buffer_size', 50000))
        self.ewc = EWCConsolidator(model, lambda_ewc=getattr(config, 'ewc_lambda', 0.4))
        
        self.phase = "wake"
        self.global_step = 0
        self.sleep_interval = getattr(config, 'sleep_interval', 1000)
        
    def step_counter(self):
        self.global_step += 1
        if self.phase == "wake" and self.global_step % self.sleep_interval == 0:
            self.phase = "sleep"
            print(f"🌙 Entering SLEEP phase at step {self.global_step}")

    def train_step_wake(self, batch) -> torch.Tensor:
        """
        Execute one training step in Wake phase.
        Supports both standard models and NaturalBrain architecture.
        """
        input_ids, labels, prosody = batch
        
        # Forward pass
        output = self.model(input_ids)
        
        # Handle NaturalBrain (tuple output) vs Standard (maybe tuple or tensor)
        info = {}
        if isinstance(output, tuple):
            logits = output[0]
            info = output[1]
        else:
            logits = output
        
        # Extract place activity if available (for sparsity loss)
        # NaturalBrain hides this in zones, so we might skip explicit sparsity penalty 
        # on the main loss and rely on the SNN's internal homeostasis.
        place_cell_activity = None
        
        # Calculate loss
        loss = self.criterion(logits, labels, place_cell_activity)
        
        # Add EWC penalty
        if self.ewc.fisher:
            loss += self.ewc.penalty(self.model)
            
        # Update Brain Homeostasis (NaturalBrain Feature)
        if hasattr(self.model, 'update_homeostasis'):
            # Proxy accuracy using loss
            est_acc = torch.exp(-loss).item()
            self.model.update_homeostasis({'accuracy': est_acc})
            
            # Log Hormones occasionally
            if self.global_step % 100 == 0:
                h = info.get('hormones', {})
                cort = h.get('cortisol', 0.0)
                dopa = h.get('dopamine', 0.0)
                print(f"   🧪 Hormones: Cortisol={cort:.3f} | Dopamine={dopa:.3f}")
            
        # Store in Replay Buffer
        self.replay_buffer.add(input_ids, labels, loss.item())
        
        return loss

    def train_step_sleep(self) -> Optional[torch.Tensor]:
        """Execute one replay step in Sleep phase."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        if not batch: return None
        
        device = next(self.model.parameters()).device
        input_ids = torch.stack([item[0] for item in batch]).to(device)
        labels = torch.stack([item[1] for item in batch]).to(device)
        
        # Forward Replay
        output = self.model(input_ids)
        logits = output[0] if isinstance(output, tuple) else output
        loss = self.criterion.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward Replay
        input_rev = torch.flip(input_ids, dims=[1])
        labels_rev = torch.flip(labels, dims=[1])
        
        output_rev = self.model(input_rev)
        logits_rev = output_rev[0] if isinstance(output_rev, tuple) else output_rev
        loss_rev = self.criterion.ce_loss(logits_rev.view(-1, logits_rev.size(-1)), labels_rev.view(-1))
        
        return loss + loss_rev