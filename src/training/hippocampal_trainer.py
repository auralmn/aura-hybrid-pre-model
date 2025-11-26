"""
Hippocampal Transformer Trainer

Implements the biologically plausible training loop with:
1. Wake Phase: Standard learning + memory encoding
2. Sleep Phase: Memory replay + consolidation
3. EWC: Elastic Weight Consolidation to prevent catastrophic forgetting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Dict
import random
from collections import deque
import numpy as np
import copy

class ReplayBuffer:
    """
    Episodic Memory Store (Replay Buffer).
    Stores experiences (input_ids, labels, loss) and samples based on priority.
    """
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        # Store as list of tuples: (input_ids, labels, loss)
        self.buffer = []
        
    def add(self, input_ids: torch.Tensor, labels: torch.Tensor, loss: float):
        """Add experience to buffer. Breaks batches into individual samples."""
        batch_size = input_ids.size(0)
        
        # Store each sample individually
        for i in range(batch_size):
            if len(self.buffer) >= self.capacity:
                # FIFO eviction
                self.buffer.pop(0)
                
            # Store detached tensors to save memory
            self.buffer.append((
                input_ids[i].detach().cpu(),  # [seq_len]
                labels[i].detach().cpu(),      # [seq_len]
                float(loss)
            ))
        
    def sample(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        """
        Sample experiences based on priority (loss).
        Higher loss = higher probability of being sampled (prioritized replay).
        """
        if not self.buffer:
            return []
            
        losses = np.array([item[2] for item in self.buffer])
        # Add small epsilon to avoid 0 probability
        probs = losses + 1e-6
        probs = probs / probs.sum()
        
        indices = np.random.choice(
            len(self.buffer), 
            size=min(batch_size, len(self.buffer)), 
            p=probs,
            replace=False
        )
        
        return [self.buffer[i] for i in indices]
        
    def __len__(self):
        return len(self.buffer)


class EWCConsolidator:
    """
    Elastic Weight Consolidation (EWC).
    Computes Fisher Information Matrix to constrain important weights.
    """
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.optpar = {}
        
    def compute_fisher(self, dataloader, device):
        """Compute Fisher Information Matrix."""
        self.fisher = {}
        self.optpar = {}
        
        # Initialize Fisher dict
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data)
                self.optpar[name] = param.data.clone()
                
        self.model.eval()
        
        count = 0
        for input_ids, labels in dataloader:
            self.model.zero_grad()
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            
            # Forward pass
            if hasattr(self.model, 'forward_train'):
                outputs = self.model.forward_train(input_ids)
            else:
                outputs = self.model(input_ids)
                
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            # Calculate loss (NLL)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            
            # Accumulate squared gradients
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2
            
            count += 1
            if count >= 100:
                break
                
        # Normalize
        for name in self.fisher:
            self.fisher[name] /= count
            
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Calculate EWC penalty loss."""
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher:
                fisher = self.fisher[name].to(param.device)
                optpar = self.optpar[name].to(param.device)
                loss += (fisher * (param - optpar) ** 2).sum()
        return self.lambda_ewc * loss


class HippocampalTransformerTrainer:
    """
    Trainer orchestrating Wake/Sleep phases and Memory Consolidation.
    """
    def __init__(self, model, config, hippocampus):
        self.model = model
        self.config = config
        self.hippocampus = hippocampus
        
        self.replay_buffer = ReplayBuffer(capacity=getattr(config, 'replay_buffer_size', 50000))
        self.ewc = EWCConsolidator(model, lambda_ewc=getattr(config, 'ewc_lambda', 0.4))
        
        self.phase = "wake"
        self.global_step = 0
        self.sleep_interval = getattr(config, 'sleep_interval', 1000)
        self.sleep_steps = getattr(config, 'sleep_steps', 100)
        
    def step_counter(self):
        """Increment step and check for phase transition."""
        self.global_step += 1
        
        if self.phase == "wake":
            if self.global_step % self.sleep_interval == 0:
                self.phase = "sleep"
                print(f"🌙 Entering SLEEP phase at step {self.global_step}")
        elif self.phase == "sleep":
            pass
            
    def should_sleep(self) -> bool:
        """Check if sleep phase should start."""
        return self.global_step > 0 and self.global_step % self.sleep_interval == 0
        
    def train_step_wake(self, batch):
        """Standard training step + memory storage."""
        input_ids, labels = batch
        
        outputs = self.model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        self.replay_buffer.add(input_ids, labels, loss.item())
        
        if self.ewc.fisher:
            loss += self.ewc.penalty(self.model)
            
        return loss
        
    def train_step_sleep(self):
        """Replay step (Sleep phase)."""
        batch = self.replay_buffer.sample(self.config.batch_size)
        if not batch:
            return None
            
        # Unpack batch - now each item is [seq_len] not [batch, seq_len]
        input_ids = torch.stack([item[0] for item in batch]).to(next(self.model.parameters()).device)
        labels = torch.stack([item[1] for item in batch]).to(next(self.model.parameters()).device)
        
        # Forward Replay
        outputs = self.model(input_ids)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward Replay (Reverse Sequence)
        input_ids_rev = torch.flip(input_ids, dims=[1])
        labels_rev = torch.flip(labels, dims=[1])
        
        outputs_rev = self.model(input_ids_rev)
        if isinstance(outputs_rev, tuple):
            logits_rev = outputs_rev[0]
        else:
            logits_rev = outputs_rev
        loss_rev = nn.CrossEntropyLoss()(logits_rev.view(-1, logits_rev.size(-1)), labels_rev.view(-1))
        
        return loss + loss_rev
