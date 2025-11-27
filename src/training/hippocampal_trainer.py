"""
Hippocampal Transformer Trainer (GPU-Native)

Optimized to remove all NumPy dependencies.
- ReplayBuffer uses torch.randperm for sampling.
- Fully compatible with the GPU-native architecture.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import time

from training.losses import HippocampalLoss

class ReplayBuffer:
    """
    Experience Replay Buffer (PyTorch Native).
    Stores experiences and samples them using Torch random generators.
    """
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        # Buffer stores tuples of (input_ids, labels, loss_val)
        # We store them as CPU tensors to avoid VRAM OOM, but all ops are Torch
        self.buffer: List[Tuple[torch.Tensor, torch.Tensor, float]] = []
        
    def add(self, input_ids: torch.Tensor, labels: torch.Tensor, loss: float):
        """Add batch to buffer."""
        # input_ids: [Batch, Seq]
        batch_size = input_ids.size(0)
        
        # Detach and move to CPU to save GPU memory for the model
        # This is standard practice; moving back to GPU for replay is cheap compared to
        # training step compute.
        input_cpu = input_ids.detach().cpu()
        labels_cpu = labels.detach().cpu()
        
        for i in range(batch_size):
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0) # FIFO eviction
            
            # Store single sample
            self.buffer.append((input_cpu[i], labels_cpu[i], loss))
            
    def sample(self, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample batch_size items without replacement.
        """
        n = len(self.buffer)
        if n == 0:
            return []
            
        k = min(n, batch_size)
        
        # Random sampling using PyTorch
        # torch.randperm is equivalent to np.random.permutation
        indices = torch.randperm(n)[:k].tolist()
        
        return [self.buffer[i] for i in indices]
        
    def __len__(self):
        return len(self.buffer)

class EWCConsolidator:
    """
    Elastic Weight Consolidation (PyTorch Native).
    Computes Fisher Information to constrain important weights.
    """
    def __init__(self, model: nn.Module, lambda_ewc: float = 0.4):
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.optpar = {}
        
    def compute_fisher(self, dataloader, device):
        """
        Compute Fisher Information Matrix.
        dataloader: List of (input, label) tuples (mock loader from buffer)
        """
        self.fisher = {}
        self.optpar = {}
        
        # Initialize zero tensors for Fisher
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher[name] = torch.zeros_like(param.data)
                self.optpar[name] = param.data.clone()
                
        self.model.eval()
        count = 0
        
        # Iterate through sampled data
        for inputs, labels in dataloader:
            self.model.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            # Note: We don't need memory creation here, just loss gradient
            outputs = self.model(inputs, use_memory=False)
            
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
                
            # Compute likelihood gradient (standard CE loss)
            loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            
            # Accumulate squared gradients (Fisher diagonal approximation)
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher[name] += param.grad.data ** 2
            
            count += 1
            # Limit samples to avoid stalling (e.g. 50 batches is usually enough estimation)
            if count >= 50:
                break
                
        # Normalize
        if count > 0:
            for name in self.fisher:
                self.fisher[name] /= count
            
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Calculate EWC loss penalty."""
        loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher:
                # EWC Loss: sum( F_i * (theta_i - theta_star_i)^2 )
                fisher = self.fisher[name].to(param.device)
                optpar = self.optpar[name].to(param.device)
                loss += (fisher * (param - optpar) ** 2).sum()
        return loss * self.lambda_ewc

class HippocampalTransformerTrainer:
    """
    Main Trainer Class.
    Orchestrates the training phases and memory management.
    """
    def __init__(self, model, config, hippocampus):
        self.model = model
        self.config = config
        self.hippocampus = hippocampus
        
        # Initialize Loss with config params (defaults if missing)
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
        """Advance step and check for sleep phase transition."""
        self.global_step += 1
        if self.phase == "wake" and self.global_step % self.sleep_interval == 0:
            self.phase = "sleep"
            print(f"🌙 Entering SLEEP phase at step {self.global_step}")

    def train_step_wake(self, batch) -> torch.Tensor:
        """
        Execute one training step in Wake phase.
        - Forward pass with memory enabled
        - Compute loss
        - Store experience in buffer
        """
        input_ids, labels, prosody = batch
        
        # Forward pass (Wake mode: use_memory=True to encode new memories)
        logits, place_cell_activity = self.model(
            input_ids, 
            prosody=prosody, 
            use_memory=True
        )
        
        # Calculate hybrid loss
        loss = self.criterion(logits, labels, place_cell_activity)
        
        # Add EWC penalty if established
        if self.ewc.fisher:
            loss += self.ewc.penalty(self.model)
            
        # Store raw data for replay (loss used for priority if we implement PER later)
        self.replay_buffer.add(input_ids, labels, loss.item())
        
        return loss

    def train_step_sleep(self) -> Optional[torch.Tensor]:
        """
        Execute one replay step in Sleep phase.
        - Sample from buffer
        - Forward replay
        - Backward replay (temporal shuffling)
        - No memory creation (use_memory=False typically, but here True for retrieval practice)
        """
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        if not batch:
            return None
        
        device = next(self.model.parameters()).device
        
        # Stack inputs (batch[i][0] is input_ids)
        input_ids = torch.stack([item[0] for item in batch]).to(device)
        labels = torch.stack([item[1] for item in batch]).to(device)
        
        # 1. Forward Replay
        logits, _ = self.model(input_ids, use_memory=True)
        # Sleep loss: Just CrossEntropy, ignore sparsity/entropy constraints for consolidation
        loss = self.criterion.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # 2. Backward Replay (Simulates biological reverse replay)
        # Reverse the sequence time dimension
        input_rev = torch.flip(input_ids, dims=[1])
        labels_rev = torch.flip(labels, dims=[1])
        
        logits_rev, _ = self.model(input_rev, use_memory=True)
        loss_rev = self.criterion.ce_loss(logits_rev.view(-1, logits_rev.size(-1)), labels_rev.view(-1))
        
        return loss + loss_rev