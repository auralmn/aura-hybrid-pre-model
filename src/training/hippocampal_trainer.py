"""
Hippocampal Transformer Trainer (Natural Brain Compatible)

Optimized to remove all NumPy dependencies.
- ReplayBuffer uses torch.randperm for sampling.
- Fully compatible with the GPU-native Natural Brain architecture.
- Includes gradient checkpointing, cosine LR schedule, and monitoring.
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple, Optional, Any, Dict
import time

from src.training.losses import HippocampalLoss


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1
):
    """
    Cosine decay learning rate schedule with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        max_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of initial LR
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
                
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
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
    
    Features:
    - Gradient checkpointing support
    - Cosine LR schedule with warmup
    - Memory warmup (disable hippocampal memory during early training)
    - Gradient clipping and monitoring
    - Mixed precision training support
    """
    def __init__(self, model, config, hippocampus, optimizer: Optional[torch.optim.Optimizer] = None):
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
        
        # Optimization settings
        self.warmup_steps = getattr(config, 'warmup_steps', 2000)
        self.max_steps = getattr(config, 'max_steps', 100000)
        self.memory_warmup_steps = getattr(config, 'memory_warmup_steps', 5000)
        self.gradient_clip = getattr(config, 'gradient_clip', 1.0)
        self.min_lr_ratio = getattr(config, 'min_lr_ratio', 0.1)
        
        # LR Scheduler (created if optimizer provided)
        self.optimizer = optimizer
        self.scheduler = None
        if optimizer is not None:
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                warmup_steps=self.warmup_steps,
                max_steps=self.max_steps,
                min_lr_ratio=self.min_lr_ratio
            )
        
        # Monitoring
        self.metrics: Dict[str, float] = {
            'loss': 0.0,
            'grad_norm': 0.0,
            'lr': 0.0,
            'perplexity': 0.0
        }
        
        # Enable gradient checkpointing if configured
        if getattr(config, 'use_gradient_checkpointing', False):
            if hasattr(model, 'set_gradient_checkpointing'):
                model.set_gradient_checkpointing(True)
                print("Gradient checkpointing enabled")
        
    def step_counter(self):
        self.global_step += 1
        if self.phase == "wake" and self.global_step % self.sleep_interval == 0:
            self.phase = "sleep"
            print(f"Entering SLEEP phase at step {self.global_step}")
    
    def should_use_memory(self) -> bool:
        """Determine if hippocampal memory should be active."""
        return self.global_step >= self.memory_warmup_steps
    
    def clip_gradients(self) -> float:
        """Clip gradients and return the norm."""
        if self.gradient_clip > 0:
            return torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                max_norm=self.gradient_clip
            ).item()
        return 0.0
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        if self.scheduler is not None:
            return self.scheduler.get_last_lr()[0]
        elif self.optimizer is not None:
            return self.optimizer.param_groups[0]['lr']
        return 0.0
    
    def log_metrics(self, loss: float, grad_norm: float):
        """Update and optionally print metrics."""
        self.metrics['loss'] = loss
        self.metrics['grad_norm'] = grad_norm
        self.metrics['lr'] = self.get_lr()
        self.metrics['perplexity'] = math.exp(min(loss, 20))  # Clamp to avoid overflow
        
        if self.global_step % 100 == 0:
            mem_status = "ON" if self.should_use_memory() else "OFF"
            print(f"Step {self.global_step} | Loss: {loss:.4f} | PPL: {self.metrics['perplexity']:.2f} | "
                  f"Grad: {grad_norm:.2f} | LR: {self.metrics['lr']:.2e} | Memory: {mem_status}")

    def train_step_wake(self, batch) -> torch.Tensor:
        """
        Execute one training step in Wake phase.
        Supports both standard models and NaturalBrain architecture.
        
        Returns:
            loss: The computed loss tensor (not yet backpropagated)
        """
        input_ids, labels, prosody = batch
        
        # Determine if memory should be active (warmup period)
        use_memory = self.should_use_memory()
        
        # Forward pass with prosody and conditional memory
        output = self.model(input_ids, prosody=prosody, use_memory=use_memory)
        
        # Handle NaturalBrain (tuple output) vs Standard (maybe tuple or tensor)
        info = {}
        if isinstance(output, tuple):
            logits = output[0]
            place_cell_activity = output[1] if len(output) > 1 else None
            info = output[2] if len(output) > 2 else {}
        else:
            logits = output
            place_cell_activity = None
        
        # Shift for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss = self.criterion(shift_logits, shift_labels, place_cell_activity)
        
        # Add EWC penalty (only after warmup)
        if self.ewc.fisher and self.global_step > self.warmup_steps:
            loss = loss + self.ewc.penalty(self.model)
            
        # Update Brain Homeostasis (NaturalBrain Feature)
        if hasattr(self.model, 'update_homeostasis'):
            est_acc = torch.exp(-loss.detach()).item()
            self.model.update_homeostasis({'accuracy': est_acc})
            
            if self.global_step % 100 == 0 and isinstance(info, dict):
                h = info.get('hormones', {})
                cort = h.get('cortisol', 0.0)
                dopa = h.get('dopamine', 0.0)
                print(f"   Hormones: Cortisol={cort:.3f} | Dopamine={dopa:.3f}")
            
        # Store in Replay Buffer
        self.replay_buffer.add(input_ids, labels, loss.item())
        
        return loss
    
    def train_step(self, batch) -> Dict[str, float]:
        """
        Complete training step with backward pass, gradient clipping, and scheduler step.
        
        Args:
            batch: Tuple of (input_ids, labels, prosody)
            
        Returns:
            Dictionary with metrics
        """
        if self.optimizer is None:
            raise ValueError("Optimizer not set. Pass optimizer to __init__ or use train_step_wake for loss only.")
        
        self.optimizer.zero_grad()
        
        # Forward and compute loss
        loss = self.train_step_wake(batch)
        
        # Backward
        loss.backward()
        
        # Clip gradients
        grad_norm = self.clip_gradients()
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Log metrics
        self.log_metrics(loss.item(), grad_norm)
        
        # Update step counter
        self.step_counter()
        
        return {
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'lr': self.get_lr(),
            'perplexity': self.metrics['perplexity']
        }

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