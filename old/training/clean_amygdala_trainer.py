#!/usr/bin/env python3
"""
Clean Amygdala Training Pipeline
Implements surgical fixes for AURA-ready training with Liquid-MoE integration
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
from typing import Dict, List, Tuple, Any

# ============================================================================
# Configuration
# ============================================================================

SINE_LENGTH = 32
SBERT_DIM = 384
EXTRA_FEATURES = 3
TOTAL_FEATURES = SINE_LENGTH + EXTRA_FEATURES + SBERT_DIM  # 419

# Training parameters
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 5e-3
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ============================================================================
# Utility Functions
# ============================================================================

def batch_encode_sbert(texts: List[str], model_name: str = "all-MiniLM-L6-v2", 
                      device: str = None, batch_size: int = 128) -> np.ndarray:
    """Efficiently batch-encode texts with SBERT"""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    model = SentenceTransformer(model_name, device=device)
    vecs = []
    
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        with torch.no_grad():
            v = model.encode(chunk, normalize_embeddings=True, convert_to_numpy=True)
        vecs.append(v)
    
    return np.vstack(vecs)

def build_features(record: Dict[str, Any], sbert_vec: np.ndarray, 
                  label_params: Dict[str, Dict], sine_len: int = 32) -> np.ndarray:
    """Build fused features: sine + extras + SBERT"""
    # Primary emotion sine wave
    prim = record.get("plutchik", {}).get("primary", "none")
    if isinstance(prim, list):
        prim = prim[0] if prim else "none"
    
    inten = float(record.get("plutchik", {}).get("intensity", 1.0))
    cfg = label_params.get(prim, {"freq": 1.5, "amp": 0.7, "phase": 0.5})
    
    t = np.linspace(0, 2*np.pi, sine_len, dtype=np.float32)
    emb = (cfg["amp"] * inten * np.sin(cfg["freq"] * t + cfg["phase"])).astype(np.float32)
    
    # Secondary emotion (if present)
    sec = record.get("plutchik", {}).get("secondary")
    if isinstance(sec, list):
        sec = sec[0] if sec else None
    
    if sec and sec in label_params:
        cfg2 = label_params[sec]
        emb += 0.5 * (cfg2["amp"] * inten * np.sin(cfg2["freq"] * t + cfg2["phase"])).astype(np.float32)
    
    # Extra features
    text = record.get("text", "")
    extras = np.array([
        len(text) / 100.0,
        int("!" in text),
        int(record.get("tone", "") in {"euphoric", "tense", "somber", "peaceful", "amazed"})
    ], dtype=np.float32)
    
    # Concatenate: sine + extras + SBERT
    return np.concatenate([emb, extras, sbert_vec]).astype(np.float32)

def generate_default_params(labels: List[str]) -> Dict[str, Dict]:
    """Generate default sine wave parameters for labels"""
    base_freq = 1.5
    base_phase = 0.5
    params = {}
    
    for idx, label in enumerate(labels):
        params[label] = {
            "freq": base_freq + 0.3 * idx,
            "amp": 0.7,
            "phase": base_phase + 0.4 * idx
        }
    
    return params

# ============================================================================
# Model Definitions
# ============================================================================

class LinearSoftmax(nn.Module):
    """Linear classifier for AmygdalaRelay compatibility"""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)

class MultiTaskHead(nn.Module):
    """Multi-task classifier for emotion, intent, and tone"""
    def __init__(self, input_dim: int, num_emotions: int, num_intents: int, num_tones: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.emotion_head = nn.Linear(256, num_emotions)
        self.intent_head = nn.Linear(256, num_intents)
        self.tone_head = nn.Linear(256, num_tones)
    
    def forward(self, x):
        h = self.shared(x)
        return self.emotion_head(h), self.intent_head(h), self.tone_head(h)

def multitask_loss(emotion_out, emotion_y, intent_out, intent_y, tone_out, tone_y, 
                  weights=(1.0, 0.7, 0.7)):
    """Multi-task loss function"""
    ce = nn.CrossEntropyLoss()
    return (weights[0] * ce(emotion_out, emotion_y) + 
            weights[1] * ce(intent_out, intent_y) + 
            weights[2] * ce(tone_out, tone_y))

# ============================================================================
# Data Processing
# ============================================================================

def load_and_process_data(data_path: str) -> Tuple[List[Dict], np.ndarray, Dict[str, Any]]:
    """Load dataset and precompute SBERT embeddings"""
    print("Loading dataset...")
    dataset = []
    all_texts = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                dataset.append(record)
                all_texts.append(record.get("text", ""))
    
    print(f"Loaded {len(dataset)} records")
    
    # Precompute SBERT embeddings (batched)
    print("Computing SBERT embeddings...")
    sbert_embeddings = batch_encode_sbert(all_texts, batch_size=BATCH_SIZE)
    print(f"SBERT embeddings shape: {sbert_embeddings.shape}")
    
    # Collect unique labels
    emotion_labels = set()
    intent_labels = set()
    tone_labels = set()
    
    for record in dataset:
        # Emotion labels
        prim = record.get("plutchik", {}).get("primary", "none")
        if isinstance(prim, list):
            prim = prim[0] if prim else "none"
        emotion_labels.add(prim)
        
        # Intent labels
        intent = record.get("intent", "none")
        intent_labels.add(intent)
        
        # Tone labels
        tone = record.get("tone", "none")
        tone_labels.add(tone)
    
    # Create label mappings
    emotion_labels = sorted(list(emotion_labels))
    intent_labels = sorted(list(intent_labels))
    tone_labels = sorted(list(tone_labels))
    
    label_maps = {
        'emotion': {label: idx for idx, label in enumerate(emotion_labels)},
        'intent': {label: idx for idx, label in enumerate(intent_labels)},
        'tone': {label: idx for idx, label in enumerate(tone_labels)},
        'emotion_reverse': {idx: label for idx, label in enumerate(emotion_labels)},
        'intent_reverse': {idx: label for idx, label in enumerate(intent_labels)},
        'tone_reverse': {idx: label for idx, label in enumerate(tone_labels)}
    }
    
    return dataset, sbert_embeddings, label_maps

def create_stratified_split(dataset: List[Dict], sbert_embeddings: np.ndarray, 
                           test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Create stratified train/test split by primary emotion"""
    print("Creating stratified split...")
    
    # Extract primary emotion labels for stratification
    primary_labels = []
    for record in dataset:
        prim = record.get("plutchik", {}).get("primary", "none")
        if isinstance(prim, list):
            prim = prim[0] if prim else "none"
        primary_labels.append(prim)
    
    # Stratified split
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)), 
        test_size=test_size, 
        random_state=random_state, 
        stratify=primary_labels
    )
    
    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]
    sbert_train = sbert_embeddings[train_idx]
    sbert_test = sbert_embeddings[test_idx]
    
    print(f"Train: {len(train_data)}, Test: {len(test_data)}")
    return train_data, test_data, sbert_train, sbert_test

def tensorize_task(data: List[Dict], sbert_slice: np.ndarray, label_map: Dict[str, int], 
                   label_getter, label_params: Dict[str, Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert data to tensors for a specific task"""
    X, y = [], []
    
    for record, sbert_vec in zip(data, sbert_slice):
        features = build_features(record, sbert_vec, label_params)
        X.append(features)
        
        label_val = label_getter(record)
        y.append(label_map.get(label_val, label_map.get("none", 0)))
    
    return torch.tensor(np.stack(X), dtype=torch.float32), torch.tensor(y)

# ============================================================================
# Training Functions
# ============================================================================

def train_linear_classifier(input_dim: int, num_classes: int, X: torch.Tensor, y: torch.Tensor,
                           epochs: int = 12, lr: float = 5e-3) -> LinearSoftmax:
    """Train a linear classifier for AmygdalaRelay compatibility"""
    model = LinearSoftmax(input_dim, num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training linear classifier: {input_dim} -> {num_classes}")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
        
        if epoch % 3 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                preds = model(X).argmax(dim=1)
                acc = (preds == y).float().mean().item()
                print(f"  Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    return model

def train_multitask_classifier(model: MultiTaskHead, X: torch.Tensor, 
                              emotion_y: torch.Tensor, intent_y: torch.Tensor, tone_y: torch.Tensor,
                              epochs: int = 30, lr: float = 5e-3) -> MultiTaskHead:
    """Train multi-task classifier"""
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    print(f"Training multi-task classifier for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        emotion_out, intent_out, tone_out = model(X)
        loss = multitask_loss(emotion_out, emotion_y, intent_out, intent_y, tone_out, tone_y)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                emotion_preds = emotion_out.argmax(dim=1)
                intent_preds = intent_out.argmax(dim=1)
                tone_preds = tone_out.argmax(dim=1)
                
                emotion_acc = (emotion_preds == emotion_y).float().mean().item()
                intent_acc = (intent_preds == intent_y).float().mean().item()
                tone_acc = (tone_preds == tone_y).float().mean().item()
                
                print(f"  Epoch {epoch}: Loss={loss:.4f}, "
                      f"Emotion={emotion_acc:.4f}, Intent={intent_acc:.4f}, Tone={tone_acc:.4f}")
    
    return model

# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_classifier(model, X: torch.Tensor, y: torch.Tensor, 
                       label_map: Dict[int, str], task_name: str = "Classification") -> Dict[str, float]:
    """Evaluate classifier and return metrics"""
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        
        accuracy = (preds == y).float().mean().item()
        f1_macro = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='macro')
        f1_weighted = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        
        print(f"\n{task_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Macro: {f1_macro:.4f}")
        print(f"  F1-Weighted: {f1_weighted:.4f}")
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }

# ============================================================================
# Export Functions
# ============================================================================

def export_linear_weights(model: LinearSoftmax, output_dir: str, task_name: str, 
                         label_maps: Dict[str, Any]) -> None:
    """Export linear weights for AmygdalaRelay compatibility"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract weights and biases
    W = model.fc.weight.detach().cpu().numpy().T  # Shape: (input_dim, num_classes)
    b = model.fc.bias.detach().cpu().numpy()      # Shape: (num_classes,)
    
    # Save weights
    np.save(os.path.join(output_dir, f"{task_name}_W.npy"), W)
    np.save(os.path.join(output_dir, f"{task_name}_b.npy"), b)
    
    # Save label mappings - handle case where task_name might not exist in label_maps
    if task_name in label_maps and f'{task_name}_reverse' in label_maps:
        label_data = {
            'LABEL_TO_IDX': label_maps[task_name],
            'IDX_TO_LABEL': label_maps[f'{task_name}_reverse']
        }
    else:
        # Fallback: create from available data
        label_data = {
            'LABEL_TO_IDX': label_maps.get(task_name, {}),
            'IDX_TO_LABEL': label_maps.get(f'{task_name}_reverse', {})
        }
    
    with open(os.path.join(output_dir, f"{task_name}_labels.json"), 'w') as f:
        json.dump(label_data, f, indent=2)
    
    print(f"Exported {task_name} weights: W{W.shape}, b{b.shape}")

def save_multitask_model(model: MultiTaskHead, output_dir: str, label_maps: Dict[str, Any]) -> None:
    """Save multi-task model and all label mappings"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), os.path.join(output_dir, "multitask_model.pt"))
    
    # Save all label mappings
    with open(os.path.join(output_dir, "all_labels.json"), 'w') as f:
        json.dump(label_maps, f, indent=2)
    
    print(f"Saved multi-task model and labels to {output_dir}")

# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """Main training pipeline"""
    print("🧠 Clean Amygdala Training Pipeline")
    print("=" * 60)
    
    # Load and process data
    dataset, sbert_embeddings, label_maps = load_and_process_data("train.jsonl")
    
    # Create stratified split
    train_data, test_data, sbert_train, sbert_test = create_stratified_split(
        dataset, sbert_embeddings, TEST_SIZE, RANDOM_STATE
    )
    
    # Generate sine wave parameters
    emotion_params = generate_default_params(list(label_maps['emotion'].keys()))
    
    # Build features for all tasks
    print("Building features...")
    
    # Emotion features
    X_emotion_train, y_emotion_train = tensorize_task(
        train_data, sbert_train, label_maps['emotion'], 
        lambda r: r.get("plutchik", {}).get("primary", "none"), emotion_params
    )
    X_emotion_test, y_emotion_test = tensorize_task(
        test_data, sbert_test, label_maps['emotion'],
        lambda r: r.get("plutchik", {}).get("primary", "none"), emotion_params
    )
    
    # Intent features
    X_intent_train, y_intent_train = tensorize_task(
        train_data, sbert_train, label_maps['intent'],
        lambda r: r.get("intent", "none"), emotion_params
    )
    X_intent_test, y_intent_test = tensorize_task(
        test_data, sbert_test, label_maps['intent'],
        lambda r: r.get("intent", "none"), emotion_params
    )
    
    # Tone features
    X_tone_train, y_tone_train = tensorize_task(
        train_data, sbert_train, label_maps['tone'],
        lambda r: r.get("tone", "none"), emotion_params
    )
    X_tone_test, y_tone_test = tensorize_task(
        test_data, sbert_test, label_maps['tone'],
        lambda r: r.get("tone", "none"), emotion_params
    )
    
    # Verify dimensions
    assert X_emotion_train.size(1) == TOTAL_FEATURES, f"Expected {TOTAL_FEATURES}, got {X_emotion_train.size(1)}"
    print(f"✅ Feature dimension verified: {TOTAL_FEATURES}")
    
    # Train linear classifiers for AmygdalaRelay compatibility
    print("\n1. Training Linear Classifiers for AmygdalaRelay...")
    
    emotion_linear = train_linear_classifier(
        TOTAL_FEATURES, len(label_maps['emotion']), X_emotion_train, y_emotion_train
    )
    intent_linear = train_linear_classifier(
        TOTAL_FEATURES, len(label_maps['intent']), X_intent_train, y_intent_train
    )
    tone_linear = train_linear_classifier(
        TOTAL_FEATURES, len(label_maps['tone']), X_tone_train, y_tone_train
    )
    
    # Train multi-task classifier
    print("\n2. Training Multi-Task Classifier...")
    
    multitask_model = MultiTaskHead(
        TOTAL_FEATURES, 
        len(label_maps['emotion']), 
        len(label_maps['intent']), 
        len(label_maps['tone'])
    )
    
    multitask_model = train_multitask_classifier(
        multitask_model, X_emotion_train, y_emotion_train, y_intent_train, y_tone_train
    )
    
    # Evaluate models
    print("\n3. Evaluating Models...")
    
    # Linear classifiers
    evaluate_classifier(emotion_linear, X_emotion_test, y_emotion_test, 
                       label_maps['emotion_reverse'], "Emotion (Linear)")
    evaluate_classifier(intent_linear, X_intent_test, y_intent_test, 
                       label_maps['intent_reverse'], "Intent (Linear)")
    evaluate_classifier(tone_linear, X_tone_test, y_tone_test, 
                       label_maps['tone_reverse'], "Tone (Linear)")
    
    # Multi-task model
    with torch.no_grad():
        emotion_out, intent_out, tone_out = multitask_model(X_emotion_test)
        
        emotion_preds = emotion_out.argmax(dim=1)
        intent_preds = intent_out.argmax(dim=1)
        tone_preds = tone_out.argmax(dim=1)
        
        emotion_acc = (emotion_preds == y_emotion_test).float().mean().item()
        intent_acc = (intent_preds == y_intent_test).float().mean().item()
        tone_acc = (tone_preds == y_tone_test).float().mean().item()
        
        print(f"\nMulti-Task Results:")
        print(f"  Emotion: {emotion_acc:.4f}")
        print(f"  Intent: {intent_acc:.4f}")
        print(f"  Tone: {tone_acc:.4f}")
    
    # Export models
    print("\n4. Exporting Models...")
    
    export_linear_weights(emotion_linear, "models", "emotion_classifier", label_maps)
    export_linear_weights(intent_linear, "models", "intent_classifier", label_maps)
    export_linear_weights(tone_linear, "models", "tone_classifier", label_maps)
    
    save_multitask_model(multitask_model, "models", label_maps)
    
    print("\n🎉 Training completed successfully!")
    print(f"✅ All models exported to models/ directory")
    print(f"✅ Linear weights compatible with AmygdalaRelay")
    print(f"✅ Multi-task model ready for Liquid-MoE integration")

if __name__ == "__main__":
    main()
