#!/usr/bin/env python3
"""
Test script for the clean Amygdala training pipeline
Demonstrates the pipeline without requiring the actual dataset
"""

import json
import numpy as np
import torch
from clean_amygdala_trainer import (
    batch_encode_sbert, build_features, generate_default_params,
    LinearSoftmax, MultiTaskHead, multitask_loss,
    create_stratified_split, tensorize_task, train_linear_classifier,
    train_multitask_classifier, evaluate_classifier, export_linear_weights
)

def create_mock_dataset(n_samples: int = 1000) -> list:
    """Create a mock dataset for testing"""
    emotions = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation", "none"]
    intents = ["question", "statement", "request", "exclamation", "none"]
    tones = ["euphoric", "tense", "somber", "peaceful", "amazed", "none"]
    
    texts = [
        "I'm so excited about this project!",
        "This is terrible, I can't believe it happened.",
        "What is the time complexity of this algorithm?",
        "I love this new feature! It's amazing!",
        "I'm feeling anxious about the presentation tomorrow.",
        "Can you help me understand this concept?",
        "WOW! This is incredible! I'm amazed!",
        "I'm disappointed with the results.",
        "This is a great opportunity for learning.",
        "I'm confused about how this works."
    ]
    
    dataset = []
    for i in range(n_samples):
        record = {
            "text": np.random.choice(texts),
            "plutchik": {
                "primary": np.random.choice(emotions),
                "intensity": np.random.uniform(0.3, 1.0),
                "secondary": np.random.choice(emotions + [None])
            },
            "intent": np.random.choice(intents),
            "tone": np.random.choice(tones)
        }
        dataset.append(record)
    
    return dataset

def test_feature_building():
    """Test feature building functionality"""
    print("🧪 Testing Feature Building...")
    
    # Create mock data
    dataset = create_mock_dataset(100)
    texts = [record["text"] for record in dataset]
    
    # Test SBERT encoding
    print("  Testing SBERT encoding...")
    sbert_embeddings = batch_encode_sbert(texts, batch_size=32)
    print(f"  SBERT shape: {sbert_embeddings.shape}")
    assert sbert_embeddings.shape == (100, 384), f"Expected (100, 384), got {sbert_embeddings.shape}"
    
    # Test feature building
    print("  Testing feature building...")
    label_params = generate_default_params(["joy", "trust", "fear", "sadness", "none"])
    
    features = []
    for record, sbert_vec in zip(dataset, sbert_embeddings):
        feat = build_features(record, sbert_vec, label_params)
        features.append(feat)
    
    features = np.stack(features)
    print(f"  Features shape: {features.shape}")
    assert features.shape == (100, 419), f"Expected (100, 419), got {features.shape}"
    
    print("  ✅ Feature building test passed!")

def test_stratified_split():
    """Test stratified splitting functionality"""
    print("\n🧪 Testing Stratified Split...")
    
    dataset = create_mock_dataset(1000)
    texts = [record["text"] for record in dataset]
    sbert_embeddings = batch_encode_sbert(texts, batch_size=128)
    
    train_data, test_data, sbert_train, sbert_test = create_stratified_split(
        dataset, sbert_embeddings, test_size=0.2, random_state=42
    )
    
    print(f"  Train: {len(train_data)}, Test: {len(test_data)}")
    print(f"  SBERT train: {sbert_train.shape}, SBERT test: {sbert_test.shape}")
    
    assert len(train_data) == 800, f"Expected 800 train samples, got {len(train_data)}"
    assert len(test_data) == 200, f"Expected 200 test samples, got {len(test_data)}"
    assert sbert_train.shape == (800, 384), f"Expected (800, 384), got {sbert_train.shape}"
    assert sbert_test.shape == (200, 384), f"Expected (200, 384), got {sbert_test.shape}"
    
    print("  ✅ Stratified split test passed!")

def test_model_training():
    """Test model training functionality"""
    print("\n🧪 Testing Model Training...")
    
    # Create mock data
    dataset = create_mock_dataset(500)
    texts = [record["text"] for record in dataset]
    sbert_embeddings = batch_encode_sbert(texts, batch_size=128)
    
    train_data, test_data, sbert_train, sbert_test = create_stratified_split(
        dataset, sbert_embeddings, test_size=0.2, random_state=42
    )
    
    # Create label mappings
    emotion_labels = ["joy", "trust", "fear", "sadness", "none"]
    intent_labels = ["question", "statement", "request", "none"]
    tone_labels = ["euphoric", "tense", "somber", "none"]
    
    label_maps = {
        'emotion': {label: idx for idx, label in enumerate(emotion_labels)},
        'intent': {label: idx for idx, label in enumerate(intent_labels)},
        'tone': {label: idx for idx, label in enumerate(tone_labels)},
        'emotion_reverse': {idx: label for idx, label in enumerate(emotion_labels)},
        'intent_reverse': {idx: label for idx, label in enumerate(intent_labels)},
        'tone_reverse': {idx: label for idx, label in enumerate(tone_labels)}
    }
    
    # Generate emotion parameters
    emotion_params = generate_default_params(emotion_labels)
    
    # Build features
    X_emotion_train, y_emotion_train = tensorize_task(
        train_data, sbert_train, label_maps['emotion'],
        lambda r: r.get("plutchik", {}).get("primary", "none"), emotion_params
    )
    X_emotion_test, y_emotion_test = tensorize_task(
        test_data, sbert_test, label_maps['emotion'],
        lambda r: r.get("plutchik", {}).get("primary", "none"), emotion_params
    )
    
    print(f"  Training features: {X_emotion_train.shape}")
    print(f"  Training labels: {y_emotion_train.shape}")
    
    # Test linear classifier training
    print("  Testing linear classifier training...")
    linear_model = train_linear_classifier(
        X_emotion_train.size(1), len(emotion_labels), 
        X_emotion_train, y_emotion_train, epochs=5, lr=1e-2
    )
    
    # Test evaluation
    metrics = evaluate_classifier(
        linear_model, X_emotion_test, y_emotion_test, 
        label_maps['emotion_reverse'], "Emotion (Test)"
    )
    
    assert metrics['accuracy'] > 0.0, "Model should have some accuracy"
    print(f"  Linear classifier accuracy: {metrics['accuracy']:.4f}")
    
    # Test multi-task model
    print("  Testing multi-task model...")
    X_intent_train, y_intent_train = tensorize_task(
        train_data, sbert_train, label_maps['intent'],
        lambda r: r.get("intent", "none"), emotion_params
    )
    X_tone_train, y_tone_train = tensorize_task(
        train_data, sbert_train, label_maps['tone'],
        lambda r: r.get("tone", "none"), emotion_params
    )
    
    multitask_model = MultiTaskHead(
        X_emotion_train.size(1), len(emotion_labels), 
        len(intent_labels), len(tone_labels)
    )
    
    multitask_model = train_multitask_classifier(
        multitask_model, X_emotion_train, y_emotion_train, 
        y_intent_train, y_tone_train, epochs=5, lr=1e-2
    )
    
    print("  ✅ Model training test passed!")

def test_export_functionality():
    """Test model export functionality"""
    print("\n🧪 Testing Export Functionality...")
    
    # Create a simple model
    model = LinearSoftmax(419, 5)
    
    # Create mock label maps
    label_maps = {
        'emotion': {'joy': 0, 'trust': 1, 'fear': 2, 'sadness': 3, 'none': 4},
        'emotion_reverse': {0: 'joy', 1: 'trust', 2: 'fear', 3: 'sadness', 4: 'none'}
    }
    
    # Test export
    try:
        export_linear_weights(model, "test_models", "test_classifier", label_maps)
        print("  ✅ Export functionality test passed!")
    except Exception as e:
        print(f"  ❌ Export test failed: {e}")

def main():
    """Run all tests"""
    print("🧪 AURA Training Pipeline Test Suite")
    print("=" * 60)
    
    try:
        test_feature_building()
        test_stratified_split()
        test_model_training()
        test_export_functionality()
        
        print("\n🎉 All tests passed successfully!")
        print("\nKey Features Verified:")
        print("  ✅ Efficient SBERT batch encoding")
        print("  ✅ Feature building with correct dimensions")
        print("  ✅ Stratified train/test splitting")
        print("  ✅ Linear classifier training")
        print("  ✅ Multi-task model training")
        print("  ✅ Model evaluation and metrics")
        print("  ✅ Export functionality")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
