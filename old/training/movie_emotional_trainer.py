# Enhanced Movie Emotional Trainer with SBERT Embeddings
# Provides much better text understanding for movie scene analysis

import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import asyncio
import trio
from dataclasses import dataclass

# Import enhanced weight saving
try:
    from enhanced_weight_saving import enhanced_save_weights, enhanced_load_weights
    ENHANCED_SAVING = True
except ImportError:
    ENHANCED_SAVING = False

@dataclass 
class AuraMovieEmotionalTrainer:
    """Enhanced movie emotional trainer with SBERT embeddings and attention support"""
    
    net: Any  # Network instance
    enable_attention: bool = False
    offline: bool = False
    device: str = "cpu"
    verbose: bool = True
    sbert_model_name: str = "all-MiniLM-L6-v2"
    
    def __post_init__(self):
        """Initialize SBERT and setup trainer"""
        self.sbert = None
        self.emotion_stats = {
            'fear': 0, 'anger': 0, 'joy': 0, 'sadness': 0, 
            'surprise': 0, 'disgust': 0, 'neutral': 0, 'love': 0
        }
        self.scene_count = 0
        self.setup_sbert()
        
        if self.verbose:
            print(f"🎬 Movie Emotional Trainer initialized")
            print(f"📱 Device: {self.device}")
            print(f"🧠 SBERT Model: {self.sbert_model_name}")
            print(f"⚡ Attention: {'Enabled' if self.enable_attention else 'Disabled'}")
            print(f"📶 Offline Mode: {'Yes' if self.offline else 'No'}")
    
    def setup_sbert(self):
        """Setup SBERT embeddings with fallback to offline mode"""
        if self.offline:
            if self.verbose:
                print("📴 Running in offline mode - using zero embeddings")
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            if self.verbose:
                print(f"📥 Loading SBERT model: {self.sbert_model_name}")
            
            # Initialize SBERT with device specification
            self.sbert = SentenceTransformer(self.sbert_model_name, device=self.device)
            
            if self.verbose:
                print(f"✅ SBERT loaded successfully on {self.device}")
                
                # Test embedding to ensure it works
                test_embedding = self.sbert.encode("test movie scene", convert_to_tensor=False)
                print(f"🔍 SBERT embedding dimension: {len(test_embedding)}")
                
        except ImportError:
            if self.verbose:
                print("⚠️  sentence-transformers not installed, falling back to offline mode")
            self.offline = True
            self.sbert = None
        except Exception as e:
            if self.verbose:
                print(f"⚠️  SBERT initialization failed: {e}, falling back to offline mode")
            self.offline = True
            self.sbert = None
    
    def extract_scene_features(self, scene_text: str, character: str = "", emotion: str = "") -> np.ndarray:
        """Extract enhanced features from movie scene using SBERT"""
        
        if self.offline or self.sbert is None:
            # Fallback: simple zero padding to 384 dimensions
            return np.zeros(384, dtype=np.float32)
        
        try:
            # Enhance the text with character and emotion context
            enhanced_text = scene_text
            
            if character:
                enhanced_text = f"Character {character}: {enhanced_text}"
            
            if emotion:
                enhanced_text = f"{enhanced_text} [Emotion: {emotion}]"
            
            # Generate SBERT embedding
            embedding = self.sbert.encode(enhanced_text, convert_to_tensor=False)
            
            # Ensure consistent 384-dimensional output
            if len(embedding) > 384:
                return np.array(embedding[:384], dtype=np.float32)
            elif len(embedding) < 384:
                padded = np.zeros(384, dtype=np.float32)
                padded[:len(embedding)] = embedding
                return padded
            else:
                return np.array(embedding, dtype=np.float32)
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Feature extraction failed for scene: {e}")
            return np.zeros(384, dtype=np.float32)
    
    def extract_emotional_features(self, scene_data: Dict[str, Any]) -> Tuple[np.ndarray, str, float]:
        """Extract features and normalize emotion labels"""
        
        # Get scene text
        scene_text = scene_data.get('scene_text', scene_data.get('text', ''))
        character = scene_data.get('character', scene_data.get('speaker', ''))
        emotion = scene_data.get('emotion', scene_data.get('label', 'neutral'))
        
        # Extract SBERT features
        features = self.extract_scene_features(scene_text, character, emotion)
        
        # Normalize emotion labels
        emotion_normalized = self.normalize_emotion_label(emotion)
        
        # Convert emotion to numerical target for training
        emotion_target = self.emotion_to_target(emotion_normalized)
        
        return features, emotion_normalized, emotion_target
    
    def normalize_emotion_label(self, emotion: str) -> str:
        """Normalize emotion labels to standard categories"""
        emotion_lower = str(emotion).lower().strip()
        
        # Map various emotion labels to standard categories
        emotion_mapping = {
            # Fear variations
            'fear': 'fear', 'scared': 'fear', 'afraid': 'fear', 'terrified': 'fear',
            'anxiety': 'fear', 'panic': 'fear', 'nervous': 'fear', 'worried': 'fear',
            
            # Anger variations  
            'anger': 'anger', 'angry': 'anger', 'mad': 'anger', 'furious': 'anger',
            'rage': 'anger', 'irritated': 'anger', 'annoyed': 'anger',
            
            # Joy variations
            'joy': 'joy', 'happy': 'joy', 'excited': 'joy', 'cheerful': 'joy',
            'pleased': 'joy', 'delighted': 'joy', 'content': 'joy',
            
            # Sadness variations
            'sadness': 'sadness', 'sad': 'sadness', 'depressed': 'sadness',
            'melancholy': 'sadness', 'grief': 'sadness', 'sorrow': 'sadness',
            
            # Surprise variations
            'surprise': 'surprise', 'surprised': 'surprise', 'shocked': 'surprise',
            'amazed': 'surprise', 'astonished': 'surprise',
            
            # Disgust variations
            'disgust': 'disgust', 'disgusted': 'disgust', 'revulsion': 'disgust',
            'repulsed': 'disgust', 'nauseated': 'disgust',
            
            # Love variations
            'love': 'love', 'affection': 'love', 'romantic': 'love',
            'caring': 'love', 'tender': 'love', 'adoration': 'love',
            
            # Neutral
            'neutral': 'neutral', 'calm': 'neutral', 'indifferent': 'neutral'
        }
        
        normalized = emotion_mapping.get(emotion_lower, 'neutral')
        
        # Update statistics
        if normalized in self.emotion_stats:
            self.emotion_stats[normalized] += 1
        
        return normalized
    
    def emotion_to_target(self, emotion: str) -> float:
        """Convert emotion to numerical target for NLMS training"""
        # Map emotions to training targets
        emotion_targets = {
            'fear': -0.8,      # Strong negative
            'anger': -0.6,     # Negative
            'sadness': -0.4,   # Mild negative
            'disgust': -0.5,   # Negative
            'neutral': 0.0,    # Neutral
            'surprise': 0.2,   # Mild positive
            'joy': 0.8,        # Strong positive
            'love': 0.9        # Very positive
        }
        
        return emotion_targets.get(emotion, 0.0)
    
    async def process_movie_scene(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single movie scene with emotional analysis"""
        
        try:
            # Extract features and emotion
            features, emotion, emotion_target = self.extract_emotional_features(scene_data)
            
            # Process through amygdala for emotional analysis
            amygdala_response = self.net._amygdala.process_emotional_salience(
                features, 
                event_data={
                    'title': scene_data.get('title', 'Unknown Scene'),
                    'character': scene_data.get('character', ''),
                    'categories': ['Movie', 'Emotional Scene']
                }
            )
            
            # Train emotional processors with the scene
            await self.net._amygdala.fear_conditioning(
                features,
                outcome='threatening' if emotion in ['fear', 'anger'] else 'positive' if emotion in ['joy', 'love'] else 'neutral',
                event_data=scene_data
            )
            
            # Update thalamic router if attention is enabled
            if self.enable_attention:
                # Route emotional content to appropriate specialists
                routing_decision = self.net._thalamic_router.analyze_conversation_intent(
                    scene_data.get('scene_text', ''), features
                )
                
                # Update routing based on emotional content
                await self.net._thalamic_router.adaptive_routing_update(
                    routing_decision,
                    {'user_satisfaction': 0.8, 'response_quality': 0.7},
                    features
                )
            
            self.scene_count += 1
            
            return {
                'scene_id': scene_data.get('id', self.scene_count),
                'emotion_detected': emotion,
                'emotion_target': emotion_target,
                'amygdala_response': amygdala_response,
                'features_shape': features.shape,
                'character': scene_data.get('character', ''),
                'success': True
            }
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to process scene: {e}")
            
            return {
                'scene_id': scene_data.get('id', 'unknown'),
                'error': str(e),
                'success': False
            }
    
    async def process_movie_scenes_dataset(self, file_path: str) -> Dict[str, Any]:
        """Process an entire movie scenes dataset"""
        
        if self.verbose:
            print(f"\n🎬 PROCESSING MOVIE SCENES DATASET")
            print("=" * 50)
            print(f"📁 File: {Path(file_path).name}")
        
        # Initialize counters
        total_scenes = 0
        successful_scenes = 0
        failed_scenes = 0
        scene_results = []
        
        # Reset emotion statistics
        self.emotion_stats = {emotion: 0 for emotion in self.emotion_stats}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Handle both JSONL and JSON formats
                if file_path.endswith('.jsonl'):
                    # JSONL format - one JSON object per line
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            scene_data = json.loads(line)
                            result = await self.process_movie_scene(scene_data)
                            scene_results.append(result)
                            
                            if result['success']:
                                successful_scenes += 1
                            else:
                                failed_scenes += 1
                            
                            total_scenes += 1
                            
                            # Progress reporting
                            if self.verbose and total_scenes % 100 == 0:
                                print(f"📊 Processed {total_scenes} scenes, {successful_scenes} successful")
                            
                        except json.JSONDecodeError as e:
                            if self.verbose:
                                print(f"⚠️  Invalid JSON on line {line_no}: {e}")
                            failed_scenes += 1
                
                else:
                    # JSON format - single object or array
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        # Array of scenes
                        for i, scene_data in enumerate(data):
                            result = await self.process_movie_scene(scene_data)
                            scene_results.append(result)
                            
                            if result['success']:
                                successful_scenes += 1
                            else:
                                failed_scenes += 1
                            
                            total_scenes += 1
                            
                            if self.verbose and total_scenes % 100 == 0:
                                print(f"📊 Processed {total_scenes} scenes")
                    
                    elif isinstance(data, dict):
                        # Single scene or scenes in a nested structure
                        scenes = data.get('scenes', [data])  # Handle both formats
                        
                        for scene_data in scenes:
                            result = await self.process_movie_scene(scene_data)
                            scene_results.append(result)
                            
                            if result['success']:
                                successful_scenes += 1
                            else:
                                failed_scenes += 1
                            
                            total_scenes += 1
        
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to process dataset: {e}")
            return {
                'error': str(e),
                'total_scenes_processed': 0
            }
        
        # Save weights after processing
        if ENHANCED_SAVING:
            try:
                weight_result = enhanced_save_weights(
                    self.net, 
                    description=f"Movie emotional training - {total_scenes} scenes processed"
                )
                if self.verbose:
                    print(f"💾 Weights saved: {weight_result.get('successful_saves', 0)} locations")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Weight saving failed: {e}")
        
        # Calculate statistics
        success_rate = successful_scenes / total_scenes if total_scenes > 0 else 0
        
        # Find most common emotion (this should fix your bug!)
        most_common_emotion = max(self.emotion_stats.items(), key=lambda x: x[1])
        
        if self.verbose:
            print(f"\n🎯 MOVIE EMOTIONAL TRAINING COMPLETE")
            print("=" * 50)
            print(f"📊 Total Scenes: {total_scenes}")
            print(f"✅ Successful: {successful_scenes}")
            print(f"❌ Failed: {failed_scenes}")
            print(f"📈 Success Rate: {success_rate:.1%}")
            print(f"\n🎭 EMOTION STATISTICS:")
            for emotion, count in sorted(self.emotion_stats.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    percentage = (count / total_scenes) * 100
                    print(f"   {emotion.capitalize()}: {count} scenes ({percentage:.1f}%)")
            
            print(f"\n🏆 Most Common Emotion: {most_common_emotion[0].capitalize()} ({most_common_emotion[1]} scenes)")
        
        return {
            'total_scenes_processed': total_scenes,
            'successful_scenes': successful_scenes,
            'failed_scenes': failed_scenes,
            'success_rate': success_rate,
            'emotion_statistics': self.emotion_stats,
            'most_common_emotion': most_common_emotion[0],
            'most_common_count': most_common_emotion[1],
            'scene_results': scene_results[:10] if self.verbose else [],  # Sample results
            'sbert_enabled': not self.offline,
            'attention_enabled': self.enable_attention,
            'device_used': self.device
        }
    
    def get_emotion_analysis_report(self) -> str:
        """Generate a detailed emotion analysis report"""
        total_scenes = sum(self.emotion_stats.values())
        
        if total_scenes == 0:
            return "No scenes processed yet."
        
        report = []
        report.append("🎭 EMOTIONAL ANALYSIS REPORT")
        report.append("=" * 40)
        
        # Sort emotions by frequency
        sorted_emotions = sorted(self.emotion_stats.items(), key=lambda x: x[1], reverse=True)
        
        for emotion, count in sorted_emotions:
            if count > 0:
                percentage = (count / total_scenes) * 100
                bar_length = int(percentage / 5)  # Scale bar to max 20 chars
                bar = "█" * bar_length + "░" * (20 - bar_length)
                report.append(f"{emotion.capitalize():>10}: {bar} {count:>4} ({percentage:>5.1f}%)")
        
        report.append(f"\nTotal Scenes Analyzed: {total_scenes}")
        
        # Emotional diversity index
        non_zero_emotions = len([count for count in self.emotion_stats.values() if count > 0])
        diversity = non_zero_emotions / len(self.emotion_stats)
        report.append(f"Emotional Diversity: {diversity:.2f} ({non_zero_emotions}/8 emotions)")
        
        return "\n".join(report)


# Example usage and testing
async def test_movie_trainer():
    """Test the movie trainer with sample data"""
    print("🧪 Testing Movie Emotional Trainer")
    
    # Mock network for testing
    class MockNetwork:
        def __init__(self):
            self._amygdala = MockAmygdala()
            self._thalamic_router = MockRouter()
    
    class MockAmygdala:
        async def fear_conditioning(self, *args, **kwargs):
            return {'success': True}
        
        def process_emotional_salience(self, *args, **kwargs):
            return {'emotional_valence': 0.5, 'emotional_intensity': 0.7}
    
    class MockRouter:
        def analyze_conversation_intent(self, *args, **kwargs):
            return {'primary_target': 'amygdala_specialist', 'routing_confidence': 0.8}
        
        async def adaptive_routing_update(self, *args, **kwargs):
            return {'success': True}
    
    # Test trainer
    mock_net = MockNetwork()
    trainer = AuraMovieEmotionalTrainer(
        net=mock_net,
        enable_attention=True,
        offline=False,  # Test with SBERT if available
        verbose=True
    )
    
    # Test scene processing
    test_scene = {
        'id': 'test_001',
        'scene_text': 'The character screamed in terror as the monster approached.',
        'character': 'John',
        'emotion': 'fear',
        'title': 'Horror Movie Scene'
    }
    
    result = await trainer.process_movie_scene(test_scene)
    print(f"✅ Test result: {result}")
    
    # Generate report
    report = trainer.get_emotion_analysis_report()
    print(f"\n📊 Report:\n{report}")


if __name__ == "__main__":
    # Run test if executed directly
    import asyncio
    import trio
    trio.run(test_movie_trainer)