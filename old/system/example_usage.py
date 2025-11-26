"""
Example usage of the AURA_GENESIS Bootloader
Demonstrates all the integrated system capabilities
"""

import asyncio
import trio
from .bootloader import AuraBootConfig, boot_aura_genesis


async def demonstrate_aura_system():
    """Demonstrate the comprehensive AURA system capabilities"""
    
    print("🚀 AURA_GENESIS System Demonstration")
    print("=" * 50)
    
    # Create configuration
    config = AuraBootConfig(
        system_name="AURA_DEMO",
        version="2.0.0",
        enable_span=True,
        enable_svc_analysis=True,
        linguistic_features_enabled=True,
        offline_mode=True,
        log_level="INFO",
        weights_dir="weights",  # Use the correct weights directory
        models_dir="models"     # Use the correct models directory
    )
    
    try:
        # Boot the system
        print("🔧 Booting AURA_GENESIS system...")
        bootloader = await boot_aura_genesis(config)
        
        print("✅ System booted successfully!")
        print(f"📊 System Status: {bootloader.get_system_status()}")
        
        # Demonstrate query processing
        print("\n💬 Testing query processing...")
        test_queries = [
            "What is artificial intelligence?",
            "Explain neural networks in simple terms",
            "How does machine learning work?"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Query: {query}")
            result = await bootloader.process_query(query)
            print(f"📝 Response: {result}")
        
        # Demonstrate SVC analysis
        print("\n🔬 Testing SVC Analysis...")
        test_texts = [
            "The cat sat on the mat",
            "Scientists discovered new particles",
            "Students learn mathematics effectively"
        ]
        
        for text in test_texts:
            print(f"\n📝 Text: {text}")
            svc_result = await bootloader.analyze_svc_structure(text)
            print(f"🔍 SVC Analysis: {svc_result}")
        
        # Demonstrate SVC insights
        print("\n📊 Getting SVC insights...")
        insights = await bootloader.get_svc_insights()
        print(f"📈 Insights: {insights}")
        
        # Demonstrate pre-trained model analysis
        print("\n🤖 Testing Pre-trained Models...")
        analysis_texts = [
            "I'm so excited about this new project!",
            "Can you help me understand this concept?",
            "This is absolutely terrible and I hate it.",
            "Thank you for your assistance today."
        ]
        
        for text in analysis_texts:
            print(f"\n📝 Text: {text}")
            
            # Emotion analysis
            emotion = await bootloader.classify_emotion(text)
            print(f"😊 Emotion: {emotion}")
            
            # Intent analysis
            intent = await bootloader.classify_intent(text)
            print(f"🎯 Intent: {intent}")
            
            # Tone analysis
            tone = await bootloader.classify_tone(text)
            print(f"🎭 Tone: {tone}")
            
            # Comprehensive analysis
            comprehensive = await bootloader.comprehensive_analysis(text)
            print(f"🔍 Comprehensive: {comprehensive}")
        
        # Demonstrate training
        print("\n🎓 Testing training capabilities...")
        sample_training_data = [
            {
                'text': 'The dog runs quickly',
                'domain': 'general',
                'realm': 'personal',
                'difficulty': 0.3
            },
            {
                'text': 'Quantum mechanics describes particle behavior',
                'domain': 'technical',
                'realm': 'academic',
                'difficulty': 0.9
            }
        ]
        
        training_result = await bootloader.train_svc_analyzer(sample_training_data)
        print(f"🎯 Training Result: {training_result}")
        
        # Show final system status
        print("\n📊 Final System Status:")
        final_status = bootloader.get_system_status()
        print(f"Health: {final_status['health']['status']}")
        print(f"Uptime: {final_status['health']['uptime']:.2f}s")
        print(f"Memory Usage: {final_status['health']['memory_usage_mb']:.2f} MB")
        print(f"Queries Processed: {final_status['metrics']['total_queries']}")
        
        # Graceful shutdown
        print("\n🛑 Shutting down system...")
        await bootloader.shutdown_system()
        print("✅ Shutdown complete!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point for the demonstration"""
    print("🧠 AURA_GENESIS Comprehensive System Demo")
    print("This demonstrates all integrated capabilities:")
    print("✓ Neural Network with SPAN integration")
    print("✓ Pre-trained Model Loading (Emotion, Intent, Tone)")
    print("✓ Weight Loading from /weights directory")
    print("✓ SVC (Subject-Verb-Complement) Analysis")
    print("✓ Chat Orchestration")
    print("✓ Enhanced Training System")
    print("✓ System Health Monitoring")
    print("✓ Performance Metrics Collection")
    print("✓ Comprehensive Text Analysis")
    print()
    
    # Run the demonstration
    trio.run(demonstrate_aura_system)


if __name__ == "__main__":
    main()
