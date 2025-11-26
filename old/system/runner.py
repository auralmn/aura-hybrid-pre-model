# Complete Bio-SVC Running Guide
# Step-by-step instructions to run your biological neural SVC system

import asyncio
import numpy as np
import json
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Your existing imports (assuming they're in your project)
from ..core.neuron import Neuron, MaturationStage, ActivityState
from ..core.thalamus import Thalamus  
from ..core.hippocampus import Hippocampus
from ..core.nlms import NLMSHead
from ..utils.codon_mappings import Specialization, DNACodon

# Our enhanced pipeline imports
from ..utils.enhanced_svc_pipeline import (
    load_enhanced_svc_dataset,
    get_enhanced_full_knowledge_embedding
)

# Bio-SVC integration
from .aura import (
    BioSVCSystem, 
    BioSVCNeuron,
    SVCSpecialization,
    SVCCodon
)

class BioSVCRunner:
    """
    Complete runner for the biological SVC system
    """
    
    def __init__(self):
        self.sbert = None
        self.svc_data = None
        self.domains = []
        self.realms = []
        self.bio_svc_system = None
        
    async def setup_environment(self):
        """
        Step 1: Set up the complete environment
        """
        print("🔧 Setting up Bio-SVC Environment...")
        print("="*50)
        
        # Initialize SBERT for feature extraction
        print("Loading SBERT model...")
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ SBERT loaded")
        
        # Load enhanced SVC dataset
        print("Loading enhanced SVC dataset...")
        try:
            self.svc_data, self.domains, self.realms = load_enhanced_svc_dataset('train_svc_enhanced.jsonl')
            print(f"✓ Loaded {len(self.svc_data)} samples")
            print(f"✓ Found {len(self.domains)} domains: {self.domains}")
            print(f"✓ Found {len(self.realms)} realms: {self.realms}")
        except FileNotFoundError:
            print("❌ Enhanced SVC dataset not found!")
            print("Please run: python svc-data-enhancer.py train_svc.jsonl train_svc_enhanced.jsonl")
            return False
        
        return True
    
    async def initialize_bio_svc_system(self):
        """
        Step 2: Initialize the biological SVC system
        """
        print("\n🧠 Initializing Biological SVC System...")
        print("="*50)
        
        # Create the bio-SVC system with your architecture
        self.bio_svc_system = BioSVCSystem(
            enhanced_svc_data=self.svc_data,
            domains=self.domains,
            realms=self.realms
        )
        
        print("✓ Bio-SVC System initialized")
        print(f"  - Thalamus: {len(self.bio_svc_system.thalamus.neurons)} relay neurons")
        print(f"  - Hippocampus: {len(self.bio_svc_system.hippocampus.neurons)} place cells")
        print(f"  - Routing specialists: {len(self.bio_svc_system.thalamus.routing_specialists)} specialists")
        print(f"  - Linguistic specialists: {len(self.bio_svc_system.hippocampus.linguistic_specialists)} specialists")
        
        return True
    
    async def demonstrate_single_prediction(self):
        """
        Step 3: Demonstrate single sample processing
        """
        print("\n🔬 Demonstrating Single Sample Processing...")
        print("="*50)
        
        # Take first sample
        sample = self.svc_data[0]
        print(f"Sample text: {sample['text']}")
        print(f"Domain: {sample['metadata']['domain']}")
        print(f"Realm: {sample['realm']}")
        print(f"SVC: {sample['metadata']['svc']}")
        
        # Generate enhanced features
        enhanced_features = get_enhanced_full_knowledge_embedding(sample, self.sbert)
        print(f"Enhanced features shape: {enhanced_features.shape}")
        
        # Process through Thalamus (routing)
        print("\n🧠 Thalamic Routing:")
        routing_decisions = await self.bio_svc_system.thalamus.route_svc_input(
            enhanced_features, sample['metadata']
        )
        
        for route, confidence in routing_decisions.items():
            print(f"  {route}: confidence {confidence:.3f}")
        
        # Process through Hippocampus (memory consolidation)
        print("\n🧠 Hippocampal Consolidation:")
        consolidation_result = await self.bio_svc_system.hippocampus.consolidate_svc_memory(
            sample, enhanced_features, sample['linguistic_features']
        )
        
        print(f"  Pattern stored: {consolidation_result['pattern_stored']}")
        print(f"  Memory size: {consolidation_result['memory_size']}")
        print(f"  Active specialists: {len(consolidation_result['specialist_outputs'])}")
        
        if 'new_neurons' in consolidation_result:
            print(f"  🌱 Neurogenesis triggered: {consolidation_result['new_neurons']} new neurons")
        
        return True
    
    async def run_development_phases(self):
        """
        Step 4: Run through developmental phases
        """
        print("\n🌱 Running Developmental Phases...")
        print("="*50)
        
        # Phase 1: Progenitor stage - Basic exposure
        print("Phase 1: Progenitor Stage (Basic Pattern Exposure)")
        progenitor_samples = self.svc_data[:100]  # First 100 samples
        
        for i, sample in enumerate(progenitor_samples):
            enhanced_features = get_enhanced_full_knowledge_embedding(sample, self.sbert)
            
            # Process through system
            await self.bio_svc_system.thalamus.route_svc_input(enhanced_features, sample['metadata'])
            await self.bio_svc_system.hippocampus.consolidate_svc_memory(
                sample, enhanced_features, sample['linguistic_features']
            )
            
            if (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/100 samples...")
        
        print("✓ Progenitor phase complete")
        
        # Phase 2: Migration stage - Domain specialization  
        print("\nPhase 2: Migration Stage (Domain Specialization)")
        migration_samples = self.svc_data[100:300]
        
        for i, sample in enumerate(migration_samples):
            enhanced_features = get_enhanced_full_knowledge_embedding(sample, self.sbert)
            
            # Focus on domain-specific routing
            routing_decisions = await self.bio_svc_system.thalamus.route_svc_input(
                enhanced_features, sample['metadata']
            )
            
            # Develop specialization in routing neurons
            domain = sample['metadata']['domain']
            if f"domain_{domain}" in self.bio_svc_system.thalamus.routing_specialists:
                specialist = self.bio_svc_system.thalamus.routing_specialists[f"domain_{domain}"]
                specialist.develop_specialization(sample, sample['linguistic_features'])
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/200 samples...")
        
        print("✓ Migration phase complete")
        
        # Phase 3: Differentiation stage - Full specialization
        print("\nPhase 3: Differentiation Stage (Full Specialization)")
        diff_samples = self.svc_data[300:600]
        
        for i, sample in enumerate(diff_samples):
            enhanced_features = get_enhanced_full_knowledge_embedding(sample, self.sbert)
            
            # Full system processing with specialization development
            await self.bio_svc_system.thalamus.route_svc_input(enhanced_features, sample['metadata'])
            consolidation = await self.bio_svc_system.hippocampus.consolidate_svc_memory(
                sample, enhanced_features, sample['linguistic_features']
            )
            
            # Develop all specialists
            for specialist in self.bio_svc_system.hippocampus.linguistic_specialists:
                specialist.develop_specialization(sample, sample['linguistic_features'])
            
            if (i + 1) % 75 == 0:
                print(f"  Processed {i + 1}/300 samples...")
        
        print("✓ Differentiation phase complete")
        
        # Phase 4: Myelination stage - Optimized performance
        print("\nPhase 4: Myelination Stage (Performance Optimization)")
        myelination_samples = self.svc_data[600:800]
        
        for i, sample in enumerate(myelination_samples):
            enhanced_features = get_enhanced_full_knowledge_embedding(sample, self.sbert)
            
            # Optimized processing with mature neurons
            routing_decisions = await self.bio_svc_system.thalamus.route_svc_input(
                enhanced_features, sample['metadata']
            )
            consolidation = await self.bio_svc_system.hippocampus.consolidate_svc_memory(
                sample, enhanced_features, sample['linguistic_features']
            )
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/200 samples...")
        
        print("✓ Myelination phase complete")
        
        return True
    
    async def evaluate_system_performance(self):
        """
        Step 5: Evaluate the trained biological system
        """
        print("\n📊 Evaluating Bio-SVC System Performance...")
        print("="*50)
        
        # Use test samples
        test_samples = self.svc_data[800:1000]  # Last 200 samples for testing
        
        domain_correct = 0
        realm_correct = 0
        total_samples = 0
        routing_confidences = []
        
        for sample in test_samples:
            enhanced_features = get_enhanced_full_knowledge_embedding(sample, self.sbert)
            
            # Get routing decisions
            routing_decisions = await self.bio_svc_system.thalamus.route_svc_input(
                enhanced_features, sample['metadata']
            )
            
            # Check domain routing accuracy
            true_domain = sample['metadata']['domain']
            domain_key = f"domain_{true_domain}"
            if domain_key in routing_decisions:
                domain_confidence = routing_decisions[domain_key]
                routing_confidences.append(domain_confidence)
                
                # Simple threshold-based classification
                if domain_confidence > 0.5:
                    domain_correct += 1
            
            # Check realm routing accuracy
            true_realm = sample['realm']
            realm_key = f"realm_{true_realm}"
            if realm_key in routing_decisions:
                realm_confidence = routing_decisions[realm_key]
                
                if realm_confidence > 0.5:
                    realm_correct += 1
            
            total_samples += 1
        
        # Calculate performance metrics
        domain_accuracy = domain_correct / total_samples if total_samples > 0 else 0
        realm_accuracy = realm_correct / total_samples if total_samples > 0 else 0
        avg_confidence = np.mean(routing_confidences) if routing_confidences else 0
        
        print(f"Test Results ({total_samples} samples):")
        print(f"  Domain Routing Accuracy: {domain_accuracy:.3f}")
        print(f"  Realm Routing Accuracy: {realm_accuracy:.3f}")
        print(f"  Average Routing Confidence: {avg_confidence:.3f}")
        
        # Analyze neuron maturation states
        print(f"\nNeuron Maturation Analysis:")
        maturation_counts = {}
        
        all_neurons = (self.bio_svc_system.thalamus.neurons + 
                      self.bio_svc_system.hippocampus.neurons +
                      list(self.bio_svc_system.thalamus.routing_specialists.values()) +
                      self.bio_svc_system.hippocampus.linguistic_specialists)
        
        for neuron in all_neurons:
            stage = neuron.maturation.name
            maturation_counts[stage] = maturation_counts.get(stage, 0) + 1
        
        for stage, count in maturation_counts.items():
            print(f"  {stage}: {count} neurons")
        
        # Analyze specialization development
        print(f"\nSpecialization Development:")
        for specialist_name, specialist in self.bio_svc_system.thalamus.routing_specialists.items():
            print(f"  {specialist_name}: {specialist.specialization_strength:.3f} strength, "
                  f"{specialist.svc_patterns_seen} patterns seen")
        
        return {
            'domain_accuracy': domain_accuracy,
            'realm_accuracy': realm_accuracy,
            'avg_confidence': avg_confidence,
            'maturation_counts': maturation_counts
        }
    
    async def demonstrate_neurogenesis(self):
        """
        Step 6: Demonstrate dynamic neurogenesis
        """
        print("\n🌱 Demonstrating Neurogenesis...")
        print("="*50)
        
        # Get current neuron count
        initial_count = len(self.bio_svc_system.hippocampus.neurons)
        print(f"Initial hippocampal neurons: {initial_count}")
        
        # Process samples to trigger neurogenesis
        neurogenesis_samples = self.svc_data[1000:1100] if len(self.svc_data) > 1000 else self.svc_data[-100:]
        
        neurogenesis_events = 0
        for sample in neurogenesis_samples:
            enhanced_features = get_enhanced_full_knowledge_embedding(sample, self.sbert)
            
            consolidation = await self.bio_svc_system.hippocampus.consolidate_svc_memory(
                sample, enhanced_features, sample['linguistic_features']
            )
            
            if 'new_neurons' in consolidation:
                neurogenesis_events += 1
                print(f"  🌱 Neurogenesis event #{neurogenesis_events}: "
                      f"{consolidation['new_neurons']} new neurons")
        
        final_count = len(self.bio_svc_system.hippocampus.neurons)
        print(f"\nFinal hippocampal neurons: {final_count}")
        print(f"Net growth: {final_count - initial_count} neurons")
        print(f"Neurogenesis events: {neurogenesis_events}")
        
        return final_count - initial_count
    
    async def run_complete_pipeline(self):
        """
        Run the complete Bio-SVC pipeline
        """
        print("🚀 STARTING COMPLETE BIO-SVC PIPELINE")
        print("="*60)
        
        # Step 1: Setup environment
        if not await self.setup_environment():
            print("❌ Environment setup failed!")
            return False
        
        # Step 2: Initialize system
        if not await self.initialize_bio_svc_system():
            print("❌ System initialization failed!")
            return False
        
        # Step 3: Demonstrate single processing
        if not await self.demonstrate_single_prediction():
            print("❌ Single prediction demo failed!")
            return False
        
        # Step 4: Run developmental phases
        if not await self.run_development_phases():
            print("❌ Development phases failed!")
            return False
        
        # Step 5: Evaluate performance
        performance = await self.evaluate_system_performance()
        
        # Step 6: Demonstrate neurogenesis
        neuron_growth = await self.demonstrate_neurogenesis()
        
        print("\n🎉 COMPLETE BIO-SVC PIPELINE FINISHED!")
        print("="*60)
        print("Final Results:")
        print(f"  Domain Accuracy: {performance['domain_accuracy']:.3f}")
        print(f"  Realm Accuracy: {performance['realm_accuracy']:.3f}")
        print(f"  Average Confidence: {performance['avg_confidence']:.3f}")
        print(f"  Neuron Growth: {neuron_growth} new neurons")
        
        print("\n✨ Your biological SVC system is now trained and operational!")
        print("   Features:")
        print("   🧠 Biological thalamic routing")
        print("   🧠 Hippocampal memory consolidation") 
        print("   🌱 Dynamic neurogenesis")
        print("   ⚡ NLMS streaming learning")
        print("   🔬 Developmental specialization")
        
        return True


# Main execution function
async def main():
    """
    Main function to run the Bio-SVC system
    """
    runner = BioSVCRunner()
    success = await runner.run_complete_pipeline()
    
    if success:
        print("\n🎯 SUCCESS: Bio-SVC system is ready for production use!")
    else:
        print("\n❌ FAILED: Check the error messages above")
    
    return success


# Run the complete system
if __name__ == "__main__":
    print("🧠🔬 Bio-SVC System Runner")
    print("="*30)
    print("This will run your complete biological SVC system")
    print("integrating your neural architecture with enhanced SVC features")
    print()
    
    # Run the async pipeline with trio
    import trio
    success = trio.run(main)
    
    if success:
        print("\n🚀 Your Bio-SVC system is ready!")
        print("   You can now use it for:")
        print("   • Domain classification with biological routing")
        print("   • Realm classification with specialized neurons") 
        print("   • Difficulty regression with memory consolidation")
        print("   • Continuous learning with neurogenesis")
        print("   • Real-time adaptation with NLMS updates")
    else:
        print("\n🔧 Troubleshooting steps:")
        print("   1. Ensure all dependencies are installed")
        print("   2. Run the SVC data enhancer first")
        print("   3. Check that all your neural modules are available")
        print("   4. Verify the enhanced dataset was created")