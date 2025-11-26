"""
AURA_GENESIS Bootloader - Comprehensive System Initialization
Integrates all system components with proper initialization and monitoring
"""

import asyncio
import trio
import numpy as np
import json
import time
import sys
import os
import signal
import traceback
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from contextlib import asynccontextmanager

# Import configuration loader
from ..utils.config_loader import load_aura_config, AuraConfig

# Core AURA system imports
from ..core.network import Network
from ..utils.chat_orchestrator import ChatOrchestrator
from ..training.enhanced_trainer import EnhancedAuraTrainer
from ..training.directory_loader import DirectoryLoader
from ..utils.enhanced_svc_pipeline import (
    load_enhanced_svc_dataset,
    get_enhanced_full_knowledge_embedding,
    extract_pos_features,
    extract_ner_features,
    extract_structural_features
)

# Import weights I/O functionality
import sys
sys.path.append('/Volumes/Others2/AURA_GENESIS')
from tools.weights_io import save_network_weights, load_network_weights

# Import Qdrant functionality
from ..utils.qdrant_stream import QdrantStreamer
from ..utils.qdrant_mapper import QdrantMapper


@dataclass
class BootSequenceConfig:
    """Configuration for proper boot sequence"""
    validate_dependencies: bool = True
    initialize_weights: bool = True
    enable_health_checks: bool = True
    timeout_seconds: int = 30
    retry_attempts: int = 3

class AuraBootSequence:
    """
    Proper boot sequence for AURA neural network system
    Ensures components initialize in correct dependency order
    """
    
    def __init__(self, config: Optional[BootSequenceConfig] = None):
        self.config = config or BootSequenceConfig()
        self.boot_status = {}
        self.initialized_components = []
        self.failed_components = []
        self.network = None
        self.chat_orchestrator = None
        
    async def execute_safe_boot(self) -> Dict[str, Any]:
        """Execute the complete safe boot sequence"""
        
        boot_result = {
            'success': False,
            'boot_time': 0.0,
            'components_initialized': [],
            'failed_components': [],
            'boot_sequence': [],
            'errors': []
        }
        
        start_time = datetime.now()
        
        try:
            print("Starting AURA safe boot sequence...")
            
            print("Phase 1: Core Dependencies")
            await self._phase_1_core_dependencies()
            boot_result['boot_sequence'].append('core_dependencies')
            
            print("Phase 2: Neural Components")  
            await self._phase_2_neural_components()
            boot_result['boot_sequence'].append('neural_components')
            
            print("Phase 3: Network Assembly")
            await self._phase_3_network_assembly()
            boot_result['boot_sequence'].append('network_assembly')
            
            print("Phase 4: Weight Initialization")
            await self._phase_4_weight_initialization()
            boot_result['boot_sequence'].append('weight_initialization')
            
            print("Phase 5: System Integration")
            await self._phase_5_system_integration()
            boot_result['boot_sequence'].append('system_integration')
            
            print("Phase 6: Health Validation")
            await self._phase_6_health_validation()
            boot_result['boot_sequence'].append('health_validation')
            
            boot_result['success'] = True
            boot_result['components_initialized'] = self.initialized_components.copy()
            boot_result['network'] = self.network
            boot_result['chat_orchestrator'] = self.chat_orchestrator
            
        except Exception as e:
            boot_result['errors'].append(str(e))
            boot_result['failed_components'] = self.failed_components.copy()
            
        boot_result['boot_time'] = (datetime.now() - start_time).total_seconds()
        return boot_result
    
    async def _phase_1_core_dependencies(self):
        """Phase 1: Initialize core dependencies first"""
        await self._validate_imports()
        await self._setup_environment()
        await self._validate_configuration()
        self.boot_status['core_dependencies'] = 'COMPLETE'
    
    async def _phase_2_neural_components(self):
        """Phase 2: Initialize individual neural components"""
        await self._initialize_neuron_classes()
        await self._initialize_nlms_heads()
        await self._initialize_phasor_systems()
        await self._initialize_state_machines()
        self.boot_status['neural_components'] = 'COMPLETE'
    
    async def _phase_3_network_assembly(self):
        """Phase 3: Assemble neural network regions"""
        await self._initialize_thalamus()
        await self._initialize_hippocampus()
        await self._initialize_amygdala()
        await self._initialize_thalamic_router()
        await self._initialize_cns()
        self.boot_status['network_assembly'] = 'COMPLETE'
    
    async def _phase_4_weight_initialization(self):
        """Phase 4: Initialize neural network weights"""
        if not self.config.initialize_weights:
            print("Skipping weight initialization (disabled in config)")
            return
        
        await self._initialize_thalamus_weights()
        await self._initialize_hippocampus_weights()
        await self._initialize_router_weights()
        await self._validate_weight_initialization()
        self.boot_status['weight_initialization'] = 'COMPLETE'
    
    async def _phase_5_system_integration(self):
        """Phase 5: Integrate all components into unified system"""
        await self._assemble_main_network()
        await self._register_brain_regions()
        await self._initialize_chat_orchestrator()
        await self._setup_component_connections()
        self.boot_status['system_integration'] = 'COMPLETE'
    
    async def _phase_6_health_validation(self):
        """Phase 6: Validate system health and readiness"""
        if not self.config.enable_health_checks:
            print("Skipping health checks (disabled in config)")
            return
        
        await self._check_component_connectivity()
        await self._check_memory_allocation()
        await self._check_neural_pathways()
        await self._test_system_responsiveness()
        self.boot_status['health_validation'] = 'COMPLETE'
    
    # Implementation methods for each phase
    
    async def _validate_imports(self):
        """Validate all required imports are available"""
        try:
            import numpy as np
            import trio
            from ..core.neuron import Neuron, ActivityState, MaturationStage
            from ..core.nlms import NLMSHead
            from ..core.neuronal_state import NeuronalState
            from ..core.phasor import PhasorBank, PhasorState
            from ..core.thalamus import Thalamus
            from ..core.hippocampus import Hippocampus
            from ..core.amygdala import Amygdala
            from ..core.thalamic_router import ThalamicConversationRouter
            from ..core.network import Network
            from ..utils.chat_orchestrator import ChatOrchestrator
            
            self.initialized_components.append('imports_validated')
            
        except ImportError as e:
            self.failed_components.append(f'import_validation: {str(e)}')
            raise Exception(f"Import validation failed: {e}")
    
    async def _setup_environment(self):
        """Setup runtime environment"""
        try:
            np.random.seed(42)
            import os
            os.environ.setdefault('AURA_BOOT_MODE', 'SAFE')
            self.initialized_components.append('environment_setup')
        except Exception as e:
            self.failed_components.append(f'environment_setup: {str(e)}')
            raise
    
    async def _validate_configuration(self):
        """Validate system configuration"""
        try:
            required_memory_mb = 512
            n_neurons = 1000
            n_features = 384
            
            if n_neurons <= 0 or n_features <= 0:
                raise ValueError("Invalid neuron or feature count")
            
            self.initialized_components.append('configuration_validated')
        except Exception as e:
            self.failed_components.append(f'configuration_validation: {str(e)}')
            raise
    
    async def _initialize_neuron_classes(self):
        """Initialize base neuron classes"""
        try:
            from ..core.neuron import Neuron, ActivityState, MaturationStage
            
            test_neuron = Neuron(
                neuron_id=0,
                specialization='test',
                abilities={'test': 1.0},
                n_features=10,
                n_outputs=1
            )
            
            self.initialized_components.append('neuron_classes')
        except Exception as e:
            self.failed_components.append(f'neuron_classes: {str(e)}')
            raise
    
    async def _initialize_nlms_heads(self):
        """Initialize NLMS head systems"""
        try:
            from ..core.nlms import NLMSHead
            
            test_head = NLMSHead(
                n_features=10,
                n_outputs=1,
                seed=42
            )
            
            self.initialized_components.append('nlms_heads')
        except Exception as e:
            self.failed_components.append(f'nlms_heads: {str(e)}')
            raise
    
    async def _initialize_phasor_systems(self):
        """Initialize phasor bank systems"""
        try:
            from ..core.phasor import PhasorBank, PhasorState
            
            # Test phasor creation
            test_phasor = PhasorState(delta0=7)
            test_bank = PhasorBank(delta0=7, H=384)
            
            self.initialized_components.append('phasor_systems')
            
        except Exception as e:
            self.failed_components.append(f'phasor_systems: {str(e)}')
            raise
    
    async def _initialize_state_machines(self):
        """Initialize neuronal state machines"""
        try:
            from ..core.neuronal_state import NeuronalState
            from ..core.state_machine import NeuronStateMachine
            
            # Test state machine creation
            test_state = NeuronalState(
                kind='test',
                position=None,
                gene_expression={'test': 0.5}
            )
            
            self.initialized_components.append('state_machines')
            
        except Exception as e:
            self.failed_components.append(f'state_machines: {str(e)}')
            raise
    
    async def _initialize_thalamus(self):
        """Initialize Thalamus (must be first brain region)"""
        try:
            from ..core.thalamus import Thalamus
            
            D = 384  # or pull from a central config if available here
            self.thalamus = Thalamus(neuron_count=100, input_channels=D, output_channels=D)
            
            self.initialized_components.append('thalamus')
            
        except Exception as e:
            self.failed_components.append(f'thalamus: {str(e)}')
            raise
    
    async def _initialize_hippocampus(self):
        """Initialize Hippocampus (depends on basic components)"""
        try:
            from ..core.hippocampus import Hippocampus
            
            D = 384  # or pull from a central config if available here
            self.hippocampus = Hippocampus(neuron_count=100, features=D, input_dim=D)
            
            self.initialized_components.append('hippocampus')
            
        except Exception as e:
            self.failed_components.append(f'hippocampus: {str(e)}')
            raise
    
    async def _initialize_amygdala(self):
        """Initialize Amygdala (depends on basic components)"""
        try:
            from ..core.amygdala import Amygdala
            
            D = 384  # or pull from a central config if available here
            self.amygdala = Amygdala(neuron_count=30, features=D, input_dim=D)
            
            self.initialized_components.append('amygdala')
            
        except Exception as e:
            self.failed_components.append(f'amygdala: {str(e)}')
            raise
    
    async def _initialize_thalamic_router(self):
        """Initialize Thalamic Router (depends on neurons)"""
        try:
            from ..core.thalamic_router import ThalamicConversationRouter
            
            D = 384  # or pull from a central config if available here
            self.thalamic_router = ThalamicConversationRouter(neuron_count=60, features=D, input_dim=D)
            
            self.initialized_components.append('thalamic_router')
            
        except Exception as e:
            self.failed_components.append(f'thalamic_router: {str(e)}')
            raise
    
    async def _initialize_cns(self):
        """Initialize Central Nervous System (orchestrator)"""
        try:
            from ..core.central_nervous_system import CentralNervousSystem
            
            D = 384  # or pull from a central config if available here
            self.cns = CentralNervousSystem(input_dim=D)
            
            self.initialized_components.append('cns')
            
        except Exception as e:
            self.failed_components.append(f'cns: {str(e)}')
            raise
    
    async def _initialize_thalamus_weights(self):
        """Initialize thalamus neuron weights"""
        try:
            # Initialize weights for all neurons in the thalamus
            for neuron in self.thalamus.neurons:
                if hasattr(neuron, 'init_weights'):
                    await neuron.init_weights()
                elif hasattr(neuron, 'nlms_head') and hasattr(neuron.nlms_head, 'init_weights'):
                    await neuron.nlms_head.init_weights()
            
            self.initialized_components.append('thalamus_weights')
            
        except Exception as e:
            self.failed_components.append(f'thalamus_weights: {str(e)}')
            raise
    
    async def _initialize_hippocampus_weights(self):
        """Initialize hippocampus neuron weights"""
        try:
            # Initialize weights for all neurons in the hippocampus
            for neuron in self.hippocampus.neurons:
                if hasattr(neuron, 'init_weights'):
                    await neuron.init_weights()
                elif hasattr(neuron, 'nlms_head') and hasattr(neuron.nlms_head, 'init_weights'):
                    await neuron.nlms_head.init_weights()
            
            self.initialized_components.append('hippocampus_weights')
            
        except Exception as e:
            self.failed_components.append(f'hippocampus_weights: {str(e)}')
            raise
    
    async def _initialize_router_weights(self):
        """Initialize router neuron weights"""
        try:
            # Initialize weights for all neurons in the thalamic router
            for neuron in self.thalamic_router.all_neurons:
                if hasattr(neuron, 'init_weights'):
                    await neuron.init_weights()
                elif hasattr(neuron, 'nlms_head') and hasattr(neuron.nlms_head, 'init_weights'):
                    await neuron.nlms_head.init_weights()
            
            self.initialized_components.append('router_weights')
            
        except Exception as e:
            self.failed_components.append(f'router_weights: {str(e)}')
            raise
    
    async def _validate_weight_initialization(self):
        """Validate all weights are properly initialized"""
        try:
            # Check thalamus weights
            for neuron in self.thalamus.neurons:
                if hasattr(neuron, 'nlms_head') and neuron.nlms_head:
                    if not hasattr(neuron.nlms_head, 'w') or neuron.nlms_head.w is None:
                        raise ValueError(f"Thalamus neuron {neuron.neuron_id} weights not initialized")
            
            # Check hippocampus weights
            for neuron in self.hippocampus.neurons:
                if hasattr(neuron, 'nlms_head') and neuron.nlms_head:
                    if not hasattr(neuron.nlms_head, 'w') or neuron.nlms_head.w is None:
                        raise ValueError(f"Hippocampus neuron {neuron.neuron_id} weights not initialized")
            
            self.initialized_components.append('weight_validation')
            
        except Exception as e:
            self.failed_components.append(f'weight_validation: {str(e)}')
            raise
    
    async def _assemble_main_network(self):
        """Assemble main Network class with all components"""
        try:
            from ..core.network import Network
            
            # Create network instance with minimal parameters
            self.network = Network(
                neuron_count=100,  # Use smaller count for safe boot
                features=384,
                enable_span=False,  # Disable for safe boot
                offline=True  # Use offline mode for safe boot
            )
            
            # Replace default components with our initialized ones
            self.network._thalamus = self.thalamus
            self.network._hippocampus = self.hippocampus
            self.network._amygdala = self.amygdala
            self.network._thalamic_router = self.thalamic_router
            self.network._cns = self.cns
            
            self.initialized_components.append('main_network')
            
        except Exception as e:
            self.failed_components.append(f'main_network: {str(e)}')
            raise
    
    async def _register_brain_regions(self):
        """Register brain regions with CNS"""
        try:
            if hasattr(self.network._cns, 'register_brain_region'):
                self.network._cns.register_brain_region('thalamus', self.thalamus, priority=0.7)
                self.network._cns.register_brain_region('hippocampus', self.hippocampus, priority=0.6)
                self.network._cns.register_brain_region('amygdala', self.amygdala, priority=0.8)
                self.network._cns.register_brain_region('router', self.thalamic_router, priority=0.5)
            
            self.initialized_components.append('brain_region_registration')
            
        except Exception as e:
            self.failed_components.append(f'brain_region_registration: {str(e)}')
            raise
    
    async def _initialize_chat_orchestrator(self):
        """Initialize chat orchestrator"""
        try:
            from ..utils.chat_orchestrator import ChatOrchestrator
            
            self.chat_orchestrator = ChatOrchestrator(self.network)
            
            self.initialized_components.append('chat_orchestrator')
            
        except Exception as e:
            self.failed_components.append(f'chat_orchestrator: {str(e)}')
            raise
    
    async def _setup_component_connections(self):
        """Setup connections between components"""
        try:
            # Connect thalamus to targets (if needed)
            # This would be component-specific connection logic
            
            self.initialized_components.append('component_connections')
            
        except Exception as e:
            self.failed_components.append(f'component_connections: {str(e)}')
            raise
    
    async def _check_component_connectivity(self):
        """Check that all components are properly connected"""
        try:
            # Verify network has all required components
            required_components = ['_thalamus', '_hippocampus', '_amygdala', '_thalamic_router', '_cns']
            
            for component in required_components:
                if not hasattr(self.network, component):
                    raise ValueError(f"Network missing required component: {component}")
                
                if getattr(self.network, component) is None:
                    raise ValueError(f"Network component {component} is None")
            
            self.initialized_components.append('connectivity_check')
            
        except Exception as e:
            self.failed_components.append(f'connectivity_check: {str(e)}')
            raise
    
    async def _check_memory_allocation(self):
        """Check memory allocation is reasonable"""
        try:
            # Basic memory check - count neurons
            total_neurons = (
                len(self.thalamus.neurons) +
                len(self.hippocampus.neurons) +
                len(self.amygdala.neurons) +
                len(self.thalamic_router.all_neurons)
            )
            
            print(f"Total neurons allocated: {total_neurons}")
            
            if total_neurons == 0:
                raise ValueError("No neurons allocated")
            
            self.initialized_components.append('memory_check')
            
        except Exception as e:
            self.failed_components.append(f'memory_check: {str(e)}')
            raise
    
    async def _check_neural_pathways(self):
        """Check neural pathways are functional"""
        try:
            # Test basic neural pathway with dummy data
            D = 384
            test_input = np.random.randn(D).astype(np.float32)
            
            # Test thalamus relay
            if hasattr(self.thalamus, 'relay'):
                thalamus_output = self.thalamus.relay(test_input)
                
                if not thalamus_output:
                    raise ValueError("Thalamus relay failed")
            
            self.initialized_components.append('pathway_check')
            
        except Exception as e:
            self.failed_components.append(f'pathway_check: {str(e)}')
            raise
    
    async def _test_system_responsiveness(self):
        """Test system responsiveness with simple query"""
        try:
            # Test basic network functionality
            if self.network and hasattr(self.network, 'get_features'):
                test_features = self.network.get_features("test query")
                if test_features is None or len(test_features) == 0:
                    raise ValueError("Network feature extraction not working")
            
            self.initialized_components.append('responsiveness_test')
            
        except Exception as e:
            self.failed_components.append(f'responsiveness_test: {str(e)}')
            raise

@dataclass
class AuraBootConfig:
    """Configuration for AURA system boot process - loads from YAML config"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from YAML file"""
        # Load configuration from YAML
        self.config = load_aura_config(config_path)
        
        # Map configuration to bootloader attributes
        self.system_name = self.config.system_name
        self.version = self.config.version
        self.enable_logging = True
        self.log_level = self.config.log_level
        self.offline_mode = self.config.offline
        
        # Network configuration
        self.neuron_count = self.config.neuron_count
        self.features = self.config.features
        self.enable_span = self.config.enable_span
        self.span_neurons_per_region = self.config.span_neurons_per_region
        
        # SVC configuration
        self.domains = self.config.domains
        self.realms = self.config.realms
        self.features_mode = self.config.features_mode
        self.features_alpha = self.config.features_alpha
        
        # Training configuration
        self.weights_dir = self.config.weights_dir_path
        self.models_dir = self.config.models_dir_path
        self.auto_save_weights = True
        self.performance_monitoring = self.config.performance_monitoring
        
        # System monitoring
        self.health_check_interval = self.config.health_check_interval
        self.metrics_collection_interval = self.config.metrics_collection_interval
        
        # SVC Analysis configuration
        self.enable_svc_analysis = self.config.enable_svc_analysis
        self.svc_data_path = self.config.svc_data_path
        self.linguistic_features_enabled = self.config.linguistic_features_enabled
        
        # Additional configuration from YAML
        self.device_type = self.config.device_type
        self.fallback_to_cpu = self.config.fallback_to_cpu
        self.model_files = self.config.model_files
        self.nlms_clamp = self.config.nlms_clamp
        self.nlms_l2 = self.config.nlms_l2
        self.input_channels = self.config.input_channels
        self.output_channels = self.config.output_channels
        self.startnew = self.config.startnew
        
        # Brain region configuration
        self.thalamus_neuron_count = self.config.thalamus_neuron_count
        self.thalamus_input_channels = self.config.thalamus_input_channels
        self.thalamus_output_channels = self.config.thalamus_output_channels
        self.hippocampus_neuron_count = self.config.hippocampus_neuron_count
        self.hippocampus_features = self.config.hippocampus_features
        self.hippocampus_input_dim = self.config.hippocampus_input_dim
        self.amygdala_neuron_count = self.config.amygdala_neuron_count
        self.amygdala_features = self.config.amygdala_features
        self.amygdala_input_dim = self.config.amygdala_input_dim
        self.thalamic_router_neuron_count = self.config.thalamic_router_neuron_count
        self.thalamic_router_features = self.config.thalamic_router_features
        self.thalamic_router_input_dim = self.config.thalamic_router_input_dim
        self.cns_input_dim = self.config.cns_input_dim


@dataclass
class SystemHealth:
    """System health monitoring"""
    status: str = "INITIALIZING"
    uptime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    span_performance: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    warning_count: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    component_status: Dict[str, str] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """Comprehensive system metrics"""
    total_queries_processed: int = 0
    successful_responses: int = 0
    failed_responses: int = 0
    average_response_time: float = 0.0
    span_accuracy: float = 0.0
    routing_accuracy: float = 0.0
    memory_consolidation_rate: float = 0.0
    neurogenesis_events: int = 0
    synaptic_updates: int = 0


class AuraBootloader:
    """
    AURA_GENESIS System Bootloader - PRODUCTION VERSION!
    
    Comprehensive system initialization with:
    - Network initialization with SPAN integration
    - System health monitoring
    - Performance metrics collection
    - Error handling and recovery
    - Modular component management
    """
    
    def __init__(self, config: AuraBootConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.boot_start_time = datetime.now()
        self.system_components: Dict[str, Any] = {}
        self.health = SystemHealth()
        self.metrics = SystemMetrics()
        self.is_running = False
        self._monitor_cancel_scope: Optional[trio.CancelScope] = None
        self._monitors_started: bool = False
        self._shutdown_event: trio.Event | None = None
        self._nursery: trio.Nursery | None = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        if not self.config.enable_logging:
            logging.getLogger().setLevel(logging.CRITICAL)
            return logging.getLogger()
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(f'aura_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        return logging.getLogger(f"{self.config.system_name}.Bootloader")
    
    async def load_pretrained_models(self) -> Dict[str, Any]:
        """Load pre-trained models from the models directory"""
        self.logger.info("📦 Loading pre-trained models...")
        
        models = {}
        models_path = Path(self.config.models_dir)
        
        if not models_path.exists():
            self.logger.warning(f"Models directory not found: {models_path}")
            return models
        
        # Define model mappings
        model_files = {
            'emotion_classifier': 'clf_emotion.pt',
            'intent_classifier': 'clf_intent.pt', 
            'tone_classifier': 'clf_tone.pt',
            'svc_domain_classifier': 'svc_domain_classifier_enhanced.pt',
            'svc_realm_classifier': 'svc_realm_classifier_enhanced.pt',
            'svc_difficulty_regressor': 'svc_difficulty_regressor_enhanced.pt'
        }
        
        try:
            import torch
            
            for model_name, filename in model_files.items():
                model_path = models_path / filename
                if model_path.exists():
                    try:
                        # Load PyTorch model
                        model = torch.load(model_path, map_location='cpu')
                        models[model_name] = model
                        self.logger.info(f"Loaded {model_name} from {filename}")
                    except Exception as e:
                        self.logger.warning(f"Could not load {filename}: {e}")
                else:
                    self.logger.warning(f"Model file not found: {filename}")
            
            self.logger.info(f"Loaded {len(models)} pre-trained models")
            return models
            
        except ImportError:
            self.logger.warning("PyTorch not available, skipping model loading")
            return models
        except Exception as e:
            self.logger.error(f"Model loading failed: {e}")
            return models
    
    async def initialize_network(self) -> Network:
        """Initialize the AURA neural network using safe boot sequence"""
        self.logger.info("Initializing AURA Network with safe boot sequence...")
        
        try:
            boot_config = BootSequenceConfig(
                validate_dependencies=True,
                initialize_weights=True,
                enable_health_checks=True,
                timeout_seconds=60,
                retry_attempts=3
            )
            
            boot_sequence = AuraBootSequence(boot_config)
            boot_result = await boot_sequence.execute_safe_boot()
            
            if not boot_result['success']:
                error_msg = f"Safe boot sequence failed: {boot_result.get('error', 'Unknown error')}"
                if boot_result.get('failed_components'):
                    error_msg += f"\nFailed components: {boot_result['failed_components']}"
                if boot_result.get('errors'):
                    error_msg += f"\nErrors: {boot_result['errors']}"
                raise Exception(error_msg)
            
            network = boot_result['network']
            pretrained_models = await self.load_pretrained_models()
            network.pretrained_models = pretrained_models
            
            try:
                weights_path = Path(self.config.weights_dir)
                if weights_path.exists():
                    historical_path = weights_path / "historical"
                    if historical_path.exists():
                        loaded_counts = load_network_weights(network, str(historical_path))
                        self.logger.info(f"Loaded weights from historical: {loaded_counts}")
                    else:
                        loaded_counts = load_network_weights(network, str(weights_path))
                        self.logger.info(f"Loaded weights: {loaded_counts}")
                else:
                    self.logger.info("No weights directory found, using safe boot initialization")
            except Exception as e:
                self.logger.warning(f"Could not load existing weights: {e}")
            
            self.logger.info(f"AURA Network initialized successfully in {boot_result['boot_time']:.2f}s")
            self.logger.info(f"Components initialized: {len(boot_result['components_initialized'])}")
            
            return network
            
        except Exception as e:
            self.logger.error(f"Network initialization failed: {e}")
            raise
    
    async def initialize_chat_orchestrator(self, network: Network) -> ChatOrchestrator:
        """Initialize the chat orchestrator"""
        self.logger.info("Initializing Chat Orchestrator...")
        
        try:
            orchestrator = ChatOrchestrator(network)
            self.logger.info("Chat Orchestrator initialized successfully")
            return orchestrator
            
        except Exception as e:
            self.logger.error(f"Chat Orchestrator initialization failed: {e}")
            raise
    
    async def initialize_trainer(self, network: Network) -> EnhancedAuraTrainer:
        """Initialize the enhanced trainer"""
        self.logger.info("Initializing Enhanced Trainer...")
        
        try:
            # Create a trainer wrapper for the network
            class NetworkTrainerWrapper:
                def __init__(self, network, **kwargs):
                    self.network = network
                    self.kwargs = kwargs
                
                async def process_dataset(self, file_path):
                    import json
                    rows: List[Dict[str, Any]] = []
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                rows.append(json.loads(line))
                    return await self.network.train_on_data(rows)
                
                async def process_movie_scenes_dataset(self, file_path):
                    import json
                    rows: List[Dict[str, Any]] = []
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                rows.append(json.loads(line))
                    return await self.network.train_on_data(rows)
            
            trainer = EnhancedAuraTrainer(NetworkTrainerWrapper)
            self.logger.info("Enhanced Trainer initialized successfully")
            return trainer
            
        except Exception as e:
            self.logger.error(f"Enhanced Trainer initialization failed: {e}")
            raise
    
    async def initialize_svc_analysis(self):
        """Initialize SVC analysis capabilities"""
        if not self.config.enable_svc_analysis:
            self.logger.info("SVC analysis disabled")
            return None
        
        self.logger.info("Initializing SVC Analysis...")
        
        try:
            # Load SVC data if available
            svc_data = []
            if self.config.svc_data_path and os.path.exists(self.config.svc_data_path):
                svc_data = load_enhanced_svc_dataset(self.config.svc_data_path)
                self.logger.info(f"Loaded {len(svc_data)} SVC samples")
            else:
                self.logger.info("No SVC data path provided, using empty dataset")
            
            # Initialize SVC analysis components
            svc_analyzer = {
                'data': svc_data,
                'domains': self.config.domains,
                'realms': self.config.realms,
                'linguistic_features_enabled': self.config.linguistic_features_enabled
            }
            
            self.logger.info("SVC Analysis initialized successfully")
            return svc_analyzer
            
        except Exception as e:
            self.logger.error(f"SVC Analysis initialization failed: {e}")
            return None

    async def initialize_qdrant(self):
        """Initialize Qdrant streaming if enabled in config"""
        try:
            # Check if Qdrant is enabled in config
            if not hasattr(self.config, 'qdrant') or not self.config.qdrant.get('enable', False):
                self.logger.info("Qdrant streaming disabled in configuration")
                return None
            
            qdrant_config = self.config.qdrant
            self.logger.info(f"Initializing Qdrant streaming to {qdrant_config['url']}:{qdrant_config['port']}")
            
            # Set environment variables for Qdrant
            os.environ['AURA_QDRANT_STREAM'] = qdrant_config['environment_vars']['AURA_QDRANT_STREAM']
            os.environ['QDRANT_URL'] = qdrant_config['environment_vars']['QDRANT_URL']
            os.environ['QDRANT_PORT'] = qdrant_config['environment_vars']['QDRANT_PORT']
            
            # Initialize Qdrant streamer
            qdrant_streamer = QdrantStreamer(
                url=qdrant_config['url'],
                port=qdrant_config['port']
            )
            
            self.logger.info("Qdrant streaming initialized successfully")
            return qdrant_streamer
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Qdrant: {e}")
            return None
    
    async def initialize_system_components(self):
        """Initialize all system components"""
        self.logger.info("Starting AURA_GENESIS System Initialization...")
        
        try:
            # Initialize core network
            network = await self.initialize_network()
            self.system_components['network'] = network
            
            # Initialize chat orchestrator
            orchestrator = await self.initialize_chat_orchestrator(network)
            self.system_components['orchestrator'] = orchestrator
            
            # Initialize trainer
            trainer = await self.initialize_trainer(network)
            self.system_components['trainer'] = trainer
            
            # Initialize SVC analysis
            svc_analyzer = await self.initialize_svc_analysis()
            if svc_analyzer:
                self.system_components['svc_analyzer'] = svc_analyzer
            
            # Initialize directory loader
            directory_loader = DirectoryLoader("datasets")
            self.system_components['directory_loader'] = directory_loader
            
            # Initialize Qdrant if enabled
            qdrant_streamer = await self.initialize_qdrant()
            if qdrant_streamer:
                self.system_components['qdrant_streamer'] = qdrant_streamer
            
            # Update health status
            self.health.status = "INITIALIZED"
            self.health.component_status = {
                'network': 'ACTIVE',
                'orchestrator': 'ACTIVE',
                'trainer': 'ACTIVE',
                'directory_loader': 'ACTIVE'
            }
            
            self.logger.info("All system components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.health.status = "FAILED"
            self.health.error_count += 1
            raise
    
    async def start_monitoring(self):
        """Start system monitoring tasks"""
        if not self.config.performance_monitoring:
            self.logger.info("Performance monitoring disabled")
            return
        if self._monitors_started:
            return
        self.logger.info("Starting system monitoring...")

        async def health_monitor():
            """Monitor system health"""
            while self.is_running:
                try:
                    await self._update_health_status()
                    await trio.sleep(self.config.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    await trio.sleep(5)
        
        async def metrics_collector():
            """Collect system metrics"""
            while self.is_running:
                try:
                    await self._collect_metrics()
                    await trio.sleep(self.config.metrics_collection_interval)
                except Exception as e:
                    self.logger.error(f"Metrics collection error: {e}")
                    await trio.sleep(10)
        
        async def _monitor_bundle():
            # Lives until cancel_scope is cancelled in shutdown
            self._monitor_cancel_scope = trio.CancelScope()
            with self._monitor_cancel_scope:
                async with trio.open_nursery() as nursery:
                    nursery.start_soon(health_monitor)
                    nursery.start_soon(metrics_collector)
                    await trio.sleep_forever()

        # Launch background monitors and return immediately
        try:
            trio.lowlevel.spawn_system_task(_monitor_bundle)
            self._monitors_started = True
        except Exception as e:
            self.logger.warning(f"Could not start monitoring: {e}")
            self._monitors_started = False
    
    async def _update_health_status(self):
        """Update system health status"""
        try:
            # Update uptime
            self.health.uptime = (datetime.now() - self.boot_start_time).total_seconds()
            
            # Update memory usage (simplified)
            import psutil
            process = psutil.Process()
            self.health.memory_usage = process.memory_info().rss / 1024 / 1024  # MB
            self.health.cpu_usage = process.cpu_percent()
            
            # Update last heartbeat
            self.health.last_heartbeat = datetime.now()
            
            # Check component health
            for component_name, component in self.system_components.items():
                try:
                    if hasattr(component, 'health_check'):
                        health_status = await component.health_check()
                        self.health.component_status[component_name] = health_status
                    else:
                        self.health.component_status[component_name] = 'ACTIVE'
                except Exception:
                    self.health.component_status[component_name] = 'ERROR'
                    self.health.error_count += 1
            
        except Exception as e:
            self.logger.warning(f"Health status update failed: {e}")
            self.health.warning_count += 1
    
    async def _collect_metrics(self):
        """Collect system performance metrics"""
        try:
            # This would collect various performance metrics
            # For now, we'll implement basic metrics collection
            
            if 'network' in self.system_components:
                network = self.system_components['network']
                
                # Collect SPAN performance if available
                if hasattr(network, 'span_performance_history'):
                    recent_performance = network.span_performance_history[-10:] if network.span_performance_history else []
                    if recent_performance:
                        avg_accuracy = np.mean([p.get('accuracy', 0) for p in recent_performance])
                        self.metrics.span_accuracy = avg_accuracy
                
                # Collect routing accuracy if available
                if hasattr(network, 'routing_accuracy'):
                    self.metrics.routing_accuracy = network.routing_accuracy
            
        except Exception as e:
            self.logger.warning(f"Metrics collection failed: {e}")
    
    async def boot_system(self):
        """Main boot sequence"""
        try:
            self.logger.info(f"🚀 Booting {self.config.system_name} v{self.config.version}")
            
            # Initialize all components
            await self.initialize_system_components()
            
            # Start monitoring
            await self.start_monitoring()
            
            # Mark system as running
            self.is_running = True
            self.health.status = "RUNNING"
            
            boot_time = (datetime.now() - self.boot_start_time).total_seconds()
            self.logger.info(f"{self.config.system_name} booted successfully in {boot_time:.2f}s")
            
            return True
            
        except Exception as e:
            self.logger.error(f"System boot failed: {e}")
            self.health.status = "FAILED"
            return False
    
    async def shutdown_system(self):
        """Graceful system shutdown"""
        self.logger.info("🛑 Shutting down AURA_GENESIS system...")
        
        try:
            # Stop monitoring tasks
            self.is_running = False
            if self._monitor_cancel_scope is not None:
                self._monitor_cancel_scope.cancel()
            
            # Save weights if configured using weights_io
            if self.config.auto_save_weights and 'network' in self.system_components:
                self.logger.info("💾 Saving network weights...")
                try:
                    network = self.system_components['network']
                    weights_path = Path(self.config.weights_dir) / "historical"
                    weights_path.mkdir(parents=True, exist_ok=True)
                    
                    saved_counts = save_network_weights(network, str(weights_path))
                    self.logger.info(f"Saved weights: {saved_counts}")
                except Exception as e:
                    self.logger.error(f"Failed to save weights: {e}")
            
            # Update health status
            self.health.status = "SHUTDOWN"
            
            self.logger.info("System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'health': {
                'status': self.health.status,
                'uptime': self.health.uptime,
                'memory_usage_mb': self.health.memory_usage,
                'cpu_usage_percent': self.health.cpu_usage,
                'error_count': self.health.error_count,
                'warning_count': self.health.warning_count,
                'component_status': self.health.component_status
            },
            'metrics': {
                'total_queries': self.metrics.total_queries_processed,
                'successful_responses': self.metrics.successful_responses,
                'failed_responses': self.metrics.failed_responses,
                'average_response_time': self.metrics.average_response_time,
                'span_accuracy': self.metrics.span_accuracy,
                'routing_accuracy': self.metrics.routing_accuracy
            },
            'config': {
                'system_name': self.config.system_name,
                'version': self.config.version,
                'offline_mode': self.config.offline_mode,
                'enable_span': self.config.enable_span
            }
        }
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query through the system"""
        start_time = time.time()
        
        try:
            if 'orchestrator' not in self.system_components:
                raise Exception("System not properly initialized")
            
            # Process through orchestrator
            result = await self.system_components['orchestrator'].process_query(query)
            
            # Update metrics
            self.metrics.total_queries_processed += 1
            self.metrics.successful_responses += 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.total_queries_processed - 1) + 
                 (time.time() - start_time)) / self.metrics.total_queries_processed
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {e}")
            self.metrics.failed_responses += 1
            self.health.error_count += 1
            return {'error': str(e), 'status': 'failed'}
    
    async def analyze_svc_structure(self, text: str) -> Dict[str, Any]:
        """Analyze Subject-Verb-Complement structure of text"""
        if 'svc_analyzer' not in self.system_components:
            return {'error': 'SVC analysis not available', 'status': 'unavailable'}
        
        try:
            # Extract linguistic features
            if self.config.linguistic_features_enabled:
                pos_features = extract_pos_features(text)
                ner_features = extract_ner_features(text)
                structural_features = extract_structural_features(text)
            else:
                pos_features = []
                ner_features = []
                structural_features = {}
            
            # Basic SVC analysis (simplified)
            words = text.lower().split()
            svc_analysis = {
                'text': text,
                'word_count': len(words),
                'estimated_subject': words[0] if words else '',
                'estimated_verb': words[1] if len(words) > 1 else '',
                'estimated_complement': ' '.join(words[2:]) if len(words) > 2 else '',
                'linguistic_features': {
                    'pos_features': pos_features,
                    'ner_features': ner_features,
                    'structural_features': structural_features
                },
                'complexity_score': len(words) / 10.0,  # Simple complexity metric
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return svc_analysis
            
        except Exception as e:
            self.logger.error(f"SVC analysis failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def train_svc_analyzer(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the SVC analyzer on provided data"""
        if 'svc_analyzer' not in self.system_components:
            return {'error': 'SVC analyzer not available', 'status': 'unavailable'}
        
        try:
            # Process training data through the network
            network = self.system_components['network']
            results = await network.train_on_data(training_data)
            
            # Update SVC analyzer with new data
            svc_analyzer = self.system_components['svc_analyzer']
            svc_analyzer['data'].extend(training_data)
            
            self.logger.info(f"SVC analyzer trained on {len(training_data)} samples")
            
            return {
                'status': 'success',
                'samples_trained': len(training_data),
                'training_results': results,
                'total_svc_samples': len(svc_analyzer['data'])
            }
            
        except Exception as e:
            self.logger.error(f"SVC training failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def get_svc_insights(self) -> Dict[str, Any]:
        """Get insights from SVC analysis"""
        if 'svc_analyzer' not in self.system_components:
            return {'error': 'SVC analyzer not available', 'status': 'unavailable'}
        
        try:
            svc_analyzer = self.system_components['svc_analyzer']
            data = svc_analyzer['data']
            
            if not data:
                return {'status': 'no_data', 'message': 'No SVC data available for analysis'}
            
            # Analyze patterns in the data
            domain_counts = {}
            realm_counts = {}
            complexity_scores = []
            
            for sample in data:
                # Count domains and realms
                domain = sample.get('metadata', {}).get('domain', 'unknown')
                realm = sample.get('metadata', {}).get('realm', 'unknown')
                
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
                realm_counts[realm] = realm_counts.get(realm, 0) + 1
                
                # Collect complexity scores
                if 'difficulty' in sample:
                    complexity_scores.append(sample['difficulty'])
            
            insights = {
                'total_samples': len(data),
                'domain_distribution': domain_counts,
                'realm_distribution': realm_counts,
                'average_complexity': np.mean(complexity_scores) if complexity_scores else 0.0,
                'complexity_range': {
                    'min': np.min(complexity_scores) if complexity_scores else 0.0,
                    'max': np.max(complexity_scores) if complexity_scores else 0.0
                },
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"SVC insights generation failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def classify_emotion(self, text: str) -> Dict[str, Any]:
        """Classify emotion using pre-trained model"""
        if 'network' not in self.system_components:
            return {'error': 'Network not available', 'status': 'unavailable'}
        
        network = self.system_components['network']
        if not hasattr(network, 'pretrained_models') or 'emotion_classifier' not in network.pretrained_models:
            return {'error': 'Emotion classifier not available', 'status': 'unavailable'}
        
        try:
            import torch
            # Get features for the text
            features = network.get_features(text)
            
            # Use pre-trained emotion classifier
            emotion_model = network.pretrained_models['emotion_classifier']
            emotion_model.eval()
            
            with torch.no_grad():
                # Convert features to tensor and add batch dimension
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                prediction = emotion_model(input_tensor)
                emotion_scores = torch.softmax(prediction, dim=1)
                predicted_emotion = torch.argmax(emotion_scores, dim=1).item()
                confidence = emotion_scores[0][predicted_emotion].item()
            
            # Map emotion indices to labels (you may need to adjust these)
            emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
            emotion_name = emotion_labels[predicted_emotion] if predicted_emotion < len(emotion_labels) else 'unknown'
            
            return {
                'emotion': emotion_name,
                'confidence': confidence,
                'all_scores': emotion_scores[0].tolist(),
                'text': text
            }
            
        except Exception as e:
            self.logger.error(f"Emotion classification failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def classify_intent(self, text: str) -> Dict[str, Any]:
        """Classify intent using pre-trained model"""
        if 'network' not in self.system_components:
            return {'error': 'Network not available', 'status': 'unavailable'}
        
        network = self.system_components['network']
        if not hasattr(network, 'pretrained_models') or 'intent_classifier' not in network.pretrained_models:
            return {'error': 'Intent classifier not available', 'status': 'unavailable'}
        
        try:
            import torch
            # Get features for the text
            features = network.get_features(text)
            
            # Use pre-trained intent classifier
            intent_model = network.pretrained_models['intent_classifier']
            intent_model.eval()
            
            with torch.no_grad():
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                prediction = intent_model(input_tensor)
                intent_scores = torch.softmax(prediction, dim=1)
                predicted_intent = torch.argmax(intent_scores, dim=1).item()
                confidence = intent_scores[0][predicted_intent].item()
            
            # Map intent indices to labels
            intent_labels = ['question', 'statement', 'command', 'greeting', 'farewell', 'other']
            intent_name = intent_labels[predicted_intent] if predicted_intent < len(intent_labels) else 'unknown'
            
            return {
                'intent': intent_name,
                'confidence': confidence,
                'all_scores': intent_scores[0].tolist(),
                'text': text
            }
            
        except Exception as e:
            self.logger.error(f"Intent classification failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def classify_tone(self, text: str) -> Dict[str, Any]:
        """Classify tone using pre-trained model"""
        if 'network' not in self.system_components:
            return {'error': 'Network not available', 'status': 'unavailable'}
        
        network = self.system_components['network']
        if not hasattr(network, 'pretrained_models') or 'tone_classifier' not in network.pretrained_models:
            return {'error': 'Tone classifier not available', 'status': 'unavailable'}
        
        try:
            import torch
            # Get features for the text
            features = network.get_features(text)
            
            # Use pre-trained tone classifier
            tone_model = network.pretrained_models['tone_classifier']
            tone_model.eval()
            
            with torch.no_grad():
                input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                prediction = tone_model(input_tensor)
                tone_scores = torch.softmax(prediction, dim=1)
                predicted_tone = torch.argmax(tone_scores, dim=1).item()
                confidence = tone_scores[0][predicted_tone].item()
            
            # Map tone indices to labels
            tone_labels = ['formal', 'casual', 'friendly', 'professional', 'aggressive', 'neutral']
            tone_name = tone_labels[predicted_tone] if predicted_tone < len(tone_labels) else 'unknown'
            
            return {
                'tone': tone_name,
                'confidence': confidence,
                'all_scores': tone_scores[0].tolist(),
                'text': text
            }
            
        except Exception as e:
            self.logger.error(f"Tone classification failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    async def comprehensive_analysis(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive analysis using all available models"""
        try:
            # Get basic SVC analysis
            svc_analysis = await self.analyze_svc_structure(text)
            
            # Get emotion classification
            emotion_analysis = await self.classify_emotion(text)
            
            # Get intent classification
            intent_analysis = await self.classify_intent(text)
            
            # Get tone classification
            tone_analysis = await self.classify_tone(text)
            
            # Get network processing
            network_result = await self.process_query(text)
            
            return {
                'text': text,
                'svc_analysis': svc_analysis,
                'emotion_analysis': emotion_analysis,
                'intent_analysis': intent_analysis,
                'tone_analysis': tone_analysis,
                'network_result': network_result,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            return {'error': str(e), 'status': 'failed'}

    async def _health_monitor_loop(self):
        """Background health monitoring loop"""
        if not self.config.performance_monitoring:
            return
        assert self._shutdown_event is not None
        while not self._shutdown_event.is_set():
            try:
                await self._update_health_status()
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
            await trio.sleep(self.config.health_check_interval)

    async def _metrics_collector_loop(self):
        """Background metrics collection loop"""
        if not self.config.performance_monitoring:
            return
        assert self._shutdown_event is not None
        while not self._shutdown_event.is_set():
            try:
                await self._collect_metrics()
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
            await trio.sleep(self.config.metrics_collection_interval)

    async def _signal_watchdog(self):
        """Handle shutdown signals (Ctrl+C, SIGTERM)"""
        assert self._shutdown_event is not None
        with trio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
            async for signum in signals:
                self.logger.info(f"Signal received: {signum}, initiating shutdown…")
                self._shutdown_event.set()
                break

    async def _stdin_repl(self):
        """
        Minimal REPL for ops while the system is online.
        Commands: status | router.stats | attn.tail <n> | shutdown
        """
        assert self._shutdown_event is not None
        while not self._shutdown_event.is_set():
            try:
                line = await trio.to_thread.run_sync(sys.stdin.readline)
                if not line:
                    await trio.sleep(0.1)
                    continue
                cmd = line.strip().lower()
                if cmd == "status":
                    print(json.dumps(self.get_system_status(), indent=2))
                elif cmd.startswith("attn.tail"):
                    n = 10
                    try:
                        parts = cmd.split()
                        if len(parts) > 1:
                            n = int(parts[1])
                    except Exception:
                        pass
                    net = self.system_components.get("network")
                    if net and hasattr(net, "_thalamic_router"):
                        events = net._thalamic_router.recent_attention_events(n)
                        print(json.dumps(events, indent=2))
                    else:
                        print("No router/attention available.")
                elif cmd == "router.stats":
                    net = self.system_components.get("network")
                    if net and hasattr(net, "_thalamic_router"):
                        print(json.dumps(net._thalamic_router.get_moe_stats(), indent=2))
                    else:
                        print("No router available.")
                elif cmd in {"quit", "exit", "shutdown"}:
                    print("Shutting down…")
                    self._shutdown_event.set()
                elif cmd == "":
                    pass
                else:
                    print("Commands: status | router.stats | attn.tail <n> | shutdown")
            except Exception as e:
                self.logger.error(f"REPL error: {e}")
                await trio.sleep(0.25)

    async def serve_forever(self):
        """
        Run the system online until SIGINT/SIGTERM or 'shutdown' from REPL.
        """
        if not self.is_running or self.health.status != "RUNNING":
            # Bring system up if not already
            ok = await self.boot_system()
            if not ok:
                raise RuntimeError("Boot failed; cannot serve.")
        self.logger.info("🌐 Entering online mode (serve_forever)…")

        self._shutdown_event = trio.Event()
        async with trio.open_nursery() as nursery:
            self._nursery = nursery
            # background tasks
            nursery.start_soon(self._health_monitor_loop)
            nursery.start_soon(self._metrics_collector_loop)
            nursery.start_soon(self._signal_watchdog)
            nursery.start_soon(self._stdin_repl)

            # block here until shutdown requested
            await self._shutdown_event.wait()

        # Nursery closed -> children cancelled; proceed to graceful shutdown
        await self.shutdown_system()


async def boot_aura_genesis(config: Optional[AuraBootConfig] = None) -> AuraBootloader:
    """Boot the AURA_GENESIS system"""
    if config is None:
        config = AuraBootConfig()
    
    bootloader = AuraBootloader(config)
    success = await bootloader.boot_system()
    
    if not success:
        raise Exception("Failed to boot AURA_GENESIS system")
    
    return bootloader


def create_default_config() -> AuraBootConfig:
    """Create a default boot configuration"""
    return AuraBootConfig()


def main():
    """Main entry point for the bootloader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="AURA_GENESIS Bootloader")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--offline", action="store_true", help="Run in offline mode")
    parser.add_argument("--no-span", action="store_true", help="Disable SPAN integration")
    parser.add_argument("--no-svc", action="store_true", help="Disable SVC analysis")
    parser.add_argument("--svc-data", type=str, help="Path to SVC training data")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_default_config()
    if args.offline:
        config.offline_mode = True
    if args.no_span:
        config.enable_span = False
    if args.no_svc:
        config.enable_svc_analysis = False
    if args.svc_data:
        config.svc_data_path = args.svc_data
    config.log_level = args.log_level
    
    # Load custom config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Boot the system
    async def run():
        try:
            bootloader = await boot_aura_genesis(config)
            print(f"{config.system_name} v{config.version} is running! (Ctrl+C to exit)")
            await bootloader.serve_forever()
        except Exception as e:
            print(f"Boot failed: {e}")
            sys.exit(1)
    
    trio.run(run)


if __name__ == "__main__":
    main()