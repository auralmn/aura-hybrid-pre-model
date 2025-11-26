"""
Configuration loader for AURA_GENESIS system
Handles loading and validation of YAML configuration files
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AuraConfig:
    """AURA system configuration loaded from YAML"""
    
    # System information
    system_name: str = "AURA_GENESIS"
    version: str = "2.0.0"
    description: str = "Advanced Neural Network System"
    
    # Boot configuration
    validate_dependencies: bool = True
    initialize_weights: bool = True
    enable_health_checks: bool = True
    timeout_seconds: int = 60
    retry_attempts: int = 3
    safe_mode: bool = True
    
    # Network architecture
    neuron_count: int = 1000
    features: int = 384
    input_channels: int = 384
    output_channels: int = 384
    enable_span: bool = False
    span_neurons_per_region: int = 10
    domains: list = field(default_factory=list)
    realms: list = field(default_factory=list)
    domain_labels_path: str = "/Volumes/Others2/AURA_GENESIS/svc_domain_labels.json"
    offline: bool = True
    nlms_clamp: tuple = (0.0, 1.0)
    nlms_l2: float = 1e-4
    features_mode: str = "sbert"
    features_alpha: float = 0.7
    weights_dir: str = "svc_nlms_weights"
    startnew: bool = False
    
    # Brain region configuration
    thalamus_neuron_count: int = 100
    thalamus_input_channels: int = 384
    thalamus_output_channels: int = 384
    hippocampus_neuron_count: int = 100
    hippocampus_features: int = 384
    hippocampus_input_dim: int = 384
    amygdala_neuron_count: int = 30
    amygdala_features: int = 384
    amygdala_input_dim: int = 384
    thalamic_router_neuron_count: int = 60
    thalamic_router_features: int = 384
    thalamic_router_input_dim: int = 384
    cns_input_dim: int = 384
    
    # Paths
    weights_dir_path: str = "/Volumes/Others2/AURA_GENESIS/weights"
    models_dir_path: str = "/Volumes/Others2/AURA_GENESIS/models"
    svc_data_path: Optional[str] = None
    log_dir: str = "logs"
    cache_dir: str = "cache"
    
    # Models
    model_files: Dict[str, str] = field(default_factory=lambda: {
        'emotion_classifier': 'clf_emotion.pt',
        'intent_classifier': 'clf_intent.pt',
        'tone_classifier': 'clf_tone.pt',
        'svc_domain_classifier': 'svc_domain_classifier_enhanced.pt',
        'svc_realm_classifier': 'svc_realm_classifier_enhanced.pt',
        'svc_difficulty_regressor': 'svc_difficulty_regressor_enhanced.pt'
    })
    
    # SVC Analysis
    enable_svc_analysis: bool = True
    linguistic_features_enabled: bool = True
    
    # Performance monitoring
    performance_monitoring: bool = True
    health_check_interval: int = 30
    metrics_collection_interval: int = 60
    memory_threshold_mb: int = 1024
    cpu_threshold_percent: float = 80.0
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "aura_system.log"
    log_max_size_mb: int = 100
    log_backup_count: int = 5
    
    # Device
    device_type: str = "mps"
    fallback_to_cpu: bool = True
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    
    # Security
    enable_encryption: bool = False
    api_key_required: bool = False
    rate_limiting: bool = False
    max_requests_per_minute: int = 100
    
    # Development
    debug_mode: bool = False
    verbose_logging: bool = False
    profile_performance: bool = False
    save_debug_info: bool = False


class ConfigLoader:
    """Configuration loader and validator"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config_data: Dict[str, Any] = {}
    
    def _find_config_file(self) -> str:
        """Find the configuration file in common locations"""
        possible_paths = [
            "config.yaml",
            "config.yml",
            "aura_config.yaml",
            "aura_config.yml",
            "/Volumes/Others2/AURA_GENESIS/config.yaml"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Default to config.yaml in current directory
        return "config.yaml"
    
    def load_config(self) -> AuraConfig:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                self.config_data = yaml.safe_load(file)
            
            logger.info(f"Loaded configuration from {self.config_path}")
            return self._parse_config()
            
        except FileNotFoundError:
            logger.warning(f"Configuration file not found: {self.config_path}")
            logger.info("Using default configuration")
            return AuraConfig()
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def _load_domain_labels(self, config: AuraConfig) -> None:
        """Load domain labels from JSON file"""
        try:
            if os.path.exists(config.domain_labels_path):
                with open(config.domain_labels_path, 'r') as f:
                    domain_data = json.load(f)
                
                # Extract domains from DOMAIN_TO_IDX mapping
                domains = list(domain_data.get('DOMAIN_TO_IDX', {}).keys())
                config.domains = domains
                
                # For realms, we can use a subset or create categories
                # Let's create some high-level realm categories based on domains
                realm_categories = {
                    'Science': ['Physics', 'Chemistry', 'Biology', 'Astronomy', 'Astrophysics', 'Biochemistry', 'Cell Biology', 'Climate Science', 'Earth Science', 'Ecology', 'Environmental Science', 'Evolutionary Biology', 'Genetics', 'Immunology', 'Neuroscience', 'Quantum Physics', 'Science'],
                    'Technology': ['Artificial Intelligence', 'Blockchain', 'Computer Science', 'Computer Security', 'Electronics', 'Energy', 'Fintech', 'Nanotechnology', 'Robotics', 'Technology'],
                    'Arts & Culture': ['Art', 'Art History', 'Arts', 'Cultural Studies', 'Cultural Systems', 'Culture', 'Fashion', 'Film', 'Folklore', 'Music', 'Music History', 'Performing Arts'],
                    'Humanities': ['Ancient History', 'Archaeology', 'Architecture', 'European History', 'History', 'History of Technology', 'Legal Studies', 'Linguistics', 'Literature', 'Medieval History', 'Modern History', 'Philosophy', 'Philosophy of Science', 'Religion', 'US History', 'World History', 'Sociolinguistics', 'Sociology'],
                    'Social Sciences': ['Economics', 'Finance', 'Geography', 'Geology', 'Political Science', 'Politics'],
                    'Medicine & Health': ['Medicine', 'Virology']
                }
                
                # Create realms list from categories
                config.realms = list(realm_categories.keys())
                
                logger.info(f"Loaded {len(domains)} domains and {len(config.realms)} realms from {config.domain_labels_path}")
            else:
                logger.warning(f"Domain labels file not found: {config.domain_labels_path}")
                # Fallback to default domains
                config.domains = ["academic", "technical", "creative", "personal", "professional"]
                config.realms = ["science", "technology", "arts", "humanities", "business"]
                
        except Exception as e:
            logger.error(f"Error loading domain labels: {e}")
            # Fallback to default domains
            config.domains = ["academic", "technical", "creative", "personal", "professional"]
            config.realms = ["science", "technology", "arts", "humanities", "business"]

    def _parse_config(self) -> AuraConfig:
        """Parse loaded YAML data into AuraConfig object"""
        config = AuraConfig()
        
        # System information
        if 'system' in self.config_data:
            system = self.config_data['system']
            config.system_name = system.get('name', config.system_name)
            config.version = system.get('version', config.version)
            config.description = system.get('description', config.description)
        
        # Boot configuration
        if 'boot' in self.config_data:
            boot = self.config_data['boot']
            config.validate_dependencies = boot.get('validate_dependencies', config.validate_dependencies)
            config.initialize_weights = boot.get('initialize_weights', config.initialize_weights)
            config.enable_health_checks = boot.get('enable_health_checks', config.enable_health_checks)
            config.timeout_seconds = boot.get('timeout_seconds', config.timeout_seconds)
            config.retry_attempts = boot.get('retry_attempts', config.retry_attempts)
            config.safe_mode = boot.get('safe_mode', config.safe_mode)
        
        # Network configuration
        if 'network' in self.config_data:
            network = self.config_data['network']
            config.neuron_count = network.get('neuron_count', config.neuron_count)
            config.features = network.get('features', config.features)
            config.input_channels = network.get('input_channels', config.input_channels)
            config.output_channels = network.get('output_channels', config.output_channels)
            config.enable_span = network.get('enable_span', config.enable_span)
            config.span_neurons_per_region = network.get('span_neurons_per_region', config.span_neurons_per_region)
            config.domains = network.get('domains', config.domains)
            config.realms = network.get('realms', config.realms)
            config.offline = network.get('offline', config.offline)
            config.nlms_clamp = tuple(network.get('nlms_clamp', config.nlms_clamp))
            config.nlms_l2 = network.get('nlms_l2', config.nlms_l2)
            config.features_mode = network.get('features_mode', config.features_mode)
            config.features_alpha = network.get('features_alpha', config.features_alpha)
            config.weights_dir = network.get('weights_dir', config.weights_dir)
            config.startnew = network.get('startnew', config.startnew)
        
        # Brain region configuration
        if 'brain_regions' in self.config_data:
            regions = self.config_data['brain_regions']
            
            if 'thalamus' in regions:
                thalamus = regions['thalamus']
                config.thalamus_neuron_count = thalamus.get('neuron_count', config.thalamus_neuron_count)
                config.thalamus_input_channels = thalamus.get('input_channels', config.thalamus_input_channels)
                config.thalamus_output_channels = thalamus.get('output_channels', config.thalamus_output_channels)
            
            if 'hippocampus' in regions:
                hippocampus = regions['hippocampus']
                config.hippocampus_neuron_count = hippocampus.get('neuron_count', config.hippocampus_neuron_count)
                config.hippocampus_features = hippocampus.get('features', config.hippocampus_features)
                config.hippocampus_input_dim = hippocampus.get('input_dim', config.hippocampus_input_dim)
            
            if 'amygdala' in regions:
                amygdala = regions['amygdala']
                config.amygdala_neuron_count = amygdala.get('neuron_count', config.amygdala_neuron_count)
                config.amygdala_features = amygdala.get('features', config.amygdala_features)
                config.amygdala_input_dim = amygdala.get('input_dim', config.amygdala_input_dim)
            
            if 'thalamic_router' in regions:
                router = regions['thalamic_router']
                config.thalamic_router_neuron_count = router.get('neuron_count', config.thalamic_router_neuron_count)
                config.thalamic_router_features = router.get('features', config.thalamic_router_features)
                config.thalamic_router_input_dim = router.get('input_dim', config.thalamic_router_input_dim)
            
            if 'cns' in regions:
                cns = regions['cns']
                config.cns_input_dim = cns.get('input_dim', config.cns_input_dim)
        
        # Paths
        if 'paths' in self.config_data:
            paths = self.config_data['paths']
            config.weights_dir_path = paths.get('weights_dir', config.weights_dir_path)
            config.models_dir_path = paths.get('models_dir', config.models_dir_path)
            config.svc_data_path = paths.get('svc_data_path', config.svc_data_path)
            config.log_dir = paths.get('log_dir', config.log_dir)
            config.cache_dir = paths.get('cache_dir', config.cache_dir)
        
        # Models
        if 'models' in self.config_data:
            config.model_files = self.config_data['models']
        
        # SVC Analysis
        if 'svc_analysis' in self.config_data:
            svc = self.config_data['svc_analysis']
            config.enable_svc_analysis = svc.get('enable', config.enable_svc_analysis)
            config.linguistic_features_enabled = svc.get('linguistic_features_enabled', config.linguistic_features_enabled)
            config.domain_labels_path = svc.get('domain_labels_path', config.domain_labels_path)
            # Note: domains and realms will be loaded from JSON file, not from YAML
        
        # Performance monitoring
        if 'monitoring' in self.config_data:
            monitoring = self.config_data['monitoring']
            config.performance_monitoring = monitoring.get('enable', config.performance_monitoring)
            config.health_check_interval = monitoring.get('health_check_interval', config.health_check_interval)
            config.metrics_collection_interval = monitoring.get('metrics_collection_interval', config.metrics_collection_interval)
            config.memory_threshold_mb = monitoring.get('memory_threshold_mb', config.memory_threshold_mb)
            config.cpu_threshold_percent = monitoring.get('cpu_threshold_percent', config.cpu_threshold_percent)
        
        # Logging
        if 'logging' in self.config_data:
            logging_config = self.config_data['logging']
            config.log_level = logging_config.get('level', config.log_level)
            config.log_format = logging_config.get('format', config.log_format)
            config.log_file = logging_config.get('file', config.log_file)
            config.log_max_size_mb = logging_config.get('max_size_mb', config.log_max_size_mb)
            config.log_backup_count = logging_config.get('backup_count', config.log_backup_count)
        
        # Device
        if 'device' in self.config_data:
            device = self.config_data['device']
            config.device_type = device.get('type', config.device_type)
            config.fallback_to_cpu = device.get('fallback_to_cpu', config.fallback_to_cpu)
        
        # Training
        if 'training' in self.config_data:
            training = self.config_data['training']
            config.batch_size = training.get('batch_size', config.batch_size)
            config.learning_rate = training.get('learning_rate', config.learning_rate)
            config.epochs = training.get('epochs', config.epochs)
            config.validation_split = training.get('validation_split', config.validation_split)
            config.early_stopping_patience = training.get('early_stopping_patience', config.early_stopping_patience)
        
        # Security
        if 'security' in self.config_data:
            security = self.config_data['security']
            config.enable_encryption = security.get('enable_encryption', config.enable_encryption)
            config.api_key_required = security.get('api_key_required', config.api_key_required)
            config.rate_limiting = security.get('rate_limiting', config.rate_limiting)
            config.max_requests_per_minute = security.get('max_requests_per_minute', config.max_requests_per_minute)
        
        # Development
        if 'development' in self.config_data:
            dev = self.config_data['development']
            config.debug_mode = dev.get('debug_mode', config.debug_mode)
            config.verbose_logging = dev.get('verbose_logging', config.verbose_logging)
            config.profile_performance = dev.get('profile_performance', config.profile_performance)
            config.save_debug_info = dev.get('save_debug_info', config.save_debug_info)
        
        # Load domain labels from JSON file
        self._load_domain_labels(config)
        
        return config
    
    def validate_config(self, config: AuraConfig) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Validate numeric ranges
        if config.neuron_count <= 0:
            errors.append("neuron_count must be positive")
        
        if config.features <= 0:
            errors.append("features must be positive")
        
        if config.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        
        if config.retry_attempts < 0:
            errors.append("retry_attempts must be non-negative")
        
        # Validate paths
        if not os.path.exists(config.weights_dir_path):
            logger.warning(f"Weights directory does not exist: {config.weights_dir_path}")
        
        if not os.path.exists(config.models_dir_path):
            logger.warning(f"Models directory does not exist: {config.models_dir_path}")
        
        # Validate device type
        valid_devices = ['cpu', 'cuda', 'cuda']
        if config.device_type not in valid_devices:
            errors.append(f"device_type must be one of {valid_devices}")
        
        # Validate log level
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.log_level.upper() not in valid_log_levels:
            errors.append(f"log_level must be one of {valid_log_levels}")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True


def load_aura_config(config_path: Optional[str] = None) -> AuraConfig:
    """Convenience function to load AURA configuration"""
    loader = ConfigLoader(config_path)
    config = loader.load_config()
    
    if not loader.validate_config(config):
        raise ValueError("Configuration validation failed")
    
    return config


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = load_aura_config()
    
    print(f"System: {config.system_name} v{config.version}")
    print(f"Neuron count: {config.neuron_count}")
    print(f"Features: {config.features}")
    print(f"Device: {config.device_type}")
    print(f"Domains: {config.domains}")
    print(f"Realms: {config.realms}")
