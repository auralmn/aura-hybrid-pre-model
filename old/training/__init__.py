"""
Training and learning components
"""

from .aura_trainer import main, EnhancedAuraTrainer, print_directory_results
from .directory_loader import DirectoryLoader, discover_and_preview
from .causal_trainer import AuraCausalHistoryTrainer
from .conv_trainer import AuraConversationTrainer
from .historical_trainer import HistoricalTrainer
from .historical_education_trainer import AuraHistoricalEducationTrainer
from .historical_teacher import HistoricalTeacher
from .movie_emotional_trainer import AuraMovieEmotionalTrainer
from .socratic_trainer import SocraticTrainer
from .span_integration import SPANNeuron, SPANPattern

__all__ = [
    'main',
    'EnhancedAuraTrainer',
    'print_directory_results',
    'DirectoryLoader',
    'discover_and_preview',
    'AuraCausalHistoryTrainer',
    'AuraConversationTrainer',
    'HistoricalTrainer',
    'AuraHistoricalEducationTrainer',
    'HistoricalTeacher',
    'AuraMovieEmotionalTrainer',
    'SocraticTrainer',
    'SPANNeuron',
    'SPANPattern'
]
