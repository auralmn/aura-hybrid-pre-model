"""
AURA Trainer - Main entry point
This module now uses the enhanced trainer for all functionality
"""

from .enhanced_trainer import main, EnhancedAuraTrainer, print_directory_results
from .directory_loader import DirectoryLoader, discover_and_preview

# Re-export all functionality for backward compatibility
__all__ = [
    'main',
    'EnhancedAuraTrainer', 
    'print_directory_results',
    'DirectoryLoader',
    'discover_and_preview'
]

if __name__ == "__main__":
    main()
