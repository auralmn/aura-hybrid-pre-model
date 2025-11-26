#!/usr/bin/env python3
"""
Neuromorphic Brain System Orchestrator

- Integrates EventBus, NeuromorphicProcessor, neuromorphic brain zones
- Optionally wires ContinuousLearningOrchestrator and plasticity engine from src.base.snn_processor
- Safe to import; no background tasks started automatically
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from src.base.events import EventBus
from src.base.snn_processor import NeuromorphicProcessor, ProcessingMode
from src.base.snn_brain_zones import (
    NeuromorphicBrainZone,
    BrainZoneType,
    BrainZoneConfig,
)
from src.base.snn_brain_stats import BrainStats

try:
    from services.continuous_learning import (
        ContinuousLearningOrchestrator,
        create_default_feeds,
        RSSFeedConfig,
        FeedCategory,
        ProcessingPriority,
    )
except Exception:  # optional dependency import
    ContinuousLearningOrchestrator = None  # type: ignore
    create_default_feeds = None  # type: ignore
    RSSFeedConfig = None  # type: ignore
    FeedCategory = None  # type: ignore
    ProcessingPriority = None  # type: ignore


class NeuromorphicBrainSystem:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.event_bus = EventBus()
        self.neuromorphic_processor: Optional[NeuromorphicProcessor] = None
        self.brain_zones: Dict[str, NeuromorphicBrainZone] = {}
        self.brain_stats: Optional[BrainStats] = None
        self.learning_orchestrator: Optional[ContinuousLearningOrchestrator] = None

        self.config: Dict[str, Any] = {
            'd_model': 512,
            'processing_mode': ProcessingMode.NEUROMORPHIC,
            'zones': self._default_zone_configs(),
            'enable_rss': False,  # off by default to not affect tests
            'log_level': logging.INFO,
        }
        if config:
            self.config.update(config)

        self._setup_logging(self.config['log_level'])
        self.log = logging.getLogger(__name__)

    def _setup_logging(self, level: int) -> None:
        if not logging.getLogger().handlers:
            logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    def _default_zone_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            'prefrontal_cortex': {
                'zone_type': BrainZoneType.PREFRONTAL_CORTEX,
                'max_neurons': 512,
                'min_neurons': 256,
                'num_layers': 3,
                'use_spiking': True,
                'd_model': self.config['d_model'] if 'd_model' in self.config else 512,
            },
            'temporal_cortex': {
                'zone_type': BrainZoneType.TEMPORAL_CORTEX,
                'max_neurons': 512,
                'min_neurons': 256,
                'num_layers': 3,
                'use_spiking': True,
                'd_model': self.config['d_model'] if 'd_model' in self.config else 512,
            },
            'hippocampus': {
                'zone_type': BrainZoneType.HIPPOCAMPUS,
                'max_neurons': 384,
                'min_neurons': 192,
                'num_layers': 2,
                'use_spiking': True,
                'd_model': self.config['d_model'] if 'd_model' in self.config else 512,
            },
            'cerebellum': {
                'zone_type': BrainZoneType.CEREBELLUM,
                'max_neurons': 384,
                'min_neurons': 192,
                'num_layers': 2,
                'use_spiking': True,
                'd_model': self.config['d_model'] if 'd_model' in self.config else 512,
            },
        }

    async def initialize(self) -> None:
        # Processor
        self.neuromorphic_processor = NeuromorphicProcessor(
            d_model=int(self.config['d_model']),
            processing_mode=self.config['processing_mode'],
            event_bus=self.event_bus,
        )
        # Zones
        for name, spec in self.config['zones'].items():
            cfg = BrainZoneConfig(
                name=name,
                zone_type=spec['zone_type'],
                max_neurons=spec['max_neurons'],
                min_neurons=spec['min_neurons'],
                num_layers=spec['num_layers'],
                use_spiking=spec['use_spiking'],
                d_model=spec['d_model'],
            )
            zone = NeuromorphicBrainZone(cfg)
            self.brain_zones[name] = zone
        # Wire zones
        self.neuromorphic_processor.set_zone_processors(self.brain_zones)
        # Stats
        self.brain_stats = BrainStats()

        # Optional RSS orchestrator
        if self.config.get('enable_rss') and ContinuousLearningOrchestrator is not None:
            self.learning_orchestrator = ContinuousLearningOrchestrator(self.neuromorphic_processor, self.event_bus)
            if create_default_feeds is not None:
                for feed in create_default_feeds():
                    self.learning_orchestrator.add_feed(feed)

    async def start(self) -> None:
        # Start optional subsystems
        if self.learning_orchestrator is not None:
            asyncio.create_task(self.learning_orchestrator.start())

    async def stop(self) -> None:
        if self.learning_orchestrator is not None:
            await self.learning_orchestrator.stop()

    def process(self, x) -> Tuple[Any, Dict[str, Any]]:
        assert self.neuromorphic_processor is not None
        return self.neuromorphic_processor.process(x)


__all__ = [
    'NeuromorphicBrainSystem',
    'FeedCategory',
    'ProcessingPriority',
    'RSSFeedConfig',
]
