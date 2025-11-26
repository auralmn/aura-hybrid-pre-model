#!/usr/bin/env python3
"""
Continuous learning orchestrator integrating RSS feeds with the neuromorphic system.

Notes:
- Optional runtime deps (aiohttp, feedparser) are imported lazily inside methods.
- Designed to be non-intrusive for unit tests; no background loops are started automatically.
- Integrates with base EventBus and NeuromorphicProcessor without introducing example-specific names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

import asyncio
import hashlib
import logging
import time

import torch
import numpy as np

from src.base.events import EventBus
from src.base.snn_processor import NeuromorphicProcessor
from encoders.fast_hash_embedder import FastHashEmbedder
from src.training.stdp_learning import STDPLearner

class FeedCategory(Enum):
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    POLITICS = "politics"
    ECONOMICS = "economics"
    PHILOSOPHY = "philosophy"
    ARTS = "arts"
    SPORTS = "sports"
    HEALTH = "health"
    ENVIRONMENT = "environment"
    EDUCATION = "education"


class ProcessingPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class RSSFeedConfig:
    url: str
    category: FeedCategory
    priority: ProcessingPriority = ProcessingPriority.MEDIUM
    update_interval_minutes: int = 30
    max_items_per_fetch: int = 10
    target_brain_zones: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    enabled: bool = True
    last_updated: Optional[datetime] = None

    def __post_init__(self):
        if not self.target_brain_zones:
            zone_mapping = {
                FeedCategory.SCIENCE: ['prefrontal_cortex', 'temporal_cortex'],
                FeedCategory.TECHNOLOGY: ['prefrontal_cortex', 'parietal_cortex'],
                FeedCategory.POLITICS: ['prefrontal_cortex', 'amygdala'],
                FeedCategory.ECONOMICS: ['prefrontal_cortex', 'basal_ganglia'],
                FeedCategory.PHILOSOPHY: ['prefrontal_cortex', 'temporal_cortex'],
                FeedCategory.ARTS: ['temporal_cortex', 'occipital_cortex'],
                FeedCategory.SPORTS: ['cerebellum', 'basal_ganglia'],
                FeedCategory.HEALTH: ['insular_cortex', 'prefrontal_cortex'],
                FeedCategory.ENVIRONMENT: ['prefrontal_cortex', 'hippocampus'],
                FeedCategory.EDUCATION: ['hippocampus', 'prefrontal_cortex'],
            }
            self.target_brain_zones = zone_mapping.get(self.category, ['prefrontal_cortex'])


@dataclass
class ContentItem:
    title: str
    content: str
    url: str
    source: str
    category: FeedCategory
    timestamp: datetime
    priority: ProcessingPriority
    target_zones: List[str]
    content_hash: str = ""
    processed: bool = False
    processing_time: Optional[float] = None
    brain_response: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.content_hash:
            key = f"{self.title}:{self.content}:{self.url}"
            self.content_hash = hashlib.sha256(key.encode()).hexdigest()[:16]


class ContinuousLearningOrchestrator:
    """Coordinates RSS monitoring and neuromorphic processing.

    Public API:
        - add_feed/remove_feed
        - start/stop (async)
        - load_config/save_config
    """

    def __init__(self, processor: NeuromorphicProcessor, event_bus: EventBus, config_file: Optional[str] = None):
        self.processor = processor
        self.event_bus = event_bus
        self.feeds: Dict[str, RSSFeedConfig] = {}
        self.content_queue: asyncio.Queue[ContentItem] = asyncio.Queue()
        self.processed_items: Dict[str, ContentItem] = {}
        self.is_running = False
        self._tasks: List[asyncio.Task] = []

        # Tunables
        self.stimulation_threshold = 0.3

        # Optional local vocabulary directory for continuous learning
        self.vocab_dir: Optional[str] = None
        self._seen_vocab_files: Dict[str, float] = {}

        # Stats
        self.stats: Dict[str, Any] = {
            'items_processed': 0,
            'feeds_monitored': 0,
            'brain_activations': 0,
            'learning_events': 0,
            'errors': 0,
            'start_time': None,
        }

        self.log = logging.getLogger(__name__)
        if not self.log.handlers:
            logging.basicConfig(level=logging.INFO)

        if config_file:
            self.load_config(config_file)

        # Subscribe minimal events
        self.event_bus.subscribe('neuron_fired', self._on_neuron_fired)

        # Initialize STDP learner for text
        self.stdp_learner = STDPLearner(
            learning_rate_plus=0.01,
            learning_rate_minus=0.012,
            time_window=5
        )
        # Persistent embedder to avoid recreation
        self.embedder = FastHashEmbedder(dim=self.processor.d_model)

    def _save_homeostasis_all(self) -> None:
        """Persist homeostasis state for all neuromorphic zones if available."""
        try:
            import os
            os.makedirs('brain_states', exist_ok=True)
            for zname, zone in (self.processor.zone_processors or {}).items():
                neu = getattr(zone, 'neuromorphic_processor', None)
                if neu and hasattr(neu, 'save_homeostasis_state'):
                    neu.save_homeostasis_state(os.path.join('brain_states', f'{zname}_homeostasis.json'))
        except Exception as e:
            self._err(f"save homeostasis: {e}")

    # ---------------- API ----------------
    def add_feed(self, feed_config: RSSFeedConfig) -> None:
        feed_id = hashlib.sha256(feed_config.url.encode()).hexdigest()[:8]
        self.feeds[feed_id] = feed_config
        self.stats['feeds_monitored'] = len(self.feeds)

    def remove_feed(self, url: str) -> bool:
        feed_id = hashlib.sha256(url.encode()).hexdigest()[:8]
        if feed_id in self.feeds:
            del self.feeds[feed_id]
            self.stats['feeds_monitored'] = len(self.feeds)
            return True
        return False

    def set_vocab_dir(self, path: str) -> None:
        """Enable local text ingestion from a directory (recursively scans for *.txt)."""
        import os
        if path and os.path.isdir(path):
            self.vocab_dir = path
            self.log.info(f"Local vocab directory set: {path}")
        else:
            self.log.warning(f"Invalid vocab directory: {path}")

    async def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        # Auto-enable vocab_src directory if present and not set explicitly
        try:
            import os
            if self.vocab_dir is None and os.path.isdir('vocab_src'):
                self.set_vocab_dir('vocab_src')
        except Exception:
            pass

        self._tasks = [
            asyncio.create_task(self._loop_feeds()),
            asyncio.create_task(self._loop_process_queue()),
            asyncio.create_task(self._loop_background())
        ]
        if self.vocab_dir:
            self._tasks.append(asyncio.create_task(self._loop_vocab_dir()))
        # Immediate save so brain_states reflects current biases
        self._save_homeostasis_all()
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        if not self.is_running:
            return
        self.is_running = False
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        # Save on shutdown
        self._save_homeostasis_all()

    # ------------- Loops -------------
    async def _loop_feeds(self) -> None:
        self.log.info("Starting RSS monitoring loop")
        while self.is_running:
            try:
                for feed_id, feed in self.feeds.items():
                    if not feed.enabled:
                        continue
                    if self._should_update(feed):
                        await self._fetch_feed(feed_id, feed)
                await asyncio.sleep(60)
            except Exception as e:
                self._err(f"feed loop: {e}")
                await asyncio.sleep(300)

    async def _loop_process_queue(self) -> None:
        self.log.info("Starting content processing loop")
        while self.is_running:
            try:
                try:
                    item = await asyncio.wait_for(self.content_queue.get(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                t0 = time.time()
                resp = await self._process_item(item)
                item.processed = True
                item.processing_time = time.time() - t0
                item.brain_response = resp
                self.event_bus.emit('content_processed', {
                    'content_hash': item.content_hash,
                    'category': item.category.value,
                    'processing_time': item.processing_time,
                    'brain_response': resp,
                })
                self.stats['items_processed'] += 1
                # Periodic save every 10 items
                if (self.stats['items_processed'] % 10) == 0:
                    self._save_homeostasis_all()
            except Exception as e:
                self._err(f"process loop: {e}")

    async def _loop_background(self) -> None:
        self.log.info("Starting background stimulation loop")
        while self.is_running:
            try:
                await self._background_activity()
                # Periodically persist homeostatic state per zone
                try:
                    import os
                    os.makedirs('brain_states', exist_ok=True)
                    for zname, zone in (self.processor.zone_processors or {}).items():
                        neu = getattr(zone, 'neuromorphic_processor', None)
                        if neu and hasattr(neu, 'save_homeostasis_state'):
                            neu.save_homeostasis_state(os.path.join('brain_states', f'{zname}_homeostasis.json'))
                except Exception:
                    pass
                await asyncio.sleep(30)
            except Exception as e:
                self._err(f"background loop: {e}")
                await asyncio.sleep(60)

    async def _loop_vocab_dir(self) -> None:
        """Continuously scan vocab_dir for new *.txt files and enqueue as learning items."""
        self.log.info("Starting local vocab ingestion loop")
        import os
        while self.is_running and self.vocab_dir:
            try:
                queued = 0
                for root, _dirs, files in os.walk(self.vocab_dir):
                    for name in files:
                        if not name.lower().endswith('.txt'):
                            continue
                        fpath = os.path.join(root, name)
                        try:
                            mtime = os.path.getmtime(fpath)
                            key = fpath
                            if key in self._seen_vocab_files and self._seen_vocab_files[key] >= mtime:
                                continue
                            # Read a bounded amount of text
                            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read(8000)
                            title = os.path.splitext(os.path.basename(fpath))[0]
                            item = ContentItem(
                                title=title,
                                content=text,
                                url=f"file://{fpath}",
                                source='local_vocab',
                                category=FeedCategory.EDUCATION,
                                timestamp=datetime.now(),
                                priority=ProcessingPriority.MEDIUM,
                                target_zones=['temporal_cortex', 'prefrontal_cortex']
                            )
                            await self.content_queue.put(item)
                            self._seen_vocab_files[key] = mtime
                            queued += 1
                            if queued >= 50:
                                break
                        except Exception as e:
                            self._err(f"vocab enqueue {fpath}: {e}")
                    if queued >= 50:
                        break
                if queued:
                    self.log.info(f"Queued {queued} local vocab items")
                await asyncio.sleep(30)
            except Exception as e:
                self._err(f"vocab loop: {e}")
                await asyncio.sleep(60)

    # ------------- Helpers -------------
    def _should_update(self, feed: RSSFeedConfig) -> bool:
        if feed.last_updated is None:
            return True
        return (datetime.now() - feed.last_updated) >= timedelta(minutes=feed.update_interval_minutes)

    async def _fetch_feed(self, feed_id: str, feed: RSSFeedConfig) -> None:
        try:
            import aiohttp  # lazy import
            import feedparser  # lazy import
            async with aiohttp.ClientSession(trust_env=True) as session:
                async with session.get(feed.url, timeout=30) as resp:
                    if resp.status != 200:
                        return
                    text = await resp.text()
            parsed = feedparser.parse(text)
            new_count = 0
            for entry in parsed.entries[:feed.max_items_per_fetch]:
                item = self._to_item(entry, feed)
                if item.content_hash in self.processed_items:
                    continue
                await self.content_queue.put(item)
                self.processed_items[item.content_hash] = item
                new_count += 1
            feed.last_updated = datetime.now()
            if new_count:
                self.log.info(f"Queued {new_count} items from {feed.category.value}")
        except Exception as e:
            self._err(f"fetch {feed.url}: {e}")

    def _to_item(self, entry: Any, feed: RSSFeedConfig) -> ContentItem:
        title = entry.get('title', '')
        content = entry.get('summary', '') or entry.get('description', '')
        url = entry.get('link', '')
        # strip HTML
        try:
            import re
            content = re.sub('<[^<]+?>', '', content)
        except Exception:
            pass
        return ContentItem(
            title=title,
            content=content,
            url=url,
            source=feed.url,
            category=feed.category,
            timestamp=datetime.now(),
            priority=feed.priority,
            target_zones=feed.target_brain_zones.copy(),
        )

    async def _process_item(self, item: ContentItem) -> Dict[str, Any]:
        text = f"{item.title}. {item.content}"
        
        # Encode text and get indices for STDP
        # We use the persistent embedder
        x_emb, token_indices = self.embedder.encode_with_indices(text)
        x = x_emb.view(1, 1, -1)
        
        # Apply STDP learning
        stdp_stats = self.stdp_learner.process_sequence(token_indices)
        if stdp_stats.get('updates', 0) > 0:
            # Log significant learning events
            if stdp_stats['updates'] > 10:
                self.log.debug(f"STDP learning: {stdp_stats['updates']} updates for '{item.title}'")
                self.stats['learning_events'] += stdp_stats['updates']

        # Build plan with our text and enforce top_k
        plan = self.processor.build_plan(text, top_k=len(item.target_zones))
        # Fallback: if router didn't select any of the target zones, process targets directly
        selected = {z for z, _ in plan}
        targets = [z for z in item.target_zones if z in self.processor.zone_processors]
        if not selected.intersection(targets) and targets:
            w = 1.0 / len(targets)
            plan = [(z, w) for z in targets]
        zone_responses: Dict[str, Any] = {}
        total_activation = 0.0
        for zone_name, w in plan:
            if zone_name not in item.target_zones:
                continue
            if zone_name not in self.processor.zone_processors:
                continue
            zone = self.processor.zone_processors[zone_name]
            try:
                with torch.no_grad():
                    out, activity = zone(x, context={'source': 'rss'})
                act = float(activity.get('avg_firing_rate', 0.0))
                act = max(0.0, min(act, 0.999))
                spk = float(act * out.numel()) if out.numel() > 0 else 0.0
            except Exception as e:
                zone_responses[zone_name] = {'error': str(e)}
                continue
            total_activation += float(w)
            zone_responses[zone_name] = {
                'activation_level': act,
                'spike_count': spk,
                'zone_healthy': act > 0.01,
                'learning_occurred': spk > 0,
            }
            if act > self.stimulation_threshold:
                print(f"Neuron {zone_name} fired with activation level {act} {spk}")
                self.event_bus.emit('neuron_fired', {
                    'zone': zone_name,
                    'firing_rate': act,
                    'timestamp': time.time(),
                })
        return {
            'processing_plan': plan,
            'zone_responses': zone_responses,
            'total_activation': total_activation,
            'stdp_updates': stdp_stats.get('updates', 0)
        }

    # _encode_text removed as we use self.embedder directly now

    async def _background_activity(self) -> None:
        if not self.processor.zone_processors:
            return
        import random
        names = list(self.processor.zone_processors.keys())
        k = min(2, len(names))
        for zone_name in random.sample(names, k):
            zone = self.processor.zone_processors[zone_name]
            stim = torch.randn(1, 1, self.processor.d_model) * 0.1
            try:
                with torch.no_grad():
                    _, activity = zone(stim, context={'source': 'background'})
                act = float(activity.get('avg_firing_rate', 0.0))
                act = max(0.0, min(act, 0.999))
                if act > 0.01:
                    self.event_bus.emit('background_activity', {
                        'zone': zone_name,
                        'activation': act,
                        'timestamp': time.time(),
                    })
            except Exception:
                continue

    def _on_neuron_fired(self, event) -> None:
        # Hook for future analytics; keep minimal
        pass

    def load_config(self, path: str) -> None:
        try:
            import json
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            for fd in data.get('feeds', []):
                cfg = RSSFeedConfig(
                    url=fd['url'],
                    category=FeedCategory(fd['category']),
                    priority=ProcessingPriority(fd.get('priority', 3)),
                    update_interval_minutes=int(fd.get('update_interval_minutes', 30)),
                    keywords=list(fd.get('keywords', [])),
                )
                self.add_feed(cfg)
        except Exception as e:
            self._err(f"load_config: {e}")

    def save_config(self, path: str) -> None:
        try:
            import json
            data = {
                'feeds': [
                    {
                        'url': f.url,
                        'category': f.category.value,
                        'priority': f.priority.value,
                        'update_interval_minutes': f.update_interval_minutes,
                        'keywords': f.keywords,
                        'enabled': f.enabled,
                    } for f in self.feeds.values()
                ]
            }
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self._err(f"save_config: {e}")

    def _err(self, msg: str) -> None:
        self.stats['errors'] += 1
        self.log.error(msg)


def create_default_feeds() -> List[RSSFeedConfig]:
    return [
        RSSFeedConfig(url="https://rss.cnn.com/rss/edition.rss", category=FeedCategory.POLITICS, priority=ProcessingPriority.MEDIUM),
        RSSFeedConfig(url="https://feeds.feedburner.com/TechCrunch", category=FeedCategory.TECHNOLOGY, priority=ProcessingPriority.HIGH),
        RSSFeedConfig(url="https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml", category=FeedCategory.SCIENCE, priority=ProcessingPriority.CRITICAL),
    ]


