"""
Continuous Learning Orchestrator (GPU-Native Integration).

Coordinates:
1. RSS/Local Content Ingestion
2. Batching for GPU Processing
3. STDP/Hebbian Updates
4. Endocrine Feedback Loop
"""

import asyncio
import hashlib
import logging
import time
import torch
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

# Import GPU-Native components
from src.core.brain import EnhancedBrain
from src.training.stdp_learning import STDPLearner
from src.encoders.fast_hash_embedder import FastHashEmbedder

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

@dataclass
class RSSFeedConfig:
    url: str
    category: FeedCategory
    priority: int = 3 # 1=Critical, 5=Background
    update_interval_minutes: int = 30
    max_items_per_fetch: int = 10
    enabled: bool = True
    last_updated: Optional[datetime] = None

@dataclass
class ContentItem:
    title: str
    content: str
    url: str
    category: FeedCategory
    timestamp: datetime
    content_hash: str = ""
    processed: bool = False
    
    def __post_init__(self):
        if not self.content_hash:
            key = f"{self.title}:{self.url}"
            self.content_hash = hashlib.sha256(key.encode()).hexdigest()[:16]

class ContinuousLearningOrchestrator:
    """
    Orchestrates the 'Life' of the Brain.
    """
    def __init__(self, brain: EnhancedBrain, event_bus, config_file: Optional[str] = None):
        self.brain = brain
        self.event_bus = event_bus
        self.feeds: Dict[str, RSSFeedConfig] = {}
        self.content_queue: asyncio.Queue[ContentItem] = asyncio.Queue()
        self.is_running = False
        self.log = logging.getLogger(__name__)
        
        # STDP Learner (GPU)
        # We maintain a separate learner for the input stream
        self.stdp = STDPLearner(
            vocab_size=32000, # Match tokenizer
            device=str(brain.device)
        )
        
        # Embedder for raw text
        self.embedder = FastHashEmbedder(dim=brain.d_model)
        
        if config_file:
            self.load_config(config_file)

    async def start(self):
        if self.is_running: return
        self.is_running = True
        self.log.info("🧠 Continuous Learning Started")
        
        self._tasks = [
            asyncio.create_task(self._feed_loop()),
            asyncio.create_task(self._processing_loop()),
            asyncio.create_task(self._homeostasis_loop())
        ]
        
    async def stop(self):
        self.is_running = False
        for t in self._tasks: t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.log.info("🧠 Continuous Learning Stopped")

    async def _feed_loop(self):
        """Fetch feeds periodically."""
        while self.is_running:
            for feed in self.feeds.values():
                if not feed.enabled: continue
                # (Simplified check logic)
                if not feed.last_updated or (datetime.now() - feed.last_updated) > timedelta(minutes=feed.update_interval_minutes):
                    await self._fetch_feed(feed)
            await asyncio.sleep(60)

    async def _fetch_feed(self, feed: RSSFeedConfig):
        # Simplified fetcher (requires aiohttp/feedparser installed)
        try:
            import aiohttp
            import feedparser
            async with aiohttp.ClientSession() as session:
                async with session.get(feed.url, timeout=10) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        parsed = feedparser.parse(text)
                        for entry in parsed.entries[:feed.max_items_per_fetch]:
                            item = ContentItem(
                                title=entry.get('title', ''),
                                content=entry.get('summary', '')[:1000],
                                url=entry.get('link', ''),
                                category=feed.category,
                                timestamp=datetime.now()
                            )
                            await self.content_queue.put(item)
                        feed.last_updated = datetime.now()
                        self.log.info(f"Fetched {len(parsed.entries)} items from {feed.url}")
        except ImportError:
            self.log.warning("Install aiohttp and feedparser for RSS support")
        except Exception as e:
            self.log.error(f"Feed error {feed.url}: {e}")

    async def _processing_loop(self):
        """Process items from queue in batches on GPU."""
        batch_size = 4 # Small batch for online learning
        buffer = []
        
        while self.is_running:
            try:
                # Collect batch
                try:
                    item = await asyncio.wait_for(self.content_queue.get(), timeout=1.0)
                    buffer.append(item)
                except asyncio.TimeoutError:
                    pass
                
                if len(buffer) >= batch_size or (buffer and self.content_queue.empty()):
                    await self._process_batch(buffer)
                    buffer = []
                    
            except Exception as e:
                self.log.error(f"Processing error: {e}")
                await asyncio.sleep(1)

    async def _process_batch(self, items: List[ContentItem]):
        """Run a batch through the brain."""
        # 1. Text -> Embeddings
        texts = [f"{i.title}. {i.content}" for i in items]
        # Mock tokenization (FastHashEmbedder handles strings -> embeddings)
        # For proper STDP, we need token indices.
        # Here we use the embedder to get dense vectors directly for the brain
        embeddings, indices = self.embedder.encode_with_indices(texts) # [Batch, Seq, Dim]
        
        # Move to GPU
        x = embeddings.to(self.brain.device)
        indices = indices.to(self.brain.device)
        
        # 2. Update STDP (Learning from input statistics)
        # Flatten indices: [Batch * Seq]
        flat_indices = indices.view(-1)
        stdp_stats = self.stdp.process_sequence(flat_indices)
        
        # 3. Brain Forward Pass
        # We use no_grad for inference/experience, but could enable grad for online learning
        with torch.no_grad():
            output, info = self.brain.process_input(x, content_context={'source': 'rss'})
        
        # 4. Feedback to Endocrine System
        # "Surprise" = Magnitude of routing change or prediction error (mocked here)
        # High activity in new zones = Surprise
        avg_activity = np.mean(list(info['zone_activities'].values()))
        
        self.brain.update_homeostasis({
            'accuracy': 0.8, # Placeholder: Assume "learning" is happening
            'energy': avg_activity
        })
        
        self.log.info(f"Processed {len(items)} items. Brain Activity: {avg_activity:.3f}")
        self.event_bus.emit('batch_processed', {'count': len(items), 'activity': avg_activity})

    async def _homeostasis_loop(self):
        """Periodic cleanup and memory consolidation."""
        while self.is_running:
            await asyncio.sleep(300) # Every 5 mins
            # Trigger sleep phase if activity is low?
            # For now, just decay memories
            if hasattr(self.brain, 'hippocampus'):
                self.brain.hippocampus.decay_memories()
                self.log.info("💤 Hippocampal decay applied")
    
    def load_config(self, path: str):
        # JSON load logic...
        pass

def create_default_feeds() -> List[RSSFeedConfig]:
    return [
        RSSFeedConfig("https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml", FeedCategory.SCIENCE),
        RSSFeedConfig("https://feeds.feedburner.com/TechCrunch", FeedCategory.TECHNOLOGY),
    ]