import numpy as np
from typing import Dict, Any, List, Optional, Union

PERIODS = [
    "Neolithic",
    "Ancient",
    "Medieval",
    "Early_Modern",
    "Industrial",
    "Modern",
]


def get_historical_period(year: int) -> str:
    if year < -3000:
        return "Neolithic"
    elif year < 500:
        return "Ancient"
    elif year < 1500:
        return "Medieval"
    elif year < 1800:
        return "Early_Modern"
    elif year < 1900:
        return "Industrial"
    else:
        return "Modern"


def parse_year(y: Union[str, int, float, None]) -> Optional[int]:
    """Parse a year that may be in forms like -551, '551BCE', '551 BCE', '551 CE', 'AD 551'.
    Returns an integer year where BCE/BC are negative, CE/AD positive.
    """
    if y is None:
        return None
    if isinstance(y, (int, float)):
        try:
            return int(y)
        except Exception:
            return None
    if isinstance(y, str):
        s = y.strip()
        # Fast path: plain signed integer string
        try:
            return int(s)
        except Exception:
            pass
        import re
        m = re.search(r"(-?\d{1,4})(?:\.\d+)?\s*(BCE|BC|CE|AD)?", s, flags=re.IGNORECASE)
        if not m:
            return None
        num = int(m.group(1))
        era = (m.group(2) or '').upper()
        if era in ('BCE', 'BC') and num > 0:
            num = -num
        # AD/CE unchanged
        return num
    return None


def encode_categorical(label: str, vocab: List[str]) -> np.ndarray:
    onehot = np.zeros(len(vocab), dtype=np.float32)
    if label in vocab:
        onehot[vocab.index(label)] = 1.0
    return onehot


def encode_temporal_features(year_start: int, year_end: int | None = None) -> np.ndarray:
    # Normalize absolute time to [0,1] over [-9000, 2024]
    normalized_year = (year_start + 9000) / 11024.0
    millennium = ((year_start + 9000) // 1000) / 11.0  # scaled
    century = ((year_start + 9000) // 100) / 110.0     # scaled
    duration = 0.0 if year_end is None else float(abs(year_end - year_start)) / 11024.0
    century_idx = int((year_start + 9000) // 100)
    century_cycle = float(np.sin(2 * np.pi * (century_idx % 10) / 10.0))
    period = get_historical_period(year_start)
    period_oh = encode_categorical(period, PERIODS)
    base = np.array([
        float(normalized_year),
        float(millennium),
        float(century),
        float(duration),
        float(century_cycle),
    ], dtype=np.float32)
    return np.concatenate([base, period_oh]).astype(np.float32)


# Simple, low-dimensional encoders for other facets
def encode_geographical_context(region: str) -> np.ndarray:
    # Map regions to a small one-hot set
    regions = [
        "Africa", "Middle_East", "Asia", "Europe", "Americas", "Oceania", "Global",
    ]
    return encode_categorical(str(region or "Global"), regions)


def encode_historical_actors(actors: List[str] | None) -> np.ndarray:
    # Bag-of-actors hashed into 32 dims
    H = 32
    v = np.zeros(H, dtype=np.float32)
    if not actors:
        return v
    for a in actors:
        h = hash(a) % H
        v[h] += 1.0
    if np.max(v) > 0:
        v = v / np.max(v)
    return v


def encode_category_interactions(categories: List[str] | None) -> np.ndarray:
    # Multi-hot for a small taxonomy
    tax = [
        "war", "trade", "religion", "technology", "culture", "politics", "science", "migration",
    ]
    if not categories:
        return np.zeros(len(tax), dtype=np.float32)
    v = np.zeros(len(tax), dtype=np.float32)
    for c in categories:
        if c in tax:
            v[tax.index(c)] = 1.0
    return v


def encode_global_context(year: int) -> np.ndarray:
    # Simple cyclical and trend features
    t = (year + 9000) / 11024.0
    return np.array([
        np.sin(2 * np.pi * t),
        np.cos(2 * np.pi * t),
        t,
        t ** 2,
    ], dtype=np.float32)


def encode_development_stage(region: str, year: int) -> np.ndarray:
    # Crude development proxy based on period
    period = get_historical_period(year)
    scale = {
        "Neolithic": 0.2,
        "Ancient": 0.4,
        "Medieval": 0.5,
        "Early_Modern": 0.7,
        "Industrial": 0.85,
        "Modern": 1.0,
    }[period]
    return np.array([scale, scale ** 2], dtype=np.float32)


def get_enhanced_historical_embedding(event: Dict[str, Any], sbert_model) -> np.ndarray:
    # Textual embeddings (optional SBERT)
    EMBED = 384
    if sbert_model is not None:
        text_emb = np.asarray(sbert_model.encode(event.get('text', ''), convert_to_tensor=False), dtype=np.float32)
        title_emb = np.asarray(sbert_model.encode(event.get('title', ''), convert_to_tensor=False), dtype=np.float32)
        impact_emb = np.asarray(sbert_model.encode(event.get('impact', ''), convert_to_tensor=False), dtype=np.float32)
    else:
        text_emb = np.zeros(EMBED, dtype=np.float32)
        title_emb = np.zeros(EMBED, dtype=np.float32)
        impact_emb = np.zeros(EMBED, dtype=np.float32)

    ys = parse_year(event.get('year_start', 0))
    ye_raw = event.get('year_end')
    ye = parse_year(ye_raw) if ye_raw is not None else None
    temporal_features = encode_temporal_features(int(ys or 0), ye)
    region_features = encode_geographical_context(str(event.get('region', 'Global')))
    actor_features = encode_historical_actors(event.get('actors'))
    category_features = encode_category_interactions(event.get('categories'))
    contextual_features = encode_global_context(int(event.get('year_start', 0)))
    development_features = encode_development_stage(str(event.get('region', 'Global')), int(event.get('year_start', 0)))

    return np.concatenate([
        text_emb, title_emb, impact_emb,
        temporal_features,
        region_features,
        actor_features,
        category_features,
        contextual_features,
        development_features,
    ]).astype(np.float32)
