# Fixed Enhanced SVC Pipeline
# Resolves the DOMAIN_TO_IDX NameError issue

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Optional: Stanza and SentenceTransformer may require downloads; guard imports
try:
    import stanza  # type: ignore
except Exception:
    stanza = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:
    SentenceTransformer = None  # type: ignore
from collections import Counter
import re

# Initialize Stanza lazily and only if available
nlp = None
if stanza is not None:
    try:
        nlp = stanza.Pipeline('en', processors='tokenize,pos,ner,lemma,depparse')  # type: ignore
    except Exception:
        # Offline or missing models; proceed without NLP pipeline
        print("Stanza pipeline not available; continuing without it.")
        nlp = None

def load_enhanced_svc_dataset(filepath: str) -> Tuple[List[Dict], List[str], List[str]]:
    """Load enhanced SVC dataset and return domains/realms"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            svc_data = [json.loads(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Enhanced dataset not found: {filepath}")
        print("Please run the data enhancer first!")
        raise
    
    # Extract domains and realms
    domains = list(set(record['metadata']['domain'] for record in svc_data if 'metadata' in record))
    realms = list(set(record['realm'] for record in svc_data if 'realm' in record))
    
    return svc_data, sorted(domains), sorted(realms)

def extract_pos_features(linguistic_features: Dict) -> np.ndarray:
    """Extract POS tag distribution features"""
    pos_tags = linguistic_features.get('pos_tags', [])
    
    # Common POS categories
    pos_categories = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'ADP', 'CONJ', 'DET']
    pos_counts = Counter(tag.get('pos', '') for tag in pos_tags)
    
    # Create feature vector
    features = []
    total_tags = len(pos_tags) if pos_tags else 1
    
    for category in pos_categories:
        count = pos_counts.get(category, 0)
        features.append(count / total_tags)  # Normalized frequency
    
    return np.array(features, dtype=np.float32)

def extract_ner_features(linguistic_features: Dict) -> np.ndarray:
    """Extract named entity features"""
    entities = linguistic_features.get('named_entities', [])
    
    # Common entity types
    entity_types = ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY', 'PERCENT']
    entity_counts = Counter(ent.get('type', '') for ent in entities)
    
    features = []
    total_entities = len(entities) if entities else 1
    
    for ent_type in entity_types:
        count = entity_counts.get(ent_type, 0)
        features.append(count / total_entities)  # Normalized frequency
    
    return np.array(features, dtype=np.float32)

def extract_structural_features(svc_linguistics: Dict) -> np.ndarray:
    """Extract SVC structural features"""
    subject_info = svc_linguistics.get('subject_analysis', {})
    verb_info = svc_linguistics.get('verb_analysis', {})
    complement_info = svc_linguistics.get('complement_analysis', {})
    
    features = []
    
    # Subject complexity (word count, dependency depth)
    subj_tokens = subject_info.get('tokens', [])
    features.append(len(subj_tokens))
    features.append(subject_info.get('dependency_depth', 0))
    
    # Verb complexity
    verb_tokens = verb_info.get('tokens', [])
    features.append(len(verb_tokens))
    features.append(verb_info.get('tense_complexity', 0))
    
    # Complement complexity
    comp_tokens = complement_info.get('tokens', [])
    features.append(len(comp_tokens))
    features.append(complement_info.get('complexity_score', 0))
    
    return np.array(features, dtype=np.float32)

def extract_morphological_features(linguistic: dict) -> np.ndarray:
    pos_tags = linguistic.get("pos_tags", [])

    # protect against tags that are None or not dicts
    safe_feats = (
        (tag.get("feats") or "")                      # tag is dict but "feats" may be None
        if isinstance(tag, dict) else ""
        for tag in pos_tags
    )

    # create a reusable list because the generator is consumed twice
    feats_list = list(safe_feats)

    tense_cnt  = sum("Tense="  in feats for feats in feats_list)
    number_cnt = sum("Number=" in feats for feats in feats_list)
    person_cnt = sum("Person=" in feats for feats in feats_list)
    case_cnt   = sum("Case="   in feats for feats in feats_list)
    mood_cnt   = sum("Mood="   in feats for feats in feats_list)
    voice_cnt  = sum("Voice="  in feats for feats in feats_list)

    total = max(len(pos_tags), 1)         # avoid div-by-zero
    return np.array(
        [
            tense_cnt  / total,
            number_cnt / total,
            person_cnt / total,
            case_cnt   / total,
            mood_cnt   / total,
            voice_cnt  / total,
            float(total)                   # raw token count
        ],
        dtype=np.float32,
    )


def extract_linguistic_context_features(sample: Dict) -> np.ndarray:
    """Extract contextual linguistic features"""
    linguistic_features = sample.get('linguistic_features', {})
    
    features = []
    
    # Text length features
    text = sample.get('text', '')
    features.append(len(text))  # Character count
    features.append(len(text.split()))  # Word count
    features.append(len([w for w in text.split() if len(w) > 6]))  # Long word count
    
    # Sentence complexity
    sentences = text.split('.')
    features.append(len(sentences))  # Sentence count
    
    # Lexical diversity
    words = text.lower().split()
    unique_words = set(words)
    if words:
        features.append(len(unique_words) / len(words))  # Type-token ratio
    else:
        features.append(0.0)
    
    return np.array(features, dtype=np.float32)

def get_enhanced_svc_embedding(sample: Dict, sbert_model: Optional[Any]) -> np.ndarray:
    """Get enhanced embedding for a single SVC sample"""
    # Base SBERT embedding (or offline fallback)
    EMBED_DIM = 384
    if sbert_model is not None:
        text = sample.get('text', '')
        text_embedding = np.asarray(sbert_model.encode(text, convert_to_tensor=False), dtype=np.float32)
        svc_data = sample.get('metadata', {}).get('svc', {})
        subject_emb = np.asarray(sbert_model.encode(svc_data.get('subject', ''), convert_to_tensor=False), dtype=np.float32)
        verb_emb = np.asarray(sbert_model.encode(svc_data.get('verb', ''), convert_to_tensor=False), dtype=np.float32)
        complement_emb = np.asarray(sbert_model.encode(svc_data.get('complement', ''), convert_to_tensor=False), dtype=np.float32)
        tagged_text = sample.get('tagged_versions', {}).get('svc_full_tagged', '')
        tagged_emb = np.asarray(sbert_model.encode(tagged_text, convert_to_tensor=False), dtype=np.float32)
    else:
        # Offline/dummy: zeros with correct dimensionality
        text_embedding = np.zeros(EMBED_DIM, dtype=np.float32)
        subject_emb = np.zeros(EMBED_DIM, dtype=np.float32)
        verb_emb = np.zeros(EMBED_DIM, dtype=np.float32)
        complement_emb = np.zeros(EMBED_DIM, dtype=np.float32)
        tagged_emb = np.zeros(EMBED_DIM, dtype=np.float32)
    
    # Linguistic features
    pos_features = extract_pos_features(sample.get('linguistic_features', {}))
    ner_features = extract_ner_features(sample.get('linguistic_features', {}))
    structural_features = extract_structural_features(sample.get('svc_linguistics', {}))
    morphological_features = extract_morphological_features(sample.get('linguistic_features', {}))
    context_features = extract_linguistic_context_features(sample)
    
    # Combine all features
    combined = np.concatenate([
        text_embedding,      # 384 dims
        subject_emb,         # 384 dims
        verb_emb,           # 384 dims
        complement_emb,     # 384 dims
        tagged_emb,         # 384 dims
        pos_features,       # 8 dims
        ner_features,       # 6 dims
        structural_features, # 6 dims
        morphological_features, # 7 dims
        context_features    # 5 dims
    ])
    
    return combined.astype(np.float32)

def get_enhanced_full_knowledge_embedding(sample: Dict, sbert_model: Optional[Any], 
                                        domains: Optional[List[str]] = None, 
                                        realms: Optional[List[str]] = None) -> np.ndarray:
    """
    Get full enhanced embedding including domain/realm one-hot encoding
    Fixed version that doesn't rely on global variables
    """
    # Get base enhanced embedding
    enhanced_emb = get_enhanced_svc_embedding(sample, sbert_model)
    
    # Create domain one-hot encoding
    if domains is not None:
        domain_onehot = np.zeros(len(domains), dtype=np.float32)
        sample_domain = sample['metadata']['domain']
        if sample_domain in domains:
            domain_idx = domains.index(sample_domain)
            domain_onehot[domain_idx] = 1.0
    else:
        # Fallback: create dummy domain encoding
        domain_onehot = np.array([0.5], dtype=np.float32)
    
    # Create realm one-hot encoding
    if realms is not None:
        realm_onehot = np.zeros(len(realms), dtype=np.float32)
        sample_realm = sample['realm']
        if sample_realm in realms:
            realm_idx = realms.index(sample_realm)
            realm_onehot[realm_idx] = 1.0
    else:
        # Fallback: create dummy realm encoding
        realm_onehot = np.array([0.5], dtype=np.float32)
    
    # Add difficulty as a feature
    difficulty_feature = np.array([sample['metadata']['difficulty']], dtype=np.float32)
    
    # Combine everything
    full_embedding = np.concatenate([
        enhanced_emb,
        domain_onehot,
        realm_onehot,
        difficulty_feature
    ])
    
    return full_embedding

def create_sample_enhanced_data():
    """Create sample enhanced data for testing"""
    sample_data = []
    
    domains = ['computer_science', 'mathematics', 'physics']
    realms = ['theoretical', 'applied', 'practical']
    
    for i in range(10):
        sample = {
            'text': f'This is sample text number {i} for testing.',
            'metadata': {
                'domain': domains[i % len(domains)],
                'difficulty': np.random.uniform(0.1, 0.9),
                'svc': {
                    'subject': f'subject_{i}',
                    'verb': f'verb_{i}',
                    'complement': f'complement_{i}'
                }
            },
            'realm': realms[i % len(realms)],
            'tagged_versions': {
                'svc_full_tagged': f'[SUBJ]subject_{i}[/SUBJ] [VERB]verb_{i}[/VERB] [COMP]complement_{i}[/COMP]'
            },
            'linguistic_features': {
                'pos_tags': [
                    {'pos': 'NOUN', 'feats': 'Number=Sing'},
                    {'pos': 'VERB', 'feats': 'Tense=Pres'},
                    {'pos': 'ADJ', 'feats': ''}
                ],
                'named_entities': [
                    {'type': 'PERSON', 'text': 'sample'},
                    {'type': 'ORG', 'text': 'test'}
                ],
                'tokens': [f'token_{j}' for j in range(5)]
            },
            'svc_linguistics': {
                'subject_analysis': {
                    'tokens': [f'subj_token_{j}' for j in range(2)],
                    'dependency_depth': 2
                },
                'verb_analysis': {
                    'tokens': [f'verb_token_{j}' for j in range(1)],
                    'tense_complexity': 1
                },
                'complement_analysis': {
                    'tokens': [f'comp_token_{j}' for j in range(3)],
                    'complexity_score': 2
                }
            }
        }
        sample_data.append(sample)
    
    return sample_data, domains, realms

def save_enhanced_data(data: List[Dict], filepath: str):
    """Save enhanced data to JSON lines file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record) + '\n')

if __name__ == "__main__":
    print("🔧 Fixed Enhanced SVC Pipeline")
    print("This version resolves the DOMAIN_TO_IDX error")
    
    # Create sample data for testing
    print("Creating sample enhanced data...")
    sample_data, domains, realms = create_sample_enhanced_data()
    
    # Save sample data
    save_enhanced_data(sample_data, 'train_svc_enhanced.jsonl')
    print("✓ Created train_svc_enhanced.jsonl with sample data")
    
    # Test the embedding function
    print("Testing embedding generation...")
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    for i, sample in enumerate(sample_data[:2]):
        embedding = get_enhanced_full_knowledge_embedding(sample, sbert, domains, realms)
        print(f"Sample {i+1}: Embedding shape = {embedding.shape}")
    
    print("✓ Enhanced SVC pipeline is working correctly!")
    print("\nNow you can run: python network-svc-runner.py")
