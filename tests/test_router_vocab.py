import pytest
from base.snn_processor import NeuromorphicProcessor
from base.brain_zone_factory import (
    create_prefrontal_cortex,
    create_temporal_cortex,
    create_hippocampus,
    create_cerebellum,
)


def test_content_router_external_lexicon_routing():
    router = NeuromorphicProcessor(d_model=32).content_router
    # Map words to zones
    lex = {
        'joy': 'amygdala',
        'sad': 'amygdala',
        'compute': 'cerebellum',
        'analysis': 'prefrontal_cortex',
        'semantic': 'temporal_cortex',
        'remember': 'hippocampus'
    }
    router.set_external_lexicon(lex)

    zones = set(router.route_text_to_zones("Joy and semantic analysis, please compute and remember."))
    # Should contain matches from external lexicon
    assert 'amygdala' in zones
    assert 'temporal_cortex' in zones
    assert 'prefrontal_cortex' in zones
    assert 'cerebellum' in zones
    assert 'hippocampus' in zones


def test_processor_uses_router_lexicon_for_zone_selection():
    proc = NeuromorphicProcessor(d_model=32)
    zones = {
        'prefrontal_cortex': create_prefrontal_cortex(d_model=32, max_neurons=32),
        'temporal_cortex': create_temporal_cortex(d_model=32, max_neurons=32),
        'hippocampus': create_hippocampus(d_model=32, max_neurons=32),
        'cerebellum': create_cerebellum(d_model=32, max_neurons=32),
    }
    proc.set_zone_processors(zones)
    proc.content_router.set_external_lexicon({'semantic': 'temporal_cortex', 'compute': 'cerebellum'})

    out = proc.process("semantic compute")  # string input path
    assert out.shape[-1] == 32

