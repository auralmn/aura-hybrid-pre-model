# Prosody Analysis Scripts

## Running the Scripts

Due to import path issues, run the scripts using Python's module mode:

```bash
# From project root
python -m pytest tests/core/language_zone/test_prosody_integration.py -v

# Or create a simple test script
```

## Alternative: Use the Jupyter Notebook

The Jupyter notebook has all visualizations included and doesn't have import issues:

```bash
jupyter notebook notebooks/snn_language_zone_demo.ipynb
```

The notebook includes:
- Prosody extraction visualization
- GIF neuron modulation comparison  
- Attention preset comparison
- GoEmotions emotion classification demo
- MNIST baseline

## Quick Fix

If you want to run the standalone scripts, temporarily add to your PYTHONPATH:

```bash
# Windows PowerShell
$env:PYTHONPATH = "c:\Users\nickn\OneDrive\Desktop\aura_clean"
python scripts/visualize_prosody.py

# Or use pytest to run the integration tests which have the same visualizations:
pytest tests/core/language_zone/test_prosody_integration.py::TestProsodyAttention -v -s
```
