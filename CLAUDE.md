# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Codebase Overview

This is a scientific data analysis toolkit for **1D atmospheric physics simulations**, specifically designed for analyzing DSMC (Direct Simulation Monte Carlo) data. The focus is on wave propagation, steepening analysis, and heating rate calculations in planetary atmospheres (primarily Mars).

## Core Architecture

### Data Processing Pipeline
1. **Data Loading** (`loader.py`, `loader_compute_rates.py`) - Load NPZ simulation files and compute backgrounds
2. **Analysis** (`helper_funcs.py`, `spectral_analysis_utils.py`) - Wave growth, spectral analysis, peak detection
3. **Specialized Analysis** (`wave_spectral_analysis.py`, `analyze_spectral_energy.py`) - Advanced wave steepening and energy analysis
4. **Visualization** (`simple_wave_viz.py`, comparison scripts) - Plotting and data presentation

### Key Modules
- **`loader_compute_rates.py`** - Enhanced data processor with background computation (standard/experimental methods) and data stitching
- **`helper_funcs.py`** - Core analysis functions including wave growth processing, heating rate calculations, and signal processing
- **`spectral_analysis_utils.py`** - FFT-based spectral analysis, harmonic detection, grid resampling
- **`wave_spectral_analysis.py`** - Multi-phase wave steepening analysis with interactive visualization

## Common Development Commands

### Testing Logic and Data Structures
Use quick Python tests for logic validation:
```bash
python3 -c "import numpy as np; from helper_funcs import process_growth_num_amp; print('Testing wave growth function...')"
```

### Running Analysis Scripts
Most scripts are designed for interactive use in Jupyter notebooks or direct execution:
```bash
python3 wave_spectral_analysis.py
python3 analyze_spectral_energy.py
```

### Data Processing Workflow
```python
# Typical usage pattern
from loader_compute_rates import DSMCDataProcessor
processor = DSMCDataProcessor(data_path)
processor.load_parameters()
processor.load_data()
background = processor.get_background_profiles()
```

## Key Dependencies

- **NumPy/SciPy** - Numerical computations and signal processing
- **Matplotlib/Seaborn** - Visualization
- **Pandas** - Data manipulation
- **Multiprocessing** - Parallel background optimization

## Domain-Specific Constants

Mars atmospheric parameters are hardcoded throughout:
- Radius: 3389.5 km
- Gravity: 3.51 m/s²  
- Temperature: 270 K
- Scale height: ~11.6 km

## Data Formats

- Input: NPZ files containing simulation parameters and temperature/density profiles
- Processing: Structured arrays with altitude grids and perturbation calculations
- Output: Analysis results with reliability scoring and comprehensive metadata