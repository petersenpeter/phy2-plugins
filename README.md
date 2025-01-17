# Plugins to Phy2
These plugins add additional features to Phy2. Originally created for Phy1 by Peter Petersen and made compatible with Phy2 by Thomas Hainmueller. Newly updated implementations made by Mingze Dou.

## Features

### Newly Updated Features 
* **ImprovedISIAnalysis** (`alt+i`): Detects ISI conflicts using multiple metrics
* **StableMahalanobisDetection** (`alt+x`): Outlier detection with interactive visualization
* **ReclusterUMAP** (`alt+k`): Modern reclustering using UMAP and template matching
* **GoodLabelsPlugin**: Improved cluster organization that sorts by quality (good > mua > noise). Provides better workflow organization despite Phy's real-time update limitations.

### Legacy Features
* **Reclustering** (`alt+shift+k`, `alt+shift+t`): KlustaKwik 2.0 based reclustering
* **Mahalanobis Distance** (`alt+shift+x`): Outlier detection (threshold: 16 std)
* **K-means Clustering** (`alt+shift+q`): Basic clustering with adjustable clusters
* **ISI Violation** (`alt+shift+i`): Visualize refractory period violations

## Dependencies
```bash
pip install pandas numpy scipy scikit-learn umap-learn
```

## Installation 
1. Place files in `~/.phy/` plugins directory
2. Install dependencies
3. Copy 'tempdir.py' to "*YourPhyDirectory*/phy/utils"

## Authors
- Original Phy1: Peter Petersen
- Phy2 compatibility: Thomas Hainmueller
- New implementations: Mingze Dou

## How to Cite
[![DOI](https://zenodo.org/badge/126424002.svg)](https://zenodo.org/badge/latestdoi/126424002)
