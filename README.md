# JWST Spectra Reconstructor

A machine learning framework for reconstructing full Trans-Neptunian Object (TNO) spectra from sparse JWST NIRCam photometry.

## Overview

This project provides tools to bridge the gap between sparse photometric observations and high-resolution spectral models. By leveraging a PCA-based manifold of known TNO spectra (from the DisCoTNO survey), we can reconstruct full reflectance spectra (0.75–5.1 µm) and estimate taxonomic classifications for objects with limited data, such as Neptune Trojans.

### Key Components

*   **Spectral Manifold**: A PCA-based representation of spectral shapes that captures ~99% of variance in TNO reflectance.
*   **Reconstruction Model**: Uses AutoGluon (TabularPredictor) with quantile regression to predict PCA components from filter photometry.
*   **Uncertainty Estimation**: Employs correlated Monte Carlo sampling to generate spectral realizations and 95% confidence intervals.
*   **Taxonomic Classifier**: An AutoGluon-based classifier that identifies object types (e.g., bowl, cliff, methanol-rich) from predicted PCA components.
*   **Filter Integration**: Comprehensive handling of JWST NIRCam filter throughput curves and bandwidths.

---

## Project Structure

```text
.
├── data/
│   ├── processed/          # PCA manifolds and correlation matrices
│   ├── spectra/            # Ground truth spectra for validation
│   └── throughput_curves/  # JWST NIRCam filter transmission data
├── models/
│   ├── 9PCs/               # Pre-trained reconstruction models
│   └── classifier_models/  # Pre-trained taxonomic classifiers
├── notebook/
│   └── reconstroctor.ipynb # End-to-end usage example
├── src/
│   ├── nircam_filters.py   # NIRCam instrument response logic
│   ├── PCA_classifier.py   # Taxonomic classification logic
│   └── spectra_generator.py # Synthetic spectrum generation (PCA + KDE)
└── README.md
```

---

## Installation

### Prerequisites
*   Python 3.10+
*   [AutoGluon](https://autogluon.mx/)

### Setup
Clone the repository and install dependencies:

```bash
pip install numpy pandas matplotlib scipy scikit-learn autogluon seaborn
```

---

## Usage

### 1. Training and Reconstruction
The primary workflow is demonstrated in `notebook/reconstroctor.ipynb`. It follows these steps:
1.  **Load Manifold**: Load the pre-calculated PCA manifold (`data/processed/manifold.pkl`).
2.  **Filter Selection**: Choose a subset of JWST NIRCam filters (e.g., F090W, F115W, F360M).
3.  **Generate Data**: Create synthetic photometry from the spectral manifold to train the reconstructor.
4.  **Fit Model**: Train an AutoGluon model for each PCA component using quantile regression.
5.  **Predict**: Input new photometry to get predicted PCA components and their uncertainties.

### 2. Taxonomy Classification
Use the `TNOpcClassifier` to categorize objects based on their reconstructed spectral signatures:

```python
from PCA_classifier import TNOpcClassifier
classifier = TNOpcClassifier(manifold.pca, model_save_path='models/classifier_models/...')
results = classifier.classify_reconstruction(predicted_pcs)
```

### 3. Visualization
The toolkit includes functions to plot reconstructed spectra against ground truth, overlay filter bands, and visualize the latent PCA space.

---

## Scientific Background

JWST NIRCam provides powerful multi-band photometry, but obtaining high-signal-to-noise spectra for faint TNOs is time-intensive. This project uses **Principal Component Analysis (PCA)** to reduce the dimensionality of complex spectra into a "latent space." **AutoGluon** is then trained to map the non-linear relationship between sparse filter fluxes and these latent components, enabling robust spectral recovery with statistically sound error bars.

---

## Data Sources
*   NIRCam filter data: [STScI JWST Docs](https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters)
*   Spectral models: DisCoTNO Survey / Neptune Trojan validation data.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
