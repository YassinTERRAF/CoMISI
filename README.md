# CoMISI: Multimodal Speaker Identification in Diverse Audio-Visual Conditions through Cross-Modal Interaction

This repository provides the official implementation of the paper:

**"CoMISI: Multimodal Speaker Identification in Diverse Audio-Visual Conditions through Cross-Modal Interaction"**  
Yassin Terraf, Youssef Iraqi  
*Neural Information Processing (ICONIP), 2026*

---

## 🔍 Overview

CoMISI introduces a novel **cross-modal interaction framework** for robust speaker identification by jointly leveraging **audio** and **visual** modalities.

Unlike traditional fusion strategies, CoMISI explicitly models interactions between modalities, improving performance under challenging conditions such as background noise and visual variability.

---

## 📂 Repository Structure

The repository is organized into core components of the CoMISI framework, along with baseline models and dataset-specific training scripts.

```
CoMISI/
├── embeddings_extraction/        # Scripts for feature/embedding extraction
│   └── <embedding scripts>
├── train_GRID/
│   ├── Clean/                   # Models trained on clean GRID dataset
│   │   ├── CrossModalRelationModel.py
│   │   ├── model_attention.py
│   │   ├── model_audio.py
│   │   ├── model_score_level.py
│   │   ├── model_visual.py
│   │   └── model.py
│   └── Noise/                   # Models trained on noisy GRID dataset
│       ├── CrossModalRelationModel.py
│       ├── model_attention.py
│       ├── model_audio.py
│       ├── model_score_level.py
│       ├── model_visual.py
│       └── model.py
├── train_RAVDESS/
│   ├── Clean/                   # Models trained on clean RAVDESS dataset
│   │   ├── CrossModalRelationModel.py
│   │   ├── model_attention.py
│   │   ├── model_audio.py
│   │   ├── model_score_level.py
│   │   ├── model_visual.py
│   │   └── model.py
│   └── Noise/                   # Models trained on noisy RAVDESS dataset
│       ├── CrossModalRelationModel.py
│       ├── model_attention.py
│       ├── model_audio.py
│       ├── model_score_level.py
│       ├── model_visual.py
│       └── model.py
├── train_GRID.py                # Training script for GRID dataset
└── train_RAVDESS.py             # Training script for RAVDESS dataset
```

---

## 🚀 Getting Started

### 1. Feature Extraction

Before training, extract embeddings from the datasets:

```bash
# GRID dataset
python extract_features_grid.py

# RAVDESS dataset
python extract_features_ravdess.py
```

---

### 2. Training

Train models under different conditions:

```bash
# GRID dataset
python train_GRID.py --condition Clean

# RAVDESS dataset
python train_RAVDESS.py --condition Clean
```

You can switch between `Clean` and `Noise` conditions depending on your experiment setup.

---

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@InProceedings{10.1007/978-981-96-6594-5_6,
  author    = {Terraf, Yassin and Iraqi, Youssef},
  title     = {CoMISI: Multimodal Speaker Identification in Diverse Audio-Visual Conditions Through Cross-Modal Interaction},
  booktitle = {Neural Information Processing},
  year      = {2026},
  publisher = {Springer Nature Singapore},
  pages     = {61--77},
  isbn      = {978-981-96-6594-5}
}
```

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests to improve the repository.

---

## 📄 License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

For questions or collaborations:

📧 yassin.terraf@um6p.ma
