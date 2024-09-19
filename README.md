
# CoMISI: Multimodal Speaker Identification in Diverse Audio-Visual Conditions through Cross-Modal Interaction

This repository contains the official implementation of the paper **"CoMISI: Multimodal Speaker Identification in Diverse Audio-Visual Conditions through Cross-Modal Interaction"** by Yassin Terraf and Youssef Iraqi, presented at the *International Conference on Neural Information Processing (ICONIP), 2024* (Accepted for publication). 

Our work introduces a novel **cross-modal interaction framework** that leverages both audio and visual modalities for speaker identification. CoMISI enhances speaker identification performance under a variety of challenging conditions by fusing both modalities in a robust and adaptive manner.

## Table of Contents
- [Repository Structure](#repository-structure)
  - [Detailed Directory Structure](#detailed-directory-structure)
- [Getting Started](#getting-started)
  - [Feature Extraction](#feature-extraction)
  - [Training Models](#training-models)
- [Citation](#citation)
- [Contributions](#contributions)
- [License](#license)
- [Contact](#contact)

## Repository Structure

The repository is organized into key components for the CoMISI framework, alongside baseline comparison models and scripts for dataset processing:

- `CoMISI/`: Contains the core PyTorch implementation of CoMISI, including model training scripts and embedding extraction.

### Detailed Directory Structure

```bash
CoMISI/
├── embeddings_extraction/        # Scripts for extracting embeddings
│   └── <Embedding extraction scripts>
├── train_GRID/
│   ├── Clean/                    # Training models on clean GRID dataset
│   │   ├── CrossModalRelationModel.py    # CMIF fusion model
│   │   ├── model_attention.py            # Attention-based fusion
│   │   ├── model_audio.py                # Audio modality model
│   │   ├── model_score_level.py          # Score-level fusion model
│   │   ├── model_visual.py               # Visual modality model
│   │   └── model.py                      # Concatenation-based fusion model
│   └── Noise/                    # Training models on noisy GRID dataset
│       ├── CrossModalRelationModel.py
│       ├── model_attention.py
│       ├── model_audio.py
│       ├── model_score_level.py
│       ├── model_visual.py
│       └── model.py
├── train_RAVDESS/
│   ├── Clean/                    # Training models on clean RAVDESS dataset
│   │   ├── CrossModalRelationModel.py
│   │   ├── model_attention.py
│   │   ├── model_audio.py
│   │   ├── model_score_level.py
│   │   ├── model_visual.py
│   │   └── model.py
│   └── Noise/                    # Training models on noisy RAVDESS dataset
│       ├── CrossModalRelationModel.py
│       ├── model_attention.py
│       ├── model_audio.py
│       ├── model_score_level.py
│       ├── model_visual.py
│       └── model.py
├── train_GRID.py                 # Script to train on GRID dataset
└── train_RAVDESS.py              # Script to train on RAVDESS dataset
```

## Getting Started

### Feature Extraction

To extract embeddings from the datasets, navigate to the `feature_extraction` directory and run the appropriate script for your needs:

- Run the following for the GRID dataset:
  
  ```bash
  python extract_features_grid.py
  ```

- Run the following for the RAVDESS dataset:
  
  ```bash
  python extract_features_ravdess.py
  ```

### Training Models

To train the CoMISI models under various conditions, use the `train_GRID.py` and `train_RAVDESS.py` scripts, specifying the dataset and condition as needed:

- For the GRID dataset:

  ```bash
  python train_GRID.py --condition Clean --options
  ```

- For the RAVDESS dataset:

  ```bash
  python train_RAVDESS.py --condition Clean --options
  ```

## Citation

If you find our work useful in your research, please consider citing:

**Yassin Terraf, Youssef Iraqi.** "CoMISI: Multimodal Speaker Identification in Diverse Audio-Visual Conditions through Cross-Modal Interaction." *Proceedings of the International Conference on Neural Information Processing (ICONIP)*, 2024. (Accepted for publication).

## Contributions

Contributions to CoMISI are welcome. Please submit pull requests or open issues to discuss proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback related to CoMISI, please contact us at yassin.terraf@um6p.ma.
