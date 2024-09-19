# CoMISI: Multimodal Speaker Identification in Diverse Audio-Visual Conditions through Cross-Modal Interaction

This repository is the official implementation of "CoMISI: Multimodal Speaker Identification in Diverse Audio-Visual Conditions through Cross-Modal Interaction." Our work introduces an innovative cross-modal interaction framework for speaker identification, leveraging both audio and visual modalities to enhance identification performance under a wide range of conditions.

## Structure

The repository is organized into major components for baseline comparison approaches and the CoMISI implementation:

- `Approaches/`: Contains scripts and models for the baseline models used for comparisons.
- `CoMISI/`: Hosts the official PyTorch implementation of the CoMISI architecture, including scripts for training and embedding extraction.

### Detailed CoMISI Structure
```
CoMISI/
├── embeddings_extraction/
│ └── <Python files for embeddings extraction>
├── train_GRID/
│ ├── Clean/
│ │ ├── CrossModalRelationModel.py # CMIF fusion
│ │ ├── model_attention.py # Attention-based fusion
│ │ ├── model_audio.py # Audio modality
│ │ ├── model_score_level.py # Score level fusion
│ │ ├── model_visual.py # Visual modality
│ │ └── model.py # Concatenation fusion
│ └── Noise/
│ ├── CrossModalRelationModel.py # CMIF fusion
│ ├── model_attention.py # Attention-based fusion
│ ├── model_audio.py # Audio modality
│ ├── model_score_level.py # Score level fusion
│ ├── model_visual.py # Visual modality
│ └── model.py # Concatenation fusion
├── train_RAVDESS/
│ ├── Clean/
│ │ ├── CrossModalRelationModel.py # CMIF fusion
│ │ ├── model_attention.py # Attention-based fusion
│ │ ├── model_audio.py # Audio modality
│ │ ├── model_score_level.py # Score level fusion
│ │ ├── model_visual.py # Visual modality
│ │ └── model.py # Concatenation fusion
│ └── Noise/
│ ├── CrossModalRelationModel.py # CMIF fusion
│ ├── model_attention.py # Attention-based fusion
│ ├── model_audio.py # Audio modality
│ ├── model_score_level.py # Score level fusion
│ ├── model_visual.py # Visual modality
│ └── model.py # Concatenation fusion
├── train_GRID.py
└── train_RAVDESS.py
```


## Getting Started

## Feature Extraction
For embedding extraction, navigate to the embeddings_extraction directory and run:

python <script_name>.py --options

## Training Models
To train the CoMISI models under various conditions, use the train_GRID.py and train_RAVDESS.py scripts, specifying the dataset and condition as needed:
python train_GRID.py --condition Clean --options
python train_RAVDESS.py --condition Clean --options

## Citation
Please cite our work if you find it useful in your research:


<Insert citation here>

  
## Contributions

Contributions to CoMISI are welcome. Please submit pull requests or open issues to discuss proposed changes.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


  
## Contact
For questions or feedback related to CoMISI, please contact us at yassin.terraf@um6p.ma.
