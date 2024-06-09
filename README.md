# Multi-Modal MIL Models for Discrete-Time Survival Prediction

This repository contains an adaptation of the PORPOISE, MCAT (Multimodal Co-Attention Transformer), and MOTCat (Multimodal Optimal Transport-based Co-Attention Transformer) models for discrete-time survival prediction tasks using whole-slide images and genetic data.

## Installation

Tested on:
- Ubuntu 22.04
- Nvidia GeForce RTX 4090
- Python 3.10
- PyTorch 2.3

Clone the repository and navigate to the directory.

```bash
git clone https://github.com/ezgiogulmus/MM-SurvModels.git
cd MM-SurvModels
```

Create a conda environment and install required packages.

```bash
conda env create -n mm_surv python=3.10 -y
conda activate mm_surv
pip install --upgrade pip 
pip install -e .
```

## Usage

First, extract patch coordinates and patch-level features using the CLAM library available at [CLAM GitHub](https://github.com/Mahmoodlab/CLAM). Then, run the following command:

```bash
python main.py --split_dir name_of_the_split_folder --model_type mcat --feats_dir path/to/features_directory
```

- `model_type`: Options are 'snn', 'deepset', 'amil', 'mi_fcn', 'mcat', 'motcat', 'porpmmf', 'porpamil'

## Acknowledgement

This code is adapted from the repositories of:
- [PORPOISE](https://github.com/mahmoodlab/PORPOISE)
- [MCAT](https://github.com/mahmoodlab/MCAT)
- [MOTCat](https://github.com/Innse/MOTCat)

## License

This repository is licensed under the [GPLv3 License](./LICENSE). Note that this project is for non-commercial academic use only, in accordance with the licenses of the original models.

## References

Chen, Richard J., et al. "Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images." *Proceedings of the IEEE/CVF International Conference on Computer Vision*, 2021, pp. 4015-4025.

Chen, Richard J., et al. "Pan-cancer integrative histology-genomic analysis via multimodal deep learning." *Cancer Cell*, 2022.

Xu, Yingxue, and Hao Chen. "Multimodal Optimal Transport-based Co-Attention Transformer with Global Structure Consistency for Survival Prediction." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023, pp. 21241-21251.
