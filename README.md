# Reproducible Deep Learning Software for Efficient Computer Vision 
PyTorch implementation and pre-trained models of TRUNK for the EMNIST, CIFAR10, and SVHN datasets. For details, see the papers: [Reproducible Deep Learning Software for Efficient Computer Vision](https://arxiv.org/pdf/2505.03165).

Recently, the field of deep learning has witnessed significant breakthroughs, spanning various applications and fundamentally transforming technological capabilities. However, alongside these advancements, there have been increasing concerns about reproducing the results of these deep learning methods. The difficulty of reproducibility may arise due to several reasons, including having differences from the original execution environment, missing or incompatible software libraries, proprietary data and source code, lack of transparency in the data-processing and training pipeline, and the stochastic nature in some software. A study conducted by the Nature journal reveals that more than 70% of researchers failed to reproduce other researcher's experiments and over 50% failed to reproduce their own experiments. Given the critical role that deep learning plays in many applications, irreproducibility poses significant challenges for researchers and practitioners. To address these concerns, this paper presents a systematic approach at analyzing and improving the reproducibility of deep learning models by demonstrating these guidelines using a case study. We illustrate the patterns and anti-patterns involved with these guidelines for improving the reproducibility of deep learning models. These guidelines encompass establishing a robust methodology to replicate the original software environment, implementing end-to-end training and testing algorithms, disclosing architectural designs, enhancing transparency in data processing and training pipelines, and our primary contribution: conducting a sensitivity analysis to understand the model's performance across diverse conditions. By implementing these strategies, we aim to bridge the gap between research and practice, so that innovations in deep learning can be effectively reproduced and deployed.  

This github repository contains all the code used to generate the results mentioned or displayed in the aforementioned paper, divided into two directories:
- LPCV Background: Contains the code used to introduce LPCV techniques mentioned in the background section of the paper
- TRUNK: Contains the code for training and testing the TRUNK network, as well as the pre-trained models

## Installation
The implementations provided in this repository requires a python version 3.9.18, a PyTorch version 2.3 with CUDA 12.1, and other 3rd party packages. To setup the required dependencies to reproduce the results of this repository, follow these instructions

[conda][1] (**Recommended**) - clone the repository and then activate the mnn conda environment using the provided environment definition
```bash
conda env create -f environment.yml
conda activate trunk
```

[pip][2] - clone the repository and then install the dependencies using the provided command
```bash
pip install -r requirements.txt
```

[1]: https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html
[2]: https://pip.pypa.io/en/stable/getting-started/
