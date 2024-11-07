<div align="center">
    <img alt="BIPOST logo" src="./docs/bipost_logo.PNG" style="width: 1000px;" />
</div>

<hr>

A computationally efficient framework for of LLM post-training involving bi-objective and bilevel fine-tuning.

<div align="center">
    <img alt="BIPOST framework" src="./docs/bipost_framework.PNG" style="height: 280px;" />
</div>

## Introduction

Preference learning (e.g., with DPO or RLHF) and supervised fine-tuning (SFT). The common sequential approach treats each stage independently, but this can cause the model to forget the first objective when optimizing for the second. We introduce BiPOST, a computationally efficient implementation of a bi-objective/bilevel LLM post-training framework that enhances model performance compared to single-objective/single-level LLM training. BiPOST offers a one-stop LLM tuning framework: a pre-trained LLM is optimized for bi-objective in one stage, with following features:

- **Flexibility**: The combination of objectives is flexible and is not limited to the popular case of direct preference optimization (DPO) and supervised fine-tuning (SFT). 
  
- **Data filtering**: In addition to simultaneous bi-objective learning, BiPOST offers an optional bi-level data filtering/selection function that automatically filters out low-quality data given a user-specified golden dataset.

- **Improved post-training performance**: As compared to sequential training, BiPOST offers better performance as it mitigates the forgetting issue.

- **Similar computational cost as sequential training**: Although the forgetting issue is mitigated using simultaneous learning of bi-objective, the computational resource usage is comparable to that of traditional sequential training.


## Example Results


TBD


## Installation

1. Create conda environment

```bash
conda create -n bipost python=3.10
conda activate bipost
```

2. Clone the repository
```bash
git clone https://github.com/Post-LLM/BIPOST.git
```

3. To install BiPOST, Navigate to the top-level of the repo and run
```bash
pip install -e .
```

## Quick Start


TBD

## Acknowledgement

We would like to thank all packages this repo is built on, especially

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): for the vanilla SFT and DPO implementation and their great extention capability.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): for the efficient distributed training functions.
