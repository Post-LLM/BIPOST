<div align="center">
    <img alt="BIPOST logo" src="./docs/bipost_logo.png" style="height: 110px;" />
</div>

<hr>

A computationally efficient framework for of LLM post-training involving bi-objective and bilevel fine-tuning.

<div align="center">
    <img alt="BIPOST framework" src="./docs/bipost_framework.PNG" style="height: 280px;" />
</div>
<br/><br/>

## Introduction

We introduce BiPOST, a computationally efficient implementation of a bi-objective/bilevel LLM post-training framework that enhances model performance compared to single-objective/single-level LLM training. BiPOST offers a one-stop LLM tuning framework: a pre-trained LLM is optimized for bi-objective in one stage, with comparable memory and runtime cost to sequentially optimizing each objective. The combination of objectives is flexible, and is not limited to the popular case of direct preference optimization (DPO) and supervised fine-tuning (SFT). In addition, BiPOST offers an optional data filtering/selection function that automatically filters out low-quality data given a user-specified golden dataset. As compared to sequential training, BiPOST offers better performance as it mitigates the forgetting issue while keeping the computational complexity under control.


- **Improved post-training performance**: 

- **Similar computational cost as sequential training**:

## Installation

Create conda environment

```bash
conda create -n bipost python=3.10
conda activate bipost
```

To install bipost, navigate to the top-level of the repo and run
```bash
pip install -e .
```

## Acknowledgement

We would like to thank all packages this repo is built on, especially

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): for the vanilla SFT and DPO implementation and their great extention capability.
- [DeepSpeed](https://github.com/microsoft/DeepSpeed): for the efficient distributed training functions.
