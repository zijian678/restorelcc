# RestoreLCC

**Restoring Pruned Large Language Models via Lost Component Compensation**  
*NeurIPS 2025 (Spotlight)*

RestoreLCC recovers the **pruned LLMs** while **preserving their sparsity and inference speed**. The method compensates for important attention heads with components lost during pruning via lightweight restoration to enhance the pruned LLMs' performance.

---

## Table of Contents

- [Preparation](#preparation)
- [Quick Start](#quick-start)  
- [Data](#data)
- [Key Modules](#key-modules)   
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---


## Preparation

1. **Obtain a pruned model.**  
   We prune a base LLM using one of the following methods:
   - **Wanda** — <https://arxiv.org/abs/2306.11695> • code: <https://github.com/locuslab/wanda>  
   - **SparseGPT** — <https://proceedings.mlr.press/v202/frantar23a>  
   - **SlimGPT** — <https://proceedings.neurips.cc/paper_files/paper/2024/hash/c1c44e46358e0fb94dc94ec495a7fb1a-Abstract-Conference.html>

2. **Install the evaluation harness (lm_eval).**  
   Follow the Wanda repo’s instructions to **modify and install** the EleutherAI LM Evaluation Harness:  
   - LM Harness: <https://github.com/EleutherAI/lm-evaluation-harness>  
   - Wanda guide: <https://github.com/locuslab/wanda>

> ℹ️ RestoreLCC expects a pruned model checkpoint as input.

---


## Quick Start

Restore lost components with either the Python script or the provided shell script:

```bash
# Python entrypoint
python RestoreLCC_train.py
```

or

```bash
# Shell wrapper
bash train.sh
```
Key Arguments:
-base_model_name, -pruned_model_path, -num_train_samples, -spec_task

Key Parameters:
-lr (1e-3 to 1e-5), -num_epoch, -use_topk_heads

---

## Data

We include **2k Alpaca** samples due to repository size limits. This subset yields **comparable restoration** in our experiments.

- Full dataset (optional, for stronger coverage): **Stanford Alpaca**  
  <https://github.com/tatsu-lab/stanford_alpaca>

---


## Key Modules

- **Lost-component extraction**  
  `utils/components.py` — Implements the computation of main components lost due to pruning (see paper **Eqs. 3 & 4**).

- **Restoration training (magnitudes + learned component)**  
  `models/modeling_llama` — Lines **258–401** contain the training logic for magnitudes and the learned component.

---


## Citation
Please contact me via feng0119 AT e.ntu.edu.sg if you have any questions.

If you find this work useful, please cite:

```bibtex
@inproceedings{RestoreLCC2025,
  title     = {Restoring Pruned Large Language Models via Lost Component Compensation},
  author    = {Zijian, Feng and Hanzhang, Zhou and Zixiao, Zhu and Tianjiao, Li and Jia Jim Deryl, Chua and Lee Onn, Mak and Gee Wah, Ng and Kezhi Mao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  note      = {NeurIPS 2025 Spotlight}
}
```

---


## Acknowledgements

We thank the authors and maintainers of **Wanda**, **SparseGPT**, **SlimGPT**, and the **EleutherAI LM Evaluation Harness** for their excellent tools.


```
@inproceedings{
      yin2024lofit,
      title={LoFiT: Localized Fine-tuning on {LLM} Representations},
      author={Fangcong Yin and Xi Ye and Greg Durrett},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
      year={2024},
      url={https://openreview.net/forum?id=dfiXFbECSZ}
}

@article{sun2023simple,
  title={A simple and effective pruning approach for large language models},
  author={Sun, Mingjie and Liu, Zhuang and Bair, Anna and Kolter, J Zico},
  journal={arXiv preprint arXiv:2306.11695},
  year={2023}
}

@misc{eval-harness,
  author       = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and Wang, Kevin and Zou, Andy},
  title        = {The Language Model Evaluation Harness},
  month        = 07,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v0.4.3},
  doi          = {10.5281/zenodo.12608602},
  url          = {https://zenodo.org/records/12608602}
}

@inproceedings{NEURIPS2024_c1c44e46,
 author = {Ling, Gui and Wang, Ziyang and Yan, Yuliang and Liu, Qingwen},
 booktitle = {Advances in Neural Information Processing Systems},
 publisher = {Curran Associates, Inc.},
 title = {SlimGPT: Layer-wise Structured Pruning for Large Language Models},
 volume = {37},
 year = {2024}
}

```
