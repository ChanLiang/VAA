<h1 align='center'>
Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning
</h1>

<p align='center'>
<a href=https://arxiv.org/abs/2506.03850"><img src="https://img.shields.io/badge/arXiv-2506.03850-b31b1b.svg" alt="Paper"></a> 
<a href="https://github.com/ChanLiang/VAA"><img src="https://img.shields.io/badge/-Github-grey?logo=github" alt="Github"></a>
<a href="https://icml.cc"><img src="https://img.shields.io/badge/Pub-ICML'25-blue" alt="ICML 2025 Publication" alt="Conf"></a> 
</p>

Implementation for our ICML 2025 paper **[Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning](https://arxiv.org/abs/2506.03850)**. 

**VAA** is a safety alignment method that upweights and reinforces vulnerable data to enhance safety retention during customized fine-tuning.


## Getting Started

### 1. Environment Setup

Create the Conda environment using the provided `environment.yml`:

```bash
conda env create -f environment.yml
```

If you encounter missing dependencies, please install them manually.

### 2. Data and Resources

Prepare experimental datasets using:

```bash
bash script/build_dataset.sh
```
  
Models should be downloaded and placed in the paths as specified in scripts.


### 3. Safety Alignment

Run the following scripts to perform safety alignment with VAA and baselines:

```bash
# model = qwen | llama
bash script/vaa_pipeline_{model}.sh # Supports ERM and vaccine
bash script/repnoise_{model}.sh
bash script/booster_{model}.sh
```

### 4. Harmful Fine-Tuning & Evaluation

Run the following script to perform HFT and evaluate the model on various downstream datasets. 

These steps are also integrated into the main training pipeline.

```bash
bash script/run_all_downstream.sh
```

## Citation

If you find our work helpful, please cite:

```bibtex
@misc{chen2025VAA,
      title={Vulnerability-Aware Alignment: Mitigating Uneven Forgetting in Harmful Fine-Tuning}, 
      author={Liang Chen and Xueting Han and Li Shen and Jing Bai and Kam-Fai Wong},
      year={2025},
      eprint={2506.03850},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.03850}, 
}
```

This implementation is partially builds upon [vaccine](https://github.com/git-disl/Vaccine) framework. We thank the authors for releasing their codebase.

```bibtex
@article{huang2024vaccine,
  title={Vaccine: Perturbation-aware alignment for large language model},
  author={Huang, Tiansheng and Hu, Sihao and Liu, Ling},
  journal={arXiv preprint arXiv:2402.01109},
  year={2024}
}
```
