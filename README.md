
# Discovering physical laws with parallel symbolic enumeration

*Official implementation of PSE with its core PSRN (Parallel Symbolic Regression Network) module*

**Authors:** Kai Ruan, Yilong Xu, Ze-Feng Gao, Yike Guo, Hao Sun, Ji-Rong Wen, Yang Liu

[![Article](https://img.shields.io/badge/Article-Nature_Computational_Science-E30613.svg?style=flat-square&logo=nature)](https://www.nature.com/articles/s43588-025-00904-8)
![License](https://img.shields.io/badge/License-MIT-2196F3.svg)
![AI4Science](https://img.shields.io/badge/AI4Science-8A2BE2)
![Stars](https://img.shields.io/github/stars/intell-sci-comput/PTS)
![Forks](https://img.shields.io/github/forks/intell-sci-comput/PTS)
<!-- [![arXiv](https://img.shields.io/badge/arXiv-2407.04405-b31b1b.svg)](https://arxiv.org/abs/2407.04405) -->


<!-- ![fig1.png](./assets/fig1.png)

![SRbench.png](./assets/SRbench.png) -->

![PSRN.jpg](./assets/PSRN.jpg)


This repository contains the official PyTorch implementation of PSE (Parallel Symbolic Enumeration): A fast and efficient symbolic expression discovery method powered by PSRN (Parallel Symbolic Regression Network). PSRN evaluates millions of symbolic expressions simultaneously on GPU with automated subtree reuse.

## 📥 Installation

**Prerequisite**: Python >=3.9, <=3.12

```bash
pip install psrn
```

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{ruan2025discovering,
  author     = {Ruan, Kai and Xu, Yilong and Gao, Ze-Feng and Guo, Yike and Sun, Hao and Wen, Ji-Rong and Liu, Yang},
  title      = {Discovering physical laws with parallel symbolic enumeration},
  journal    = {Nature Computational Science},
  year       = {2025},
  doi        = {10.1038/s43588-025-00904-8},
  url        = {https://www.nature.com/articles/s43588-025-00904-8}
}
```
