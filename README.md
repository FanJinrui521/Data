# Variable-specific physical-prior integration for paired 500 m snow depth and soil moisture reconstruction over the Qinghai–Tibet Plateau

This repository provides the core implementation and a lightweight demonstration dataset associated with the manuscript:

**Variable-specific physical-prior integration for paired 500 m snow depth and soil moisture reconstruction over the Qinghai–Tibet Plateau**

Authors: Jinrui Fan, Huan Yu, Ning Li, Ruili Gu, Jie Meng, Xu Ouyang, Qing Wang, and Shuaifeng Peng.

## Overview

This study develops a variable-specific framework for integrating physical priors into paired 500 m snow depth (SD) and soil moisture (SM) reconstruction over the Qinghai–Tibet Plateau.

The framework evaluates six reference and candidate pathways:

- M0: coarse-reference field
- P0: physical-prior field
- M1: no-prior baseline
- M2: prior as predictor
- M3: residual anchoring
- M4: directional constraint
- M5: adaptive fusion

The final reconstruction uses M5 adaptive fusion for SD and M2 prior-as-predictor for SM.

This repository contains the core code required to demonstrate:

1. SD and SM physical-prior integration;
2. M0/P0/M1–M5 pathway comparison;
3. performance-weighted ensemble reconstruction;
4. parent-scale residual-consistency adjustment;
5. station-based and raster-level evaluation.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── code/
│   └── SM_PM_demo.py
├── paper_demo/
│   ├── README_data.md
│   ├── input/
│   │   ├── coarse_sd/
│   │   ├── coarse_sm/
│   │   ├── predictors/
│   │   └── stations/
│   └── expected_output/
└── output/
