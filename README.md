# Core code for variable-specific physical-prior integration

This repository provides the core methodological implementation associated with the manuscript:

**Variable-specific physical-prior integration for paired 500 m snow depth and soil moisture reconstruction over the Qinghai–Tibet Plateau**

## Contents

- `sm_PM_demo.py`  
  Core implementation of the soil-moisture physical prior, including FAO-56 reference evapotranspiration, water-availability features, coarse-scale ridge regression, and physical-feature-space residual transfer.

- `sm_prior_integration_ablation_demo.py`  
  Core implementation of the controlled prior-integration pathways used to compare the no-prior baseline, prior-as-predictor, residual anchoring, directional constraint, and adaptive fusion.

- `requirements.txt`  
  Python dependencies required by the supplied scripts.

- `LICENSE`  
  License for the source code.

## Scope

The repository provides the principal computational logic used in the manuscript for methodological transparency and code inspection.

The complete input archive is not redistributed because the full reconstruction relies on large third-party Earth-observation products, reanalysis datasets, and station observations governed by their original licenses and access conditions.

Therefore, the supplied scripts are not intended to reproduce all manuscript maps, figures, or numerical results without the complete input datasets and preprocessing workflow described in the manuscript and Supplementary Information.

## Input data

The complete analysis used passive-microwave soil-moisture data, MODIS products, ERA5/ERA5-Land meteorological forcing, CHIRPS precipitation, terrain variables, SoilGrids properties, freeze–thaw information, and in situ observations.

Users should obtain these datasets from their original repositories and prepare them according to the variable definitions, units, spatial support, and preprocessing rules documented in the manuscript and Supplementary Information.

## Environment

Install the required packages using:

```bash
pip install -r requirements.txt
