# Obesity Analysis: Hereditary vs. Lifestyle Predictors

## Project Overview
This project investigates the relative influence of hereditary factors versus lifestyle behaviors on obesity levels.
We aim to answer two primary research questions:

1. **Relative Strength:** Is obesity more strongly predicted by family history or by lifestyle choices (diet, physical activity)?
2. **Compounding Effect:** Do hereditary factors and lifestyle behaviors interact to create a risk greater than the sum of their parts?

## Repository Structure
The project is organized as follows:

```

.
├── Makefile
├── data/
├── results/
├── src/
│   ├── binary_logistic_regression.py
│   ├── chi_square.py
│   ├── contingency_heatmap.py
│   ├── family_history.py
│   ├── lifestyle.py
│   └── ordinal_logistic_regression.py
├── typings/
├── devenv.lock
├── devenv.nix
├── devenv.yaml
├── matplotlibrc
├── pyproject.toml
├── pyrightconfig.json
└── README.md

````

## Prerequisites

Ensure you have Python installed. Install the required dependencies using `uv`:

```bash
$ uv sync
````

*Key libraries: `pandas`, `statsmodels`, `seaborn`, `matplotlib`, `scipy`.*

## Reproducing Results

This project uses a `Makefile` to automate the analysis pipeline.

### 1. Run Complete Analysis

To clean the data, run all statistical models, and generate visualizations in one step:

```bash
$ make
```

*This will execute all scripts in `src/` and populate the `results/` directory.*

### 2. Clean Environment

To remove all generated results and temporary files (useful for verifying reproducibility from scratch):

```bash
$ make clean
```
