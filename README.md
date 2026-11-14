# Reweighing: reproduction of the discrimination-free classification experiment

Reproduction of the Reweighing method from Kamiran & Calders (2012), *Data preprocessing techniques for classification without discrimination*, made for Assignment 4 of Data Wrangling (INFOMDW). Code: https://github.com/zangphan96/DW-assignment-4

## Background
Historical data often encodes unequal treatment: one group received the favourable label less often than the rest, and a classifier trained on it reproduces this gap. Reweighing is a pre-processing method that fixes the training data instead of the learner. For each combination of group value s and class label c, every training instance gets the weight

w(s, c) = P(s) · P(c) / P(s, c)

so that, in the weighted data, the label is statistically independent of the group (discrimination score 0). Features and labels stay unchanged, and any learner that accepts instance weights can be used afterwards.

## Experiment
Reweighing is applied with AIF360 and followed by a logistic regression. It is compared with the same model trained without weights. For each dataset the experiment is repeated for 10 random seeds (0-9), with a 70% / 30% train / test split stratified on label and protected group. The weights are fitted on the training set only. Measured on the test set:
- **DP_Diff**: statistical parity difference (0 is fair)
- **DIR**: disparate impact ratio (1 is fair)
- **Consistency**: AIF360 consistency with k = 5 (1 means similar individuals get the same outcome)
- accuracy, balanced accuracy and F1

The same fairness measures are also computed on the original labels of the full dataset.

## Data
| Dataset | Source | File | Rows used | Protected attribute (privileged) |
|---|---|---|---|---|
| Adult | [UCI](https://archive.ics.uci.edu/dataset/2/adult) | `adult.data` | 30,162 | sex (male) |
| German Credit | [UCI Statlog](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | `german.data` | 1,000 | age (25 or older) |
| COMPAS | [ProPublica](https://github.com/propublica/compas-analysis) | `compas-scores-two-years.csv` | 5,278 | race (Caucasian) |

Put the downloaded files in a `data/` folder: `adult.data`, `german.data` and `compas.csv` (the ProPublica file, renamed). The standard ProPublica filters are applied in the code.

## How to run
1. Download the data into `data/`.
2. Install the libraries (Python 3.10.9):
   ```
   pip install aif360==0.6.1 scikit-learn==1.2.2 pandas==2.1.2 numpy==1.26.4 matplotlib==3.7.0 notebook
   ```
3. Start `jupyter notebook` in the folder that contains `Reweighing_Experiment_AIF360.ipynb` and `data/`, open the notebook and choose **Kernel > Restart & Run All**.

**Main parameters:** logistic regression with L2 penalty, `C = 1.0`, `max_iter = 2000`, decision threshold 0.5, standardised numeric and one-hot categorical features (the protected attribute stays a feature).

**Output:**
- `results_raw_aif360.csv`: one row per dataset, stage and seed
- `results_summary_aif360.csv`: mean and standard deviation over the 10 seeds
- `results_figure_aif360.png`: the results figure

## Results
Test-set means over 10 seeds:

| Dataset | Stage | DP_Diff | DIR | Accuracy |
|---|---|---|---|---|
| Adult (sex) | Baseline LR | -0.192 | 0.291 | 0.849 |
| | Reweighing + LR | -0.092 | 0.586 | 0.843 |
| German (age) | Baseline LR | -0.145 | 0.817 | 0.758 |
| | Reweighing + LR | -0.121 | 0.847 | 0.754 |
| COMPAS (race) | Baseline LR | -0.283 | 0.620 | 0.674 |
| | Reweighing + LR | -0.014 | 0.977 | 0.665 |

Reweighing moves DP_Diff and DIR towards fair values on Adult and COMPAS for a small loss in accuracy. On German Credit the change is smaller than the variation between splits. Consistency does not change noticeably (differences of at most 0.004).
