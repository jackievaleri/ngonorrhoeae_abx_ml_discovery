# `models/` — Trained Models for *N. gonorrhoeae* Antibiotics Discovery

This directory holds the trained models used in the project. The headline
result is a four-round series of D-MPNN graph neural networks (chemprop) that
each ingest the previous round's experimental validation data and produce the
next round of predicted hits. A set of shallow / non-GNN baselines (RFC, SVM,
FFN, AttentiveFP, ChemBERTa) accompanies the Round 1 GNN for comparison.

Each model directory is named by date and pairs 1:1 with a [data/data_prep_for_ml/](../data/data_prep_for_ml/)
subdirectory of the same date — that's how to map a model checkpoint back to
the exact train/test split it was fit on.

## Models not in git → Zenodo

Model weights (`*.pt` for chemprop GNNs, `*.pkl` for sklearn baselines) are
excluded by [.gitignore](../.gitignore) — the repo only carries the metadata
(`args.json`, `test_scores.csv`, hyperopt logs, prediction CSVs) so the
training history is reviewable without the binaries. The full checkpoint
trees are mirrored at the project's Zenodo deposit (the `models.zip` archive
is also gitignored). To run prediction with these models, pull `models.zip`
from Zenodo and unpack it over this directory; see the quick-start in
[minimal_example_start_here/START_HERE.sh](../minimal_example_start_here/START_HERE.sh).

A typical chemprop checkpoint tree looks like:
`{model_dir}/fold_{0..N}/model_{0..M}/model.pt` — folds are random seeds for
the data split, models within a fold are the ensemble.

---

## D-MPNN GNN models — round-by-round

Trained by [src/2D_train_final_models_all_data.sh](../src/2D_train_final_models_all_data.sh) on top of the hyperparameters
selected in [2D_hyperparameter_optimization_with_bayesian_optimization.sh](../src/2D_hyperparameter_optimization_with_bayesian_optimization.sh)
(except `FINAL151`, which used a manual grid in [2D_hyperopt_pk_screen_11152021.sh](../src/2D_hyperopt_pk_screen_11152021.sh)).
All later GNN finals use scaffold-balanced 80/10/10 splits with `num_folds 5`
and `ensemble_size 10` (50 model checkpoints per directory).

### `pk_screen_models_11152021/` — *PK-only baseline GNN*
Trained on [data/data_prep_for_ml/data_prep_for_ml_pk_screen/](../data/data_prep_for_ml/data_prep_for_ml_pk_screen/) (`FULL_11_15_2021.{csv,npz}`, ~1.7K compounds).

| Subdir / file | Description |
| --- | --- |
| `FINAL151/` | Final GNN: `init_lr 1e-3, dropout 0.3, hidden 1200, ffn 3, depth 4`, 50 folds × ensemble 1. Selected as trial 151 from the manual hyperopt grid. |
| [pk_gnn_hyperopt_results.csv](pk_screen_models_11152021/pk_gnn_hyperopt_results.csv) | All hyperopt trial results from the manual grid. |
| [151_test_set_performance.csv](pk_screen_models_11152021/151_test_set_performance.csv) | Test-set auROC/auPR for the chosen `FINAL151` configuration. |

### `pk_37k_screen_models_03192022/` — *Round 1 GNN (PK + 37K)*
Trained on [data/data_prep_for_ml/data_prep_for_ml_pk_37k_screen/](../data/data_prep_for_ml/data_prep_for_ml_pk_37k_screen/) (`FULL_03_19_2022.{csv,npz}`, ~38K compounds).

| Subdir / file | Description |
| --- | --- |
| `FINALbayHO04052022/` | Final GNN, **random split**: `dropout 0.15, hidden 2300, ffn 3, depth 3`, 5 folds × ensemble 10. |
| `FINALbayHO04052022_with_scaffold_split/` | Same hyperparameters as above, retrained with **scaffold-balanced** split. |
| `FINALbayHO04052022_30_models_no_scaffold_split/` | 30-fold ensemble of the same architecture under random split (used for variance analysis in [2D_compare_models_plot.ipynb](../src/2D_compare_models_plot.ipynb)). |
| `FINALbayHO04052022_30_models_with_scaffold_split/` | 30-fold ensemble, scaffold-balanced. |
| `depth_3_dropout_0.15..._hidden_size_2300/` | Hyperopt run that produced the chosen architecture (intermediate; carried for reproducibility). |
| `depth_3_dropout_0.15..._hidden_size_2300_preds.csv` | Test-set predictions from the hyperopt run above. |
| [best_bayesian_models_04052022.csv](pk_37k_screen_models_03192022/best_bayesian_models_04052022.csv) | Top hyperopt trials ranked by mean prc-auc. |

### `pk_37k_first_round_val_screen_models_10262022/` — *Round 2 GNN (PK + 37K + 1st-round val)*
Trained on [data/data_prep_for_ml/data_prep_for_ml_pk_37k_first_round_val_screen/](../data/data_prep_for_ml/data_prep_for_ml_pk_37k_first_round_val_screen/) (`FULL_10_26_2022.{csv,npz}`).
**This is the model used by the [minimal_example_start_here/](../minimal_example_start_here/) quick-start.**

| Subdir / file | Description |
| --- | --- |
| `FINALbayHO11152022/` | Final GNN: `dropout 0.25, hidden 800, ffn 2, depth 5`, 5 folds × ensemble 10, scaffold-balanced. Mean test prc-auc 0.502, auROC 0.907. |
| `depth_5_dropout_0.25_ffn_num_layers_2_hidden_size_800/` | Hyperopt run that produced the chosen architecture. |
| `depth_5_dropout_0.25_ffn_num_layers_2_hidden_size_800_preds.csv` | Test-set predictions from the hyperopt run above. |
| [best_bayesian_models_11152022.csv](pk_37k_first_round_val_screen_models_10262022/best_bayesian_models_11152022.csv) | Top hyperopt trials ranked by mean prc-auc. |
| [hyperopt_seeds.txt](pk_37k_first_round_val_screen_models_10262022/hyperopt_seeds.txt), [quiet.log](pk_37k_first_round_val_screen_models_10262022/quiet.log), [verbose.log](pk_37k_first_round_val_screen_models_10262022/verbose.log) | Bayesian hyperopt logs. |

### `pk_37k_three_rounds_val_models_03312023/` — *Round 3 GNN (PK + 37K + 3 rounds val)*
Trained on [data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/](../data/data_prep_for_ml/data_prep_for_ml_pk_37k_three_rounds_val/) (`FULL_03_31_2023.csv`).

| Subdir / file | Description |
| --- | --- |
| `FINALbayHO04112023/` | Final GNN: `dropout 0.25, hidden 400, ffn 3, depth 6`, 5 folds × ensemble 10, scaffold-balanced. Mean test prc-auc 0.480, auROC 0.911. |
| `trial_seed_17/` | Hyperopt run that produced the chosen architecture. |
| `trial_seed_17_preds.csv` | Test-set predictions from the hyperopt run above. |
| [best_bayesian_models_03312023.csv](pk_37k_three_rounds_val_models_03312023/best_bayesian_models_03312023.csv) | Top hyperopt trials ranked by mean prc-auc. |
| [hyperopt_seeds.txt](pk_37k_three_rounds_val_models_03312023/hyperopt_seeds.txt), [quiet.log](pk_37k_three_rounds_val_models_03312023/quiet.log), [verbose.log](pk_37k_three_rounds_val_models_03312023/verbose.log) | Bayesian hyperopt logs. |

---

## `other_models/` — Round 1 baselines

Non-GNN and alternative-architecture baselines, all trained on the same Round 1
PK + 37K split ([data/data_prep_for_ml/data_prep_for_ml_pk_37k_screen/](../data/data_prep_for_ml/data_prep_for_ml_pk_37k_screen/)). Used by
[2D_compare_models_plot.ipynb](../src/2D_compare_models_plot.ipynb) to benchmark the GNN against shallow and
transformer-style alternatives.

Within each `*_hyperopt_pk_37k/` directory:
- A bare numeric directory (e.g. `122/`) is the chosen hyperopt trial.
- `*_final_<n>/` retrains the chosen trial across many folds with the
  **scaffold-balanced** split (the comparison reported in the manuscript).
- `*_RANDOM_final_<n>/` retrains the chosen trial with a **random** split.
- `*_hyperopt_results.csv` lists all trials and their CV scores.
- `<n>_test_set_*_preds.csv` is the test-set prediction CSV for the chosen trial.

| Subdir | Architecture | Trained by | Best trial |
| --- | --- | --- | --- |
| [rfc_hyperopt_pk_37k/](other_models/rfc_hyperopt_pk_37k/) | Random Forest Classifier (sklearn, 4096-bit Morgan r=2, class-weighted, 250 trees) | [src/2D_hyperopt_rfc_pk_37k.sh](../src/2D_hyperopt_rfc_pk_37k.sh) | trial `122` |
| [svm_hyperopt_pk_37k/](other_models/svm_hyperopt_pk_37k/) | Support Vector Machine (sklearn, 4096-bit Morgan r=2, class-weighted) | [src/2D_hyperopt_svm_pk_37k.sh](../src/2D_hyperopt_svm_pk_37k.sh) | trial `15` |
| [ffn_hyperopt_pk_37k/](other_models/ffn_hyperopt_pk_37k/) | Feedforward Neural Network on Morgan fingerprints (chemprop with `--features_only --depth 0`) | [src/2D_hyperopt_ffn_pk_37k.sh](../src/2D_hyperopt_ffn_pk_37k.sh) | trial `20` |
| [attentivefp_pk_37k/](other_models/attentivefp_pk_37k/) | AttentiveFP (DGL-LifeSci) — random and scaffold-split runs, plus `all_ho_results.csv` | [src/additional_analyses/](../src/additional_analyses/) and [2D_attentivefp_attention_model.ipynb](../src/2D_attentivefp_attention_model.ipynb) | per `summary_results.csv` |
| [chemberta_pk_37k/](other_models/chemberta_pk_37k/) | ChemBERTa (transformer language model on SMILES) — random and scaffold-split runs | [2D_chemberta_language_model.ipynb](../src/2D_chemberta_language_model.ipynb) | per `summary_results.csv` |

---

## File conventions inside any chemprop model directory

| File | Description |
| --- | --- |
| `args.json` | Full chemprop CLI args used to train this model (the source of truth for all hyperparameters). |
| `fold_<i>/model_<j>/model.pt` | **[Zenodo only]** Trained checkpoint. `i` indexes the data-split seed, `j` indexes the ensemble member. |
| `fold_<i>/model_<j>/events.out.tfevents.*` | TensorBoard event files from training. |
| `fold_<i>/test_scores.json` | Per-fold test metrics. |
| `test_scores.csv` | Aggregated mean ± stdev test metrics across all folds (prc-auc and auc). |
| `quiet.log` / `verbose.log` | Training stdout — quiet is one-line-per-epoch, verbose includes per-batch diagnostics. |

For the sklearn baselines under `other_models/{rfc,svm}_hyperopt_pk_37k/`,
checkpoints are `*.pkl` instead of `*.pt` (also **[Zenodo only]**) and
otherwise follow the same `fold_<i>/model_<j>/` layout.
