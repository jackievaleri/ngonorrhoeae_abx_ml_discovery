# `data/` — Datasets for *N. gonorrhoeae* Antibiotics Discovery

This directory holds every input dataset used in the project: raw screening
results, cleaned/merged screen files, ML-ready train/test splits (with
pre-computed `rdkit_2d_normalized` features), prediction libraries, and the
validation data returned from each round of experimental follow-up.

Datasets and models are versioned by date in their filenames (e.g.
`FULL_03_19_2022.csv`, `pk_37k_screen_models_03192022/`). The dataset date
matches the model directory date — that's how to map a split to the model it
trained.

## Data not in git -> Zenodo

Large library files and `_for_sklearn` features matrices are excluded by
[.gitignore](../.gitignore) to keep the repo lightweight. They are mirrored at
the project's Zenodo deposit (the `data.zip` / `models.zip` archives produced
from this directory are also gitignored). Files marked **[Zenodo only]** below
must be pulled from there before re-running the matching scripts.

---

## Top-level files

| File | Date | Purpose |
| --- | --- | --- |
| [04052022_CLEANED_v5_antibiotics_across_many_classes.csv](04052022_CLEANED_v5_antibiotics_across_many_classes.csv) | 2022-04-05 | Curated, class-annotated reference set of known antibiotics (ChEMBL ID, Name, SMILES, Class). Used by [2A_2B_tsne_plots_quantify_diversity.ipynb](../src/2A_2B_tsne_plots_quantify_diversity.ipynb) as the "known antibiotics" overlay in the t-SNE figure. |

---

## `cleaned_screening_data/`

Cleaned, deduplicated screening tables (`Name, SMILES, hit`). Each row
corresponds to one assayed compound; `hit` is the binary label used for model
training. These are the canonical screen files — produced from the raw screen
in [screening_data/](screening_data/) and the library annotation files in
[library_info/](library_info/) by [Methods_prep_data_for_ml.ipynb](../src/Methods_prep_data_for_ml.ipynb).

| File | Composition | Feeds model |
| --- | --- | --- |
| [pk_screen_cleaned.csv](cleaned_screening_data/pk_screen_cleaned.csv) | Pharmakon (PK) library only (~1.7K compounds) | [models/pk_screen_models_11152021/](../models/pk_screen_models_11152021/) — *the Round 0 GNN does not appear in the manuscript but is left for interested readers* |
| [pk_37k_screen_cleaned.csv](cleaned_screening_data/pk_37k_screen_cleaned.csv) | PK + 37K combined screen (~38K) — **Round 1** training set | [models/pk_37k_screen_models_03192022/](../models/pk_37k_screen_models_03192022/) |
| [pk_37k_first_round_val_cleaned.csv](cleaned_screening_data/pk_37k_first_round_val_cleaned.csv) | PK + 37K + 1st round of experimentally validated predictions — **Round 2** training set | [models/pk_37k_first_round_val_screen_models_10262022/](../models/pk_37k_first_round_val_screen_models_10262022/) (used by [minimal_example_start_here/](../minimal_example_start_here/)) |
| [pk_37k_three_rounds_val_cleaned.csv](cleaned_screening_data/pk_37k_three_rounds_val_cleaned.csv) | PK + 37K + 3 rounds of validated predictions (~38.8K) — **Round 3** training set | [models/pk_37k_three_rounds_val_models_03312023/](../models/pk_37k_three_rounds_val_models_03312023/) — *not used in the manuscript, left for interested readers* |

---

## `data_prep_for_ml/`

Per-model train/test splits and pre-computed feature matrices. Each
subdirectory is paired 1:1 with a model directory under [models/](../models/).
`.csv` files are the SMILES tables passed to `chemprop`/sklearn; matching
`.npz` files are the cached `rdkit_2d_normalized` 200-D feature vectors
generated via `chemprop-master/scripts/save_features.py` (see
[minimal_example_start_here/START_HERE.sh](../minimal_example_start_here/START_HERE.sh)).
`_for_sklearn.csv` files are reformatted versions for the shallow models (RFC,
SVM, FFN) trained in [2D_hyperparameter_optimization_scripts_for_chemprop_models.ipynb](../src/2D_hyperparameter_optimization_scripts_for_chemprop_models.ipynb).

### `data_prep_for_ml_pk_screen/` — 2021-11-15
**NOTE: This data prep feeds the Round 0 (PK-only) baseline GNN, which does not appear in the manuscript and is otherwise superseded by the PK + 37K Round 1 model. Left for interested readers.**

PK-only splits feeding [models/pk_screen_models_11152021/](../models/pk_screen_models_11152021/).
Contains `FULL_11_15_2021.{csv,npz}`, `TRAIN_11_15_2021.{csv,npz}`,
`TEST_11_15_2021.{csv,npz}`.

### `data_prep_for_ml_pk_37k_screen/` — 2022-03-19  *(Round 1)*
PK + 37K splits feeding [models/pk_37k_screen_models_03192022/](../models/pk_37k_screen_models_03192022/) and the
shallow-model baselines in [models/other_models/](../models/other_models/).
Contains `FULL_03_19_2022.{csv,npz}`, `TRAIN_03_19_2022.{csv,npz}`,
`TEST_03_19_2022.{csv,npz}`, `with_scaffold_FULL_03_19_2022.csv` (Bemis-Murcko
scaffolds appended; used for scaffold-balanced splits).
**[Zenodo only]** `FULL_03_19_2022_for_sklearn.csv`, `TRAIN_03_19_2022_for_sklearn.csv`.

### `data_prep_for_ml_pk_37k_first_round_val_screen/` — 2022-10-26  *(Round 2)*
PK + 37K + 1st round of validated predictions feeding
[models/pk_37k_first_round_val_screen_models_10262022/](../models/pk_37k_first_round_val_screen_models_10262022/).
This is the model used by the [minimal_example_start_here/](../minimal_example_start_here/) quick-start.
Contains `FULL_10_26_2022.{csv,npz}`, `TRAIN_10_26_2022.{csv,npz}`,
`TEST_10_26_2022.{csv,npz}`.

### `data_prep_for_ml_pk_37k_three_rounds_val/` — 2023-03-31  *(Round 3)*
**NOTE: This data prep is not used in the manuscript, but left for interested readers.** The Round 3 retraining was an exploratory follow-up to the Round 2 model that drives the manuscript predictions.

PK + 37K + 3 rounds of validated predictions feeding
[models/pk_37k_three_rounds_val_models_03312023/](../models/pk_37k_three_rounds_val_models_03312023/).
Contains `FULL_03_31_2023.csv`, `TRAIN_03_31_2023.csv`, `TEST_03_31_2023.csv`
(no `.npz` — features are regenerated at train/predict time).

---

## `library_info/`

Source compound libraries and identifier-to-SMILES mappings. These are the
inputs to data prep (mapping plate IDs / Broad IDs to SMILES) and the
prediction targets used by [4A_5A_make_predictions_using_best_models.ipynb](../src/4A_5A_make_predictions_using_best_models.ipynb).

| File | Date | Description |
| --- | --- | --- |
| [PK180301.xls](library_info/PK180301.xls) | 2021-10-06 | Pharmakon plate map from Broad (plate/well → name). |
| [pk_np_smiles_mapping_manual.csv](library_info/pk_np_smiles_mapping_manual.csv) | 2025-08-25 | Manually validated Name → SMILES → Activity mapping for the PK library. |
| [NeisseriaGonorrhoeaePharmakonScreen.csv](screening_data/NeisseriaGonorrhoeaePharmakonScreen.csv) | (see `screening_data/`) | — |
| [Broad Compound Registration Melis to date.xlsx](library_info/Broad%20Compound%20Registration%20Melis%20to%20date.xlsx) | 2023-03-29 | Broad registration metadata for the validation compounds ordered/tested by Melis. |
| [37Kclean.csv](library_info/37Kclean.csv) | 2025-08-25 | Cleaned 37K screening library (SMILES). |
| [37Kclean.npz](library_info/37Kclean.npz) | 2023-04-14 | `rdkit_2d_normalized` features for the 37K library. |
| [broad800k.csv](library_info/broad800k.csv) | 2020-09-10 | ~800K Broad compound library used as the primary virtual screen target. |
| broad800k.npz | 2020-09-10 | **[Zenodo only]** Features for the 800K library. |
| [250k_50k_rndm_selected.csv](library_info/250k_50k_rndm_selected.csv) | 2025-08-04 | 50K random sample drawn from the 250K ZINC drug-like set |
| `250k_rndm_zinc_drugs_clean_sorted copy.txt` | 2025-08-25 | **[Zenodo only]** Full 250K ZINC drug-like library (source for the random sample above). |
| `PublicStructures.txt` | 2021-04-15 | **[Zenodo only]** Public compound-structure dump used during library annotation. |
| `cleaned_full_all_dbs_04_19_2022.csv` | 2022-04-19 | **[Zenodo only]** Combined, cleaned multi-database library (~all DBs merged). |
| `cleaned_full_all_dbs_04_19_2022.npz` | 2022-04-20 | **[Zenodo only]** Features for the combined library above. |

---

## `screening_data/`

Raw experimental screen output from the Lewis lab — pre-cleaning. The cleaned
versions live in [cleaned_screening_data/](cleaned_screening_data/).

| File | Date | Description |
| --- | --- | --- |
| [NeisseriaGonorrhoeaePharmakonScreen.csv](screening_data/NeisseriaGonorrhoeaePharmakonScreen.csv) | 2021-10-28 | Raw PK screen output (`Name, SMILES, NG_hit`). |
| [Ngonorrhoeae_PharmakonAnd37K_all_forJackie.csv](screening_data/Ngonorrhoeae_PharmakonAnd37K_all_forJackie.csv) | 2025-08-25 | Raw merged PK + 37K screen output with plate/well metadata (`Library, Plate, Well, Name, SMILES, NG_hit`). |

---

## `validated_model_predictions/`

Experimental MIC follow-up data on compounds prioritized by the models — i.e.
the labels for the validation rounds that get fed back into the next training
set. Files supplied by Melis (Lewis lab); cleaned `.csv`/`.npz` versions are
produced in [4A_5A_filter_predictions_to_prioritize_compounds_for_validation.ipynb](../src/4A_5A_filter_predictions_to_prioritize_compounds_for_validation.ipynb).

| File | Date | Description |
| --- | --- | --- |
| [B800K_EasyMedHard_ValidationData for Jackie.xlsx](validated_model_predictions/B800K_EasyMedHard_ValidationData%20for%20Jackie.xlsx) | 2022-09-28 | Easy/Medium/Hard tier validation results from the 800K virtual screen. |
| [easy_medium_hard_broad800k_val_sets.xlsx](validated_model_predictions/easy_medium_hard_broad800k_val_sets.xlsx) | 2022-10-03 | Companion sheet defining the Easy/Med/Hard validation tier membership. |
| [cleaned_easy_med_hard_val_sets_800k_10_03_2022.csv](validated_model_predictions/cleaned_easy_med_hard_val_sets_800k_10_03_2022.csv) | 2022-10-03 | Cleaned Easy/Med/Hard validation set (`hit, SMILES, name`). |
| [cleaned_easy_med_hard_val_sets_800k_10_03_2022.npz](validated_model_predictions/cleaned_easy_med_hard_val_sets_800k_10_03_2022.npz) | 2022-10-03 | `rdkit_2d_normalized` features for the cleaned Easy/Med/Hard set. |
| [2022-10-21 Broad800K validation with MP order 1 data summary Jackie.xlsx](validated_model_predictions/2022-10-21%20Broad800K%20validation%20with%20MP%20order%201%20data%20summary%20Jackie.xlsx) | 2022-10-26 | Round 1 validation summary — fed into the Round 2 training set (`pk_37k_first_round_val_cleaned.csv`). |
| [2023-03-27 Data update for Jackie.xlsx](validated_model_predictions/2023-03-27%20Data%20update%20for%20Jackie.xlsx) | 2023-03-28 | Rounds 2 + 3 validation update — fed into the Round 3 training set. |
| [cleaned_round2_round3_val.csv](validated_model_predictions/cleaned_round2_round3_val.csv) | 2023-04-19 | Cleaned Round 2 + Round 3 validation labels (`Name, hit, SMILES`). |
