# Interpret Compound Rationales to Understand Activity
#
# Overall Goal
# - Run Chemprop's `interpret` over hit-predicted molecules to extract the
#   rationale substructure (and substructure score) most responsible for each
#   compound's predicted activity.
#
# Relation to Manuscript
# - Produces the rationale text outputs visualized in Figure S4B by
#   S4B_interpret_substructures_as_rationales_for_model_prediction.ipynb.
#
# Inputs
# - Best D-MPNN ensemble: models/FINALbayHO04052022/ (PK + 37K final, from
#   2D_train_final_models_all_data.sh).
# - Per-round molecule sets in out/interpretation/mols_for_interpretation/:
#     * 800k_mols_for_interpretation.csv (Broad 800K validation round)
#     * molport_val1_mols_for_interpretation.csv (Molport round 1)
#     * molport_val2_val3_mols_for_interpretation.csv (Molport rounds 2 + 3)
#
# Run Configuration
# - chemprop_interpret with --property_id 1 (predicted-hit class), rationale
#   size 8-10 atoms, --prop_delta 0.1, rdkit_2d_normalized features, no scaling.
# - Output is redirected to per-round .txt files in
#   out/interpretation/interpretation_results/.
#
# Prerequisites
# - conda activate chemprop and run as: bash [filename].sh
# - To use chemprop_interpret, RDKit must be downgraded
#   (see https://github.com/chemprop/chemprop/issues/178):
#   conda install -c conda-forge rdkit=2019.09.1

export MODEL_PATH=../models/FINALbayHO04052022/;
export DATA_PATH=../out/interpretation/mols_for_interpretation/;
export OUT_PATH=../out/interpretation/interpretation_results/;

# interpret for the molecules in the 800K validation round
chemprop_interpret --data_path "$DATA_PATH"800k_mols_for_interpretation.csv --checkpoint_dir $MODEL_PATH --property_id 1 --smiles_column SMILES --max_atoms 10 --min_atoms 8 --prop_delta 0.1 --features_generator rdkit_2d_normalized --no_features_scaling > "$OUT_PATH"results_800k_mols_for_interpretation.txt

# interpret for the molecules in the first round of the Molport validation
chemprop_interpret --data_path "$DATA_PATH"molport_val1_mols_for_interpretation.csv --checkpoint_dir $MODEL_PATH --property_id 1 --smiles_column SMILES --max_atoms 10 --min_atoms 8 --prop_delta 0.1 --features_generator rdkit_2d_normalized --no_features_scaling > "$OUT_PATH"results_molport_val1_mols_for_interpretation.txt

# interpret for the molecules in the second and third rounds of validation
chemprop_interpret --data_path "$DATA_PATH"molport_val2_val3_mols_for_interpretation.csv --checkpoint_dir $MODEL_PATH --property_id 1 --smiles_column SMILES --max_atoms 10 --min_atoms 8 --prop_delta 0.1 --features_generator rdkit_2d_normalized --no_features_scaling > "$OUT_PATH"results_molport_val2_val3_mols_interpretation.txt
