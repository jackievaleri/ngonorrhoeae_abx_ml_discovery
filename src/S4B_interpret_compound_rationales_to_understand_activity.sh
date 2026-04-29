# To use, conda activate chemprop and then bash [filename].sh
# Note that to use the interpret function, needed to downgrade rdkit (as per https://github.com/chemprop/chemprop/issues/178): conda install -c conda-forge rdkit=2019.09.1

export MODEL_PATH=../models/FINALbayHO04052022/;
export DATA_PATH=../out/interpretation/mols_for_interpretation/;
export OUT_PATH=../out/interpretation/interpretation_results/;

# interpret for the molecules in the 800K validation round
chemprop_interpret --data_path "$DATA_PATH"800k_mols_for_interpretation.csv --checkpoint_dir $MODEL_PATH --property_id 1 --smiles_column SMILES --max_atoms 10 --min_atoms 8 --prop_delta 0.1 --features_generator rdkit_2d_normalized --no_features_scaling > "$OUT_PATH"results_800k_mols_for_interpretation.txt

# interpret for the molecules in the first round of the Molport validation
chemprop_interpret --data_path "$DATA_PATH"molport_val1_mols_for_interpretation.csv --checkpoint_dir $MODEL_PATH --property_id 1 --smiles_column SMILES --max_atoms 10 --min_atoms 8 --prop_delta 0.1 --features_generator rdkit_2d_normalized --no_features_scaling > "$OUT_PATH"results_molport_val1_mols_for_interpretation.txt

# interpret for the molecules in the second and third rounds of validation
chemprop_interpret --data_path "$DATA_PATH"molport_val2_val3_mols_for_interpretation.csv --checkpoint_dir $MODEL_PATH --property_id 1 --smiles_column SMILES --max_atoms 10 --min_atoms 8 --prop_delta 0.1 --features_generator rdkit_2d_normalized --no_features_scaling > "$OUT_PATH"results_molport_val2_val3_mols_interpretation.txt
