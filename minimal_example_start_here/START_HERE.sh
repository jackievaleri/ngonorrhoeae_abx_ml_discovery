# 1. Set up the environment
# Note older version of chemprop and python to match what the model was trained with
# But likely could be done with newer versions too
conda create -n test_chemprop151 python=3.8 -y
conda activate test_chemprop151
pip install chemprop==1.5.1
pip install git+https://github.com/bp-kelley/descriptastorus

# 2. Pre-compute rdkit_2d_normalized features and save to .npz
#    (the model expects features_path, not an on-the-fly generator)
# Requires local version of v1.5.1 chemprop
# Source code can be downloaded from old release here: https://github.com/chemprop/chemprop/releases/tag/v1.5.1
# Could likely be done with most recent version too! This just had a handy scripts/save_features.py script
# Here we have submodule with name "chemprop-master" that contains the source code for v1.5.1
# Change the path / version / code as needed
python chemprop-master/scripts/save_features.py \
    --data_path minimal_example_start_here/dummy_mols.csv \
    --features_generator rdkit_2d_normalized \
    --save_path minimal_example_start_here/dummy_mols_fts \
    --smiles_column smiles

# 3. Predict, pointing at the saved features
#    --no_features_scaling matches how the model was trained
# Note that this uses Round 2 model which was trained on PK+37K+1st round val
chemprop_predict \
    --test_path minimal_example_start_here/dummy_mols.csv \
    --preds_path minimal_example_start_here/dummy_mols_output.csv \
    --checkpoint_dir models/pk_37k_first_round_val_screen_models_10262022/FINALbayHO11152022/ \
    --features_path minimal_example_start_here/dummy_mols_fts.npz \
    --no_features_scaling

# Predictions land in minimal_example_start_here/dummy_mols_output.csv
# Takes a while if you are on CPU or have a lot of molecules