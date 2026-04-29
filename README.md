# N. gonorrhoeae Antibiotics Discovery

Discovery of antibiotics active against N. gonorrhoeae using experimental and machine learning screening. Below is a description of Jupyter notebooks found in this repository.

## minimal_example_start_here/ - Start here!

This folder provides very minimal code and dummy molecules to use one of the models for prediction. You should be able to "quick start" here.

## 2A_2B_tsne_plots_quantify_diversity.ipynb - t-SNE Analysis

This notebook uses t-distributed stochastic neighbor embedding (t-SNE) to visualize high-dimensional data in a lower-dimensional space. We perform t-SNE analysis on a dataset of training data and known antibiotics, considering both hits and non-hits, to understand the chemical space.

Additionally, we quantify structural diversity using Bemis-Murcko scaffolds, average Tanimoto similarity of fingerprints, and sum bottleneck diversity metrics.

## Methods_prep_data_for_ml.ipynb - Data Preparation for Machine Learning

This notebook focuses on preparing data for machine learning models. We process multiple training datasets and validation datasets.

## 2D_hyperparameter_optimization_scripts_for_chemprop_models.ipynb - Model Training and Hyperparameter Optimization

This notebook involves several steps for training and optimizing machine learning models. First, we perform hyperparameter optimization to fine-tune our models for the best performance. We then train shallow models, including Random Forest Classifier (RFC), Support Vector Machine (SVM), and Feedforward Neural Network (FFN), using the prepared dataset. Additionally, we train a Graph Neural Network (GNN), incorporating Bayesian Hyperparameter Optimization to tune the model. Companion notebooks 2D_chemberta_language_model.ipynb and 2D_attentivefp_attention_model.ipynb include comparison code for ChemBERTa and Attentive Fingerprint (AttentiveFP) models.

## 2D_compare_models_plot.ipynb - Model Comparison

In this notebook, we create and analyze comparison plots for the various models trained in the previous steps. These visualizations help in evaluating model performance and identifying the best-performing model.

## 4A_5A_make_predictions_using_best_models.ipynb - Using Models for Prediction

This notebook demonstrates how to use the trained models to make predictions on new data. We curate predictions for a large dataset of 800,000 compounds and a commercial compound library. The companion notebook 4A_5A_filter_predictions_to_prioritize_compounds_for_validation.ipynb filters these predictions to prioritize compounds for experimental validation as shown in Figure 4A and Figure 5A.

## S2_try_different_negative_datasets_combined.ipynb - Negative Dataset Test

This notebook involves testing the models on a negative dataset to evaluate their performance and robustness.

## S4B_interpret_substructures_as_rationales_for_model_prediction.ipynb - Substructure Interpretation

In this notebook, we interpret the substructures of the chemical compounds that contribute to the predictions made by our models. Understanding these substructures helps in identifying key features that influence model decisions and provides insights into the underlying biochemical mechanisms.

## Additional analyses

### biochem_differences_between_screen_hits_and_nonhits.ipynb

This notebook compares experimental hits with non-hits to explore available data and identify key biochemical differences. We also analyze the incorrect predictions made by our model to gain insights into potential areas for improvement.

### speed_of_fingerprints.ipynb

This notebook compares the generation time for producing 2D Morgan fingerprints (RDKit) vs. 3D E3FP fingerprints. E3FP generation is more computationally intensive due to conformer generation and 3D feature calculation, so we focus only on 2D fingerprints for the analyses in the manuscript.
