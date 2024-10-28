# N. gonorrhoeae Antibiotics Discovery
Discovery of antibiotics active against N. gonorrhoeae using experimental and machine learning screening. Below is a description of Jupyter notebooks found in this repository.

## 01 - Experimental Data Exploration
This notebook compares experimental hits with non-hits to explore available data and identify key biochemical differences. Additionally, we compare a combined dataset of tens of thousands of small molecules to further explore high-throughput screening data. We also analyze the incorrect predictions made by our model to gain insights into potential areas for improvement.

## 02 - t-SNE Analysis
This notebook uses t-distributed stochastic neighbor embedding (t-SNE) to visualize high-dimensional data in a lower-dimensional space. We perform t-SNE analysis on a dataset of training data and known antibiotics, considering both hits and non-hits, to understand the chemical space.

## 03 - Data Preparation for Machine Learning
This notebook focuses on preparing data for machine learning models. We process multiple training datasets and validation datasets.

## 04 - Model Training and Hyperparameter Optimization
This notebook involves several steps for training and optimizing machine learning models. First, we perform hyperparameter optimization to fine-tune our models for the best performance. We then train shallow models, including Random Forest Classifier (RFC), Support Vector Machine (SVM), and Feedforward Neural Network (FFN), using the prepared dataset. Additionally, we train a Graph Neural Network (GNN), incorporating Bayesian Hyperparameter Optimization to tune the model. This notebook also includes comparison code for ChemBERTa and Attentive Fingerprint (AttentiveFP) models.

## 05 - Model Comparison
In this notebook, we create and analyze comparison plots for the various models trained in the previous steps. These visualizations help in evaluating model performance and identifying the best-performing model.

## 06 - Using Models for Prediction
This notebook demonstrates how to use the trained models to make predictions on new data. We curate predictions for a large dataset of 800,000 compounds and a commercial compound library.

## 07 - Substructure Interpretation
In this notebook, we interpret the substructures of the chemical compounds that contribute to the predictions made by our models. Understanding these substructures helps in identifying key features that influence model decisions and provides insights into the underlying biochemical mechanisms.

## 08 - Negative Dataset Test
This notebook involves testing the models on a negative dataset to evaluate their performance and robustness.
