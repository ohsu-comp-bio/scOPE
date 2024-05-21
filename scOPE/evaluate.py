import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Import custom modules
sys.path.append('../')
from scOPE import preprocessing
from scOPE import utilities
from scOPE import models
from scOPE import evaluate


def predict_single_cell_mutation(single_cell_data, model):
    '''Predicts the mutation presence or absence in single-cell data using the provided model.'''
    # Ensure the single-cell data is in the correct shape (transpose if necessary)
    if single_cell_data.shape[0] > single_cell_data.shape[1]:
        single_cell_data = single_cell_data.T
    # Use the model to predict the mutation status
    predictions = model.predict(single_cell_data)
    return predictions


def predict_mutations_in_single_cells(gene_models_dir, single_cell_data):
    '''
    Applies each gene mutation prediction model to the single-cell RNA-seq data.
    
    Parameters:
    - gene_models_dir: Directory where the trained models are saved.
    - single_cell_data: DataFrame containing the single-cell RNA-seq data.
    
    Returns:
    - A dictionary with gene names as keys and prediction results as values.
    '''
    predictions_dict = {}
    
    # Iterate through the files in the gene models directory
    for model_file in os.listdir(gene_models_dir):
        if model_file.endswith('_logistic_ridge_model.pkl'):
            
            # Extract gene name from the file name
            gene_name = model_file.split('_logistic_ridge_model.pkl')[0]
            
            # Load the model
            model_path = os.path.join(gene_models_dir, model_file)
            model = utilities.load_model(gene_models_dir, gene_name)
            
            # Predict mutations in single-cell data
            predictions = predict_single_cell_mutation(single_cell_data, model)
            
            # Store predictions in the dictionary
            predictions_dict[gene_name] = predictions
            
            print(f"Predictions for {gene_name} completed.")
    
    return predictions_dict


def evaluate_model(y_true, y_pred, y_prob, gene_name):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    # Print classification report
    print(f"Classification Report for {gene_name}:\n", classification_report(y_true, y_pred))
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {gene_name}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"ROC Curve for {gene_name}")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    
    # Plot Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(recall_vals, precision_vals, label=f'Precision-Recall Curve (AP = {avg_precision:.2f})')
    plt.title(f"Precision-Recall Curve for {gene_name}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'average_precision': avg_precision
    }
