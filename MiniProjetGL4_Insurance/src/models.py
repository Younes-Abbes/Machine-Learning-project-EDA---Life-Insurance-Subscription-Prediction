"""
Models Module for Life Insurance Subscription Prediction
=========================================================
This module contains:
- Model definitions and training
- Cross-validation with multiple metrics
- Model comparison and selection
- Hyperparameter tuning
- Model persistence

Author: GL4 Data Mining Mini-Project Team
Date: 2026
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Sklearn imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# XGBoost and LightGBM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    A comprehensive model trainer for insurance subscription prediction.
    """
    
    def __init__(self):
        """Initialize the model trainer with predefined models."""
        self.models = self._initialize_models()
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def _initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models with specified parameters.
        
        Returns:
        --------
        dict
            Dictionary of model name -> model object
        """
        models = {
            'Logistic Regression': LogisticRegression(
                C=1.0, 
                solver='liblinear',
                random_state=42,
                max_iter=1000
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                metric='euclidean',
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
        }
        return models
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                             cv: int = 5, verbose: bool = True) -> Dict[str, float]:
        """
        Perform cross-validation for a single model.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to evaluate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target variable
        cv : int
            Number of cross-validation folds
        verbose : bool
            Print progress
            
        Returns:
        --------
        dict
            Dictionary of metric -> mean score
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Calculate metrics using cross_val_score
        accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
        roc_auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        
        results = {
            'Accuracy': {
                'mean': accuracy_scores.mean(),
                'std': accuracy_scores.std(),
                'scores': accuracy_scores
            },
            'F1-Score': {
                'mean': f1_scores.mean(),
                'std': f1_scores.std(),
                'scores': f1_scores
            },
            'ROC-AUC': {
                'mean': roc_auc_scores.mean(),
                'std': roc_auc_scores.std(),
                'scores': roc_auc_scores
            }
        }
        
        if verbose:
            print(f"  Accuracy:  {results['Accuracy']['mean']:.4f} (+/- {results['Accuracy']['std']:.4f})")
            print(f"  F1-Score:  {results['F1-Score']['mean']:.4f} (+/- {results['F1-Score']['std']:.4f})")
            print(f"  ROC-AUC:   {results['ROC-AUC']['mean']:.4f} (+/- {results['ROC-AUC']['std']:.4f})")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        cv: int = 5) -> Dict[str, Dict]:
        """
        Train and cross-validate all models.
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training feature matrix
        y_train : np.ndarray
            Training target variable
        cv : int
            Number of cross-validation folds
            
        Returns:
        --------
        dict
            Dictionary of model name -> results
        """
        print("=" * 70)
        print("TRAINING AND CROSS-VALIDATING ALL MODELS")
        print("=" * 70)
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Model: {name}")
            print('='*50)
            
            results = self.cross_validate_model(model, X_train, y_train, cv=cv)
            self.results[name] = results
        
        return self.results
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a formatted DataFrame.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with model comparison results
        """
        data = []
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': f"{metrics['Accuracy']['mean']:.4f} ± {metrics['Accuracy']['std']:.4f}",
                'F1-Score': f"{metrics['F1-Score']['mean']:.4f} ± {metrics['F1-Score']['std']:.4f}",
                'ROC-AUC': f"{metrics['ROC-AUC']['mean']:.4f} ± {metrics['ROC-AUC']['std']:.4f}",
                'Accuracy_mean': metrics['Accuracy']['mean'],
                'F1_mean': metrics['F1-Score']['mean'],
                'AUC_mean': metrics['ROC-AUC']['mean']
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('F1_mean', ascending=False).reset_index(drop=True)
        
        return df
    
    def select_best_model(self, metric: str = 'F1-Score') -> Tuple[str, Any]:
        """
        Select the best model based on specified metric.
        
        Parameters:
        -----------
        metric : str
            Metric to use for selection ('Accuracy', 'F1-Score', 'ROC-AUC')
            
        Returns:
        --------
        tuple
            (best_model_name, best_model)
        """
        best_score = -1
        best_name = None
        
        for name, results in self.results.items():
            score = results[metric]['mean']
            if score > best_score:
                best_score = score
                best_name = name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\nBest model based on {metric}: {best_name}")
        print(f"  Score: {best_score:.4f}")
        
        return best_name, self.best_model
    
    def hyperparameter_tuning(self, model_name: str, X_train: np.ndarray, 
                              y_train: np.ndarray, param_grid: Dict = None) -> Any:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to tune
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training target
        param_grid : dict
            Parameter grid for search
            
        Returns:
        --------
        sklearn estimator
            Best estimator from grid search
        """
        print(f"\n{'='*50}")
        print(f"Hyperparameter Tuning: {model_name}")
        print('='*50)
        
        # Default parameter grids
        default_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.15]
            },
            'Logistic Regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            },
            'K-Nearest Neighbors': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            },
            'LightGBM': {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.05, 0.1, 0.15]
            }
        }
        
        if param_grid is None:
            param_grid = default_grids.get(model_name, {})
        
        model = self.models[model_name]
        
        grid_search = GridSearchCV(
            model, param_grid,
            cv=3,  # Reduced for speed
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        # Update model with best estimator
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def train_final_model(self, model_name: str, X_train: np.ndarray, 
                          y_train: np.ndarray) -> Any:
        """
        Train the final model on full training data.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to train
        X_train : np.ndarray
            Full training features
        y_train : np.ndarray
            Full training target
            
        Returns:
        --------
        sklearn estimator
            Trained model
        """
        print(f"\nTraining final model: {model_name}")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        
        self.best_model = model
        self.best_model_name = model_name
        
        print("Training complete!")
        
        return model
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate a trained model on test data.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test target
            
        Returns:
        --------
        dict
            Dictionary with evaluation results
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred),
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        print("\n" + "=" * 50)
        print("TEST SET EVALUATION")
        print("=" * 50)
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"F1-Score:  {results['f1']:.4f}")
        print(f"ROC-AUC:   {results['roc_auc']:.4f}")
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        print("\nClassification Report:")
        print(results['classification_report'])
        
        return results
    
    def get_feature_importance(self, model_name: str = None, 
                               feature_names: List[str] = None) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        feature_names : list
            List of feature names
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with feature importances sorted
        """
        if model_name is None:
            model_name = self.best_model_name
            
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return None
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        df = df.sort_values('Importance', ascending=False).reset_index(drop=True)
        
        return df
    
    def save_model(self, model=None, filepath: str = 'models/best_model.pkl'):
        """
        Save the model to a pickle file.
        
        Parameters:
        -----------
        model : sklearn estimator
            Model to save. If None, saves best_model
        filepath : str
            Path to save the model
        """
        if model is None:
            model = self.best_model
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/best_model.pkl') -> Any:
        """
        Load a model from a pickle file.
        
        Parameters:
        -----------
        filepath : str
            Path to the model file
            
        Returns:
        --------
        sklearn estimator
            Loaded model
        """
        model = joblib.load(filepath)
        self.best_model = model
        print(f"Model loaded from {filepath}")
        return model
    
    def get_roc_curve_data(self, model, X_test: np.ndarray, 
                           y_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get ROC curve data for plotting.
        
        Parameters:
        -----------
        model : sklearn estimator
            Trained model
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test target
            
        Returns:
        --------
        tuple
            (fpr, tpr, auc)
        """
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        
        return fpr, tpr, auc


# Main execution for testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=5,
        n_redundant=2, n_classes=2, random_state=42
    )
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train all models
    results = trainer.train_all_models(X, y, cv=5)
    
    # Get results
    df_results = trainer.get_results_dataframe()
    print("\n" + "=" * 70)
    print("MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(df_results[['Model', 'Accuracy', 'F1-Score', 'ROC-AUC']].to_string(index=False))
    
    # Select best model
    best_name, best_model = trainer.select_best_model(metric='F1-Score')
