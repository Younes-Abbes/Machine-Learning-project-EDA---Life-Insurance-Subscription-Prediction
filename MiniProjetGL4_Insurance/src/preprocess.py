"""
Preprocessing Module for Life Insurance Subscription Prediction
================================================================
This module handles all data preprocessing steps:
- Loading and cleaning data
- Outlier treatment (IQR method)
- Encoding categorical variables
- Feature scaling
- SMOTE oversampling for class imbalance
- Train/test split

Author: GL4 Data Mining Mini-Project Team
Date: 2026
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    A comprehensive data preprocessor for insurance subscription prediction.
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV data file
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        
        # Define column categories
        self.categorical_cols = ['Gender', 'Vehicle_Age', 'Vehicle_Damage']
        self.binary_cols = ['Driving_License', 'Previously_Insured']
        self.numeric_cols = ['Age', 'Annual_Premium', 'Vintage', 'Region_Code', 'Policy_Sales_Channel']
        self.target_col = 'Response'
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load the dataset from CSV file.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataframe
        """
        if data_path:
            self.data_path = data_path
            
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Drop 'id' column if exists
        if 'id' in self.df.columns:
            self.df = self.df.drop('id', axis=1)
            print("Dropped 'id' column")
            
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def get_data_info(self) -> dict:
        """
        Get comprehensive information about the dataset.
        
        Returns:
        --------
        dict
            Dictionary containing data information
        """
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'class_distribution': self.df[self.target_col].value_counts().to_dict()
        }
        return info
    
    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the dataset.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame without duplicates
        """
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_rows - len(self.df)
        print(f"Removed {removed} duplicate rows")
        return self.df
    
    def handle_outliers_iqr(self, column: str) -> pd.DataFrame:
        """
        Handle outliers using IQR method (clip values).
        
        Parameters:
        -----------
        column : str
            Column name to process
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with clipped outliers
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = ((self.df[column] < lower_bound) | (self.df[column] > upper_bound)).sum()
        
        self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"Outliers handled in '{column}': {outliers_before} values clipped")
        print(f"  - Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
        
        return self.df
    
    def encode_categorical(self) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with encoded categorical variables
        """
        print("\nEncoding categorical variables...")
        
        # Gender: LabelEncoder
        if 'Gender' in self.df.columns:
            le_gender = LabelEncoder()
            self.df['Gender'] = le_gender.fit_transform(self.df['Gender'])
            self.label_encoders['Gender'] = le_gender
            print(f"  - Gender: {dict(zip(le_gender.classes_, range(len(le_gender.classes_))))}")
        
        # Vehicle_Age: Manual mapping (ordinal)
        if 'Vehicle_Age' in self.df.columns:
            vehicle_age_map = {'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}
            self.df['Vehicle_Age'] = self.df['Vehicle_Age'].map(vehicle_age_map)
            print(f"  - Vehicle_Age: {vehicle_age_map}")
        
        # Vehicle_Damage: Yes/No to 1/0
        if 'Vehicle_Damage' in self.df.columns:
            damage_map = {'Yes': 1, 'No': 0}
            self.df['Vehicle_Damage'] = self.df['Vehicle_Damage'].map(damage_map)
            print(f"  - Vehicle_Damage: {damage_map}")
        
        return self.df
    
    def scale_features(self, columns: list = None) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Parameters:
        -----------
        columns : list
            List of columns to scale. If None, uses self.numeric_cols
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with scaled features
        """
        if columns is None:
            columns = self.numeric_cols
            
        print(f"\nScaling features: {columns}")
        
        # Store original values for reference
        self.original_stats = {col: {'mean': self.df[col].mean(), 'std': self.df[col].std()} 
                              for col in columns}
        
        self.df[columns] = self.scaler.fit_transform(self.df[columns])
        
        return self.df
    
    def apply_smote(self, X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple:
        """
        Apply SMOTE oversampling to handle class imbalance.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        tuple
            (X_resampled, y_resampled)
        """
        print("\nApplying SMOTE oversampling...")
        print(f"  - Before: Class 0: {(y == 0).sum()}, Class 1: {(y == 1).sum()}")
        
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"  - After: Class 0: {(y_resampled == 0).sum()}, Class 1: {(y_resampled == 1).sum()}")
        
        return X_resampled, y_resampled
    
    def full_preprocess(self, apply_smote: bool = True, test_size: float = 0.2, 
                        random_state: int = 42) -> tuple:
        """
        Execute the full preprocessing pipeline.
        
        Parameters:
        -----------
        apply_smote : bool
            Whether to apply SMOTE oversampling
        test_size : float
            Proportion of test set
        random_state : int
            Random state for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        print("=" * 60)
        print("FULL PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # 1. Remove duplicates
        self.remove_duplicates()
        
        # 2. Handle outliers in Annual_Premium
        print("\nHandling outliers...")
        self.handle_outliers_iqr('Annual_Premium')
        
        # 3. Encode categorical variables
        self.encode_categorical()
        
        # 4. Store feature names before scaling
        feature_cols = [col for col in self.df.columns if col != self.target_col]
        self.feature_names = feature_cols
        
        # 5. Scale numerical features
        self.scale_features()
        
        # 6. Prepare X and y
        X = self.df[feature_cols]
        y = self.df[self.target_col]
        
        # 7. Train/test split
        print(f"\nSplitting data (test_size={test_size}, stratify=y)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"  - Train: {X_train.shape[0]} samples")
        print(f"  - Test: {X_test.shape[0]} samples")
        
        # 8. Apply SMOTE on training data only
        if apply_smote:
            X_train, y_train = self.apply_smote(X_train, y_train, random_state)
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test, 
                           output_dir: str = 'data/processed'):
        """
        Save processed data to pickle files.
        
        Parameters:
        -----------
        X_train, X_test, y_train, y_test : arrays
            Processed data splits
        output_dir : str
            Output directory path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(X_train, output_path / 'X_train.pkl')
        joblib.dump(X_test, output_path / 'X_test.pkl')
        joblib.dump(y_train, output_path / 'y_train.pkl')
        joblib.dump(y_test, output_path / 'y_test.pkl')
        joblib.dump(self.scaler, output_path / 'scaler.pkl')
        joblib.dump(self.label_encoders, output_path / 'label_encoders.pkl')
        joblib.dump(self.feature_names, output_path / 'feature_names.pkl')
        
        print(f"\nProcessed data saved to {output_path}")
    
    def load_processed_data(self, input_dir: str = 'data/processed') -> tuple:
        """
        Load processed data from pickle files.
        
        Parameters:
        -----------
        input_dir : str
            Input directory path
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        input_path = Path(input_dir)
        
        X_train = joblib.load(input_path / 'X_train.pkl')
        X_test = joblib.load(input_path / 'X_test.pkl')
        y_train = joblib.load(input_path / 'y_train.pkl')
        y_test = joblib.load(input_path / 'y_test.pkl')
        self.scaler = joblib.load(input_path / 'scaler.pkl')
        self.label_encoders = joblib.load(input_path / 'label_encoders.pkl')
        self.feature_names = joblib.load(input_path / 'feature_names.pkl')
        
        print(f"Processed data loaded from {input_path}")
        return X_train, X_test, y_train, y_test


def preprocess_single_input(input_data: dict, scaler, feature_names: list) -> np.ndarray:
    """
    Preprocess a single input for prediction (used in Streamlit app).
    
    Parameters:
    -----------
    input_data : dict
        Dictionary with feature names and values
    scaler : StandardScaler
        Fitted scaler object
    feature_names : list
        List of feature names in correct order
        
    Returns:
    --------
    np.ndarray
        Preprocessed input ready for prediction
    """
    # Create DataFrame from input
    df = pd.DataFrame([input_data])
    
    # Ensure correct column order
    df = df[feature_names]
    
    # Scale numerical features
    numeric_cols = ['Age', 'Annual_Premium', 'Vintage', 'Region_Code', 'Policy_Sales_Channel']
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    
    return df.values


# Main execution
if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor()
    
    # Load data
    df = preprocessor.load_data('data/train.csv')
    
    # Get data info
    info = preprocessor.get_data_info()
    print("\nData Info:")
    print(f"Shape: {info['shape']}")
    print(f"Missing values: {sum(info['missing_values'].values())}")
    print(f"Duplicates: {info['duplicates']}")
    print(f"Class distribution: {info['class_distribution']}")
    
    # Full preprocessing
    X_train, X_test, y_train, y_test = preprocessor.full_preprocess()
    
    # Save processed data
    preprocessor.save_processed_data(X_train, X_test, y_train, y_test)
