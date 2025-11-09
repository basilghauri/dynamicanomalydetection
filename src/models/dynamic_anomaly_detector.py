"""
Dynamic Anomaly Detection Model for Transaction Data
Handles variable number of features and adapts automatically
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import joblib
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DynamicAnomalyDetector:
    """
    Anomaly detection model that can adapt to changing feature sets.
    
    Features:
    - Automatically handles new features when they appear
    - Maintains backward compatibility with existing features
    - Uses Isolation Forest for robust anomaly detection
    - Supports incremental learning
    """
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100,
        max_samples: Union[int, float] = 'auto',
        random_state: int = 42,
        scaler_type: str = 'robust'
    ):
        """
        Initialize the dynamic anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
            n_estimators: Number of trees in Isolation Forest
            max_samples: Number of samples to draw for each tree
            random_state: Random seed for reproducibility
            scaler_type: Type of scaler ('robust' or 'standard')
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state
        self.scaler_type = scaler_type
        
        # Core components
        self.model: Optional[IsolationForest] = None
        self.scaler: Optional[Union[StandardScaler, RobustScaler]] = None
        self.imputer: Optional[SimpleImputer] = None
        
        # Feature management
        self.known_features: List[str] = []
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted: bool = False
        
    def _initialize_scaler(self):
        """Initialize the appropriate scaler."""
        if self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
    
    def _initialize_imputer(self):
        """Initialize imputer for missing values."""
        self.imputer = SimpleImputer(strategy='median')
    
    def _extract_features(self, data: Union[pd.DataFrame, Dict, List[Dict]]) -> pd.DataFrame:
        """
        Extract features from various input formats.
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            
        Returns:
            DataFrame with features
        """
        if isinstance(data, pd.DataFrame):
            df = data.copy()
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return df
    
    def _align_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align features with known feature set, adding missing features with defaults.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with aligned features
        """
        aligned_df = pd.DataFrame()
        
        # Add known features
        for feature in self.known_features:
            if feature in df.columns:
                aligned_df[feature] = df[feature]
            else:
                # Use default value based on feature statistics
                if feature in self.feature_stats:
                    default_value = self.feature_stats[feature].get('median', 0.0)
                else:
                    default_value = 0.0
                aligned_df[feature] = default_value
                logger.warning(f"Feature '{feature}' not found, using default: {default_value}")
        
        # Add new features that weren't seen before
        new_features = [f for f in df.columns if f not in self.known_features]
        if new_features:
            logger.info(f"Detected {len(new_features)} new features: {new_features}")
            for feature in new_features:
                aligned_df[feature] = df[feature]
                # Initialize statistics for new feature
                self.feature_stats[feature] = {
                    'mean': float(df[feature].mean()) if df[feature].dtype in ['int64', 'float64'] else 0.0,
                    'median': float(df[feature].median()) if df[feature].dtype in ['int64', 'float64'] else 0.0,
                    'std': float(df[feature].std()) if df[feature].dtype in ['int64', 'float64'] else 1.0
                }
            self.known_features.extend(new_features)
        
        return aligned_df
    
    def _update_feature_stats(self, df: pd.DataFrame):
        """Update feature statistics from new data."""
        for feature in df.columns:
            if df[feature].dtype in ['int64', 'float64']:
                if feature not in self.feature_stats:
                    self.feature_stats[feature] = {}
                self.feature_stats[feature].update({
                    'mean': float(df[feature].mean()),
                    'median': float(df[feature].median()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max())
                })
    
    def fit(self, data: Union[pd.DataFrame, Dict, List[Dict]]):
        """
        Fit the anomaly detection model on training data.
        
        Args:
            data: Training data (DataFrame, dict, or list of dicts)
        """
        logger.info("Fitting anomaly detection model...")
        
        # Extract and prepare data
        df = self._extract_features(data)
        
        # Initialize components if needed
        if self.scaler is None:
            self._initialize_scaler()
        if self.imputer is None:
            self._initialize_imputer()
        
        # Update known features and statistics
        if not self.known_features:
            self.known_features = list(df.columns)
            logger.info(f"Initial feature set: {self.known_features}")
        
        self._update_feature_stats(df)
        
        # Align features
        aligned_df = self._align_features(df)
        
        # Handle missing values
        aligned_df = pd.DataFrame(
            self.imputer.fit_transform(aligned_df),
            columns=aligned_df.columns,
            index=aligned_df.index
        )
        
        # Scale features
        scaled_data = self.scaler.fit_transform(aligned_df)
        
        # Fit Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(scaled_data)
        
        self.is_fitted = True
        logger.info(f"Model fitted successfully with {len(self.known_features)} features")
    
    def predict(
        self,
        data: Union[pd.DataFrame, Dict, List[Dict]],
        return_scores: bool = True
    ) -> Union[np.ndarray, tuple]:
        """
        Predict anomalies in the input data.
        
        Args:
            data: Input data (DataFrame, dict, or list of dicts)
            return_scores: Whether to return anomaly scores
            
        Returns:
            Anomaly predictions (-1 for anomaly, 1 for normal)
            If return_scores=True, also returns anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract and prepare data
        df = self._extract_features(data)
        
        # Align features (handles new features automatically)
        aligned_df = self._align_features(df)
        
        # Handle missing values
        aligned_df = pd.DataFrame(
            self.imputer.transform(aligned_df),
            columns=aligned_df.columns,
            index=aligned_df.index
        )
        
        # Scale features
        scaled_data = self.scaler.transform(aligned_df)
        
        # Predict
        predictions = self.model.predict(scaled_data)
        
        if return_scores:
            scores = self.model.score_samples(scaled_data)
            # Convert scores to anomaly scores (lower = more anomalous)
            anomaly_scores = -scores
            return predictions, anomaly_scores
        else:
            return predictions
    
    def partial_fit(self, data: Union[pd.DataFrame, Dict, List[Dict]]):
        """
        Incrementally update the model with new data.
        Note: Isolation Forest doesn't support true incremental learning,
        so this refits the model with accumulated data.
        
        Args:
            data: New training data
        """
        logger.info("Performing partial fit...")
        # For true incremental learning, you'd need to maintain a data buffer
        # For now, we'll refit with the new data
        # In production, you might want to implement a sliding window approach
        self.fit(data)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance (based on feature statistics and model).
        Note: Isolation Forest doesn't provide direct feature importance,
        so we return feature statistics.
        
        Returns:
            Dictionary mapping features to their importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance = {}
        for feature in self.known_features:
            if feature in self.feature_stats:
                # Use coefficient of variation as a proxy for importance
                stats = self.feature_stats[feature]
                if stats.get('std', 0) > 0:
                    cv = abs(stats['std'] / stats['mean']) if stats['mean'] != 0 else stats['std']
                    importance[feature] = cv
                else:
                    importance[feature] = 0.0
        
        # Normalize importance scores
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def save(self, filepath: str):
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'known_features': self.known_features,
            'feature_stats': self.feature_stats,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'random_state': self.random_state,
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load(self, filepath: str):
        """
        Load the model from disk.
        
        Args:
            filepath: Path to load the model from
        """
        load_path = Path(filepath)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        model_data = joblib.load(load_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.imputer = model_data['imputer']
        self.known_features = model_data['known_features']
        self.feature_stats = model_data['feature_stats']
        self.contamination = model_data['contamination']
        self.n_estimators = model_data['n_estimators']
        self.max_samples = model_data['max_samples']
        self.random_state = model_data['random_state']
        self.scaler_type = model_data['scaler_type']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {load_path} with {len(self.known_features)} features")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model state.
        
        Returns:
            Dictionary with model information
        """
        return {
            'is_fitted': self.is_fitted,
            'num_features': len(self.known_features),
            'features': self.known_features,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'feature_stats': self.feature_stats
        }

