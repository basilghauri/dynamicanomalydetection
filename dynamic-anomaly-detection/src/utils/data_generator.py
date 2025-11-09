"""
Utility script to generate sample transaction data for testing
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import random


def generate_transaction_data(
    n_samples: int = 1000,
    n_features: int = 10,
    anomaly_ratio: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic transaction data with anomalies.
    
    Args:
        n_samples: Number of transactions to generate
        n_features: Number of features per transaction
        anomaly_ratio: Proportion of anomalies
        random_state: Random seed
        
    Returns:
        DataFrame with transaction data
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    n_anomalies = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomalies
    
    data = []
    
    # Generate normal transactions
    for i in range(n_normal):
        transaction = {
            'transaction_id': f'txn_{i:06d}',
            'amount': np.random.normal(100, 30),
            'merchant_category': random.choice(['retail', 'restaurant', 'gas', 'grocery', 'online']),
            'hour_of_day': np.random.randint(0, 24),
            'day_of_week': np.random.randint(0, 7),
            'transaction_type': random.choice(['debit', 'credit', 'transfer']),
            'user_age': np.random.normal(35, 10),
            'account_balance': np.random.normal(5000, 2000),
            'transaction_frequency': np.random.poisson(5),
            'location_latitude': np.random.normal(40.0, 0.1),
            'location_longitude': np.random.normal(-74.0, 0.1),
        }
        
        # Add additional numeric features
        for j in range(10, n_features):
            transaction[f'feature_{j}'] = np.random.normal(0, 1)
        
        data.append(transaction)
    
    # Generate anomalous transactions
    for i in range(n_anomalies):
        transaction = {
            'transaction_id': f'txn_anomaly_{i:06d}',
            'amount': np.random.normal(500, 200),  # Higher amounts
            'merchant_category': random.choice(['casino', 'luxury', 'unknown']),
            'hour_of_day': np.random.choice([2, 3, 4]),  # Unusual hours
            'day_of_week': np.random.randint(0, 7),
            'transaction_type': random.choice(['debit', 'credit', 'transfer']),
            'user_age': np.random.normal(35, 10),
            'account_balance': np.random.normal(500, 500),  # Low balance
            'transaction_frequency': np.random.poisson(20),  # High frequency
            'location_latitude': np.random.normal(40.0, 5.0),  # Far from normal
            'location_longitude': np.random.normal(-74.0, 5.0),
        }
        
        # Add additional numeric features with unusual patterns
        for j in range(10, n_features):
            transaction[f'feature_{j}'] = np.random.normal(5, 2)  # Outlier values
        
        data.append(transaction)
    
    # Shuffle data
    random.shuffle(data)
    
    return pd.DataFrame(data)


def generate_dynamic_transaction_data(
    base_features: List[str],
    new_features: List[str],
    n_samples: int = 100
) -> pd.DataFrame:
    """
    Generate transaction data with new features added.
    Simulates the scenario where new attributes are introduced.
    
    Args:
        base_features: List of base feature names
        new_features: List of new feature names to add
        n_samples: Number of samples
        
    Returns:
        DataFrame with both base and new features
    """
    np.random.seed(42)
    
    data = []
    for i in range(n_samples):
        transaction = {}
        
        # Add base features
        for feature in base_features:
            if 'amount' in feature.lower():
                transaction[feature] = np.random.normal(100, 30)
            elif 'age' in feature.lower():
                transaction[feature] = np.random.normal(35, 10)
            elif 'balance' in feature.lower():
                transaction[feature] = np.random.normal(5000, 2000)
            else:
                transaction[feature] = np.random.normal(0, 1)
        
        # Add new features
        for feature in new_features:
            transaction[feature] = np.random.normal(0, 1)
        
        data.append(transaction)
    
    return pd.DataFrame(data)

