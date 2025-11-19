"""
Improved Feature Engineering for Hospital Mortality Prediction
Focus: Clinically meaningful features that may improve model performance
"""

import numpy as np
import pandas as pd


def add_refined_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clinically-informed features with less noise than previous FE
    
    Key principles:
    1. Use established clinical thresholds
    2. Focus on physiological instability
    3. Create interaction features for known risk combinations
    4. Avoid too many correlated features
    """
    df = df.copy()
    eps = 1e-3
    
    # ========================================
    # 1. VITAL SIGN INSTABILITY (variability)
    # ========================================
    # Only create ranges for the most critical vitals
    
    df['HR_variability'] = df['HeartRate_Max'] - df['HeartRate_Min']
    df['MAP_variability'] = df['MeanBP_Max'] - df['MeanBP_Min']
    df['RR_variability'] = df['RespRate_Max'] - df['RespRate_Min']
    
    # Temperature variability (fever/hypothermia swings)
    df['Temp_variability'] = df['TempC_Max'] - df['TempC_Min']
    
    # ========================================
    # 2. CRITICAL THRESHOLD FLAGS
    # ========================================
    # Based on established clinical criteria (SIRS, qSOFA, etc.)
    
    # Cardiovascular
    df['severe_tachycardia'] = (df['HeartRate_Max'] >= 130).astype(int)
    df['severe_hypotension'] = (df['SysBP_Min'] < 90).astype(int)
    df['map_low'] = (df['MeanBP_Min'] < 65).astype(int)  # organ perfusion threshold
    
    # Respiratory  
    df['severe_tachypnea'] = (df['RespRate_Max'] >= 30).astype(int)
    df['severe_hypoxia'] = (df['SpO2_Min'] < 90).astype(int)  # critical threshold
    
    # Temperature
    df['hypothermia'] = (df['TempC_Min'] < 36.0).astype(int)
    df['high_fever'] = (df['TempC_Max'] >= 38.5).astype(int)
    
    # Glucose
    df['severe_hyperglycemia'] = (df['Glucose_Max'] >= 250).astype(int)
    df['hypoglycemia'] = (df['Glucose_Min'] < 70).astype(int)
    
    # ========================================
    # 3. COMPOSITE CLINICAL SCORES
    # ========================================
    
    # Modified SIRS criteria (Systemic Inflammatory Response Syndrome)
    sirs_components = [
        (df['HeartRate_Max'] > 90).astype(int),
        (df['RespRate_Max'] > 20).astype(int),
        ((df['TempC_Max'] > 38.0) | (df['TempC_Min'] < 36.0)).astype(int),
    ]
    df['sirs_score'] = sum(sirs_components)
    
    # qSOFA-like score (quick Sequential Organ Failure Assessment)
    qsofa_components = [
        (df['RespRate_Max'] >= 22).astype(int),
        (df['SysBP_Min'] <= 100).astype(int),
        # Mental status would be here but we don't have GCS
    ]
    df['qsofa_like_score'] = sum(qsofa_components)
    
    # Shock index (HR/SBP) - KEY hemodynamic indicator
    df['shock_index_mean'] = df['HeartRate_Mean'] / (df['SysBP_Mean'].clip(lower=eps))
    df['shock_index_max'] = df['HeartRate_Max'] / (df['SysBP_Min'].clip(lower=eps))  # worst case
    
    # ========================================
    # 4. PHYSIOLOGICAL COHERENCE
    # ========================================
    # These capture relationships between vitals
    
    # Pulse pressure (SBP - DBP) - cardiac output indicator
    df['pulse_pressure'] = df['SysBP_Mean'] - df['DiasBP_Mean']
    df['narrow_pulse_pressure'] = (df['pulse_pressure'] < 25).astype(int)
    
    # Compensatory tachycardia (high HR with low BP = bad)
    df['inadequate_compensation'] = (
        (df['HeartRate_Mean'] > 100) & (df['SysBP_Mean'] < 100)
    ).astype(int)
    
    # Respiratory distress with hypoxia (very concerning)
    df['respiratory_failure_risk'] = (
        (df['RespRate_Max'] >= 25) & (df['SpO2_Min'] < 93)
    ).astype(int)
    
    # ========================================
    # 5. TIME-BASED / TRAJECTORY FEATURES
    # ========================================
    # Capture if patient is getting worse
    
    # Using min/max as proxies for early/late (rough approximation)
    df['BP_deterioration'] = (df['SysBP_Max'] - df['SysBP_Min']) / (df['SysBP_Max'].clip(lower=eps))
    df['SpO2_deterioration'] = (df['SpO2_Max'] - df['SpO2_Min']) / (df['SpO2_Max'].clip(lower=eps))
    
    # ========================================
    # 6. EXTREME VALUE FLAGS
    # ========================================
    # Flag when ANY vital reaches extreme value
    
    df['any_extreme_vital'] = (
        (df['severe_tachycardia'] == 1) |
        (df['severe_hypotension'] == 1) |
        (df['severe_tachypnea'] == 1) |
        (df['severe_hypoxia'] == 1) |
        (df['hypothermia'] == 1) |
        (df['high_fever'] == 1)
    ).astype(int)
    
    return df


def add_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ultra-minimal feature engineering - only the most proven features
    Use this if extensive FE hurts performance
    """
    df = df.copy()
    eps = 1e-3
    
    # Only the most critical
    df['shock_index'] = df['HeartRate_Mean'] / (df['SysBP_Mean'].clip(lower=eps))
    df['severe_hypotension'] = (df['SysBP_Min'] < 90).astype(int)
    df['severe_hypoxia'] = (df['SpO2_Min'] < 90).astype(int)
    df['pulse_pressure'] = df['SysBP_Mean'] - df['DiasBP_Mean']
    
    return df


def select_features_by_importance(X, y, feature_names, top_k=50):
    """
    Use simple RF to rank features, return top K
    Useful for reducing dimensionality
    """
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X, y)
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    top_features = importances.head(top_k)['feature'].tolist()
    
    return top_features, importances


def remove_correlated_features(df, threshold=0.95):
    """
    Remove highly correlated features to reduce redundancy
    """
    # Only look at numeric columns
    num_df = df.select_dtypes(include=[np.number])
    
    # Compute correlation matrix
    corr_matrix = num_df.corr().abs()
    
    # Upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    print(f"Dropping {len(to_drop)} highly correlated features (>{threshold}):")
    for col in to_drop:
        print(f"  - {col}")
    
    return df.drop(columns=to_drop)


# ============================================
# Feature Selection Strategies
# ============================================

def get_feature_set_1():
    """Original vitals only - baseline"""
    return [
        'HeartRate_Min', 'HeartRate_Max', 'HeartRate_Mean',
        'SysBP_Min', 'SysBP_Max', 'SysBP_Mean',
        'DiasBP_Min', 'DiasBP_Max', 'DiasBP_Mean',
        'MeanBP_Min', 'MeanBP_Max', 'MeanBP_Mean',
        'RespRate_Min', 'RespRate_Max', 'RespRate_Mean',
        'TempC_Min', 'TempC_Max', 'TempC_Mean',
        'SpO2_Min', 'SpO2_Max', 'SpO2_Mean',
        'Glucose_Min', 'Glucose_Max', 'Glucose_Mean',
    ]


def get_feature_set_2():
    """Vitals + demographics + care unit"""
    base = get_feature_set_1()
    return base + [
        'GENDER', 'ADMISSION_TYPE', 'INSURANCE', 
        'MARITAL_STATUS', 'ETHNICITY', 'FIRST_CAREUNIT',
        'Diff',  # age at admission
    ]


def get_feature_set_3():
    """Minimal FE: only shock index and critical flags"""
    base = get_feature_set_2()
    fe = [
        'shock_index_mean', 'shock_index_max',
        'severe_hypotension', 'severe_hypoxia',
        'severe_tachycardia', 'severe_tachypnea',
        'map_low',
    ]
    return base + fe


def get_feature_set_4():
    """Full refined FE"""
    base = get_feature_set_2()
    fe = [
        # Variability
        'HR_variability', 'MAP_variability', 'RR_variability', 'Temp_variability',
        
        # Flags
        'severe_tachycardia', 'severe_hypotension', 'map_low',
        'severe_tachypnea', 'severe_hypoxia',
        'hypothermia', 'high_fever',
        'severe_hyperglycemia', 'hypoglycemia',
        
        # Scores
        'sirs_score', 'qsofa_like_score',
        'shock_index_mean', 'shock_index_max',
        
        # Physiological
        'pulse_pressure', 'narrow_pulse_pressure',
        'inadequate_compensation', 'respiratory_failure_risk',
        
        # Deterioration
        'BP_deterioration', 'SpO2_deterioration',
        
        # Extreme
        'any_extreme_vital',
    ]
    return base + fe


# ============================================
# Evaluation Helper
# ============================================

def compare_feature_sets(X_train, y_train, X_valid, y_valid, 
                        feature_sets_dict, preprocessor):
    """
    Compare different feature set strategies
    
    Parameters:
    -----------
    feature_sets_dict : dict
        {'set_name': list_of_features}
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    
    results = {}
    
    for set_name, features in feature_sets_dict.items():
        # Filter to available features
        available_features = [f for f in features if f in X_train.columns]
        
        X_train_subset = X_train[available_features]
        X_valid_subset = X_valid[available_features]
        
        # Simple RF
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('clf', rf),
        ])
        
        # Fit and evaluate
        pipe.fit(X_train_subset, y_train)
        
        y_train_proba = pipe.predict_proba(X_train_subset)[:, 1]
        y_valid_proba = pipe.predict_proba(X_valid_subset)[:, 1]
        
        train_auc = roc_auc_score(y_train, y_train_proba)
        valid_auc = roc_auc_score(y_valid, y_valid_proba)
        
        results[set_name] = {
            'n_features': len(available_features),
            'train_auc': train_auc,
            'valid_auc': valid_auc,
            'overfit': train_auc - valid_auc,
        }
        
        print(f"{set_name}: {len(available_features)} features, "
              f"Valid AUC: {valid_auc:.4f}")
    
    return pd.DataFrame(results).T
