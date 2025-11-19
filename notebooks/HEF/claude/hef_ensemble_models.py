"""
Advanced Ensemble Models for Hospital Mortality Prediction
Focuses on interpretability and causal inference potential
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns


class HEFEnsembleModels:
    """Collection of ensemble approaches for mortality prediction"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def create_base_models(self):
        """Create diverse base models for ensembling"""
        
        # 1. Random Forest - your current best
        rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=self.random_state,
            oob_score=True,  # useful for monitoring
        )
        
        # 2. Extra Trees - more random, often better generalization
        et = ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1,
            random_state=self.random_state,
        )
        
        # 3. Gradient Boosting - sequential learning
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            subsample=0.8,
            max_features='sqrt',
            random_state=self.random_state,
        )
        
        # 4. Regularized Logistic Regression - linear baseline, interpretable
        lr = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            C=0.1,
            l1_ratio=0.5,
            class_weight='balanced',
            max_iter=1000,
            random_state=self.random_state,
            n_jobs=-1,
        )
        
        return {'rf': rf, 'et': et, 'gb': gb, 'lr': lr}
    
    def create_voting_ensemble(self, base_models):
        """Soft voting ensemble - averages probabilities"""
        voting = VotingClassifier(
            estimators=[
                ('rf', base_models['rf']),
                ('et', base_models['et']),
                ('gb', base_models['gb']),
                ('lr', base_models['lr']),
            ],
            voting='soft',
            weights=[2, 2, 2, 1],  # can tune these
            n_jobs=-1,
        )
        return voting
    
    def create_stacking_ensemble(self, base_models):
        """Stacking - meta-learner on base predictions"""
        
        # Use subset of base models as first layer
        estimators = [
            ('rf', base_models['rf']),
            ('et', base_models['et']),
            ('gb', base_models['gb']),
        ]
        
        # Simple logistic regression as meta-learner
        meta_learner = LogisticRegression(
            penalty='l2',
            C=1.0,
            class_weight='balanced',
            random_state=self.random_state,
            max_iter=1000,
        )
        
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,  # internal CV for meta-features
            n_jobs=-1,
        )
        return stacking
    
    def fit_and_evaluate(self, X_train, y_train, X_valid, y_valid, 
                        preprocessor, model_name, model):
        """Fit a model with preprocessing and evaluate"""
        from sklearn.pipeline import Pipeline
        
        # Create pipeline
        pipe = Pipeline([
            ('preprocess', preprocessor),
            ('model', model),
        ])
        
        # Fit
        print(f"\nFitting {model_name}...")
        pipe.fit(X_train, y_train)
        
        # Predict
        y_train_proba = pipe.predict_proba(X_train)[:, 1]
        y_valid_proba = pipe.predict_proba(X_valid)[:, 1]
        
        # Evaluate
        train_auc = roc_auc_score(y_train, y_train_proba)
        valid_auc = roc_auc_score(y_valid, y_valid_proba)
        
        print(f"{model_name} - Train AUC: {train_auc:.4f}, Valid AUC: {valid_auc:.4f}")
        
        # Store results
        self.models[model_name] = pipe
        self.results[model_name] = {
            'train_auc': train_auc,
            'valid_auc': valid_auc,
            'y_valid_proba': y_valid_proba,
        }
        
        return pipe
    
    def calibrate_model(self, model, X_train, y_train, method='isotonic'):
        """Calibrate probabilities for better reliability"""
        calibrated = CalibratedClassifierCV(
            model,
            method=method,  # 'isotonic' or 'sigmoid'
            cv=5,
        )
        return calibrated
    
    def analyze_feature_importance(self, model_name, feature_names, top_n=20):
        """Extract and plot feature importances for tree-based models"""
        model = self.models[model_name]
        
        # Get the actual estimator (after preprocessing)
        if hasattr(model, 'named_steps'):
            estimator = model.named_steps['model']
        else:
            estimator = model
        
        # Extract importances
        if hasattr(estimator, 'feature_importances_'):
            importances = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            importances = np.abs(estimator.coef_[0])
        else:
            print(f"Cannot extract importances from {model_name}")
            return None
        
        # Create dataframe
        feat_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        return feat_imp
    
    def plot_roc_curves(self, y_valid):
        """Compare ROC curves of all models"""
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(y_valid, result['y_valid_proba'])
            auc = result['valid_auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Model Comparison')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_precision_recall_curves(self, y_valid):
        """Compare precision-recall curves"""
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            precision, recall, _ = precision_recall_curve(
                y_valid, result['y_valid_proba']
            )
            plt.plot(recall, precision, label=name)
        
        baseline = y_valid.mean()
        plt.axhline(baseline, color='k', linestyle='--', 
                   label=f'Baseline ({baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def get_best_model(self):
        """Return the model with highest validation AUC"""
        best_name = max(self.results.items(), 
                       key=lambda x: x[1]['valid_auc'])[0]
        return best_name, self.models[best_name]
    
    def create_final_ensemble_weights(self):
        """Create weighted average based on validation performance"""
        # Performance-based weighting
        weights = {}
        total_auc = sum(r['valid_auc'] for r in self.results.values())
        
        for name, result in self.results.items():
            weights[name] = result['valid_auc'] / total_auc
        
        return weights
    
    def predict_weighted_ensemble(self, X, weights=None):
        """Make predictions using weighted average of all models"""
        if weights is None:
            weights = self.create_final_ensemble_weights()
        
        predictions = np.zeros(len(X))
        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1]
            predictions += weights[name] * pred
        
        return predictions


def sample_weight_by_minority(y_train, minority_weight=3.0):
    """
    Create sample weights that emphasize minority class
    Alternative to class_weight='balanced'
    """
    weights = np.ones(len(y_train))
    weights[y_train == 1] = minority_weight
    return weights


def create_stratified_folds(X, y, n_splits=5, random_state=42):
    """Create stratified k-fold splits for CV"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, 
                         random_state=random_state)
    return list(skf.split(X, y))


def evaluate_with_cv(model, X, y, cv_splits, scoring='roc_auc'):
    """Evaluate model with cross-validation"""
    scores = cross_val_score(
        model, X, y, 
        cv=cv_splits,
        scoring=scoring,
        n_jobs=-1,
    )
    return scores.mean(), scores.std()


# ============ For Causal Inference Prep ============

def extract_treatment_effect_features(X, y, model, feature_names):
    """
    Identify features most predictive of outcome
    Useful for identifying potential confounders in causal analysis
    """
    # Fit model
    model.fit(X, y)
    
    # Get importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # Create ranking
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    
    return feature_importance


def identify_risk_profiles(X, y, model, n_clusters=5):
    """
    Cluster patients by predicted risk
    Useful for identifying distinct patient subgroups
    """
    from sklearn.cluster import KMeans
    
    # Get predictions
    proba = model.predict_proba(X)[:, 1]
    
    # Add predicted risk to features
    X_with_risk = X.copy()
    X_with_risk['predicted_risk'] = proba
    
    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_with_risk)
    
    # Analyze clusters
    cluster_stats = pd.DataFrame({
        'cluster': clusters,
        'actual_outcome': y,
        'predicted_risk': proba,
    }).groupby('cluster').agg({
        'actual_outcome': ['count', 'mean'],
        'predicted_risk': 'mean',
    })
    
    return clusters, cluster_stats


def plot_feature_importance(feat_imp, top_n=20, title='Feature Importance'):
    """Plot feature importance bar chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_features = feat_imp.head(top_n)
    sns.barplot(data=top_features, y='feature', x='importance', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Importance')
    plt.tight_layout()
    
    return fig
