"""
Machine Learning for IPD Strategy Prediction
============================================
Uses ML to predict if a strategy will be successful.
"""

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from ipd_core import (
    IPDGame, Strategy, create_strategy_from_bitstring,
    generate_random_strategy, REFERENCE_STRATEGIES, TFT, ALLD, ALLC
)
from optimization import FitnessEvaluator


@dataclass
class MLResult:
    """Results from ML training and evaluation"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    feature_importance: Dict[str, float]
    train_size: int
    test_size: int


def extract_features(strategy: Strategy) -> Dict[str, float]:
    """
    Extract features from a strategy bitstring.
    
    Features:
    - initial_cooperate: 1 if initial move is cooperate
    - coop_rate: Overall cooperation rate in strategy
    - response_to_cc: Probability of cooperating after mutual cooperation
    - response_to_cd: Probability of cooperating after being defected on
    - response_to_dc: Probability of cooperating after defecting
    - response_to_dd: Probability of cooperating after mutual defection
    - nice: 1 if starts with cooperation
    - provokable: 1 if punishes defection
    - forgiving: 1 if returns to cooperation
    """
    bits = strategy.bitstring
    features = {}
    
    if strategy.memory_depth == 1 and len(bits) >= 5:
        # Initial move
        features['initial_cooperate'] = 1.0 if bits[0] == '0' else 0.0
        
        # Response patterns (index 1-4 correspond to CC, CD, DC, DD)
        features['response_to_cc'] = 1.0 if bits[1] == '0' else 0.0
        features['response_to_cd'] = 1.0 if bits[2] == '0' else 0.0
        features['response_to_dc'] = 1.0 if bits[3] == '0' else 0.0
        features['response_to_dd'] = 1.0 if bits[4] == '0' else 0.0
        
        # Overall cooperation rate
        features['coop_rate'] = bits.count('0') / len(bits)
        
        # TFT-like properties
        features['nice'] = features['initial_cooperate']
        features['provokable'] = 1.0 if features['response_to_cd'] == 0.0 else 0.0
        features['forgiving'] = 1.0 if features['response_to_dc'] == 1.0 else 0.0
        features['tft_like'] = 1.0 if bits == '00101' else 0.0
        
    else:
        # For deeper memory, use aggregate features
        features['initial_cooperate'] = 1.0 if bits[0] == '0' else 0.0
        features['coop_rate'] = bits.count('0') / len(bits)
        features['response_to_cc'] = 0.5
        features['response_to_cd'] = 0.5
        features['response_to_dc'] = 0.5
        features['response_to_dd'] = 0.5
        features['nice'] = features['initial_cooperate']
        features['provokable'] = 0.5
        features['forgiving'] = 0.5
        features['tft_like'] = 0.0
    
    return features


def generate_training_data(
    n_samples: int,
    opponent_strategies: List[Strategy],
    num_rounds: int = 100,
    memory_depth: int = 1,
    good_threshold_percentile: float = 80
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Strategy], List[float]]:
    """
    Generate training data for ML models.
    
    Returns:
        X: Feature matrix
        y: Labels (1=good, 0=bad)
        feature_names: List of feature names
    """
    # Generate random strategies
    strategies = [generate_random_strategy(memory_depth) for _ in range(n_samples)]

    # For memory depth 1 there are only 2^5=32 unique strategies, so very high ML
    # scores are expected once the dataset is large enough to cover most configurations.
    if memory_depth == 1 and n_samples > 1000:
        print(
            "[ML Note] memory_depth=1 has only 32 unique strategies (2^5). "
            "High accuracy is expected due to the limited search space."
        )
    
    # Evaluate fitness
    evaluator = FitnessEvaluator(opponent_strategies, num_rounds)
    fitnesses = evaluator.evaluate_population(strategies)
    
    # Extract features
    feature_list = []
    for s in strategies:
        features = extract_features(s)
        feature_list.append(features)
    
    # Create feature matrix
    feature_names = list(feature_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in feature_list])
    
    # Create labels (top percentile = good)
    threshold = np.percentile(fitnesses, good_threshold_percentile)
    y = np.array([1 if f >= threshold else 0 for f in fitnesses])
    
    return X, y, feature_names, strategies, fitnesses


class StrategyPredictor:
    """ML-based predictor for strategy success"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              feature_names: List[str],
              test_size: float = 0.2,
              memory_depth: Optional[int] = None) -> List[MLResult]:
        """
        Train multiple ML models and return results.
        """
        self.feature_names = feature_names
        
        # Split data with stratification whenever each class has >=2 samples.
        # This reduces leakage-like artifacts from imbalanced train/test class splits.
        unique, counts = np.unique(y, return_counts=True)
        stratify = y if len(unique) > 1 and int(np.min(counts)) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=stratify
        )

        if memory_depth == 1 and len(X_train) > 1000:
            print(
                "[ML Note] Perfect/near-perfect accuracy can be expected at memory_depth=1 "
                "because only 32 unique strategies exist (2^5)."
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'NeuralNetwork': MLPClassifier(hidden_layer_sizes=(50, 25), 
                                          max_iter=1000, random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42)
        }
        
        results = []
        
        for name, model in models.items():
            # Train
            if name in ['SVM', 'LogisticRegression', 'NeuralNetwork']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(feature_names, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                importance = dict(zip(feature_names, np.abs(model.coef_[0])))
            else:
                importance = {name: 0.0 for name in feature_names}
            
            result = MLResult(
                model_name=name,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                feature_importance=importance,
                train_size=len(X_train),
                test_size=len(X_test)
            )
            
            results.append(result)
            self.models[name] = model
        
        self.is_trained = True
        return results
    
    def predict(self, strategy: Strategy, model_name: str = 'RandomForest') -> float:
        """Predict probability of strategy being good"""
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        features = extract_features(strategy)
        X = np.array([[features[name] for name in self.feature_names]])
        
        model = self.models[model_name]
        
        if model_name in ['SVM', 'LogisticRegression', 'NeuralNetwork']:
            X = self.scaler.transform(X)
        
        prob = model.predict_proba(X)[0][1]
        return prob
    
    def find_good_strategy(self, 
                          n_attempts: int = 100,
                          model_name: str = 'RandomForest',
                          memory_depth: int = 1) -> Strategy:
        """
        Use ML to guide search for good strategies.
        Generate candidates and pick the one with highest predicted score.
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        candidates = [generate_random_strategy(memory_depth) for _ in range(n_attempts)]
        
        best_candidate = None
        best_score = -1
        
        for candidate in candidates:
            score = self.predict(candidate, model_name)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate


def analyze_patterns(good_strategies: List[Strategy], 
                     bad_strategies: List[Strategy]) -> Dict:
    """
    Analyze patterns that distinguish good from bad strategies.
    """
    analysis = {
        'good_count': len(good_strategies),
        'bad_count': len(bad_strategies),
        'patterns': {}
    }
    
    # Extract features for both groups
    good_features = [extract_features(s) for s in good_strategies]
    bad_features = [extract_features(s) for s in bad_strategies]
    
    # Compare averages
    for key in good_features[0].keys():
        good_avg = np.mean([f[key] for f in good_features])
        bad_avg = np.mean([f[key] for f in bad_features])
        analysis['patterns'][key] = {
            'good_avg': good_avg,
            'bad_avg': bad_avg,
            'difference': good_avg - bad_avg
        }
    
    # Most common bitstrings in good strategies
    good_bitstrings = [s.bitstring for s in good_strategies]
    from collections import Counter
    bitstring_counts = Counter(good_bitstrings)
    analysis['common_good_strategies'] = bitstring_counts.most_common(5)
    
    return analysis


if __name__ == "__main__":
    print("=" * 60)
    print("Machine Learning Prediction Test")
    print("=" * 60)
    
    # Generate training data
    print("\n1. Generating Training Data (1000 samples)...")
    opponents = [TFT, ALLD, ALLC]
    
    X, y, feature_names, strategies, fitnesses = generate_training_data(
        n_samples=1000,
        opponent_strategies=opponents,
        num_rounds=100,
        memory_depth=1,
        good_threshold_percentile=80
    )
    
    print(f"   Features: {feature_names}")
    print(f"   Good strategies: {sum(y)}/{len(y)}")
    print(f"   Fitness range: {min(fitnesses):.2f} - {max(fitnesses):.2f}")
    
    # Train models
    print("\n2. Training ML Models...")
    predictor = StrategyPredictor()
    results = predictor.train(X, y, feature_names, test_size=0.2, memory_depth=1)
    
    print(f"\n{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r.model_name:<20}{r.accuracy:<12.3f}{r.precision:<12.3f}"
              f"{r.recall:<12.3f}{r.f1:<12.3f}")
    
    # Feature importance
    print("\n3. Feature Importance (Random Forest):")
    rf_result = [r for r in results if r.model_name == 'RandomForest'][0]
    sorted_features = sorted(rf_result.feature_importance.items(), 
                            key=lambda x: x[1], reverse=True)
    for name, importance in sorted_features:
        print(f"   {name:<20}: {importance:.3f}")
    
    # Pattern analysis
    print("\n4. Pattern Analysis:")
    good_strategies = [s for s, label in zip(strategies, y) if label == 1]
    bad_strategies = [s for s, label in zip(strategies, y) if label == 0]
    
    analysis = analyze_patterns(good_strategies, bad_strategies)
    print(f"   Good strategies tend to:")
    for pattern, data in analysis['patterns'].items():
        if data['difference'] > 0.1:
            print(f"   - Have higher {pattern}: {data['good_avg']:.2f} vs {data['bad_avg']:.2f}")
    
    # Use ML to find a good strategy
    print("\n5. ML-Guided Strategy Search:")
    ml_strategy = predictor.find_good_strategy(n_attempts=100, memory_depth=1)
    evaluator = FitnessEvaluator(opponents, 100)
    actual_fitness = evaluator.evaluate(ml_strategy)
    predicted_prob = predictor.predict(ml_strategy)
    
    print(f"   Found strategy: {ml_strategy.bitstring}")
    print(f"   Predicted good probability: {predicted_prob:.3f}")
    print(f"   Actual fitness: {actual_fitness:.2f}")
