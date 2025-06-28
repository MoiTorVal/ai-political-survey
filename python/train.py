#!/usr/bin/env python3
"""
Political Affiliation Survey - Machine Learning Training Module

This module implements multiple ML approaches for classifying political affiliations
based on survey responses with feature engineering and weighted scoring.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class PoliticalAffiliationClassifier:
    """
    A comprehensive classifier for political affiliation prediction
    with feature engineering and weighted responses.
    """
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'naive_bayes': MultinomialNB(),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.best_model = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.feature_weights = {}
        self.question_texts = []
        
    def load_questions(self, json_path):
        """Load questions from JSON file for feature engineering."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.question_texts = [q['question'] for q in data['survey']['questions']]
        except FileNotFoundError:
            print(f"Warning: {json_path} not found. Using default feature names.")
            self.question_texts = [f"Question_{i}" for i in range(1, 13)]
    
    def load_data(self, csv_path):
        """Load and preprocess survey data."""
        try:
            # Read CSV data
            df = pd.read_csv(csv_path, header=None)
            
            if df.empty:
                raise ValueError("No data found in CSV file")
            
            # Split features and target
            X = df.iloc[:, :-1]  # All columns except last (responses)
            y = df.iloc[:, -1]   # Last column (political affiliation)
            
            # Convert letter responses to numerical
            response_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            for col in X.columns:
                X[col] = X[col].map(response_mapping)
            
            # Handle any missing mappings
            X = X.fillna(0)
            
            # Encode target labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            print(f"Loaded {len(df)} survey responses")
            print(f"Features shape: {X.shape}")
            print(f"Target distribution: {Counter(y)}")
            
            return X, y_encoded, y
            
        except FileNotFoundError:
            print(f"Error: {csv_path} not found!")
            return None, None, None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None, None
    
    def engineer_features(self, X):
        """
        Advanced feature engineering with weighted responses.
        Creates interaction features and political leaning scores.
        """
        X_engineered = X.copy()
        
        # Create political leaning scores based on question type
        # Economic questions (Q1, Q2, Q9) - Conservative vs Progressive
        economic_questions = [0, 1, 8]  # 0-indexed
        X_engineered['economic_conservatism'] = X.iloc[:, economic_questions].apply(
            lambda row: sum([4-val if i in [0, 1] else val for i, val in enumerate(row)]), axis=1
        )
        
        # Social questions (Q3, Q10, Q11) - Conservative vs Progressive  
        social_questions = [2, 9, 10]
        X_engineered['social_conservatism'] = X.iloc[:, social_questions].apply(
            lambda row: sum([4-val for val in row]), axis=1
        )
        
        # Security/Authority questions (Q4, Q6, Q8) - Authoritarian vs Libertarian
        authority_questions = [3, 5, 7]
        X_engineered['authoritarianism'] = X.iloc[:, authority_questions].apply(
            lambda row: sum([val if i == 0 else 4-val for i, val in enumerate(row)]), axis=1
        )
        
        # Government involvement score (Q2, Q7)
        gov_questions = [1, 6]
        X_engineered['gov_involvement'] = X.iloc[:, gov_questions].apply(
            lambda row: sum(row), axis=1
        )
        
        # Calculate interaction features
        X_engineered['econ_social_interaction'] = (
            X_engineered['economic_conservatism'] * X_engineered['social_conservatism']
        )
        
        # Fix column names to be all strings
        X_engineered.columns = X_engineered.columns.astype(str)
        
        return X_engineered
    
    def calculate_feature_weights(self, X, y):
        """
        Calculate TF-IDF inspired weights for survey responses.
        """
        weights = {}
        
        for col_idx, col in enumerate(X.columns):
            if col_idx < len(self.question_texts):
                question_weights = {}
                
                for response in [1, 2, 3, 4]:  # A, B, C, D
                    # Calculate how distinctive this response is for each class
                    class_scores = {}
                    
                    for party_idx, party in enumerate(self.label_encoder.classes_):
                        # Count this response for this party
                        party_mask = (y == party_idx)
                        response_count = sum((X.iloc[:, col_idx] == response) & party_mask)
                        total_party_responses = sum(party_mask)
                        
                        if total_party_responses > 0:
                            # TF-IDF inspired score
                            tf = response_count / total_party_responses
                            
                            # IDF: how rare is this response across all parties
                            total_response_count = sum(X.iloc[:, col_idx] == response)
                            idf = np.log(len(y) / (total_response_count + 1))
                            
                            class_scores[party] = tf * idf
                        else:
                            class_scores[party] = 0
                    
                    question_weights[response] = class_scores
                
                weights[col_idx] = question_weights
        
        self.feature_weights = weights
        return weights
    
    def train_models(self, X, y):
        """Train multiple models and select the best one."""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model_scores = {}
        trained_models = {}
        
        print("\nTraining Models:")
        print("=" * 50)
        
        for name, model in self.models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, y_train, cv=3)
                
                # Test score
                test_score = model.score(X_test, y_test)
                
                model_scores[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_score': test_score
                }
                
                trained_models[name] = model
                
                print(f"{name.upper()}:")
                print(f"  CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                print(f"  Test Score: {test_score:.3f}")
                print()
                
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        # Select best model based on CV score
        if model_scores:
            best_name = max(model_scores.keys(), key=lambda x: model_scores[x]['cv_mean'])
            self.best_model = trained_models[best_name]
            self.best_model_name = best_name
            
            print(f"Best Model: {best_name.upper()}")
            print(f"CV Score: {model_scores[best_name]['cv_mean']:.3f}")
            
            return X_train, X_test, y_train, y_test
        
        return None, None, None, None
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation."""
        if self.best_model is None:
            print("No trained model available!")
            return
        
        # Predictions
        y_pred = self.best_model.predict(X_test)
        
        print("\nModel Evaluation:")
        print("=" * 50)
        print(f"Model: {self.best_model_name.upper()}")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print()
        
        print("Classification Report:")
        target_names = self.label_encoder.classes_
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Feature importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            print("\nTop 5 Most Important Features:")
            feature_names = [f"Q{i+1}" for i in range(len(self.best_model.feature_importances_))]
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(min(5, len(indices))):
                idx = indices[i]
                if idx < len(self.question_texts):
                    question_text = self.question_texts[idx][:60] + "..." if len(self.question_texts[idx]) > 60 else self.question_texts[idx]
                else:
                    question_text = feature_names[idx]
                print(f"  {i+1}. {question_text}: {importances[idx]:.3f}")
    
    def save_model(self, model_path='../models/'):
        """Save the trained model and preprocessing components."""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        if self.best_model is not None:
            # Save model
            joblib.dump(self.best_model, f"{model_path}/best_model.pkl")
            joblib.dump(self.label_encoder, f"{model_path}/label_encoder.pkl")
            joblib.dump(self.feature_weights, f"{model_path}/feature_weights.pkl")
            
            # Save model info
            model_info = {
                'model_name': self.best_model_name,
                'classes': self.label_encoder.classes_.tolist(),
                'n_features': len(self.question_texts)
            }
            
            with open(f"{model_path}/model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print(f"\nModel saved to {model_path}")
        else:
            print("No model to save!")

def main():
    """Main training pipeline."""
    print("Political Affiliation Survey - ML Training")
    print("=" * 50)
    
    # Initialize classifier
    classifier = PoliticalAffiliationClassifier()
    
    # Load questions for feature engineering
    classifier.load_questions('../data/questions.json')
    
    # Load data
    X, y, y_original = classifier.load_data('../data/survey_results.csv')
    
    if X is None:
        print("Cannot proceed without data. Please run the survey to collect responses.")
        return
    
    if len(X) < 5:
        print(f"Warning: Only {len(X)} responses available. Need more data for reliable training.")
        print("Consider collecting more survey responses.")
        return
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    X_engineered = classifier.engineer_features(X)
    
    # Calculate feature weights
    print("Calculating feature weights...")
    classifier.calculate_feature_weights(X, y)
    
    # Train models
    X_train, X_test, y_train, y_test = classifier.train_models(X_engineered, y)
    
    if X_train is not None:
        # Evaluate best model
        classifier.evaluate_model(X_test, y_test)
        
        # Save model
        classifier.save_model()
        
        print("\nTraining completed successfully!")
        print("You can now use predict.py to make predictions on new survey responses.")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
