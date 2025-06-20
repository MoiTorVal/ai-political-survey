#!/usr/bin/env python3
"""
Political Affiliation Survey - Prediction Module

This module loads a trained model and makes predictions on new survey responses.
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
from collections import Counter

class PoliticalAffiliationPredictor:
    """
    Predictor class for making political affiliation predictions
    using a pre-trained model.
    """
    
    def __init__(self, model_path='../models/'):
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.feature_weights = None
        self.model_info = None
        self.question_texts = []
        
    def load_model(self):
        """Load the trained model and preprocessing components."""
        try:
            # Load model components
            self.model = joblib.load(f"{self.model_path}/best_model.pkl")
            self.label_encoder = joblib.load(f"{self.model_path}/label_encoder.pkl")
            self.feature_weights = joblib.load(f"{self.model_path}/feature_weights.pkl")
            
            # Load model info
            with open(f"{self.model_path}/model_info.json", 'r') as f:
                self.model_info = json.load(f)
            
            print(f"Loaded {self.model_info['model_name']} model")
            print(f"Classes: {', '.join(self.model_info['classes'])}")
            return True
            
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            print("Please run train.py first to train a model.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_questions(self, json_path='../data/questions.json'):
        """Load questions from JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                self.question_texts = [q['question'] for q in data['survey']['questions']]
            return True
        except FileNotFoundError:
            print(f"Warning: {json_path} not found.")
            return False
    
    def engineer_features(self, X):
        """
        Apply the same feature engineering as in training.
        """
        X_engineered = X.copy()
        
        # Create political leaning scores based on question types
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
        
        return X_engineered
    
    def predict_single_response(self, responses):
        """
        Predict political affiliation for a single survey response.
        
        Args:
            responses: List of responses ['A', 'B', 'C', 'D', ...] or [1, 2, 3, 4, ...]
        
        Returns:
            dict: Prediction results with probabilities and explanation
        """
        if self.model is None:
            print("Model not loaded! Call load_model() first.")
            return None
        
        try:
            # Convert letter responses to numerical if needed
            if isinstance(responses[0], str):
                response_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
                responses = [response_mapping.get(r.upper(), 0) for r in responses]
            
            # Create DataFrame
            X = pd.DataFrame([responses])
            
            # Apply feature engineering
            X_engineered = self.engineer_features(X)
            
            # Make prediction
            prediction = self.model.predict(X_engineered)[0]
            prediction_proba = self.model.predict_proba(X_engineered)[0]
            
            # Get predicted class name
            predicted_class = self.label_encoder.classes_[prediction]
            
            # Create results dictionary
            results = {
                'predicted_affiliation': predicted_class,
                'confidence': float(max(prediction_proba)),
                'probabilities': {
                    self.label_encoder.classes_[i]: float(prob) 
                    for i, prob in enumerate(prediction_proba)
                },
                'weighted_scores': self.calculate_weighted_scores(responses),
                'explanation': self.generate_explanation(responses, predicted_class)
            }
            
            return results
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def calculate_weighted_scores(self, responses):
        """Calculate weighted scores for each political affiliation."""
        if not self.feature_weights:
            return {}
        
        weighted_scores = {party: 0 for party in self.label_encoder.classes_}
        
        for question_idx, response in enumerate(responses):
            if question_idx in self.feature_weights:
                question_weights = self.feature_weights[question_idx]
                if response in question_weights:
                    for party, weight in question_weights[response].items():
                        weighted_scores[party] += weight
        
        return weighted_scores
    
    def generate_explanation(self, responses, predicted_class):
        """Generate human-readable explanation for the prediction."""
        explanation_parts = []
        
        # Analyze key response patterns
        response_mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
        
        # Economic conservatism analysis
        economic_responses = [responses[0], responses[1], responses[8]]  # Q1, Q2, Q9
        econ_conservative_count = sum(1 for r in economic_responses if r in [1, 2])
        
        if econ_conservative_count >= 2:
            explanation_parts.append("economically conservative responses")
        else:
            explanation_parts.append("economically progressive responses")
        
        # Social conservatism analysis
        social_responses = [responses[2], responses[9], responses[10]]  # Q3, Q10, Q11
        social_conservative_count = sum(1 for r in social_responses if r in [1, 2])
        
        if social_conservative_count >= 2:
            explanation_parts.append("socially conservative views")
        else:
            explanation_parts.append("socially progressive views")
        
        # Government involvement
        gov_responses = [responses[1], responses[6]]  # Q2, Q7
        gov_involvement = sum(gov_responses) / len(gov_responses)
        
        if gov_involvement <= 2:
            explanation_parts.append("preference for limited government")
        else:
            explanation_parts.append("support for active government role")
        
        explanation = f"Predicted as {predicted_class.title()} based on {', '.join(explanation_parts)}."
        return explanation
    
    def predict_from_csv(self, csv_path):
        """Make predictions on multiple responses from a CSV file."""
        try:
            # Read CSV
            df = pd.read_csv(csv_path, header=None)
            
            # Separate features from labels (if present)
            if df.shape[1] > 12:  # Has labels
                X = df.iloc[:, :-1]
                y_true = df.iloc[:, -1]
                has_labels = True
            else:
                X = df
                y_true = None
                has_labels = False
            
            # Convert responses to numerical
            response_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4}
            for col in X.columns:
                X[col] = X[col].map(response_mapping)
            X = X.fillna(0)
            
            # Apply feature engineering
            X_engineered = self.engineer_features(X)
            
            # Make predictions
            predictions = self.model.predict(X_engineered)
            probabilities = self.model.predict_proba(X_engineered)
            
            # Create results
            results = []
            for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
                predicted_class = self.label_encoder.classes_[pred]
                confidence = max(proba)
                
                result = {
                    'row': i + 1,
                    'predicted_affiliation': predicted_class,
                    'confidence': float(confidence),
                    'probabilities': {
                        self.label_encoder.classes_[j]: float(p) 
                        for j, p in enumerate(proba)
                    }
                }
                
                if has_labels:
                    result['actual_affiliation'] = y_true.iloc[i]
                    result['correct'] = (predicted_class.lower() == y_true.iloc[i].lower())
                
                results.append(result)
            
            # Calculate accuracy if labels are available
            if has_labels:
                predicted_labels = [self.label_encoder.classes_[pred] for pred in predictions]
                accuracy = sum(1 for pred, true in zip(predicted_labels, y_true) 
                             if pred.lower() == true.lower()) / len(y_true)
                
                print(f"Accuracy on {len(results)} samples: {accuracy:.3f}")
            
            return results
            
        except Exception as e:
            print(f"Error making predictions from CSV: {e}")
            return None

def interactive_prediction():
    """Interactive mode for single prediction."""
    predictor = PoliticalAffiliationPredictor()
    
    if not predictor.load_model():
        return
    
    predictor.load_questions()
    
    print("\nPolitical Affiliation Prediction")
    print("=" * 50)
    print("Please answer each question with A, B, C, or D")
    print()
    
    responses = []
    
    # Get responses for each question
    for i, question in enumerate(predictor.question_texts, 1):
        print(f"{i}. {question}")
        
        while True:
            response = input("Your answer (A/B/C/D): ").strip().upper()
            if response in ['A', 'B', 'C', 'D']:
                responses.append(response)
                break
            else:
                print("Please enter A, B, C, or D")
        print()
    
    # Make prediction
    result = predictor.predict_single_response(responses)
    
    if result:
        print("Prediction Results:")
        print("=" * 50)
        print(f"Predicted Political Affiliation: {result['predicted_affiliation'].title()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print()
        
        print("Probability Breakdown:")
        for party, prob in result['probabilities'].items():
            print(f"  {party.title()}: {prob:.1%}")
        print()
        
        print("Explanation:")
        print(f"  {result['explanation']}")

def main():
    """Main prediction interface."""
    import sys
    
    if len(sys.argv) > 1:
        # Batch prediction mode
        csv_path = sys.argv[1]
        
        predictor = PoliticalAffiliationPredictor()
        if predictor.load_model():
            results = predictor.predict_from_csv(csv_path)
            
            if results:
                print(f"\\nPredictions for {len(results)} responses:")
                for result in results:
                    print(f"Row {result['row']}: {result['predicted_affiliation'].title()} "
                          f"({result['confidence']:.1%} confidence)")
    else:
        # Interactive mode
        interactive_prediction()

if __name__ == "__main__":
    main()
