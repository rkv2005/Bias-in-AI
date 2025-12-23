"""
Classification Bias Detection
Analyzes fairness metrics in sentiment classification task
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

class ClassificationBiasDetector:
    def __init__(self):
        print("Initializing Classification Bias Detector...")
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression(max_iter=1000)
    
    def create_synthetic_dataset(self):
        """Create synthetic dataset with demographic attributes"""
        print("Creating synthetic dataset...")
        
        # Positive sentiment examples
        positive_male = [
            "He is an excellent engineer and very skilled.",
            "His work on the project was outstanding.",
            "He demonstrated great leadership abilities.",
            "His technical expertise is impressive.",
            "He completed the task efficiently.",
        ] * 20
        
        positive_female = [
            "She is an excellent engineer and very skilled.",
            "Her work on the project was outstanding.",
            "She demonstrated great leadership abilities.",
            "Her technical expertise is impressive.",
            "She completed the task efficiently.",
        ] * 20
        
        # Negative sentiment examples
        negative_male = [
            "He failed to meet the project deadlines.",
            "His work quality was subpar and disappointing.",
            "He showed poor communication skills.",
            "His technical knowledge is inadequate.",
            "He struggled with basic tasks.",
        ] * 20
        
        negative_female = [
            "She failed to meet the project deadlines.",
            "Her work quality was subpar and disappointing.",
            "She showed poor communication skills.",
            "Her technical knowledge is inadequate.",
            "She struggled with basic tasks.",
        ] * 20
        
        # Combine data
        texts = positive_male + positive_female + negative_male + negative_female
        labels = [1]*len(positive_male) + [1]*len(positive_female) + \
                 [0]*len(negative_male) + [0]*len(negative_female)
        genders = ['male']*len(positive_male) + ['female']*len(positive_female) + \
                  ['male']*len(negative_male) + ['female']*len(negative_female)
        
        df = pd.DataFrame({
            'text': texts,
            'label': labels,
            'gender': genders
        })
        
        print(f"✓ Created dataset with {len(df)} examples")
        return df
    
    def train_model(self, df):
        """Train sentiment classification model"""
        print("Training classification model...")
        
        X = self.vectorizer.fit_transform(df['text'])
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ Model trained. Accuracy: {accuracy:.3f}")
        
        return df
    
    def compute_fairness_metrics(self, df):
        """Compute fairness metrics across demographic groups"""
        print("Computing fairness metrics...")
        
        X = self.vectorizer.transform(df['text'])
        df['prediction'] = self.model.predict(X)
        df['prediction_proba'] = self.model.predict_proba(X)[:, 1]
        
        # Metrics by gender
        male_df = df[df['gender'] == 'male']
        female_df = df[df['gender'] == 'female']
        
        male_accuracy = accuracy_score(male_df['label'], male_df['prediction'])
        female_accuracy = accuracy_score(female_df['label'], female_df['prediction'])
        
        # Positive prediction rate
        male_positive_rate = (male_df['prediction'] == 1).mean()
        female_positive_rate = (female_df['prediction'] == 1).mean()
        
        # Demographic parity difference
        demographic_parity = abs(male_positive_rate - female_positive_rate)
        
        # Equalized odds (TPR difference)
        male_tpr = ((male_df['label'] == 1) & (male_df['prediction'] == 1)).sum() / (male_df['label'] == 1).sum()
        female_tpr = ((female_df['label'] == 1) & (female_df['prediction'] == 1)).sum() / (female_df['label'] == 1).sum()
        
        tpr_difference = abs(male_tpr - female_tpr)
        
        results = {
            'male_accuracy': male_accuracy,
            'female_accuracy': female_accuracy,
            'accuracy_difference': abs(male_accuracy - female_accuracy),
            'male_positive_rate': male_positive_rate,
            'female_positive_rate': female_positive_rate,
            'demographic_parity': demographic_parity,
            'male_tpr': male_tpr,
            'female_tpr': female_tpr,
            'tpr_difference': tpr_difference,
            'predictions_df': df
        }
        
        print("✓ Fairness metrics computed")
        
        return results
    
    def analyze_bias(self):
        """Run complete classification bias analysis"""
        df = self.create_synthetic_dataset()
        df = self.train_model(df)
        results = self.compute_fairness_metrics(df)
        
        return results
    
    def save_results(self, results, filepath='results/classification_bias_results.pkl'):
        """Save results to file"""
        import os
        os.makedirs('results', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary
        summary = {
            'Metric': ['Male Accuracy', 'Female Accuracy', 'Accuracy Difference',
                      'Male Positive Rate', 'Female Positive Rate', 'Demographic Parity',
                      'Male TPR', 'Female TPR', 'TPR Difference'],
            'Value': [results['male_accuracy'], results['female_accuracy'], 
                     results['accuracy_difference'], results['male_positive_rate'],
                     results['female_positive_rate'], results['demographic_parity'],
                     results['male_tpr'], results['female_tpr'], results['tpr_difference']]
        }
        pd.DataFrame(summary).to_csv('results/classification_fairness.csv', index=False)
        print(f"✓ Results saved to {filepath}")

def main():
    detector = ClassificationBiasDetector()
    results = detector.analyze_bias()
    detector.save_results(results)
    
    print("\n" + "="*60)
    print("CLASSIFICATION BIAS ANALYSIS COMPLETE")
    print("="*60)
    print(f"Male Accuracy: {results['male_accuracy']:.3f}")
    print(f"Female Accuracy: {results['female_accuracy']:.3f}")
    print(f"Demographic Parity: {results['demographic_parity']:.3f}")
    print(f"Equalized Odds (TPR Diff): {results['tpr_difference']:.3f}")

if __name__ == "__main__":
    main()
