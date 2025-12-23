"""
Embedding Bias Detection using WEAT (Word Embedding Association Test)
Analyzes gender bias in BERT word embeddings
"""

import numpy as np
import pandas as pd
from scipy import spatial
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import warnings
warnings.filterwarnings('ignore')

class EmbeddingBiasDetector:
    def __init__(self, model_name='bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")
        
        # Define word sets
        self.career_words = [
            'executive', 'management', 'professional', 'corporation', 'salary',
            'office', 'business', 'career', 'promotion', 'engineer', 'programmer',
            'scientist', 'doctor', 'lawyer', 'architect'
        ]
        
        self.family_words = [
            'home', 'parents', 'children', 'family', 'cousins', 
            'marriage', 'wedding', 'relatives', 'nurture', 'babies',
            'kindergarten', 'domestic', 'household', 'caregiver', 'daycare'
        ]
        
        self.male_words = [
            'he', 'him', 'his', 'man', 'male', 'boy', 'father', 
            'son', 'brother', 'uncle', 'grandfather', 'gentleman'
        ]
        
        self.female_words = [
            'she', 'her', 'hers', 'woman', 'female', 'girl', 'mother',
            'daughter', 'sister', 'aunt', 'grandmother', 'lady'
        ]
        
        self.stereotyped_male_occupations = [
            'engineer', 'programmer', 'pilot', 'surgeon', 'architect',
            'mechanic', 'carpenter', 'electrician', 'plumber', 'scientist'
        ]
        
        self.stereotyped_female_occupations = [
            'nurse', 'teacher', 'secretary', 'receptionist', 'librarian',
            'social worker', 'counselor', 'dietitian', 'hairdresser', 'decorator'
        ]
    
    def get_bert_embedding(self, word):
        """Get BERT embedding for a single word"""
        inputs = self.tokenizer(word, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.flatten()
    
    def compute_embeddings(self):
        """Compute embeddings for all word sets"""
        print("Computing embeddings...")
        
        self.career_embeddings = np.array([self.get_bert_embedding(w) for w in self.career_words])
        self.family_embeddings = np.array([self.get_bert_embedding(w) for w in self.family_words])
        self.male_embeddings = np.array([self.get_bert_embedding(w) for w in self.male_words])
        self.female_embeddings = np.array([self.get_bert_embedding(w) for w in self.female_words])
        self.male_occ_embeddings = np.array([self.get_bert_embedding(w) for w in self.stereotyped_male_occupations])
        self.female_occ_embeddings = np.array([self.get_bert_embedding(w) for w in self.stereotyped_female_occupations])
        
        print("✓ Embeddings computed")
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Compute cosine similarity between two vectors"""
        return 1 - spatial.distance.cosine(vec1.flatten(), vec2.flatten())
    
    def weat_association(self, w, A, B):
        """Compute association of word w with attribute sets A and B"""
        similarities_A = [self.cosine_similarity(w, a) for a in A]
        similarities_B = [self.cosine_similarity(w, b) for b in B]
        return np.mean(similarities_A) - np.mean(similarities_B)
    
    def weat_score(self, X, Y, A, B):
        """Compute WEAT score between target sets X, Y and attribute sets A, B"""
        x_associations = [self.weat_association(x, A, B) for x in X]
        y_associations = [self.weat_association(y, A, B) for y in Y]
        return np.mean(x_associations) - np.mean(y_associations)
    
    def weat_effect_size(self, X, Y, A, B):
        """Compute effect size (Cohen's d)"""
        x_associations = [self.weat_association(x, A, B) for x in X]
        y_associations = [self.weat_association(y, A, B) for y in Y]
        
        all_associations = x_associations + y_associations
        std_dev = np.std(all_associations)
        
        if std_dev == 0:
            return 0
        
        return (np.mean(x_associations) - np.mean(y_associations)) / std_dev
    
    def analyze_bias(self):
        """Run complete bias analysis"""
        print("\nAnalyzing bias...")
        
        # Test 1: Career-Family Gender Bias
        score_1 = self.weat_score(self.career_embeddings, self.family_embeddings,
                                   self.male_embeddings, self.female_embeddings)
        effect_size_1 = self.weat_effect_size(self.career_embeddings, self.family_embeddings,
                                               self.male_embeddings, self.female_embeddings)
        
        # Test 2: Occupational Stereotypes
        score_2 = self.weat_score(self.male_occ_embeddings, self.female_occ_embeddings,
                                   self.male_embeddings, self.female_embeddings)
        effect_size_2 = self.weat_effect_size(self.male_occ_embeddings, self.female_occ_embeddings,
                                               self.male_embeddings, self.female_embeddings)
        
        # Individual occupation bias
        all_occupations = self.stereotyped_male_occupations + self.stereotyped_female_occupations
        all_occ_embeddings = np.vstack([self.male_occ_embeddings, self.female_occ_embeddings])
        
        occupation_results = []
        for i, occupation in enumerate(all_occupations):
            embedding = all_occ_embeddings[i]
            
            male_similarity = np.mean([self.cosine_similarity(embedding, m) 
                                       for m in self.male_embeddings])
            female_similarity = np.mean([self.cosine_similarity(embedding, f) 
                                         for f in self.female_embeddings])
            bias_score = male_similarity - female_similarity
            
            occupation_results.append({
                'Occupation': occupation,
                'Male_Similarity': male_similarity,
                'Female_Similarity': female_similarity,
                'Bias_Score': bias_score,
                'Bias_Direction': 'Male-biased' if bias_score > 0 else 'Female-biased'
            })
        
        bias_df = pd.DataFrame(occupation_results)
        
        results = {
            'career_family_score': score_1,
            'career_family_effect': effect_size_1,
            'occupation_score': score_2,
            'occupation_effect': effect_size_2,
            'occupation_bias_df': bias_df
        }
        
        print(f"✓ Career-Family Bias Effect Size: {effect_size_1:.3f}")
        print(f"✓ Occupation Stereotype Effect Size: {effect_size_2:.3f}")
        
        return results
    
    def save_results(self, results, filepath='results/embedding_bias_results.pkl'):
        """Save results to file"""
        import os
        os.makedirs('results', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        # Save CSV
        results['occupation_bias_df'].to_csv('results/occupation_bias.csv', index=False)
        print(f"✓ Results saved to {filepath}")

def main():
    detector = EmbeddingBiasDetector()
    detector.compute_embeddings()
    results = detector.analyze_bias()
    detector.save_results(results)
    
    print("\n" + "="*60)
    print("EMBEDDING BIAS DETECTION COMPLETE")
    print("="*60)
    print(f"Career-Family Gender Bias: {results['career_family_effect']:.3f}")
    print(f"Occupational Stereotypes: {results['occupation_effect']:.3f}")
    print("\nTop 5 Male-biased Occupations:")
    print(results['occupation_bias_df'].nlargest(5, 'Bias_Score')[['Occupation', 'Bias_Score']])
    print("\nTop 5 Female-biased Occupations:")
    print(results['occupation_bias_df'].nsmallest(5, 'Bias_Score')[['Occupation', 'Bias_Score']])

if __name__ == "__main__":
    main()
