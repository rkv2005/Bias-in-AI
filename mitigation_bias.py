"""
Bias Mitigation Techniques
Implements counterfactual augmentation and hard debiasing
"""

import numpy as np
import pandas as pd
from scipy import spatial
import pickle
import warnings
warnings.filterwarnings('ignore')

class BiasMitigation:
    def __init__(self):
        print("Initializing Bias Mitigation...")
    
    def create_counterfactual_pairs(self, text):
        """Create gender-swapped version of text"""
        swap_map = {
            'he': 'she', 'she': 'he',
            'him': 'her', 'her': 'him',
            'his': 'her', 'hers': 'his',
            'himself': 'herself', 'herself': 'himself',
            'man': 'woman', 'woman': 'man',
            'male': 'female', 'female': 'male',
            'boy': 'girl', 'girl': 'boy',
            'father': 'mother', 'mother': 'father',
            'son': 'daughter', 'daughter': 'son',
            'brother': 'sister', 'sister': 'brother'
        }
        
        words = text.split()
        swapped_words = []
        
        for word in words:
            word_lower = word.lower()
            # Check if word (without punctuation) is in swap_map
            clean_word = word_lower.strip('.,!?;:')
            if clean_word in swap_map:
                swapped_word = swap_map[clean_word]
                # Preserve capitalization
                if word[0].isupper():
                    swapped_word = swapped_word.capitalize()
                # Preserve punctuation
                if word[-1] in '.,!?;:':
                    swapped_word += word[-1]
                swapped_words.append(swapped_word)
            else:
                swapped_words.append(word)
        
        return ' '.join(swapped_words)
    
    def demonstrate_counterfactual_augmentation(self):
        """Demonstrate counterfactual data augmentation"""
        print("\nDemonstrating Counterfactual Data Augmentation...")
        
        example_sentences = [
            "The engineer designed the system and he tested it thoroughly.",
            "The nurse administered the medication and she checked vital signs.",
            "The CEO made a decision because he believed it was right.",
            "Her work as a programmer was exceptional and innovative.",
            "The pilot completed his flight training successfully.",
        ]
        
        augmented_data = []
        for sentence in example_sentences:
            counterfactual = self.create_counterfactual_pairs(sentence)
            augmented_data.append({
                'Original': sentence,
                'Counterfactual': counterfactual
            })
        
        augmented_df = pd.DataFrame(augmented_data)
        
        print("✓ Counterfactual augmentation examples created")
        
        return augmented_df
    
    def compute_gender_direction(self, male_embeddings, female_embeddings):
        """Compute the gender direction vector"""
        male_avg = np.mean(male_embeddings, axis=0)
        female_avg = np.mean(female_embeddings, axis=0)
        gender_direction = male_avg - female_avg
        gender_direction = gender_direction / np.linalg.norm(gender_direction)
        return gender_direction
    
    def debias_embedding(self, embedding, gender_direction):
        """Remove gender component from embedding"""
        embedding = embedding.flatten()
        gender_direction = gender_direction.flatten()
        projection = np.dot(embedding, gender_direction) * gender_direction
        debiased = embedding - projection
        return debiased
    
    @staticmethod
    def cosine_similarity(vec1, vec2):
        """Compute cosine similarity"""
        return 1 - spatial.distance.cosine(vec1.flatten(), vec2.flatten())
    
    def apply_hard_debiasing(self, occupation_embeddings, occupations, 
                            male_embeddings, female_embeddings):
        """Apply hard debiasing to occupation embeddings"""
        print("\nApplying Hard Debiasing...")
        
        # Compute gender direction
        gender_direction = self.compute_gender_direction(male_embeddings, female_embeddings)
        print(f"✓ Computed gender direction")
        
        debiased_results = []
        
        for i, occupation in enumerate(occupations):
            original_emb = occupation_embeddings[i]
            debiased_emb = self.debias_embedding(original_emb, gender_direction)
            
            # Compute bias before and after
            male_sim_before = np.mean([self.cosine_similarity(original_emb, m) 
                                       for m in male_embeddings])
            female_sim_before = np.mean([self.cosine_similarity(original_emb, f) 
                                         for f in female_embeddings])
            bias_before = male_sim_before - female_sim_before
            
            male_sim_after = np.mean([self.cosine_similarity(debiased_emb, m) 
                                      for m in male_embeddings])
            female_sim_after = np.mean([self.cosine_similarity(debiased_emb, f) 
                                        for f in female_embeddings])
            bias_after = male_sim_after - female_sim_after
            
            debiased_results.append({
                'Occupation': occupation,
                'Bias_Before': bias_before,
                'Bias_After': bias_after,
                'Bias_Reduction': abs(bias_before) - abs(bias_after),
                'Reduction_Percent': ((abs(bias_before) - abs(bias_after)) / abs(bias_before) * 100) 
                                     if bias_before != 0 else 0
            })
        
        debiased_df = pd.DataFrame(debiased_results)
        
        print(f"✓ Debiased {len(debiased_df)} embeddings")
        print(f"Average bias reduction: {debiased_df['Reduction_Percent'].mean():.1f}%")
        
        return debiased_df
    
    def run_mitigation(self):
        """Run complete mitigation analysis"""
        # Counterfactual augmentation
        augmented_df = self.demonstrate_counterfactual_augmentation()
        
        # Load embedding results for debiasing
        try:
            with open('results/embedding_bias_results.pkl', 'rb') as f:
                embedding_results = pickle.load(f)
            
            # Get embeddings from detector
            from embedding_bias import EmbeddingBiasDetector
            detector = EmbeddingBiasDetector()
            detector.compute_embeddings()
            
            all_occupations = detector.stereotyped_male_occupations + detector.stereotyped_female_occupations
            all_occ_embeddings = np.vstack([detector.male_occ_embeddings, detector.female_occ_embeddings])
            
            debiased_df = self.apply_hard_debiasing(
                all_occ_embeddings, all_occupations,
                detector.male_embeddings, detector.female_embeddings
            )
        except FileNotFoundError:
            print("Warning: Run embeddings_bias.py first for debiasing analysis")
            debiased_df = pd.DataFrame()
        
        results = {
            'augmented_df': augmented_df,
            'debiased_df': debiased_df
        }
        
        return results
    
    def save_results(self, results, filepath='results/mitigation_results.pkl'):
        """Save results to file"""
        import os
        os.makedirs('results', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        results['augmented_df'].to_csv('results/counterfactual_examples.csv', index=False)
        if not results['debiased_df'].empty:
            results['debiased_df'].to_csv('results/debiasing_results.csv', index=False)
        
        print(f"✓ Mitigation results saved to {filepath}")

def main():
    mitigation = BiasMitigation()
    results = mitigation.run_mitigation()
    mitigation.save_results(results)
    
    print("\n" + "="*60)
    print("BIAS MITIGATION COMPLETE")
    print("="*60)
    print("\n1. Counterfactual Augmentation Examples:")
    print(results['augmented_df'])
    
    if not results['debiased_df'].empty:
        print("\n2. Hard Debiasing Results:")
        print(f"Average bias reduction: {results['debiased_df']['Reduction_Percent'].mean():.1f}%")
        print("\nTop 5 improvements:")
        print(results['debiased_df'].nlargest(5, 'Reduction_Percent')[['Occupation', 'Reduction_Percent']])

if __name__ == "__main__":
    main()
