import pandas as pd
import numpy as np
import re
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from spellchecker import SpellChecker
from symspellpy import SymSpell, Verbosity
import pkg_resources
import nltk
from nltk.tokenize import word_tokenize
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class SentimentDataProcessor:
    def __init__(self, file_paths, glove_path='data/glove.6B.100d.txt', max_seq_length=26, 
                 embedding_dim=100, correct_spelling=True):
        """
        Initialize the data processor.
        
        Args:
            file_paths (list): List of paths to the data files
            glove_path (str): Path to the pre-trained GloVe embeddings
            max_seq_length (int): Maximum sequence length for padding/truncating
            embedding_dim (int): Dimension of word embeddings
            correct_spelling (bool): Whether to correct misspelled words not in GloVe
        """
        self.file_paths = file_paths
        self.glove_path = glove_path
        self.max_seq_length = max_seq_length
        self.embedding_dim = embedding_dim
        self.correct_spelling = correct_spelling
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        self.embedding_matrix = None
        self.word_corrections = {}
        
    def read_data(self, file_path):
        """Read data from text file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    parts = line.split('\t')
                    if len(parts) == 2:
                        sentence, label = parts
                        data.append((sentence, int(label)))
        return data
    
    def load_all_data(self):
        """Load data from all the provided file paths."""
        all_data = []
        
        # Process each file
        for file_path in self.file_paths:
            source_name = os.path.basename(file_path).split('_')[0]
            data = self.read_data(file_path)
            for sentence, label in data:
                all_data.append({
                    'source': source_name,
                    'sentence': sentence,
                    'label': label
                })
        
        # Create DataFrame
        return pd.DataFrame(all_data)
    
    def tokenize(self, text):
        """Tokenize the text to words."""
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        tokens = word_tokenize(text)
        return tokens
    
    def build_vocabulary(self, df):
        """Build vocabulary from the dataset."""
        print("Building vocabulary...")
        # Get all unique words
        all_tokens = []
        for sentence in df['sentence']:
            tokens = self.tokenize(sentence)
            all_tokens.extend(tokens)
        
        # Create vocabulary
        word_counts = Counter(all_tokens)
        
        # Add words to the vocabulary
        for word, count in word_counts.items():
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        print(f"Vocabulary size: {len(self.word_to_idx)}")
        return word_counts
    
    def load_glove_embeddings(self):
        """Load GloVe embeddings from file."""
        print(f"Loading GloVe embeddings ({self.embedding_dim}d)...")
        embeddings_index = {}
        with open(self.glove_path, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        
        print(f"Loaded {len(embeddings_index)} word vectors from GloVe.")
        return embeddings_index
    
    def find_words_not_in_glove(self, vocabulary, glove_vocab):
        """Find words in our vocabulary that aren't in GloVe."""
        words_not_in_glove = set(vocabulary.keys()) - set(glove_vocab.keys())
        return words_not_in_glove
    
    def correct_misspelled_words(self, words_not_in_glove, glove_vocab, word_counts):
        """Try to correct misspelled words using SymSpell."""
        print("Correcting misspelled words with SymSpell...")
        
        # Initialize SymSpell
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        
        # Load dictionary
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        print(f"Loaded SymSpell dictionary with {sym_spell.word_count} words")
        
        # Add common domain-specific terms (technology, brands, food) to avoid incorrect corrections
        domain_terms = [
            "bluetooth", "earbuds", "smartphone", "wifi", "tmobile", "hdmi", "apps",
            "usb", "motorola", "samsung", "iphone", "android", "laptop", "charger",
            "macarons", "burrito", "taco", "sushi", "delish", "aioli", "quinoa",
            "facebook", "twitter", "instagram", "alexa", "reddit", "netflix", 
            "uber", "yelp", "gmail", "youtube", "spotify"
        ]
        
        # Add domain terms to dictionary with high frequency to prioritize them
        for term in domain_terms:
            sym_spell.create_dictionary_entry(term, 10000)
        
        corrections = {}
        skipped_reasons = Counter()
        
        for word in tqdm(words_not_in_glove):
            # Skip very short words (likely abbreviations)
            if len(word) <= 2:
                skipped_reasons["too_short"] += 1
                continue
            
            # Skip words with numbers (likely model numbers, etc.)
            if any(c.isdigit() for c in word):
                skipped_reasons["contains_digits"] += 1
                continue
            
            # Skip words with special characters in the middle (likely intentional)
            if re.search(r'[^a-zA-Z0-9][a-zA-Z]', word[1:-1]):
                skipped_reasons["special_chars"] += 1
                continue
                
            # Skip words that look like they might be proper nouns (capitalized)
            if word[0].isupper() and word[1:].islower():
                skipped_reasons["proper_noun"] += 1
                continue
            
            # Get spelling suggestions
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            
            if suggestions:
                top_suggestion = suggestions[0].term
                
                # Only accept corrections that:
                # 1. Exist in GloVe vocabulary
                # 2. Are different from the original word
                # 3. Have a reasonable edit distance (proportional to word length)
                # 4. Are not too short (to avoid aggressive truncation)
                if (top_suggestion in glove_vocab and 
                    top_suggestion != word and 
                    len(top_suggestion) >= 3 and 
                    suggestions[0].distance <= max(1, len(word) // 3)):
                    
                    # Additional check: don't change the first letter unless it's very common
                    if top_suggestion[0] == word[0] or suggestions[0].distance <= 1:
                        corrections[word] = top_suggestion
                    else:
                        skipped_reasons["first_letter_change"] += 1
                else:
                    skipped_reasons["failed_criteria"] += 1
            else:
                skipped_reasons["no_suggestions"] += 1
        
        print(f"Found corrections for {len(corrections)} words.")
        print("Words skipped for reasons:", dict(skipped_reasons))
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Save detailed report with assessment
        with open('data/correction_report.txt', 'w', encoding='utf-8') as f:
            f.write("WORD CORRECTION REPORT\n")
            f.write("=====================\n\n")
            f.write(f"Total vocabulary: {len(word_counts)} words\n")
            f.write(f"Words not in GloVe: {len(words_not_in_glove)}\n")
            f.write(f"Words corrected: {len(corrections)}\n\n")
            f.write("Top corrections by frequency:\n")
            f.write("-----------------------------\n")
            for word, correction in sorted(corrections.items(), key=lambda x: word_counts[x[0]], reverse=True)[:50]:
                f.write(f"{word} → {correction} (frequency: {word_counts[word]})\n")
            
            f.write("\n\nSkipped words by reason:\n")
            f.write("------------------------\n")
            for reason, count in skipped_reasons.most_common():
                f.write(f"{reason}: {count}\n")
        
        # Visualize corrections
        self.visualize_corrections(words_not_in_glove, corrections, word_counts)
        
        return corrections
    
    def visualize_corrections(self, words_not_in_glove, corrections, word_counts):
        """Create visualizations for words not in GloVe and their corrections."""
        # Create visualizations directory if it doesn't exist
        os.makedirs('visualizations', exist_ok=True)
        
        # Count total vocabulary size
        vocab_size = len(word_counts)
        
        # GloVe coverage pie chart
        plt.figure(figsize=(8, 8))
        coverage_data = [vocab_size - len(words_not_in_glove), len(words_not_in_glove)]
        labels = [f'In GloVe ({(vocab_size - len(words_not_in_glove))/vocab_size*100:.1f}%)', 
                f'Not in GloVe ({len(words_not_in_glove)/vocab_size*100:.1f}%)']
        colors = ['lightgreen', 'tomato']
        plt.pie(coverage_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('GloVe Vocabulary Coverage')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('visualizations/glove_coverage.png')
        
        # GloVe coverage after corrections
        if corrections:
            plt.figure(figsize=(8, 8))
            corrected_data = [
                vocab_size - len(words_not_in_glove),  # Original coverage
                len(corrections),  # Correctable words
                len(words_not_in_glove) - len(corrections)  # Remaining uncovered
            ]
            corrected_labels = [
                f'In GloVe ({(vocab_size - len(words_not_in_glove))/vocab_size*100:.1f}%)',
                f'Correctable ({len(corrections)/vocab_size*100:.1f}%)',
                f'Uncovered ({(len(words_not_in_glove) - len(corrections))/vocab_size*100:.1f}%)'
            ]
            corrected_colors = ['lightgreen', 'lightskyblue', 'tomato']
            plt.pie(corrected_data, labels=corrected_labels, colors=corrected_colors, autopct='%1.1f%%', startangle=90)
            plt.title('GloVe Vocabulary Coverage with Corrections')
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig('visualizations/glove_coverage_with_corrections.png')
    
    def create_embedding_matrix(self, glove_embeddings):
        """Create embedding matrix for the vocabulary."""
        print("Creating embedding matrix...")
        n_words = len(self.word_to_idx)
        
        # Initialize embedding matrix with random values
        embedding_matrix = np.random.normal(scale=0.6, size=(n_words, self.embedding_dim))
        
        # Set padding token embedding to zeros
        embedding_matrix[0] = np.zeros(self.embedding_dim)
        
        oov_count = 0
        
        # Fill in embedding matrix with GloVe vectors
        for word, i in self.word_to_idx.items():
            # Check if word needs correction
            if word in self.word_corrections:
                corrected_word = self.word_corrections[word]
                embedding_vector = glove_embeddings.get(corrected_word)
            else:
                embedding_vector = glove_embeddings.get(word)
            
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
            else:
                oov_count += 1
        
        print(f"Words not found in GloVe embeddings: {oov_count}/{n_words}")
        self.embedding_matrix = embedding_matrix
        return embedding_matrix
    
    def preprocess_text(self, texts):
        """Preprocess a list of texts."""
        processed = []
        for text in texts:
            # Tokenize
            tokens = self.tokenize(text)
            
            # Truncate if necessary
            if len(tokens) > self.max_seq_length:
                tokens = tokens[:self.max_seq_length]
            
            # Convert to indices
            token_ids = []
            for token in tokens:
                # Apply corrections if available
                if token in self.word_corrections and self.correct_spelling:
                    token = self.word_corrections[token]
                
                # Get ID (or UNK if not in vocab)
                idx = self.word_to_idx.get(token, self.word_to_idx['<UNK>'])
                token_ids.append(idx)
            
            # Pad if necessary (0 is the padding token)
            if len(token_ids) < self.max_seq_length:
                token_ids.extend([0] * (self.max_seq_length - len(token_ids)))
                
            processed.append(token_ids)
        return processed
    
    def process_data(self):
        """Process all data."""
        print("Processing data...")
        # Load data
        df = self.load_all_data()
        print(f"Loaded {len(df)} examples from {len(self.file_paths)} files.")
        
        # Build vocabulary
        word_counts = self.build_vocabulary(df)
        
        # Load GloVe embeddings
        glove_embeddings = self.load_glove_embeddings()
        
        # Find words not in GloVe
        words_not_in_glove = self.find_words_not_in_glove(self.word_to_idx, glove_embeddings)
        print(f"Found {len(words_not_in_glove)} words not in GloVe embeddings.")
        
        # Correct misspelled words
        if self.correct_spelling:
            self.word_corrections = self.correct_misspelled_words(
                words_not_in_glove, glove_embeddings, word_counts)
        
        # Create embedding matrix
        self.create_embedding_matrix(glove_embeddings)
        
        # Preprocess text data
        X = self.preprocess_text(df['sentence'])
        y = df['label'].values
        
        return X, y, df
    
    def train_val_test_split(self, X, y, df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
        """Split data into train, validation and test sets."""
        assert abs(train_size + val_size + test_size - 1.0) < 1e-10, "Split sizes must sum to 1"
        
        # For reproducibility
        np.random.seed(random_state)
        
        # Create indices
        indices = np.random.permutation(len(X))
        
        # Calculate split boundaries
        train_end = int(train_size * len(X))
        val_end = train_end + int(val_size * len(X))
        
        # Split indices
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        # Split data
        X_train = [X[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        X_test = [X[i] for i in test_idx]
        
        y_train = y[train_idx]
        y_val = y[val_idx]
        y_test = y[test_idx]
        
        # Also split the original dataframe for reference
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        
        print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test
    
    def save_preprocessed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save preprocessed data to files."""
        # Ensure data directory exists
        os.makedirs('data/processed_data', exist_ok=True)
        
        # Save data splits
        train_data = {'X': X_train, 'y': y_train}
        val_data = {'X': X_val, 'y': y_val}
        test_data = {'X': X_test, 'y': y_test}
        
        with open('data/processed_data/train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)
        with open('data/processed_data/val_data.pkl', 'wb') as f:
            pickle.dump(val_data, f)
        with open('data/processed_data/test_data.pkl', 'wb') as f:
            pickle.dump(test_data, f)
        
        # Save embedding matrix
        with open('data/processed_data/embedding_matrix.pkl', 'wb') as f:
            pickle.dump(self.embedding_matrix, f)
        
        # Save configuration
        config = {
            'max_length': self.max_seq_length,
            'embedding_dim': self.embedding_dim,
            'vocab_size': len(self.word_to_idx)
        }
        with open('data/processed_data/config.pkl', 'wb') as f:
            pickle.dump(config, f)
        
        print("Preprocessed data saved successfully!")


def main():
    """Main function for data preprocessing."""
    # File paths
    file_paths = [
        'data/amazon_cells_labelled.txt',
        'data/imdb_labelled.txt',
        'data/yelp_labelled.txt'
    ]
    
    # Initialize data processor
    processor = SentimentDataProcessor(
        file_paths=file_paths,
        glove_path='data/glove.6B.100d.txt',
        max_seq_length=26,
        embedding_dim=100,
        correct_spelling=True
    )
    
    # Process data
    X, y, df = processor.process_data()
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test = processor.train_val_test_split(
        X, y, df, train_size=0.8, val_size=0.1, test_size=0.1)
    
    # Save preprocessed data
    processor.save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("Data preprocessing completed successfully!")
    return X_train, X_val, X_test, y_train, y_val, y_test, processor.embedding_matrix


if __name__ == "__main__":
    main()
