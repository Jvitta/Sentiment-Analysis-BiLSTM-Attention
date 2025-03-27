import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import Counter

# File paths
file_paths = [
    'data/amazon_cells_labelled.txt',
    'data/imdb_labelled.txt',
    'data/yelp_labelled.txt'
]

# Function to read the data
def read_data(file_path):
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

# Read all datasets
datasets = {}
for file_path in file_paths:
    file_name = os.path.basename(file_path).split('_')[0]  # Extract source name (amazon, imdb, yelp)
    datasets[file_name] = read_data(file_path)

# Create dataframes
dataframes = {}
for source, data in datasets.items():
    sentences, labels = zip(*data)
    dataframes[source] = pd.DataFrame({
        'sentence': sentences,
        'label': labels,
        'word_count': [len(sentence.split()) for sentence in sentences]
    })

# Combine all datasets
all_data = pd.concat([df.assign(source=source) for source, df in dataframes.items()], ignore_index=True)

# Calculate vocabulary size (unique words across all datasets)
def tokenize(text):
    # Simple tokenization - convert to lowercase and split by non-alphanumeric chars
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

all_words = []
unusual_chars = set()
non_ascii_pattern = re.compile(r'[^\x00-\x7F]')

# Process each sentence to get vocabulary and check for unusual characters
for sentence in all_data['sentence']:
    # Find unusual characters (non-ASCII)
    unusual_chars.update(non_ascii_pattern.findall(sentence))
    
    # Get words for vocabulary
    words = tokenize(sentence)
    all_words.extend(words)

vocabulary = set(all_words)
vocabulary_size = len(vocabulary)

# Load GloVe embeddings vocabulary
def load_glove_vocabulary(file_path):
    glove_vocab = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            glove_vocab.add(word)
    return glove_vocab

# Load GloVe vocabulary
print("Loading GloVe vocabulary...")
glove_path = 'data/glove.6B.100d.txt'
glove_vocab = load_glove_vocabulary(glove_path)
print(f"Loaded {len(glove_vocab)} words from GloVe embeddings.")

# Find words not in GloVe
words_not_in_glove = vocabulary - glove_vocab
print(f"Found {len(words_not_in_glove)} words in the dataset that are not in GloVe vocabulary.")

# Save words not in GloVe to a file
with open('words_not_in_glove.txt', 'w', encoding='utf-8') as f:
    for word in sorted(words_not_in_glove):
        f.write(f"{word}\n")

print("="*80)
print("EDA on Sentiment Labeled Sentences Dataset")
print("="*80)

# 1. Number of examples per file
print("\n1. Number of examples per file:")
for source, df in dataframes.items():
    print(f"{source}: {len(df)} examples")
print(f"Total: {len(all_data)} examples")

# 2. Distribution of scores (0 and 1)
print("\n2. Distribution of sentiment scores:")
for source, df in dataframes.items():
    neg_count = df[df['label'] == 0].shape[0]
    pos_count = df[df['label'] == 1].shape[0]
    neg_percent = neg_count / len(df) * 100
    pos_percent = pos_count / len(df) * 100
    print(f"{source}: Negative (0): {neg_count} ({neg_percent:.2f}%), Positive (1): {pos_count} ({pos_percent:.2f}%)")

# Overall distribution
neg_count = all_data[all_data['label'] == 0].shape[0]
pos_count = all_data[all_data['label'] == 1].shape[0]
neg_percent = neg_count / len(all_data) * 100
pos_percent = pos_count / len(all_data) * 100
print(f"Overall: Negative (0): {neg_count} ({neg_percent:.2f}%), Positive (1): {pos_count} ({pos_percent:.2f}%)")

# 3. Vocabulary analysis
print("\n3. Vocabulary Analysis:")
print(f"  Total vocabulary size (unique words): {vocabulary_size}")
print(f"  Words not in GloVe vocabulary: {len(words_not_in_glove)} ({len(words_not_in_glove)/vocabulary_size*100:.2f}%)")
print(f"  Examples of words not in GloVe (first 20):")
for word in sorted(list(words_not_in_glove))[:20]:
    print(f"    {word}")

# Print top 20 most common words
word_freq = Counter(all_words)
print(f"\n  Top 20 most common words:")
for word, count in word_freq.most_common(20):
    in_glove = "Yes" if word in glove_vocab else "No"
    print(f"    {word}: {count} (In GloVe: {in_glove})")

# Print top important sentiment words
print(f"\n  Example sentiment words:")
positive_words = ["good", "great", "excellent", "amazing", "love", "best"]
negative_words = ["bad", "terrible", "worst", "disappointed", "waste", "poor"]

print(f"  Positive sentiment words:")
for word in positive_words:
    if word in word_freq:
        in_glove = "Yes" if word in glove_vocab else "No"
        print(f"    {word}: {word_freq[word]} (In GloVe: {in_glove})")

print(f"\n  Negative sentiment words:")
for word in negative_words:
    if word in word_freq:
        in_glove = "Yes" if word in glove_vocab else "No"
        print(f"    {word}: {word_freq[word]} (In GloVe: {in_glove})")

# Check for unusual characters
print(f"\n  Unusual characters (non-ASCII):")
if unusual_chars:
    print(f"    Found {len(unusual_chars)} unusual characters: {''.join(unusual_chars)}")
    print("    Unicode code points:")
    for char in sorted(unusual_chars):
        print(f"      '{char}': U+{ord(char):04X}")
else:
    print("    No unusual characters found")

# 4. Sentence length statistics - Word Count
print("\n4. Sentence length statistics (word count):")
for source, df in dataframes.items():
    print(f"\n{source}:")
    print(f"  Min length: {df['word_count'].min()} words")
    print(f"  Max length: {df['word_count'].max()} words")
    print(f"  Average length: {df['word_count'].mean():.2f} words")
    print(f"  Median length: {df['word_count'].median()} words")
    print(f"  Standard deviation: {df['word_count'].std():.2f} words")
    
    # Distribution by sentiment
    neg_avg_len = df[df['label'] == 0]['word_count'].mean()
    pos_avg_len = df[df['label'] == 1]['word_count'].mean()
    print(f"  Average length (Negative): {neg_avg_len:.2f} words")
    print(f"  Average length (Positive): {pos_avg_len:.2f} words")
    
    # Percentiles for deciding sequence lengths
    percentiles = [50, 75, 90, 95, 99]
    word_percentiles = np.percentile(df['word_count'], percentiles)
    print(f"  Word count percentiles:")
    for p, val in zip(percentiles, word_percentiles):
        print(f"    {p}th percentile: {val:.0f} words")

# Overall word count statistics
print("\nOverall (Word Count):")
print(f"  Min length: {all_data['word_count'].min()} words")
print(f"  Max length: {all_data['word_count'].max()} words")
print(f"  Average length: {all_data['word_count'].mean():.2f} words")
print(f"  Median length: {all_data['word_count'].median()} words")
print(f"  Standard deviation: {all_data['word_count'].std():.2f} words")

# Overall distribution by sentiment
neg_avg_len = all_data[all_data['label'] == 0]['word_count'].mean()
pos_avg_len = all_data[all_data['label'] == 1]['word_count'].mean()
print(f"  Average length (Negative): {neg_avg_len:.2f} words")
print(f"  Average length (Positive): {pos_avg_len:.2f} words")

# Percentiles for overall word count
percentiles = [50, 75, 90, 95, 99]
word_percentiles = np.percentile(all_data['word_count'], percentiles)
print(f"  Word count percentiles (all datasets):")
for p, val in zip(percentiles, word_percentiles):
    print(f"    {p}th percentile: {val:.0f} words")

# 5. Example sentences
print("\n5. Example sentences:")
for source, df in dataframes.items():
    print(f"\n{source} examples:")
    print(f"  Negative example: \"{df[df['label'] == 0]['sentence'].iloc[0]}\"")
    print(f"  Positive example: \"{df[df['label'] == 1]['sentence'].iloc[0]}\"")

# 6. GloVe coverage analysis
print("\n6. GloVe Coverage Analysis:")
print(f"  Total unique words in dataset: {vocabulary_size}")
print(f"  Words covered by GloVe: {vocabulary_size - len(words_not_in_glove)} ({(vocabulary_size - len(words_not_in_glove))/vocabulary_size*100:.2f}%)")
print(f"  Words not covered by GloVe: {len(words_not_in_glove)} ({len(words_not_in_glove)/vocabulary_size*100:.2f}%)")

# Count frequency of words not in GloVe
not_in_glove_freq = {word: word_freq[word] for word in words_not_in_glove if word in word_freq}
sorted_not_in_glove = sorted(not_in_glove_freq.items(), key=lambda x: x[1], reverse=True)

print(f"\n  Top 20 most frequent words not in GloVe:")
for word, count in sorted_not_in_glove[:20]:
    print(f"    {word}: {count}")

# 7. Create visualizations for the report
print("\nGenerating visualizations...")

# Create visualizations directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Set style
plt.style.use('ggplot')
sns.set(style="whitegrid")

# Figure 1: Word count distributions
plt.figure(figsize=(12, 8))
for i, (source, df) in enumerate(dataframes.items()):
    plt.subplot(2, 2, i+1)
    sns.histplot(data=df, x='word_count', hue='label', multiple='stack',
                palette=['tomato', 'lightgreen'], bins=20)
    plt.title(f'{source.capitalize()} Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.legend(['Negative', 'Positive'])

# Combined word count distribution
plt.subplot(2, 2, 4)
sns.histplot(data=all_data, x='word_count', hue='label', multiple='stack',
            palette=['tomato', 'lightgreen'], bins=20)
plt.title('Combined Word Count Distribution')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.legend(['Negative', 'Positive'])
plt.tight_layout()
plt.savefig('visualizations/word_count_distributions.png')

# Figure 2: Word count boxplot by dataset
plt.figure(figsize=(10, 6))
sns.boxplot(x='source', y='word_count', data=all_data, palette='pastel')
plt.title('Word Count Comparison Across Datasets')
plt.xlabel('Dataset Source')
plt.ylabel('Word Count')
# Add maximum sequence length line
seq_length = int(np.percentile(all_data['word_count'], 95))
plt.axhline(y=seq_length, color='r', linestyle='--')
plt.text(0, seq_length+1, f'95th percentile: {seq_length} words', color='r')
plt.tight_layout()
plt.savefig('visualizations/word_count_boxplot.png')

# Figure 3: Cumulative distribution of word counts
plt.figure(figsize=(10, 6))
word_counts = np.sort(all_data['word_count'])
y = np.arange(1, len(word_counts) + 1) / len(word_counts)
plt.plot(word_counts, y)
plt.title('Cumulative Distribution of Word Counts')
plt.xlabel('Word Count')
plt.ylabel('Cumulative Probability')
plt.grid(True, linestyle='--', alpha=0.7)

# Add vertical lines at common percentiles
for p in [50, 90, 95, 99]:
    val = np.percentile(all_data['word_count'], p)
    plt.axvline(x=val, color='r', linestyle='--', alpha=0.5)
    plt.text(val+1, 0.5, f'{p}th percentile: {int(val)} words', rotation=90)

plt.tight_layout()
plt.savefig('visualizations/word_count_cdf.png')

# Figure 4: GloVe coverage pie chart
plt.figure(figsize=(8, 8))
coverage_data = [vocabulary_size - len(words_not_in_glove), len(words_not_in_glove)]
labels = [f'In GloVe ({(vocabulary_size - len(words_not_in_glove))/vocabulary_size*100:.1f}%)', 
          f'Not in GloVe ({len(words_not_in_glove)/vocabulary_size*100:.1f}%)']
colors = ['lightgreen', 'tomato']
plt.pie(coverage_data, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('GloVe Vocabulary Coverage')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.tight_layout()
plt.savefig('visualizations/glove_coverage.png')

print("\nEDA completed. Visualizations saved to 'visualizations' directory.")
print(f"Words not found in GloVe have been saved to 'words_not_in_glove.txt'")
