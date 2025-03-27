# Sentiment Analysis BiLSTM with Attention

A deep learning project implementing sentiment analysis using a Bidirectional LSTM with Attention mechanism. This model achieves 89.3% accuracy on binary sentiment classification of customer reviews from multiple sources (Amazon, IMDB, and Yelp).

## Project Overview

This project implements a sentiment analysis system that can classify text as positive or negative sentiment. Key features include:

- Bidirectional LSTM with Attention mechanism for better context understanding
- Pre-trained GloVe embeddings with intelligent handling of out-of-vocabulary words
- Automated spelling correction using SymSpell
- Bayesian hyperparameter optimization
- Comprehensive data preprocessing pipeline
- Detailed visualizations and metrics tracking

## Model Architecture

- **Word Embeddings**: 100-dimensional GloVe vectors
- **Bidirectional LSTM**: Multiple layers with optimized hidden dimensions
- **Attention Mechanism**: Helps focus on sentiment-bearing words
- **Dropout Regularization**: Prevents overfitting
- **Output Layer**: Binary classification (positive/negative sentiment)

## Requirements

```bash
# Core dependencies
torch>=1.9.0
numpy>=1.19.5
pandas>=1.3.0
matplotlib>=3.4.3
seaborn>=0.11.2
nltk>=3.6.3
symspellpy>=6.7.0
scikit-learn>=0.24.2
tqdm>=4.62.3

# Optional for hyperparameter optimization
optuna>=2.10.0
mlflow>=1.20.2
```

## Project Structure

```
sentiment-analysis-bilstm-attention/
├── data/
│   ├── amazon_cells_labelled.txt
│   ├── imdb_labelled.txt
│   ├── yelp_labelled.txt
│   ├── glove.6B.100d.txt
│   └── processed_data/
├── models/
│   └── lstm_with_attention.py
├── configs/
│   ├── model_config.json
│   └── bayesian_best_config.json
├── visualizations/
├── checkpoints/
├── data_preprocessing.py
├── train.py
├── test.py
└── model_summary.py
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analysis-bilstm-attention.git
cd sentiment-analysis-bilstm-attention
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download GloVe embeddings:
```bash
# Download glove.6B.zip from https://nlp.stanford.edu/data/glove.6B.zip
# Extract glove.6B.100d.txt to the data/ directory
```

## Usage

### 1. Data Preprocessing

```bash
python data_preprocessing.py
```
This script:
- Loads and processes text data from multiple sources
- Builds vocabulary and handles out-of-vocabulary words
- Creates embedding matrix using GloVe
- Performs train/validation/test split
- Saves processed data and configurations

### 2. Training

```bash
python train.py --config configs/bayesian_best_config.json
```
Features:
- Bayesian hyperparameter optimization
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training metrics visualization

### 3. Testing

```bash
python test.py --checkpoint checkpoints/[timestamp]/model_epoch_best.pt
```
Outputs:
- Test accuracy and metrics
- Confusion matrix
- Classification report

## Performance

The model achieves:
- Test Accuracy: 89.3%
- Precision: 89.8%
- Recall: 90.9%
- F1 Score: 90.4%

## Visualizations

The project generates various visualizations:
- Training and validation metrics over time
- Confusion matrix
- GloVe vocabulary coverage
- Attention weights visualization
- Class distribution plots

## Model Interpretability

The attention mechanism provides interpretability by highlighting which words contributed most to the classification decision. This can be visualized using the attention weights output.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- GloVe embeddings from Stanford NLP
- Dataset sources: Amazon, IMDB, and Yelp reviews
- SymSpell for spelling correction
- PyTorch community for deep learning implementation

## Contact

Jack Vittimberga - [jvittimberga@gmail.com](mailto:jvittimberga@gmail.com)
