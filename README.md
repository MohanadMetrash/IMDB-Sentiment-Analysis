# IMDB Movie Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A deep learning project for classifying IMDB movie reviews as either positive or negative. This repository contains the code for data preprocessing, model training using a hybrid CNN-BiLSTM architecture, and scripts for inference on new reviews.

![Training History Plot](/training_history.png)

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Directory Structure](#directory-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Use](#how-to-use)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project implements a sentiment analysis model to determine whether a movie review is positive or negative. The workflow includes:
- **Text Preprocessing**: Cleaning raw text by removing HTML tags, punctuation, and stopwords.
- **Word Embeddings**: Utilizing pre-trained 100-dimensional GloVe embeddings to represent words as dense vectors.
- **Hybrid Model**: A deep learning model combining Convolutional Neural Networks (CNNs) for feature extraction and Bidirectional LSTMs (BiLSTMs) for capturing sequential context.
- **Training & Evaluation**: Training the model on the IMDB dataset and evaluating its performance.
- **Inference**: Scripts to predict the sentiment of new, unseen reviews from text input or a CSV file.

## Model Architecture

The model is a Sequential Keras model composed of the following layers:

1.  **Embedding Layer**: Initializes word embeddings with pre-trained GloVe vectors. Set to be non-trainable to retain learned knowledge.
2.  **Conv1D Layers**: Two sets of 1D convolutional layers to extract local features and patterns from the text sequence.
3.  **Bidirectional LSTM**: A BiLSTM layer to capture contextual information from both forward and backward directions in the sequence.
4.  **GlobalMaxPooling1D**: Reduces the dimensionality of the LSTM output.
5.  **Dense Layers**: Fully connected layers with Dropout and Batch Normalization for regularization.
6.  **Output Layer**: A final Dense layer with a sigmoid activation function to output a probability between 0 (negative) and 1 (positive).


## Results

The model was trained for 10 epochs and achieved the following performance on the test set:

-   **Test Accuracy**: ~86%
-   **Test Loss**: ~0.35

The training and validation accuracy/loss curves show good generalization with early stopping preventing significant overfitting.


## Setup and Installation

Follow these steps to set up the project environment.

**1. Clone the repository:**
```bash
git clone https://github.com/MohanadMetrash/IMDB-Sentiment-Analysis.git
cd IMDB-Sentiment-Analysis
```

**2. Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

**3. Install the required packages:**
```bash
pip install -r requirements.txt
```

**4. Download NLTK stopwords:**
```python
import nltk
nltk.download('stopwords')
```

**5. Download GloVe Embeddings:**
The model requires the GloVe pre-trained word embeddings.
- Download `glove.6B.zip` from [Stanford's GloVe page](https://nlp.stanford.edu/projects/glove/).
- Unzip the file and place `glove.6B.100d.txt` inside the `data/` directory.

After these steps, your `data/` directory should look like this:
```
data/
├── a1_IMDB_Dataset.csv
├── a3_IMDb_Unseen_Reviews.csv
└── glove.6B.100d.txt
```

## How to Use

### Training the Model

To train the model from scratch, run the `imdb_project_exploration.ipynb` script. This will process the data, build the model, train it, and save the final `model.keras` and `tokenizer.pkl` to the `saved_model/` directory.

```bash
python src/train.py
```

### Making Predictions

Use the `predict.py` script to get sentiment predictions. It loads the pre-trained model from `saved_model/`.

**To predict a single review:**
```bash
python src/predict.py --text "This movie was absolutely fantastic, a true masterpiece of cinema!"
```
**Expected Output:**
```
Prediction: [0.98] -> Positive
```

**To predict on a CSV file:**
The script can also read a CSV file. Make sure the file has a column named `Review Text`.
```bash
python src/predict.py --file data/a3_IMDb_Unseen_Reviews.csv
```
This will generate a `predictions.csv` file in the root directory with the results.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The [IMDb Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).
- The [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/) project by Stanford University.
