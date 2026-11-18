# Deep (Stacked) RNN Architecture

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Domain-Deep%20Learning-blue)](https://en.wikipedia.org/wiki/Deep_learning)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/API-Keras-red)](https://keras.io/)
[![NLP](https://img.shields.io/badge/Task-Sentiment%20Analysis-green)](https://en.wikipedia.org/wiki/Sentiment_analysis)

A practical exploration of **Deep (Stacked) Recurrent Neural Networks** using TensorFlow and Keras. This project demonstrates how stacking recurrent layers helps neural networks process sequential data and learn representations at different levels.

The notebook implements and trains three architectures—**SimpleRNN, LSTM, and GRU**—on the IMDB movie-review sentiment classification dataset.

**Repository:** https://github.com/mani787060/deep-rnn-architecture

---

## Project Overview

Recurrent Neural Networks (RNNs) are designed to process sequential data by maintaining information from previous time steps. They are useful in applications such as natural language processing, time-series analysis, and speech recognition.

A standard recurrent architecture may use a single recurrent layer. A **Deep RNN**, also known as a **Stacked RNN**, uses multiple recurrent layers placed one after another.

In a stacked architecture, the output sequence from one recurrent layer is passed to the next recurrent layer. This allows the network to learn increasingly complex representations of the input sequence.

This project explores that concept through three Keras implementations:

1. Stacked SimpleRNN
2. Stacked LSTM
3. Stacked GRU

The models are trained for binary sentiment classification using the IMDB movie-review dataset.

---

## Key Concepts and Implementations

### 1. Stacked SimpleRNN Architecture

The first model contains:

- An Embedding layer that converts word indices into dense vectors.
- A first `SimpleRNN` layer with 5 units and `return_sequences=True`.
- A second `SimpleRNN` layer with 5 units.
- A Dense output layer with sigmoid activation.

**Architecture:**

```text
Input Sequence
      |
      v
Embedding Layer
      |
      v
SimpleRNN (5 units)
return_sequences=True
      |
      v
SimpleRNN (5 units)
      |
      v
Dense (1, Sigmoid)
      |
      v
Sentiment Prediction
```

The first recurrent layer returns the complete output sequence so that the next recurrent layer can process it.

### 2. Stacked LSTM Architecture

Long Short-Term Memory (LSTM) networks introduce gating mechanisms to control how information is retained, updated, and forgotten.

The second model contains:

- An Embedding layer.
- Two stacked LSTM layers, each with 5 units.
- `return_sequences=True` in the first LSTM layer.
- A Dense output layer with sigmoid activation.

**Architecture:**

```text
Input Sequence
      |
      v
Embedding Layer
      |
      v
LSTM (5 units)
return_sequences=True
      |
      v
LSTM (5 units)
      |
      v
Dense (1, Sigmoid)
      |
      v
Sentiment Prediction
```

This implementation provides a practical example of stacking LSTM layers for sequence classification.

### 3. Stacked GRU Architecture

Gated Recurrent Units (GRUs) use gating mechanisms to manage information flow through recurrent networks.

The third model contains:

- An Embedding layer.
- Two stacked GRU layers, each with 5 units.
- `return_sequences=True` in the first GRU layer.
- A Dense output layer with sigmoid activation.

**Architecture:**

```text
Input Sequence
      |
      v
Embedding Layer
      |
      v
GRU (5 units)
return_sequences=True
      |
      v
GRU (5 units)
      |
      v
Dense (1, Sigmoid)
      |
      v
Sentiment Prediction
```

This model demonstrates how GRU layers can also be stacked to learn sequential representations.

---

## Dataset: IMDB Movie Reviews

The project uses the built-in IMDB dataset provided by TensorFlow/Keras.

The dataset contains movie reviews labeled as either positive or negative.

The notebook loads the dataset using:

```python
from tensorflow.keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=10000
)
```

### Dataset Preparation

The notebook applies the following preprocessing:

- **Vocabulary limit:** Keeps the 10,000 most frequent words.
- **Sequence padding:** Converts variable-length reviews into sequences of equal length.
- **Maximum sequence length:** 100 tokens per review.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)
```

Padding ensures that the input sequences have a consistent shape before they are passed to the neural network.

---

## Model Configuration

The three architectures share a similar overall structure, making it easier to understand how different recurrent layers are used in stacked networks.

| Component | Configuration |
|---|---|
| Input sequence length | 100 |
| Vocabulary size | 10,000 |
| Embedding dimension | 32 |
| Number of recurrent layers | 2 |
| Recurrent units per layer | 5 |
| Output layer | Dense (1) |
| Output activation | Sigmoid |
| Loss function | Binary Crossentropy |
| Optimizer | Adam |
| Evaluation metric during training | Accuracy |
| Epochs | 5 |
| Batch size | 32 |
| Validation split | 20% |

The notebook compiles each model using binary crossentropy loss and the Adam optimizer.

---

## Training the Models

Each model is trained using the same training data and a validation split of 20%.

Example:

```python
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)
```

The same training configuration is used for the LSTM and GRU models.

The notebook includes training logs showing training loss, training accuracy, validation loss, and validation accuracy across epochs.

**Important:** These are training and validation results. The current notebook does not include a separate final evaluation using `X_test` and `y_test`, so no independent test accuracy comparison is claimed here.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Building and training neural networks |
| NumPy | Numerical computing support |
| IMDB Dataset | Movie-review sentiment classification |
| Jupyter Notebook | Interactive development environment |

---

## Repository Structure

```text
deep-rnn-architecture/
│
├── deep-RNNs.ipynb    # Stacked SimpleRNN, LSTM, and GRU implementations
└── README.md          # Project documentation
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/mani787060/deep-rnn-architecture.git
```

### 2. Navigate to the Project Directory

```bash
cd deep-rnn-architecture
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install tensorflow numpy jupyter
```

Use a Python version supported by your installed TensorFlow release.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
deep-RNNs.ipynb
```

Run the notebook cells sequentially to load the dataset, build the three architectures, inspect their model summaries, and train them.

**Note:** The IMDB dataset is downloaded automatically by Keras the first time it is loaded. An internet connection may be required for this initial download.

---

## What I Learned

Through this project, I explored:

- How recurrent neural networks process sequential data.
- The concept of stacking multiple recurrent layers.
- How information flows between recurrent layers.
- Why `return_sequences=True` is needed when stacking recurrent layers.
- How Embedding layers represent words as dense vectors.
- The architectural differences between SimpleRNN, LSTM, and GRU.
- How to build and compile recurrent models using TensorFlow/Keras.
- How to train sequence-classification models using the IMDB dataset.

---

## Future Improvements

Possible extensions to this project include:

- Evaluate all three models on the independent test dataset.
- Compare test accuracy, precision, recall, and F1-score.
- Visualize training and validation loss and accuracy.
- Experiment with different numbers of recurrent units and layers.
- Add Dropout and other regularization techniques.
- Explore gradient clipping and other training-stability techniques.
- Experiment with bidirectional recurrent layers.
- Compare model size, training time, and inference performance.

---

## References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [TensorFlow IMDB Dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb)
- [Understanding Recurrent Neural Networks](https://www.tensorflow.org/guide/keras/working_with_rnns)

---
