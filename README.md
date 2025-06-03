# Arabic Named Entity Recognition (NER) using BiLSTM

## 🧠 Domain: Natural Language Processing (NLP)

This project focuses on building an Arabic Named Entity Recognition (NER) model using a Bidirectional Long Short-Term Memory (BiLSTM) network. It identifies and classifies named entities like **Person**, **Location**, and **Organization** from Arabic text.


## 🎯 Project Objective

To develop a robust NER system for the Arabic language to support information extraction tasks by classifying tokens into named entity categories. This is especially important given the scarcity of NLP resources for Arabic.


## 🧰 Model Overview

- **Architecture**: BiLSTM (Bidirectional LSTM) model.
- **Why BiLSTM?**: It reads input sequences in both directions (forward and backward), capturing context from both sides — crucial for accurate entity classification.

## 📦 Dataset

- **Name**: [ANERCorp (Arabic Named Entity Corpus)](https://huggingface.co/datasets/asas-ai/ANERCorp)
- **Labels**:  
  - `PER`: Person  
  - `LOC`: Location  
  - `ORG`: Organization  
  - `O`: Other
  - `Misc`: Miscellaneous
- **Format**: IOB (Inside–Outside–Beginning)  
- **Preprocessing**:
  - Mapping of words and tags to integers
  - Augmentation
  - Sequence padding


## 📈 Model Training & Evaluation

- **Embedding Layer**: Transforms tokens into dense vectors.
- **BiLSTM Layer**: Captures sequential context.
- **TimeDistributed Dense Layer**: Predicts entity tag for each token.
- **Loss Function**: Sparse Categorical cross-entropy.
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-score.

## 🖥️ Streamlit Application

An interactive web app is built using Streamlit to allow users to explore and test the model in real-time.

### 🎯 Features of the Streamlit App

- **🔍 Entity Extraction**: Input your own Arabic text or choose from sample domains (Politics, Sports, Education) to extract named entities.
- **📊 Entity Statistics**: Visualize entity tag distributions using bar and pie charts.
- **🧰 Preprocessing Insight**: See how text is cleaned, normalized, and filtered before prediction.
- **📥 Export Option**: Download entity predictions as a CSV file for further analysis.
- **💡 Session Memory**: Keeps user input and predictions across tabs for seamless exploration.

## 📁 Repository Structure

```
Arabic-NER-BiLSTM/
│
├── Arabic_NER.ipynb         # Jupyter notebook for training and evaluating the model
├── app.py                   # Streamlit application
└── README.md                # Project documentation
```

## 🚀 Run the App

To run the Streamlit app locally:

```bash
streamlit run app.py

