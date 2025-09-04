# Neural Network SMS Text Classifier

This project implements a **spam vs ham SMS classifier** using a **Neural Network** built with **TensorFlow/Keras**. It classifies SMS messages as either **ham (0)** or **spam (1)**.

---

## Features

- **Preprocessing**:  
  - Text tokenization using `Tokenizer`  
  - Padding sequences to fixed length (`MAX_LENGTH`)  

- **Model Architecture**:  
  - Embedding layer for word representation  
  - LSTM layer to capture sequential patterns in messages  
  - Dropout layer to prevent overfitting  
  - Dense layer with sigmoid activation for binary classification  

- **Training**:  
  - Binary crossentropy loss  
  - Accuracy metric  
  - EarlyStopping callback on validation accuracy to prevent overfitting  
  - Class weighting to handle imbalanced datasets  

- **Prediction Function**:  
  - Predict new messages as **ham** or **spam**  
  - Returns both probability score and label  

---

## Dataset

- `train-data.tsv` → training dataset  
- `valid-data.tsv` → validation/test dataset  
- Both files contain two columns:  
  - `class` → `"ham"` or `"spam"`  
  - `message` → the SMS text  

---

## Usage

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Predict a single message
pred_text = "how are you doing today?"
prediction = predict_message(pred_text)
print(prediction)  # Example output: [0.032, 'ham']


sms-classifier/
├── train-data.tsv               # Training dataset
├── valid-data.tsv               # Validation/test dataset
├── sms_classifier.ipynb         # Colab/Notebook implementation
├── README.md                    # Project overview
└── requirements.txt             # Python dependencies


