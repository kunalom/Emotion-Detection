
# Emotion Detection using DistilBERT

This project demonstrates fine-tuning of the `distilbert-base-uncased` transformer model using Hugging Face's `transformers` and `datasets` libraries for multi-class emotion classification from text.

## 🔍 Project Overview

The goal is to classify text inputs into one of several emotion categories such as **joy**, **anger**, **sadness**, **love**, **fear**, and **surprise** using a transformer-based NLP model.

## 🚀 Technologies Used

- Python
- Hugging Face Transformers
- Hugging Face Datasets
- PyTorch
- DistilBERT (Pretrained)
- Scikit-learn (for evaluation)
- Matplotlib (for visualization)

## 📁 Dataset

We used the `emotion` dataset provided by Hugging Face Datasets. It includes 20,000+ labeled text samples categorized into 6 emotion classes.

```python
from datasets import load_dataset
dataset = load_dataset("emotion")
```

## 🏗️ Project Structure

- `Tokenization` using Hugging Face Tokenizer
- `DataLoader` creation for efficient batch processing
- `TrainingArguments` setup for controlling fine-tuning process
- `Trainer` API for training and evaluation
- Inference on custom text samples

## 🧠 Model Training

- Base Model: `distilbert-base-uncased`
- Loss Function: CrossEntropy
- Optimizer: AdamW
- Epochs: 2
- Batch Size: 64
- Evaluation Metric: Accuracy

## 📊 Results

After fine-tuning, the model performs well in detecting emotions with good generalization on validation data. You can test it using:

```python
text = "I feel fantastic today!"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model(**inputs)
logits = outputs.logits
pred = torch.argmax(logits, dim=1).item()
print(classes[pred])  # Outputs the predicted emotion
```

## 📌 Key Features

- ✅ End-to-end transformer pipeline: load, tokenize, train, evaluate, predict
- ✅ Fine-tuned on emotional text data using Hugging Face Trainer
- ✅ Ready for downstream deployment or chatbot integration

## 📎 References

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Dataset on Hugging Face](https://huggingface.co/datasets/emotion)

---

## 🧑‍💻 Author

Kunal Pathak  
Feel free to contribute or connect on [GitHub](https://github.com/).
