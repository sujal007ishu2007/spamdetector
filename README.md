# 📧 Spam Mail Detection Using Deep Learning (BiLSTM & BERT)

## 🔍 Overview

This project aims to detect spam emails using two advanced deep learning models — **BiLSTM (Bidirectional Long Short-Term Memory)** and **BERT (Bidirectional Encoder Representations from Transformers)**. It compares their performance and showcases the power of NLP in spam classification.

---

## 🧠 Models Used

- **BiLSTM**: Captures sequential word dependencies in both directions.
- **BERT**: Transformer-based model that understands deep contextual semantics using attention mechanisms.

---

## 🎯 Objectives

- Build a spam detection system using BiLSTM and BERT.
- Demonstrate advantages of deep learning in NLP tasks over traditional ML.
- Improve classification metrics like precision, recall, and F1-score.
- Evaluate models using a clean, preprocessed email dataset.

---

## 🗃️ Dataset

- **Source**: Public dataset with 5,572 emails.
- **Distribution**:
  - Spam: 747 messages
  - Non-Spam: 4,825 messages
- **Split**:
  - Training: 70%
  - Validation: 15%
  - Testing: 15%

---

## 🧹 Preprocessing Steps

- Lowercasing all text
- Removing punctuation and HTML tags
- Stop-word removal
- Tokenization
- TF-IDF vectorization for BiLSTM
- WordPiece tokenization for BERT

---

## 🛠️ Project Workflow

1. **Data Collection**
2. **Text Preprocessing**
3. **Model Implementation (BiLSTM and BERT)**
4. **Training & Evaluation**
5. **Performance Comparison**
6. **Final Classification**

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

BERT consistently outperformed BiLSTM in all metrics but required more computational resources.

---

## 📈 Results

| Model   | Accuracy | Precision | Recall | F1-Score |
|---------|----------|-----------|--------|----------|
| BiLSTM  | Moderate | Moderate  | Good   | Good     |
| BERT    | High     | High      | High   | High     |

---

## ✅ Conclusion

- **BERT** provides superior spam detection accuracy but at a higher computational cost.
- **BiLSTM** is efficient and lightweight, suitable for environments with limited resources.
- Deep learning offers a significant improvement over traditional spam filters.

---

## 🚀 Future Scope

- 🔄 Real-time integration with email clients
- 🤖 Hybrid ensemble models
- 🌍 Multi-language support
- 🧠 Model auto-update with new spam data
- 💡 Explainable AI to improve user trust

---

## 👨‍💻 Authors

- Sujal Kommawar (22BCE8223)
- Lahari Yammanuru (22BCE20270)
- Harshitha Gadupudi (22BCE20279)

---

## 📚 References

- [Efficient Email Spam Detection Using Genetic Algorithm](https://onlinelibrary.wiley.com/doi/10.1155/2022/7710005)
- [Spam Email Detection Using Deep Learning Techniques](https://www.researchgate.net/publication/351678576_Spam_Email_Detection_Using_Deep_Learning_Techniques)

---

## 📌 License

This project is for educational and research purposes only.
