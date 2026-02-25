# 📧 Email Spam Classifier

A machine learning project that classifies emails as **Spam** or **Ham (Legitimate)** using Natural Language Processing (NLP) and three different ML models. Achieves **97.94% accuracy** on the SMS Spam Collection dataset.

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Naive Bayes | 97.49% | 0.9831 |
| Logistic Regression | 96.86% | 0.9902 |
| **Random Forest** ✅ | **97.94%** | **0.9908** |

---

## 🖼️ Sample Output

```
Email     : "CONGRATULATIONS! You've won a $500 Amazon gift card! Click here to claim NOW!!!"
Result    : SPAM
Spam prob : 90.4%  |  Ham prob : 9.6%
Spam Risk [███████████████████████████░░░] 90%

Email     : "Hi Sarah, just wanted to confirm our lunch meeting tomorrow at 12:30pm."
Result    : HAM (Legitimate)
Spam prob : 1.0%   |  Ham prob : 99.0%
Spam Risk [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 1%
```

---

## 🗂️ Project Structure

```
Email Spam Classification/
├── Spam_classifier.py       # Main script
├── SMSSpamCollection        # Dataset (download separately)
├── spam_evaluation.png      # Generated charts
└── README.md
```

---

## ⚙️ How It Works

The project follows a 6-step machine learning pipeline:

**Step 1 — Load Dataset**
Loads the SMS Spam Collection dataset (5,572 real emails labeled as spam or ham).

**Step 2 — Text Preprocessing**
Cleans raw email text by converting to lowercase, removing URLs, email addresses, numbers, and punctuation.

**Step 3 — Feature Extraction (TF-IDF)**
Converts text into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency) with unigrams and bigrams. Vocabulary size: 5,000 features.

**Step 4 — Train 3 Models**
Trains and compares three machine learning classifiers: Naive Bayes, Logistic Regression, and Random Forest.

**Step 5 — Evaluate**
Evaluates each model using accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC curve.

**Step 6 — Predict New Emails**
Classifies any new email with a spam probability score and visual risk bar.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/email-spam-classifier.git
cd email-spam-classifier
```

### 2. Install dependencies
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

### 3. Download the dataset
Download the **SMS Spam Collection** dataset from:
👉 https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Extract and place the `SMSSpamCollection` file in the project folder.

### 4. Run the classifier
```bash
python Spam_classifier.py
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| scikit-learn | Machine learning models and evaluation |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib | Chart plotting |
| seaborn | Confusion matrix visualization |

---

## 📈 Evaluation Charts

Running the script automatically generates `spam_evaluation.png` containing:
- **Confusion Matrix** — shows correct vs incorrect predictions
- **Model Accuracy Comparison** — bar chart comparing all 3 models
- **ROC Curve** — shows the trade-off between catching spam and false alarms

---

## 🧪 Test Your Own Emails

Edit the `test_emails` list at the bottom of `Spam_classifier.py`:

```python
test_emails = [
    "Your custom email text here",
    "Another email to test",
]
```

Then run the script again to see predictions.

---

## 📋 Dataset Info

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Total emails | 5,572 |
| Spam emails | 747 (13.4%) |
| Ham emails | 4,825 (86.6%) |
| Train / Test split | 80% / 20% |

---

## 🔮 Future Improvements

- Use BERT or other transformer models for even higher accuracy
- Deploy as a web app using Flask or FastAPI
- Add a GUI interface for non-technical users
- Tune hyperparameters using GridSearchCV

---

## 👤 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/Sheraz-ahmed7)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
