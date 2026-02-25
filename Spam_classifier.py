# ================================================================
#         EMAIL SPAM CLASSIFIER - Complete Beginner Guide
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

print("All libraries loaded successfully!\n")


# ================================================================
# STEP 1: LOAD THE DATASET
# ================================================================

# Get the folder where THIS script is saved
script_dir   = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "SMSSpamCollection")

print(f"Looking for dataset at: {dataset_path}")

if os.path.exists(dataset_path):
    print("Real dataset found! Loading 5,500+ emails...\n")
    df = pd.read_csv(dataset_path, sep='\t', names=['label', 'text'], encoding='latin-1')
else:
    print("Real dataset NOT found. Using built-in 50 sample emails.")
    print("For 95%+ accuracy, put SMSSpamCollection in the same folder as this script.\n")

    spam_emails = [
        "Congratulations! You've won a FREE iPhone. Click here to claim NOW!!!",
        "URGENT: Your account will be suspended. Verify now at http://scam.com",
        "Win $1000 cash prize! Limited offer. Act fast! Call 1-800-FAKE",
        "FREE VIAGRA!!! Buy now discount 90% off click here!!",
        "You are selected for a cash reward. Send your bank details.",
        "Earn $5000 per week working from home! No experience needed!",
        "WINNER! Claim your prize before it expires! Reply YES now!",
        "Get rich quick! Guaranteed returns! Invest now risk-free!",
        "FREE entry in our competition to win FA Cup final tickets!",
        "Urgent! We need to verify your PayPal account immediately.",
        "You have been chosen to receive $500 Walmart gift card. Click now!",
        "Lose weight fast! Miracle pill! 100% guaranteed results!",
        "Your bank account has been compromised. Login now to secure it.",
        "Cheap meds online! No prescription needed! Order now!",
        "You are pre-approved for a $10,000 loan. No credit check!",
        "Hot singles in your area! Click to meet them now!",
        "Make money online fast! Work from home $200/hour!",
        "Congratulations, you have been randomly selected for a survey prize!",
        "Act now! Your subscription expires today. Renew immediately!",
        "Exclusive deal just for you: 80% off luxury watches. Order now!",
        "Your computer may be infected! Download our FREE antivirus now!",
        "Claim your unclaimed inheritance of $2 million. Contact us today!",
        "Double your bitcoin in 24 hours! Guaranteed profit! Act fast!",
        "FREE gift card inside! Just verify your email to claim.",
        "Warning: Unusual login detected. Verify your account NOW!",
    ]
    ham_emails = [
        "Hey, are we still meeting for lunch tomorrow?",
        "Can you send me the report when you get a chance?",
        "The meeting has been rescheduled to 3pm on Friday.",
        "Happy to help with your question, let me know if you need more info.",
        "Thanks for the birthday wishes, it really made my day!",
        "I'll be home late tonight, don't wait up for dinner.",
        "The project deadline has been extended by one week.",
        "Can you pick up some groceries on your way home?",
        "Hi, I wanted to follow up on our conversation from yesterday.",
        "Please find attached the invoice for last month's services.",
        "We're having a team dinner on Thursday, hope you can make it.",
        "Your order has been shipped! Expected delivery: Monday.",
        "Just checking in to see how you're doing after the surgery.",
        "The quarterly report is ready for your review.",
        "Don't forget about the dentist appointment at 2pm tomorrow.",
        "Can we reschedule our call to next week? I'm swamped this week.",
        "Here are the notes from today's meeting as promised.",
        "Thank you for submitting your application. We'll be in touch.",
        "Your flight confirmation number is ABC123. Have a safe trip!",
        "The library book you requested is now available for pickup.",
        "Hi, just wanted to say I really enjoyed your presentation today.",
        "Please review and sign the attached document by end of day.",
        "Your Amazon package was delivered to your front door.",
        "Reminder: Team standup at 9am tomorrow. See you all then!",
        "Let me know if you have any questions about the proposal.",
    ]
    df = pd.DataFrame({
        'label': ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails),
        'text':  spam_emails + ham_emails
    })

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print("=" * 60)
print("STEP 1: DATASET LOADED")
print("=" * 60)
print(f"Total emails : {len(df)}")
print(f"Spam emails  : {sum(df['label'] == 'spam')}")
print(f"Ham  emails  : {sum(df['label'] == 'ham')}")
print(f"\nFirst 5 rows:")
print(df.head())


# ================================================================
# STEP 2: TEXT PREPROCESSING
# ================================================================

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

print("\n" + "=" * 60)
print("STEP 2: TEXT PREPROCESSING")
print("=" * 60)
print("BEFORE:", df['text'].iloc[0])
print("AFTER :", df['cleaned_text'].iloc[0])


# ================================================================
# STEP 3: FEATURE EXTRACTION (TF-IDF)
# ================================================================

X = df['cleaned_text']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                        stop_words='english', min_df=1)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

print("\n" + "=" * 60)
print("STEP 3: FEATURE EXTRACTION (TF-IDF)")
print("=" * 60)
print(f"Training emails : {len(X_train)}")
print(f"Testing  emails : {len(X_test)}")
print(f"Vocabulary size : {X_train_tfidf.shape[1]} words/phrases")


# ================================================================
# STEP 4: TRAIN THE MODELS
# ================================================================

print("\n" + "=" * 60)
print("STEP 4: TRAINING MODELS")
print("=" * 60)

models = {
    "Naive Bayes"         : MultinomialNB(),
    "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.0
    results[name] = {'model': model, 'y_pred': y_pred,
                     'y_prob': y_prob, 'accuracy': acc, 'auc': auc}
    print(f"\n  {name}:")
    print(f"    Accuracy : {acc:.2%}")
    print(f"    ROC-AUC  : {auc:.4f}")


# ================================================================
# STEP 5: EVALUATE BEST MODEL + CHARTS
# ================================================================

print("\n" + "=" * 60)
print("STEP 5: EVALUATION")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]['accuracy'])
best      = results[best_name]

print(f"\nBest Model: {best_name} (Accuracy: {best['accuracy']:.2%})")
print("\nClassification Report:")
print(classification_report(y_test, best['y_pred'], target_names=['Ham', 'Spam']))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Email Spam Classifier - Model Evaluation", fontsize=16, fontweight='bold')

cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
            linewidths=2, linecolor='white')
axes[0].set_title(f'Confusion Matrix\n({best_name})', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Actual Label')
axes[0].set_xlabel('Predicted Label')

names  = list(results.keys())
accs   = [results[m]['accuracy'] for m in names]
colors = ['#2ecc71' if m == best_name else '#3498db' for m in names]
bars   = axes[1].bar(names, accs, color=colors, edgecolor='black', width=0.5)
axes[1].set_ylim(0, 1.15)
axes[1].set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Accuracy')
axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.4, label='Random guess (50%)')
axes[1].legend()
for bar, acc in zip(bars, accs):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
best_idx = names.index(best_name)
axes[1].text(best_idx, accs[best_idx] + 0.06, 'Best', ha='center', fontsize=10, color='green')

try:
    fpr, tpr, _ = roc_curve(y_test, best['y_prob'])
    axes[2].plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC Curve (AUC = {best["auc"]:.3f})')
    axes[2].fill_between(fpr, tpr, alpha=0.1, color='blue')
    axes[2].plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='Random (AUC = 0.5)')
    axes[2].set_xlabel('False Positive Rate')
    axes[2].set_ylabel('True Positive Rate')
    axes[2].set_title(f'ROC Curve\n({best_name})', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
except:
    axes[2].text(0.5, 0.5, 'Need more data', ha='center', va='center',
                 transform=axes[2].transAxes)

plt.tight_layout()
chart_path = os.path.join(script_dir, 'spam_evaluation.png')
plt.savefig(chart_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nChart saved: spam_evaluation.png")


# ================================================================
# STEP 6: PREDICT NEW EMAILS
# ================================================================

print("\n" + "=" * 60)
print("STEP 6: CLASSIFY NEW EMAILS")
print("=" * 60)

def predict_email(email_text):
    cleaned    = preprocess_text(email_text)
    vectorized = tfidf.transform([cleaned])
    model      = results[best_name]['model']
    prediction = model.predict(vectorized)[0]
    spam_prob  = model.predict_proba(vectorized)[0][1]
    ham_prob   = 1 - spam_prob
    label      = "SPAM" if prediction == 1 else "HAM (Legitimate)"
    print(f"\nEmail     : \"{email_text[:70]}{'...' if len(email_text)>70 else ''}\"")
    print(f"Result    : {label}")
    print(f"Spam prob : {spam_prob:.1%}  |  Ham prob : {ham_prob:.1%}")
    bar = chr(9608) * int(spam_prob * 30) + chr(9617) * (30 - int(spam_prob * 30))
    print(f"Spam Risk [{bar}] {spam_prob:.0%}")

test_emails = [
    "CONGRATULATIONS! You've won a $500 Amazon gift card! Click here to claim NOW!!!",
    "FREE loan offer! $50,000 approved instantly. No credit check. Apply now!",
    "Your PayPal account has been LIMITED. Verify your identity immediately!",
    "Hi Sarah, just wanted to confirm our lunch meeting tomorrow at 12:30pm.",
    "Please find attached the quarterly budget report. Let me know if you have questions.",
    "Don't forget team standup at 9am tomorrow. Agenda: sprint review.",
]

for email in test_emails:
    predict_email(email)

print("\n" + "=" * 60)
print("CLASSIFIER COMPLETE!")
print("=" * 60)
print(f"\n  Best Model : {best_name}")
print(f"  Accuracy   : {best['accuracy']:.2%}")
print(f"  ROC-AUC    : {best['auc']:.4f}")
print(f"\n  Chart saved: spam_evaluation.png")