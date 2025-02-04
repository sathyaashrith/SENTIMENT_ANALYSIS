import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import os

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Download necessary NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the dataset
data = pd.read_csv("sentiment_analysis.csv")
print("Dataset preview:")
display(data.head())

# Basic dataset inspection
print("\nDataset Info:")
display(data.info())

# Check for missing values
print("\nMissing Values:")
display(data.isnull().sum())

# Drop missing rows if necessary
data = data.dropna()

# Rename columns if needed (Ensure proper column names for text and labels)
data = data.rename(columns={"text_column": "text", "sentiment_column": "sentiment"})

# Show class distribution
sns.countplot(x='sentiment', data=data, palette='viridis')
plt.title("Sentiment Distribution")
plt.show()

# Ensure NLTK resources are available before preprocessing
def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_resources()

# Text preprocessing function
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

# Apply preprocessing
data['processed_text'] = data['text'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['processed_text']).toarray()
y = data['sentiment']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluation
print("\nClassification Report:")
display(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())

print("Confusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Ensure output directory exists
output_dir = "/mnt/data/output"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "sentiment_predictions.csv")

# Save predictions
predictions = pd.DataFrame({
    "Actual Sentiment": y_test.values,
    "Predicted Sentiment": y_pred
})
predictions.to_csv(output_file, index=False)
print(f"Predictions saved to '{output_file}'")

# Display predictions
display(predictions.head())
