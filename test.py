import nltk
from nltk.corpus import movie_reviews
import random
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('movie_reviews')
data = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]


random.shuffle(data)
texts = [" ".join(words) for words, lables, in data]
lables = [label for words, label in data]

#data splitting
split = int(0.8 * len(texts))
train_texts, test_texts = texts[:split],  texts[split:]
train_labels, test_labels = lables[:split], lables[split:]

#feature extraction
vectorize=TfidfVectorizer(stop_words="english",max_features=5000)
x_train=vectorize.fit_transform(train_texts)
x_test=vectorize.transform(test_texts)

model=LogisticRegression(max_iter=1000)
model.fit(x_train,train_labels)

pred=model.predict(x_test)
print(f"Accuracy:{accuracy_score(test_labels,pred)}")
print(f"Classification Report:{classification_report(test_labels,pred)}")

joblib.dump(model,"model.pkl")
joblib.dump(vectorize,"vectorize.pkl")