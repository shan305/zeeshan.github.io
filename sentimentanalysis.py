#!/usr/bin/env python
# coding: utf-8

import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.metrics import ConfusionMatrix
import ssl
import random
import matplotlib.pyplot as plt

# Download NLTK Resources
def download_nltk_resources():
    nltk.download('movie_reviews')
    nltk.download('punkt')
    nltk.download('stopwords')

# Load Movie Reviews Dataset
def load_movie_reviews():
    positive_reviews = [(list(movie_reviews.words(fileids=[f])), 'pos') for f in movie_reviews.fileids('pos')]
    negative_reviews = [(list(movie_reviews.words(fileids=[f])), 'neg') for f in movie_reviews.fileids('neg')]
    reviews = positive_reviews + negative_reviews
    random.shuffle(reviews)
    return reviews

# Feature Extraction
def extract_features(words):
    return dict([(word, True) for word in words])

# Preprocess Data
def preprocess_data(reviews):
    stop_words = set(stopwords.words('english'))
    processed_reviews = []

    for words, sentiment in reviews:
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
        processed_reviews.append((words, sentiment))

    return processed_reviews

# Split Data
def split_data(processed_reviews, split_ratio=0.8):
    split = int(len(processed_reviews) * split_ratio)
    train_data, test_data = processed_reviews[:split], processed_reviews[split:]
    return train_data, test_data

# Train Naive Bayes Classifier
def train_classifier(train_data):
    training_features = [(extract_features(words), sentiment) for words, sentiment in train_data]
    classifier = NaiveBayesClassifier.train(training_features)
    return classifier

# Evaluate the Model
def evaluate_model(classifier, test_data):
    test_features = [(extract_features(words), sentiment) for words, sentiment in test_data]
    accuracy = nltk_accuracy(classifier, test_features)
    print(f'Accuracy: {accuracy:.2%}')
    
    return test_features

# Calculate Metrics and Confusion Matrix
def calculate_metrics_and_confusion_matrix(test_labels, predicted_labels):
    cm = ConfusionMatrix(test_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)

    precision = cm['pos', 'pos'] / (cm['pos', 'pos'] + cm['neg', 'pos'])
    recall = cm['pos', 'pos'] / (cm['pos', 'pos'] + cm['pos', 'neg'])
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f'Precision: {precision:.2%}')
    print(f'Recall: {recall:.2%}')
    print(f'F1 Score: {f1_score:.2%}')

# Make Predictions
def make_predictions(examples, classifier):
    predictions = [classifier.classify(extract_features(word_tokenize(example))) for example in examples]
    for example, prediction in zip(examples, predictions):
        print(f'Example: {example}\nPrediction: {prediction}\n')

# Visualize Sentiment Distribution
def visualize_sentiment_distribution(predictions):
    sentiment_counts = {'pos': 0, 'neg': 0}
    for prediction in predictions:
        sentiment_counts[prediction] += 1

    labels = list(sentiment_counts.keys())
    counts = list(sentiment_counts.values())

    plt.bar(labels, counts, color=['green', 'red'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Predicted Sentiment Distribution for Examples')
    plt.show()

# Main Function
def main():
    download_nltk_resources()
    reviews = load_movie_reviews()
    processed_reviews = preprocess_data(reviews)
    train_data, test_data = split_data(processed_reviews)
    classifier = train_classifier(train_data)
    test_features = evaluate_model(classifier, test_data)
    
    test_labels = [sentiment for _, sentiment in test_data]
    predicted_labels = [classifier.classify(features) for features, _ in test_features]
    
    calculate_metrics_and_confusion_matrix(test_labels, predicted_labels)
    
    examples = [
        "This movie is fantastic! I loved every moment of it.",
        "The cinematography and acting were outstanding. A must-watch!",
        "The plot was confusing, and the characters were poorly developed.",
        "I regret watching this movie. It was a waste of time.",
        "I regret watching this. It was a waste of money."
    ]
    make_predictions(examples, classifier)
    
    visualize_sentiment_distribution(predicted_labels)

if __name__ == "__main__":
    main()
