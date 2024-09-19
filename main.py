import pandas as pd
from model import FeedbackAnalyzer
from explainability import explain_model

# Load dataset
data_path = 'data/amazon_reviews.csv'
df = pd.read_csv(data_path)

# Initialize and train model
analyzer = FeedbackAnalyzer()
analyzer.train(df['reviewText'], df['sentiment'])

# Analyze feedback
results = analyzer.analyze_feedback(df['reviewText'])

# Explain the model's predictions
explanations = explain_model(analyzer.model, df['reviewText'])
