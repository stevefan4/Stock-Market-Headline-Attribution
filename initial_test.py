import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import datetime
import unittest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from nltk.sentiment import SentimentIntensityAnalyzer
from scipy.stats import bayes_mvs

# ----------------------------------------------------------------------
# 🟢 Step 1: Data Preparation
# ----------------------------------------------------------------------

def generate_sample_data():
    """
    Creates a simulated dataset of daily headlines and SPX returns.
    """
    # Generate 100 days of dates starting from Nov 1, 2024
    dates = pd.date_range(start="2024-11-01", periods=100, freq="D")

    # Sample financial headlines covering different macro themes
    headlines = [
        "Fed hikes interest rates, markets react negatively",
        "Strong earnings reports push S&P 500 higher",
        "Global economic slowdown weighs on stocks",
        "Geopolitical tensions escalate, causing market volatility",
        "Tech stocks surge on positive AI breakthroughs",
        "Trump announces new tariffs, markets react cautiously",
        "Federal Reserve hints at possible rate cuts in 2025",
        "Earnings season starts with strong reports from major firms",
        "US-China trade war fears resurface after new policy updates",
        "Dogecoin rallies as Elon Musk tweets again"
    ]

    # Seed for reproducibility
    np.random.seed(42)

    # Create DataFrame with randomly selected headlines
    df = pd.DataFrame({
        "date": dates,
        "headline": np.random.choice(headlines, size=len(dates)),  # Random headlines
        "spx_return": np.random.uniform(-2, 2, len(dates))  # Simulated daily SPX returns (-2% to 2%)
    })

    df["date"] = pd.to_datetime(df["date"])  # Convert date column to proper format
    return df

# Load simulated data
df = generate_sample_data()

# ----------------------------------------------------------------------
# 🟢 Step 2: Macro Factor Classification (Original Method)
# ----------------------------------------------------------------------

# Predefined macroeconomic factors and keywords
macro_factors = {
    "Fed Policy": ["fed", "interest rates", "hike", "cut", "federal reserve"],
    "Earnings": ["earnings", "profit", "quarterly", "revenue", "reports"],
    "Macro": ["economy", "economic", "recession", "growth", "slowdown"],
    "Geopolitics": ["geopolitical", "war", "conflict", "tensions", "sanctions"],
    "Tech Sentiment": ["tech", "AI", "software", "NASDAQ", "innovation"],
    "Trump-Tariffs/Immigration/DOGE": ["trump", "tariffs", "immigration", "dogecoin", "trade war", "doge"]
}

def assign_macro_factor(headline):
    """
    Assigns a macro factor to a given financial headline based on keyword matching.
    """
    headline = headline.lower()  # Convert to lowercase for case-insensitive matching
    for factor, keywords in macro_factors.items():
        if any(word in headline for word in keywords):
            return factor  # Return the first matching macro category
    return "Other"  # If no match, classify as "Other"

# Apply classification function
df["original_factor"] = df["headline"].apply(assign_macro_factor)

# ----------------------------------------------------------------------
# 🟢 Step 3: Sentiment-Based Attribution
# ----------------------------------------------------------------------

def compute_sentiment_scores(df):
    """
    Uses NLP sentiment analysis to determine how positive or negative each headline is.
    Scores range from -1 (negative) to +1 (positive).
    """
    sia = SentimentIntensityAnalyzer()
    df["sentiment_score"] = df["headline"].apply(lambda x: sia.polarity_scores(x)["compound"])
    df["sentiment_factor_weighted"] = df["sentiment_score"] * df["spx_return"]
    return df

# Apply sentiment analysis
df = compute_sentiment_scores(df)

# ----------------------------------------------------------------------
# 🟢 Step 4: Regression-Based Attribution
# ----------------------------------------------------------------------

def compute_regression_attributions(df):
    """
    Uses a linear regression model to determine how each macro factor contributes to SPX movements.
    """
    # Convert categorical macro factors into one-hot encoded variables
    encoder = OneHotEncoder(sparse=False)
    X = encoder.fit_transform(df[["original_factor"]])
    y = df["spx_return"].values  # Target variable

    # Train the regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict attributions for each day
    df["regression_factor_weighted"] = model.predict(X)
    return df

# Apply regression-based method
df = compute_regression_attributions(df)

# ----------------------------------------------------------------------
# 🟢 Step 5: Bayesian Inference-Based Attribution
# ----------------------------------------------------------------------

def compute_bayesian_attributions(df):
    """
    Uses Bayesian inference to estimate the likely contribution of each macro factor to market movements.
    """
    bayesian_attributions = {}

    for factor in df["original_factor"].unique():
        subset = df[df["original_factor"] == factor]["spx_return"]
        if len(subset) > 1:
            mean, _, _ = bayes_mvs(subset, alpha=0.95)
            bayesian_attributions[factor] = mean.statistic  # Expected mean return
        else:
            bayesian_attributions[factor] = 0  # Default to zero for small datasets

    df["bayesian_factor_weighted"] = df["original_factor"].map(bayesian_attributions)
    return df

# Apply Bayesian attribution
df = compute_bayesian_attributions(df)

# ----------------------------------------------------------------------
# 🟢 Step 6: Visualization & Comparison
# ----------------------------------------------------------------------

def visualize_results(df):
    """
    Plots cumulative attribution scores over time for each method.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df["date"], df["spx_return"].cumsum(), label="SPX Returns", color="black", linestyle="dotted")
    plt.plot(df["date"], df["sentiment_factor_weighted"].cumsum(), label="Sentiment Attribution", linestyle="--")
    plt.plot(df["date"], df["regression_factor_weighted"].cumsum(), label="Regression Attribution", linestyle=":")
    plt.plot(df["date"], df["bayesian_factor_weighted"].cumsum(), label="Bayesian Attribution", linestyle="-.")

    plt.xlabel("Date")
    plt.ylabel("Cumulative Attribution Score")
    plt.title("Comparison of Market Attribution Models")
    plt.legend()
    plt.show()

# Plot results
visualize_results(df)

# ----------------------------------------------------------------------
# 🟢 Step 7: Unit Testing
# ----------------------------------------------------------------------

class TestAttributionMethods(unittest.TestCase):
    def test_macro_factor_assignment(self):
        self.assertEqual(assign_macro_factor("Fed cuts interest rates"), "Fed Policy")
        self.assertEqual(assign_macro_factor("Strong earnings report"), "Earnings")
        self.assertEqual(assign_macro_factor("Tech sector surges on AI"), "Tech Sentiment")
        self.assertEqual(assign_macro_factor("Dogecoin price jumps"), "Trump-Tariffs/Immigration/DOGE")
        self.assertEqual(assign_macro_factor("No recognizable keywords"), "Other")

    def test_sentiment_range(self):
        self.assertTrue(df["sentiment_score"].between(-1, 1).all())

# Run tests
unittest.main(argv=[''], verbosity=2, exit=False)
