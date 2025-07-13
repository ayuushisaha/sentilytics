##Sentiment Analysis on Movie Reviews

Project Overview:
This project implements a web-based Sentiment Analysis application using Python and the Streamlit framework. It analyzes movie reviews to determine their sentiment (e.g., positive or negative) using a machine learning pipeline. The application features text preprocessing, TF-IDF vectorization, and a Multinomial Naive Bayes classifier. It provides a user-friendly interface for live sentiment prediction and displays key model evaluation metrics.

Features:
Interactive Web Application: Built with Streamlit for a user-friendly experience.

Text Preprocessing: Cleans raw text data by converting to lowercase, removing URLs, mentions, punctuation, numbers, and stopwords.

TF-IDF Vectorization: Transforms text data into numerical feature vectors.

Multinomial Naive Bayes Classifier: A machine learning model trained to predict sentiment.

Live Sentiment Prediction: Allows users to input a movie review and get an instant sentiment prediction.

Model Evaluation Metrics: Displays accuracy, a confusion matrix, and a detailed classification report of the trained model.

Technologies Used
Python

Streamlit: For building the interactive web application.

Pandas: For data manipulation and loading the dataset.

NLTK (Natural Language Toolkit): For text preprocessing, specifically stopwords removal.

Scikit-learn: For machine learning functionalities, including:

TfidfVectorizer for text vectorization.

MultinomialNB for the classification model.

train_test_split for data partitioning.

accuracy_score, confusion_matrix, classification_report for model evaluation.

Matplotlib & Seaborn: For visualizing the confusion matrix.

Regular Expressions (re module): For advanced text cleaning patterns.

How to Run the Application Locally
Follow these steps to set up and run the Sentiment Analysis App on your local machine:

Prerequisites
Ensure you have Python installed (Python 3.8+ is recommended).

1. Clone or Download the Project
If you have Git, you can clone the repository:

git clone <repository_url> # Replace <repository_url> with your actual repo URL
cd sentiment-analysis # Navigate into the project directory

Otherwise, download the project files and extract them to a folder (e.g., sentiment-analysis).

2. Install Dependencies
Navigate to your project directory in your terminal or command prompt and install the required Python libraries:

3. Prepare the Dataset
Place your movie review dataset in a CSV file named reviews.csv in the root of your project directory (the same location as app.py). The CSV file must contain at least two columns:

text: The movie review content.

label: The corresponding sentiment label (e.g., 'positive', 'negative').

4. Run the Streamlit Application
From your project directory in the terminal/command prompt, execute the following command:

streamlit run app.py

This will start the Streamlit server and open the application in your default web browser.

Project Structure
sentiment-analysis/
├── app.py              # Main Streamlit application code
├── reviews.csv         # Dataset containing movie reviews and labels
└── README.md           # This README file
