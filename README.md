# Disaster Response Pipeline Project


## Project Overview
This project focuses on building a web-based application to classify messages for disaster response, helping crisis management teams organize incoming information effectively. The application uses machine learning models to categorize messages into predefined disaster-related categories. The project includes data processing scripts, training pipelines, and a Flask-based web application for user interaction.

---
### Data Files

- disaster_categories.csv: Contains categories associated with each message, indicating types of disasters (e.g., earthquake, flood).
- disaster_messages.csv: Holds the messages that need classification. This file contains text data that will be preprocessed and used to train and test the model.
- DisasterResponse.db: A SQLite database that stores the cleaned and processed disaster messages along with their respective categories.

---
### Scripts

- process_data.py: This script reads disaster_categories.csv and disaster_messages.csv, processes them to handle missing values, and merges them into a single dataset. The cleaned data is saved into DisasterResponse.db for further use in model training.
- train_classifier.py: This script loads the cleaned data from DisasterResponse.db, trains a machine learning model on the labeled messages, and performs hyperparameter tuning using GridSearchCV.

---
### Flask Application Files

- run.py: The main script to run the Flask web application. It loads the trained model and serves the application, allowing users to input messages for classification.
- go.html: A Flask template that extends master.html to display classification results. It shows each disaster category as a card, colored based on whether the category is relevant (green for relevant, grey for irrelevant).
- master.html: The base HTML template for the web app. It includes a navigation bar, title, input form, and placeholders for displaying graphs and message classifications.

---
## Getting Started

1. Run process_data.py to generate DisasterResponse.db
2. Run train_classifier.py to create and save the model.
3. Start the web app using run.py
4. Enter a message in the input box, and the app will classify it into disaster categories.

---
