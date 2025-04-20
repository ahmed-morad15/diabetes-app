# Diabetic Prediction System

The **Diabetic Prediction System** is a sophisticated web application powered by **Machine Learning** and **Artificial Intelligence**. Built using **Flask**, this application allows users to predict their likelihood of having diabetes based on personal medical data, such as age, BMI, blood glucose levels, and more. In addition to providing predictive analytics, the system offers personalized advice and dynamically generated questions via **Large Language Models (LLM)**, facilitating continuous engagement and support for diabetes management.

## Key Features

- **Diabetes Prediction**: Accurately predict if a person is diabetic or healthy using a pre-trained machine learning model based on their medical inputs such as age, BMI, blood glucose levels, and more.
- **Dynamic Question Generation**: Generate contextually relevant follow-up questions for diabetic patients, adapting to their previous answers for a more personalized experience.
- **Personalized Medical Advice**: Receive actionable and tailored advice regarding lifestyle, diet, and management strategies to handle diabetes based on the Q&A interaction history.
- **Chatbot for Continuous Support**: Engage with a friendly chatbot that provides concise and helpful responses regarding diabetes care, offering general guidance on diabetes management.

## Project Overview

The project consists of three main components:

1. **Machine Learning Model**: A pre-trained model that uses patient health data to classify whether an individual has diabetes or not.
2. **LLM-Based Conversational Interface**: An AI-driven assistant that interacts with the patient, asking follow-up questions and generating medical advice based on the patient's responses.
3. **Web Application**: The application interface built using **Flask** that connects the machine learning model and the LLM backend with the user interface, providing a seamless experience.

## Prerequisites

Before running this project, ensure that your environment has the following:

- Python 3.7+
- Flask
- Pandas
- Scikit-learn
- Joblib
- Requests

## Installation

Follow these steps to get the project up and running on your local machine:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/ahmed-morad15/diabetes-app.git
   cd diabetes-app
