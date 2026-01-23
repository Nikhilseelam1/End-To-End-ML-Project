# End-to-End Machine Learning Pipeline Project

##  Overview
This repository contains a **complete end-to-end Machine Learning pipeline** built using Python, following **industry-standard ML engineering and MLOps practices**.  
The project demonstrates how to move beyond notebooks and build a **modular, scalable, and production-ready ML system**.

Instead of focusing only on model training, this project emphasizes:
- clean architecture
- configuration-driven pipelines
- separation of concerns
- reproducibility and maintainability

The pipeline is designed to reflect how **real-world ML systems** are built and deployed in production environments.


##  Machine Learning Pipeline Stages

###  Data Ingestion
- Downloads raw data from a remote source
- Stores raw data as pipeline artifacts
- Extracts compressed datasets

###  Data Validation
- Validates dataset against a predefined schema
- Ensures required columns are present
- Prevents invalid or corrupted data from entering the pipeline

###  Data Transformation
- Performs preprocessing and feature engineering
- Splits data into training and testing sets
- Prepares data for model training

###  Model Training
- Trains a machine learning model using transformed data
- Stores trained model as a versioned artifact

###  Model Evaluation
- Evaluates model performance using metrics
- Compares results against defined thresholds
- Determines whether the model is production-ready

---

##  Configuration Management
All pipeline behavior is controlled using YAML configuration files:

- `config.yaml` → pipeline paths and settings
- `params.yaml` → model hyperparameters
- `schema.yaml` → expected data schema

##  How to Run the Project

### Step 1: Clone the repository
git clone https://github.com/<your-username>/End-To-End-ML-Project.git
cd End-To-End-ML-Project

### Step 2: Create and activate environment
conda create -n mlproj python=3.8 -y
conda activate mlproj

### Step 3: Install dependencies
pip install -r requirements.txt

### Step 4: Run the ML pipeline
python main.py

### Tech Stack

Python

Pandas, NumPy

Scikit-learn

YAML

Joblib

Logging

Flask

Git & GitHub



