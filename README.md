# NBA XGBoost Classifier – SageMaker Project
Binary classification problem related to NBA player performance. Model capable of predicting whether a player will perform Above Average (1) or Below Average (0) in a given game based on historical game statistics.

This repository contains an end-to-end machine learning pipeline built on Amazon SageMaker, using XGBoost to classify NBA data.
The workflow includes:

Local exploration and testing inside SageMaker Studio

A custom training script (train.py) executed on a SageMaker Training Job

A preprocessing module reused by training and inference

Deployment to a real-time SageMaker endpoint

Validation of predictions from the endpoint
