# NBA XGBoost Classifier – SageMaker Project
Binary classification problem related to NBA player performance. Model capable of predicting whether a player will perform Above Average (1) or Below Average (0) in a given game based on historical game statistics.

This repository contains an end-to-end machine learning pipeline built on Amazon SageMaker, using XGBoost to classify NBA data.
The workflow includes:

Local exploration and testing inside SageMaker Studio

A custom training script (train.py) executed on a SageMaker Training Job

A preprocessing module reused by training and inference

Deployment to a real-time SageMaker endpoint

Validation of predictions from the endpoint

| File                              | Description                                                                                                                                        |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **.gitignore**                    | Lists files and folders that Git should ignore (e.g., checkpoints, temporary files, Sagemaker output directories).                                 |
| **README.md**                     | Main project documentation. Explains model overview, training steps, and repo structure.                                                           |
| **XGBclass.ipynb**                | Jupyter Notebook used in SageMaker Studio for exploration, preprocessing steps, training/debugging iterations, and endpoint testing.               |
| **train.py**                      | Main script used by SageMaker Training Jobs. Loads data from S3, applies preprocessing, trains the XGBoost classifier, and exports `model.tar.gz`. |
| **preprocessing.py**              | Contains all preprocessing functions (feature engineering, cleaning, scaling, encoding). Imported by `train.py`.                                   |
| **requirements.txt**              | Python dependencies for the training and inference environment. Ensures reproducibility inside SageMaker.                                          |
| **predictions_binary_only.csv**   | Local test of the model using binary-class predictions (0/1) generated in the notebook.                                                            |
| **predictions_from_endpoint.csv** | Predictions received from the deployed SageMaker endpoint using the same input dataset (used to validate inference vs training consistency).       |

