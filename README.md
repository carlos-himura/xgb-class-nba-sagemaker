# NBA XGBoost Classifier – SageMaker Project
Binary classification problem related to NBA player performance. A Model capable of predicting whether a player will perform Above Average (1) or Below Average (0) in a given game based on historical game statistics.

This repository contains an end-to-end machine learning pipeline built on Amazon SageMaker, using XGBoost to classify NBA data.
The workflow includes:

Local exploration and testing inside SageMaker Studio

A custom training script (train.py) executed on a SageMaker Training Job

A preprocessing module reused by training and inference

Deployment to a real-time SageMaker endpoint

Validation of predictions from the endpoint

| File                              | Description                                                                                                                                                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **.gitignore**                    | Specifies which files and directories Git should ignore (cache files, logs, SageMaker artifacts).                                                                                                     |
| **README.md**                     | Main documentation for understanding the purpose, structure, and workflow of the project.                                                                                                             |
| **XGBclass.ipynb**                | Jupyter Notebook used inside SageMaker Studio for exploration, Hyperopt tuning experiments, manual testing, and endpoint validation.                                                                  |
| **train.py**                      | Main SageMaker training script. Loads and preprocesses data, applies class balancing using `scale_pos_weight`, integrates the best Hyperopt parameters, trains the model, and exports `model.tar.gz`. |
| **preprocessing.py**              | Custom preprocessing utilities for feature engineering and data cleaning. Imported by the training script and consistent between training and inference.                                              |
| **requirements.txt**              | List of Python dependencies used for training and inference environments.                                                                                                                             |
| **predictions_binary_only.csv**   | Local model predictions using binary classification outputs (0/1). Used to compare training behavior.                                                                                                 |
| **predictions_from_endpoint.csv** | Predictions generated from the deployed SageMaker endpoint to validate inference pipeline consistency.                                                                                                |

# Hyperparameter Optimization (Hyperopt)

Hyperopt was used to automatically search for the optimal XGBoost parameters.
The notebook runs:

- Random + Bayesian optimization

- Search space for key parameters like:
  -max_depth
  -eta
  -min_child_weight
  -subsample
  -colsample_bytree
  -gamma

The best set of parameters found by Hyperopt is passed into train.py for SageMaker training.

# Class Balancing Strategy

The dataset contains imbalance between classes.
To address this, the model uses XGBoost’s scale_pos_weight, computed as:

scale_pos_weight = negative_samples / positive_samples

This helps the model penalize the minority class appropriately and improves prediction accuracy for underrepresented outcomes.

# Model Performance on Test Set

After deploying the XGBoost classifier through the SageMaker pipeline and validating the final model, the following performance metrics were obtained:

Evaluation Metrics
- Accuracy: 0.8394
- Precision: 0.7660
- Recall: 0.8360
- F1 Score: 0.7994
- ROC AUC: 0.9300

Confusion Matrix
[[935 176]
 [113 576]]


These results indicate a strong balance between precision and recall, with particularly high ROC AUC, showing that the model discriminates well between “Above Average” and “Below Average” player performance. The high recall for the positive class suggests that the class balancing strategy and Hyperopt-optimized parameters contributed effectively to improving minority class detection.
