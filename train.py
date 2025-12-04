import argparse # Lets the Python script accepts command-line arguments — like --epochs 10
import os # Allow the interaction with the operating system: creating folders, reading env variables
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from preprocessing import preprocess_data  # Using the preprocessing script for feature engineering

# --- METRICS FUNCTION ---
def evaluate_classifier(model, X, y, dataset_name="Dataset"):
    dmatrix = xgb.DMatrix(X) # Convert X into a DMatrix (XGBoost format), and use that DMatrix to run predictions efficiently. DMatrix is because we are not using scikit-learn
    proba = model.predict(dmatrix)
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    roc = roc_auc_score(y, proba)
    cm = confusion_matrix(y, preds)

    print(f"\n📊 {dataset_name} Results")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {roc:.4f}")
    print("  Confusion Matrix:")
    print(cm)
    print("-" * 40)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "confusion_matrix": cm.tolist()
    }

# --- MAIN TRAINING FUNCTION ---
def main(args):
    print("--- Loading Data ---")
    # Directories (on the container) that SageMaker maps from S3 channels
    train_path = os.path.join(args.train, "train.csv")
    valid_path = os.path.join(args.validation, "validation.csv")
    test_path = os.path.join(args.test, "test.csv")

    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    test_df = pd.read_csv(test_path)

    # Apply preprocessing logic
    print("\n--- Preprocessing ---")
    train_df = preprocess_data(train_df, is_training=True)
    valid_df = preprocess_data(valid_df, is_training=True)
    test_df  = preprocess_data(test_df, is_training=True)

    X_train, y_train = train_df.drop("target", axis=1), train_df["target"]
    X_valid, y_valid = valid_df.drop("target", axis=1), valid_df["target"]
    X_test,  y_test  = test_df.drop("target", axis=1), test_df["target"]

    # Converts the training data (X_train, y_train) into a DMatrix — the format XGBoost uses internally during training
    dtrain = xgb.DMatrix(X_train, label=y_train) 
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest  = xgb.DMatrix(X_test, label=y_test)

    # --- Fixed Parameters ---
    # NOTE: The ratio for scale_pos_weight is calculated from your class counts (3212 / 2188)
    # This parameter is fixed here as it directly addresses the known data imbalance.
    CLASS_WEIGHT_RATIO = 3212 / 2188  # ~1.468
    params = {
        'objective': 'binary:logistic', # Defines what the model is trying to optimize during training
        'colsample_bytree': 0.9507579290048368, # Fraction of features (columns) sampled per tree. Adds randomness and prevents overfitting
        'gamma': 9.887094626323933, # Minimum loss reduction required to make a further partition
        'learning_rate': 0.1371656701428185, # Step size shrinkage used in updates for Gradient Descent
        'min_child_weight': int(10), # Minimum sum of instance weight needed in a child node. Controls overfitting.
        'max_depth': int(3.0), # Maximum depth of each tree. Controls model complexity
        'n_estimators': int(1150), # Number of boosting rounds (trees). How many trees to build in total
        'subsample': 0.8702299683381548, # Fraction of training data used per tree. Helps prevent overfitting
        'eval_metric': 'logloss', # Defines how performance is measured during evaluation (validation/testing)
        "tree_method": "hist",
        'seed': 12345,
        'scale_pos_weight': CLASS_WEIGHT_RATIO # BALANCE THE CLASSES
    }

    print("--- Training XGBoost Classification Model ---")
    evals = [(dtrain, 'train'), (dvalid, 'validation')] # Allows XGBoost to track performance and detect overfitting

    # Native XGBoost training function. train.() is a Native XGBoost API
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=params["n_estimators"],
        evals=evals
    )

    # --- Evaluate ---
    evaluate_classifier(model, X_train, y_train, "Train")
    evaluate_classifier(model, X_valid, y_valid, "Validation")
    evaluate_classifier(model, X_test, y_test, "Test")

    # --- Save Model (Required Name for XGB Container) ---
    os.makedirs(args.model_dir, exist_ok=True) # Creates the directory where SageMaker expects the trained model to be saved
    output_path = os.path.join(args.model_dir, "xgboost-model") # Builds the full file path for the model file
    model.save_model(output_path) # Writes the trained XGBoost model to disk in binary format
    print(f"✅ Model saved to {output_path}")

# --- Entry Point ---
if __name__ == "__main__": # Python convention that tells only run the code of this block if this file is being executed
    parser = argparse.ArgumentParser() # Creates a command-line argument parser using the argparse module
    # Environment variables automatically created by SageMaker inside the container
    # They tell the script (train.py) where to find the data and where to save the model
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    args = parser.parse_args() # Reads all the arguments and stores them in the args object
    main(args) # Calls the main training function, passing the parsed arguments