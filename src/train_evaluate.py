# load the train and test
# train algo
# save the metrices, params
import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from get_data import read_params
import argparse
import joblib
import json


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]

    criterion = config["estimators"]["DecisionTreeClassifier"]["params"]["criterion"]
    max_depth = config["estimators"]["DecisionTreeClassifier"]["params"]["max_depth"]
    min_samples_split = config["estimators"]["DecisionTreeClassifier"]["params"]["min_samples_split"]
    min_samples_leaf = config["estimators"]["DecisionTreeClassifier"]["params"]["min_samples_leaf"]

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    y_train = train[target]
    test_y = test[target]

    x_train = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    tr = DecisionTreeClassifier(
        criterion=criterion, 
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf, 
        random_state=random_state)
    tr.fit(x_train.values, y_train)

    predicted_qualities = tr.predict(test_x)
    
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

#####################################################
    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            'criterion':criterion, 
            'max_depth':max_depth,
            'min_samples_split':min_samples_split,
            'min_samples_leaf':min_samples_leaf
            }
        json.dump(params, f, indent=4)
#####################################################


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    print(model_path)
    joblib.dump(tr, model_path)




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)