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
import mlflow
from urllib.parse import urlparse


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

    
################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        lr = DecisionTreeClassifier(
            criterion=criterion, 
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf, 
            random_state=random_state)
        lr.fit(x_train.values, y_train)

        predicted_qualities = lr.predict(test_x)
        
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr, 
                "model", 
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(lr, "model")



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)