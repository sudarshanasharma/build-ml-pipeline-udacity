#!/usr/bin/env python
"""
This script trains a Linear Regression model on the Udacity Dataset.
"""
import argparse
import logging
import os
import shutil
import matplotlib.pyplot as plt

import mlflow
import json

import pandas as pd
import numpy as np
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

import wandb
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline, make_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

OBJECT_DATATYPE_COLUMNS = [
    "name",
    "host_name",
    "neighbourhood_group",
    "neighbourhood",
    "room_type",
    "last_review"
]


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(
        lambda d: (
            d.max() - d).dt.days,
        axis=0).to_numpy()


def plot_feature_importance(pipe, feat_names):
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["linear_regression"].feature_importances_[
        : len(feat_names) - 1]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(
        pipe["linear_regression"].feature_importances_[
            len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))

    sub_feat_imp.bar(
        range(
            feat_imp.shape[0]),
        feat_imp,
        color="r",
        align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp


def get_inference_pipeline(max_tfidf_features):
    # We will handle the categorical features first
    ordinal_categorical = ["room_type"]
    non_ordinal_categorical = ["neighbourhood_group"]
    # NOTE: The type of the room is mandatory on the websites. So we do not need to
    # impute room_type, as missing values are not possible in production
    # (nor during training). The same is not true for neighbourhood_group
    ordinal_categorical_preproc = OrdinalEncoder()

    # Build a pipeline with two steps:
    # 1 - A SimpleImputer(strategy="most_frequent") to impute missing values
    # 2 - A OneHotEncoder() step to encode the variable
    non_ordinal_categorical_preproc = make_pipeline(
        SimpleImputer(strategy="constant", fill_value="Manhattan"),
        OneHotEncoder()
    )

    # Impute the numerical columns to make sure we can handle missing values

    zero_imputed = [
        "number_of_reviews",
        "longitude",
        "latitude"
    ]
    zero_imputer = SimpleImputer(strategy="constant", fill_value=0)

    # As the skewness for minimum_nights is very high (γ1 = 25.17996962), we shall apply log
    # transform to the values , in an effort to distribute the values more uniformly and reduce
    # skewness
    skewness_fixer = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=0),
        FunctionTransformer(np.log1p, validate=True)
    )

    # we create a feature that represents the number of days passed since the last review
    # First we impute the missing review date with an old date (because there hasn't been
    # a review for a long time), and then we create a new feature from it,
    date_imputer = make_pipeline(
        SimpleImputer(
            strategy='constant',
            fill_value='2010-01-01'),
        FunctionTransformer(
            delta_date_feature,
            check_inverse=False,
            validate=False))

    # Some minimal NLP for the "name" column
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    name_tfidf = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=""),
        reshape_to_1d,
        TfidfVectorizer(
            binary=False,
            max_features=max_tfidf_features,
            stop_words='english'
        ),
    )

    # Connect everything together in the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("ordinal_cat", ordinal_categorical_preproc, ordinal_categorical),
            ("non_ordinal_cat",
             non_ordinal_categorical_preproc,
             non_ordinal_categorical),
            ("impute_zero", zero_imputer, zero_imputed),
            ("skewness_fixer", skewness_fixer, ["minimum_nights"]),
            ("transform_date", date_imputer, ["last_review"]),
            ("transform_name", name_tfidf, ["name"])
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )

    processed_features = ordinal_categorical + non_ordinal_categorical + \
        zero_imputed + ["minimum_nights", "last_review", "name"]

    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("linear_regression", LinearRegression())
        ]
    )

    return sk_pipe, processed_features


def go(args):
    run = wandb.init(job_type="train_linear_regression")
    run.config.update(args)

    # We get the train and validation artifact (args.trainval_artifact)
    # and save the returned path in train_local_path
    logger.info("Downloading training set artifact")
    trainval_local_path = run.use_artifact(args.trainval_artifact).file()

    X = pd.read_csv(trainval_local_path)
    # this removes the column "price" from X and puts it into y
    y = X.pop("price")

    logger.info(f"Minimum price: {y.min()}, Maximum price: {y.max()}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=X[args.stratify_by], random_state=args.random_seed
    )

    logger.info("Preparing sklearn pipeline")

    sk_pipe, processed_features = get_inference_pipeline(
        args.max_tfidf_features)

    # Then fit it to the X_train, y_train data
    logger.info("Fitting")

    # We fit the pipeline sk_pipe by calling the .fit method on X_train and
    # y_train
    sk_pipe.fit(X_train, y_train)

    # Here we compute r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_val, y_val)

    y_pred = sk_pipe.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    logger.info("Exporting model")

    # We save model package in the MLFlow sklearn format
    if os.path.exists("linear_regression_dir"):
        shutil.rmtree("linear_regression_dir")

    # We save the sk_pipe pipeline as a mlflow.sklearn model in the directory
    # "linear_regression_dir"

    export_path = "linear_regression_dir"
    for col in OBJECT_DATATYPE_COLUMNS:
        X_val[col] = X_val[col].astype('string')
    signature = infer_signature(X_val, y_pred)

    mlflow.sklearn.save_model(
        sk_pipe,
        export_path,
        signature=signature,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        input_example=X_val.iloc[:5]
    )

    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Trained pipeline artifact"
    )
    artifact.add_dir(export_path)
    run.log_artifact(artifact)

    # Here we save r_squared under the "r2" key
    run.summary['r2'] = r_squared
    # We log the variable "mae" under the key "mae".
    run.summary['mae'] = mae


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training of dataset")
    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify_by",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--max_tfidf_features",
        help="Maximum number of words to consider for the TFIDF",
        default=10,
        type=int
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)
