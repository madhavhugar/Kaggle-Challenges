"""Script to upload scikit-learn pickeled models"""
import os
import boto3


def get_s3_client():
    """
    Authenticates and returns AWS S3 client

    :return: AWS client object
    """
    return boto3.client('s3')


def upload_models(event, context):
    """
    Upload models to S3

    :return: None
    """
    s3 = get_s3_client()
    for each_model in ['tfidf-vectorizer.joblib', 'whatscooking.joblib', 'label-encoder.joblib']:
        s3.upload_file(Filename=f'{os.curdir}/{each_model}', Bucket='machine-learning-models-serverless', Key=each_model)
