import os
import json
import boto3
import joblib
import pandas as pd


def get_s3_client():
    """
    Authenticates and returns AWS S3 client

    :return: AWS client object
    """
    return boto3.client('s3')


def get_model(model_name):
    """
    Fetch and load sklearn model from S3 bucket

    :param model_name: model filename
    :return: ml model
    """
    s3 = get_s3_client()
    s3.download_file(Bucket='machine-learning-models-serverless', Key=model_name, Filename=f'/tmp/{model_name}')
    return joblib.load(f'/tmp/{model_name}')


def predict(event, context):
    """

    :return: None
    """
    body_parsed = json.loads(event['body'])
    for each in body_parsed['ingredients']:
        print(each)
    tfidf_vectorizer = get_model('tfidf-vectorizer.joblib')
    logisticr_model = get_model('whatscooking.joblib')
    label_encoder = get_model('label-encoder.joblib')
    ingredients = pd.Series(body_parsed['ingredients'])
    x_transformed = tfidf_vectorizer.transform(ingredients)
    vect_prediction = logisticr_model.predict(x_transformed)
    prediction = label_encoder.inverse_transform(vect_prediction)
    print(f'For the ingredients {ingredients} the predicted cuisine is {prediction[0]}')
    return create_http_response(f'{prediction[0]}\n')


def create_http_response(body):

    return {"statusCode": 200, "body": body}


def info(event, context):
    """
    Information about the /kaggle/whatsCooking API

    :param event: lambda event payload
    :param context: lambda context
    :return: string
    """
    description = 'Given a set of ingredients, predicts the cuisine'
    input_types = 'Pass the input in the form of a JSON object, containing an array of ingredients'
    request_type = 'Send the request via a POST request on the endpoint /kaggle/whatsCooking/predict'
    example_curl = 'Example:'
    curl_post = "curl -X POST https://y139r04032.execute-api.eu-central-1.amazonaws.com" \
                "/dev/kaggle/whatsCooking/predict " \
                "-d '{\"ingredients\": [\"garam masala, onion, salt\"]}' " \
                "-H  \"x-api-key: 1audCQKc3s321n03ZcrWj6doJtwYu0DE2LGDiViO\" " \
                "-H \"content-type:application/json\""

    return create_http_response(f'\n\n{description}\n{input_types}\n{request_type}\n{example_curl}\n\n{curl_post}\n\n')
