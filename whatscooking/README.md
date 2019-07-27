serverless create -t aws-python3

Information about the POST API:

https://y139r04032.execute-api.eu-central-1.amazonaws.com/dev/kaggle/whatsCooking/info

*Whats cooking* is a machine learning challenge posted on Kaggle. Given a set of ingredients we need to train a model to predict the cuisine. The machine learning aspect of the challenge is in the jupyter notebook. Below we demonstrate the engineering aspect of deploying the model.

#### Overview

Thq quickest way to deploy a model in my opinion was to host it as Function as a Service(FaaS). Being more familiar with AWS services, I decided to use their services. The solution can be applied to any other vendor which are supported by the [Serverless Framework](..).
The idea was to pickel the scikit learn models and store them on S3 bucket. In this challenge, we were required to store three models on S3. The idea was to download the models, load them, and use them to prepare the features and predict the output.  
The below image describes the architecture of the deployment. We send a POST request to the API gateway. The API gateway in turn invokes a lambda function. The lambda function downloads the models, applies the necessary transformations, predicts the output, and returns the prediction.

Some challenges faced in this were:

- Lambda functions have a size restriction of 62MB. So the python package was to be kept minimal. In our case, importing libraries like scikit-learn, numpy and pandas caused the package to become larger than 62MB. This issue was over come by adding `slim: true` under `pythonRequirements`:

```
service: whats-cooking

plugins:
  - serverless-python-requirements

custom:
  bucket: machine-learning-models-serverless
  pythonRequirements:
    slim: true
```

- Sending a POST request to API Gateway without sending the `AWS client key` and `AWS secret access key` authorization was not possible. We generated an API key which specifically allows us to send POST requests.
[create POST api key](https://github.com/dwyl/learn-aws-lambda/issues/28)


- Tf-idf vectorizer model - which basically assigns weights to the set of ingredients. We requir this to prepare out feature set before it is passed to the machine learning model.
- Logistic regression model - which accepts the vectorized input and returns a prediction. 