## Kaggle Challenges

My solutions for kaggle challenges. 

##### What's cooking - [Machine Learning](whatscooking/whats-cooking.ipynb) - [Engineering](whatscooking/)

Information about the API:

`curl https://y139r04032.execute-api.eu-central-1.amazonaws.com/dev/kaggle/whatsCooking/info`

Prediction from the deployed model. Feel free to replace the ingredients. To execute the API execute:

```
curl -X POST https://y139r04032.execute-api.eu-central-1.amazonaws.com/dev/kaggle/whatsCooking/predict -d '{"ingredients": ["garam masala, onion, salt"]}' -H  "x-api-key: 1audCQKc3s321n03ZcrWj6doJtwYu0DE2LGDiViO" -H "content-type:application/json"

##### Titanic survivor challenge - [Machine Learning](titanic-survival/src/analysis/titanic-survival-notebook.ipynb) - [Engineering](titanic-survival/)

