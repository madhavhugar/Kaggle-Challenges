## Kaggle Challenges

My solutions for kaggle challenges. 

### What's cooking 

Given a set of ingredients we predict the cuisine.

- [Machine Learning](whatscooking/whats-cooking.ipynb) 
- [Engineering](whatscooking/)

#### Invoke the API

__GET__ information about the service at `dev/kaggle/whatsCooking/info`:

```
curl https://y139r04032.execute-api.eu-central-1.amazonaws.com/dev/kaggle/whatsCooking/info
```

__POST__ features to get prediction from the deployed model at `dev/kaggle/whatsCooking/predict`. Feel free to replace the ingredients. To execute the API, run:

```
curl -X POST https://y139r04032.execute-api.eu-central-1.amazonaws.com/dev/kaggle/whatsCooking/predict -d '{"ingredients": ["garam masala, onion, salt"]}' -H  "x-api-key: 1audCQKc3s321n03ZcrWj6doJtwYu0DE2LGDiViO" -H "content-type:application/json"
```

### Titanic survivor challenge 

Given the details of a passenger in titanic, we predict if the passenger survies or not.

- [Machine Learning](titanic-survival/src/analysis/titanic-survival-notebook.ipynb) 
- [Engineering](titanic-survival/)
