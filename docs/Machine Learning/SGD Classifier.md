# SGD Classifier
# Model Training

Train the model on the training data

```python
from sklearn.linear_model import SGDClassifier as clf

sgd_model = clf()
sgd_model.fit(X_train, y_train)
```

# Predicting output

The data must be an array of array as that is what the model is trained on. 

```python
sgd_model.predict([[<data_array>]])
```

We need to check the accuracy of the predicted data, which will be taken from the `validation data`

```python
sgd_model.score(X_valid, y_valid)
```

Depending on the type of problem, the accuracy can vary to certain degrees, going as low as $40\%$ or as high as $90\%$

The learning is based off of the weights of each data column. In layman terms, the input data of each column in multiplied by the weights of the column to get the prediction. For checking the weights of each input column

```python
sgd_model.coef_
```