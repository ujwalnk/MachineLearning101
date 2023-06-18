# Random Forest Classifier
# Model Training

```python
from sklearn.ensemble import RandomForestClassifier as clf

rf_model = clf()
rf_model.fit(X_train, y_train)
```

# Predicting

```python
rf_model.predict(<your_data>)
```

Testing the accuracy of the model on the `validation data`

```python
rf_model.score(X_valid, y_valid)
rf_model.score(X_train, y_train)
```

Two kinds of scoring can be done here, one based on the trained data, and the other on the validation data.

- The accuracy on the `train data` must not be too high as that would be similar to mugging up a textbook before the exam, without understanding the essence.
- The accuracy on the `train data` must be be too low as that would be similar to going to the exam unprepared.

!!! warning
    The accuracy on the `validation data` thus will also never be too high or too low. Could lie somewhere from $60\%$ to $90\%$