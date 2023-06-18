
# Model Persistance
## Machine Learning

Saving the Model

```python
import pickle

# Store in Binary
# File can have any extension, pkl is used
pickle.dump(model, open(<file_name.pkl>, 'wb')) 
```

Loading the Model 

```python
import pickle

loaded_model = pickle.load(open(<file_name.plk>, "rb"))
```

## Neural Networks

Saving the model

```python
pd.DataFrame(<model_history>.history).to_csv("<history_file>")
```