# theano-deep-neural-net
Deep neural network implementation using theano and lasagne

Neural network implementation mostly based on the (mnist lasagne tutorial)[https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py]    

## Use
```python
from dnn.dnn import MLP

# load data
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# init nn
nn = MLP()

# fit nn
nn.fit(X_train, y_train, X_val, y_val, X_test, y_test)

# predict
predictions = nn.predict(X_test[:5])
print('Predictions: %s' % predictions)
print('Actual: %s' % y_test[:5])

```
