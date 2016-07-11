# theano-deep-neural-net
Deep neural network implementation using theano and lasagne

Neural network implementation mostly based on the [mnist lasagne tutorial](https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py)    

## Use
```python
from dnn.dnn import MLP

X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

# images are 28*28 pixels. Shown as a vector of 28*28 length
# train algo
nn = MLP(input_dim_count=28*28, output_size=10)
nn.fit(X_train, y_train, X_val, y_val, X_test, y_test, epochs=50)

# predict on first 5 of test
x_preds = X_test[0:5]
ans = nn.predict(x_preds)

# print prediction results
print(ans)
print(y_test[0:5])

```
