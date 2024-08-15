# Rynet Deep learning With rust 
## a silly attempt to Clone `PyTorch` famous framework for Machine Learning and Deep Learning.

the goal was never building the whole framework. my attemption was learing how to get "Rusty" and nowing hoe to work with `Pyo3` rust library.

### parts
- 'layer.rs' : a class template for building custom layer as data scientice desire.
- 'liear.rs' : a sub class of `Layer` for building linear layer. work like `torch.Linear`
- 'neural.rs' : a class template for building custom model as data scientice desire, as like Pytorch `nn.module`.
- 'loss.rs' : all loss functions. never finished.
- 'optimizers.rs' : all optimizers. never finished.

how to use it?
for `Neural` like this:
```python
class SimpleNN(Neuaral):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = Linear(3, 12)  
        self.fc2 = Linear(12, 4)
        self.fc3 = Linear(4, 1) 
        pass

    def forward(self, x : np.ndarray):
        # x = x.reshape(1,-1)
        x = relu(self.fc1(x))
        x = relu(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.fc3(x)
        return x
```

as for custom layer `Layer`
```python

class custom_layer(Layer):
    def __init__(self, in_features, out_features, is_bias=True , *args, **kwargs ):
        self.in_features = in_features
        self.out_features = out_features
        # super(custom_layer, self).__init__()
        pass
    def some_custom_method(self):
        print( self.in_features * self.out_features )
        pass
```

and then:
```python
cls = SimpleNN()

param = cls.parameters()
y_hat = cls.forward(x)

criterion = MSELoss()
loss = criterion(y_hat , y)
print(loss)
```