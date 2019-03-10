# Multi-Layer-Neural-Network
This is an implementation from scratch of a deep feedforward Neural Network using Python. 
Note: This is a work in progress and things will be added gradually. It is not intended for production, just for learning purposes.

### Weights initialization ###
The Multi-Layer neural network can support several initialization methods such as:
  - Random initialization.
  - Xavier initialization.
  - He-et-Al. initialization.

Demo notebook is to be found [here](https://github.com/qarchli/Multi-Layer-Neural-Network/blob/master/Initialization%20DEMO.ipynb).

### Regularization ###
The Multi-Layer neural network can be regularized to reduce overfitting, using methods such as:
  - L2 Regularization, by specifying the regularization parameter λ (lambd in the constructor).
  - Dropout, by specifying the probability of keeping a neuron activated (keep_prob in the constructor). *

Demo notebook is to be found [here](https://github.com/qarchli/Multi-Layer-Neural-Network/blob/master/Regularization%20DEMO.ipynb).

### Optimization methods ###
In the learning phase, the MLNN optimal parameters can be learnt by using:
  - Batch, Mini-Batch or Stochastic gradient descent, by specifying the mini-batch size parameter in the constructor.<br>
  Combined with advanced parameter update methods such as:
  - Momentum, with hyperparameter β (beta1 in the constructor)
  - RMSProp, with hyperparameter β (beta1 in the constructor)
  - Adam, with hyperparameters β1 and β2 (beta1 and beta2 in the constructor).

Demo notebook is to be found [here](https://github.com/qarchli/Multi-Layer-Neural-Network/blob/master/Optimization%20methods%20DEMO.ipynb).
 
 ### Gradient checking: backpropagation debugging tool ###
It is a method that is added to check whether the gradients computed in the backpropagation are correct or not. <br>
It consists of comparing the gradients returned by backprop with their numerical approximations, with an approximation error of ε=1e-07.

Demo notebook is to be found [here](https://github.com/qarchli/Multi-Layer-Neural-Network/blob/master/Gradient%20Checking%20DEMO.ipynb).

