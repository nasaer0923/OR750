# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:08:27 2018

@author: yudi-chen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import timeit

enc = OneHotEncoder(sparse=False)
np.random.seed(1) # set a seed so that the results are consistent

class MLP(object):
    """multi layer perceptron neural network
    
    Parameters:
        C -- Penalty used in L2 normalization, constant
        mini_batch_size -- Sample size for each mini batch, constant
        max_iteration -- max iteration
        drop_out -- drop out probability range(0, 1)
        beta1 -- parameter in ADAM
        beta2 -- parameter in ADAM
        activation -- activation function
            linear: default
            tanh:
            relu:
        lr -- learning rate
        epsilon - 1E-5
        init -- parameters initialization methods
            normal:
            zeros:
            Xavier:
            He:
        verbose -- logical (True or False)
            Enable verbose output of the training process
            
    Attributes:
        parameters -- weights and bias
        cost -- cross entropy cost
        time -- training time
        epoch -- training epoch number
    """
    
    def __init__(self,
                 n_h=10,
                 C=0, 
                 mini_batch_size=0,
                 max_iteration=int(1E4),
                 drop_out=0,
                 lr=0.1,
                 epsilon=1E-4,
                 beta1=0.9,
                 beta2=0.999,
                 activation="linear",
                 optimizer="GD",
                 init="normal",
                 verbose=False
                 ):
        
        self.n_h = n_h
        self.C = C
        self.drop_out = drop_out
        self.mini_batch_size = mini_batch_size
        self.max_iteration = max_iteration
        self.activaction = activation
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.optimizer = optimizer
        self.init = init
        self.verbose = verbose
        
        
    def softmax(self, X):
        """
        Arguments:
            X -- The matrix used for softmax calculation (n_classes, n_samples)
        """
        exps = np.exp(X)
        value = exps / np.sum(exps, axis=0, keepdims=True)
        return value
    
    
    def relu(self, X, derivative=False):
        """
        relu activation function
        """
        return_X = X
        if derivative:
            return_X[X<0] = 0
            return_X[X>=0] = 1
        else:
            return_X[X<0] = 0
            
        return return_X
    
    
    def initialize_parameters(self, n_x, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_y -- size of the output layer
        """
    
        if self.init == "normal":
            W1 = np.random.normal(size=(self.n_h, n_x))
            b1 = np.random.normal(size=(self.n_h, 1))
            W2 = np.random.normal(size=(n_y, self.n_h))
            b2 = np.random.normal(size=(n_y, 1))
        elif self.init == "zeros":
            W1 = np.zeros((self.n_h, n_x))
            b1 = np.zeros((self.n_h, 1))
            W2 = np.zeros((n_y, self.n_h))
            b2 = np.zeros((n_y, 1))
        elif self.init == "Xavier":
            # initialize the weights from a Gaussian distribution with zeros mean
            # and a variance of 1/N
            std = np.sqrt(1/n_x)
            W1 = np.random.normal(scale=std, size=(self.n_h, n_x))
            b1 = np.random.normal(size=(self.n_h, 1))
            W2 = np.random.normal(scale=std, size=(n_y, self.n_h))
            b2 = np.random.normal(size=(n_y, 1))
        elif self.init == "He":
            std = np.sqrt(2/n_x)
            W1 = np.random.normal(scale=std, size=(self.n_h, n_x))
            b1 = np.random.normal(size=(self.n_h, 1))
            W2 = np.random.normal(scale=std, size=(n_y, self.n_h))
            b2 = np.random.normal(size=(n_y, 1))
        else:
            raise Exception("Oops, wrong input of initialization method!!!")
            
        assert (W1.shape == (self.n_h, n_x))
        assert (b1.shape == (self.n_h, 1))
        assert (W2.shape == (n_y, self.n_h))
        assert (b2.shape == (n_y, 1))
    
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        
        self.parameters = parameters
        
        return self
    
    
    def forward_propagation(self, X):
        """
        Argument:
        X -- input data of size (n_features, n_samples)
    
        Returns:
        A2 -- The softmax output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = self.parameters["W1"]
        b1 = self.parameters["b1"]
        W2 = self.parameters["W2"]
        b2 = self.parameters["b2"]
    
        # Implement Forward Propagation to calculate A2 (probabilities)
        Z1 = np.dot(W1, X) + b1
        if self.activaction == "linear":
            A1 = Z1
        elif self.activaction == "tanh":
            A1 = np.tanh(Z1)
        elif self.activaction == "relu":
            A1 = self.relu(Z1)
        else:
            raise Exception("Oops, wrong input of activation method!!!")
        
        if self.drop_out != 0:
            mask = np.random.rand(A1.shape[0], A1.shape[1]) > self.drop_out
            self.mask = mask
            A1 = A1*mask
            A1 /= (1-self.drop_out)
            
        Z2 = np.dot(W2, A1) + b2 
        A2 = self.softmax(Z2)
    
        assert(A2.shape == (b2.shape[0], X.shape[1]))
    
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": A2}
    
        return A2, cache
    
    
    def compute_cost(self, X, Y):
        """
        Computes the cross-entropy cost
        Arguments:
        X -- The sigmoid output of the second activation, of shape (n_classes, n_samples)
        Y -- The class labels (n_classes, n_samples)
        """
        m = Y.shape[1]
        p = self.softmax(X)
        
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
        
        log_likelihood = -np.log(p[Y.argmax(axis=0), range(m)])
        cost = np.sum(log_likelihood) / m + self.C*(np.mean(np.mean(W1**2))+np.mean(np.mean(W2**2)))
    
        assert(isinstance(cost, float))
        self.cost = cost
        return self
    
    
    def backward_propagation(self, cache, X, Y):
        """
        Arguments:
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
    
        Returns:
        grads -- python dictionary containing your gradients
        """
        m = X.shape[1]
        W1 = self.parameters["W1"]
        W2 = self.parameters["W2"]
    
        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache["A1"]
        A2 = cache["A2"]
        
        if self.drop_out != 0:
            A1 = A1*self.mask
    
        # Backward propagation: calculate dW1, db1, dW2, db2.
        dZ2 = A2 - Y
        dW2 = 1.0/m*np.dot(dZ2, A1.T) + self.C*2*W2
        db2 = 1.0/m*np.sum(dZ2, axis=1, keepdims=True)
        
        if self.activaction == "linear":
            dZ1 = np.dot(W2.T, dZ2)
        elif self.activaction == "tanh":
            dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1, 2))
        elif self.activaction == "relu":
            dZ1 = np.dot(W2.T, dZ2)*self.relu(A1, derivative=True)
        else:
            raise Exception("Oops, wrong input of activation function!!!")
            
        dW1 = 1.0/m*np.dot(dZ1, X.T) +  +self.C*2*W1
        db1 = 1.0/m*np.sum(dZ1, axis=1, keepdims=True)
    
        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}
    
        return grads
    
    
    def ADAM(self, grads):
        """
        update parameters using ADAM
        """
        adam_params = []
        idx = 0
        for ii in grads:
            dX = grads[ii]
            if self.t == 1:
                mt_hist = np.zeros_like(dX)
                vt_hist = np.zeros_like(dX)
            else:
                mt_hist = self.adam_params[idx][ii+"_mt"]
                vt_hist = self.adam_params[idx][ii+"_vt"]
            mt = self.beta1*mt_hist+(1-self.beta1)*dX
            vt = self.beta2*vt_hist+(1-self.beta2)*(dX**2)
            mt_hat = mt/(1-self.beta1**self.t)
            vt_hat = vt/(1-self.beta2**self.t)
            
            self.parameters[ii[1:]] = self.parameters[ii[1:]] - self.lr*mt_hat/(np.sqrt(vt_hat)+self.epsilon)
            adam_params.append({ii+"_mt": mt, ii+"_vt": vt})
            
            idx += 1
        
        self.adam_params = adam_params
    
    
    def GD(self, grads):
        """
        update parameters using gradient descent
        """
        for ii in grads:
            dX = grads[ii]
            X = self.parameters[ii[1:]]
            self.parameters[ii[1:]] = X - self.lr*dX
    
    
    def update_parameters(self, grads):
        """
        """

        # Update rule for each parameter
        if self.optimizer == "GD":
            self.GD(grads)
        elif self.optimizer == "ADAM":
            self.t += 1
            self.ADAM(grads)
        else:
            raise Exception("Oops, wrong input of optimizer!!!")

        
    def fit(self, X, Y):
        """
        Arguments:
        X -- dataset of shape (n_features, n_samples)
        Y -- labels of shape (n_classes, n_samples)
        """
        n_x = X.shape[0]
        n_y = Y.shape[0]
        m = X.shape[1]
    
        # Initialize parameters, then retrieve W1, b1, W2, b2. 
        self.initialize_parameters(n_x, n_y)
    
        # Loop (gradient descent)
        self.epoch = 0
        error = 1
        self.t = 0
        start = timeit.default_timer()
        while self.epoch < self.max_iteration and error > self.epsilon:
            
            if self.mini_batch_size != 0:
                batch_num = np.floor(m/self.mini_batch_size)
                for ii in range(int(batch_num)):
                    idx_start = self.mini_batch_size*ii
                    idx_end = min(self.mini_batch_size*(ii+1), m-1)
                    X_batch = X[:, idx_start:idx_end]
                    Y_batch = Y[:, idx_start:idx_end]
                    
                    A2, cache = self.forward_propagation(X_batch)
                    grads = self.backward_propagation(cache, X_batch, Y_batch)
                    self.update_parameters(grads)
                    
                A2, cache = self.forward_propagation(X)
                self.compute_cost(A2, Y)
            else:
                A2, cache = self.forward_propagation(X)
                self.compute_cost(A2, Y)
                grads = self.backward_propagation(cache, X, Y)
                self.update_parameters(grads)
            
            if self.epoch < 10:
                error = 1
                cost_hist = self.cost
            else:
                error = np.abs((self.cost-cost_hist) / self.cost)
                cost_hist = self.cost
                
            # Print the cost every 1000 iterations
            if self.verbose and self.epoch % 10 == 0:
                print ("Cost and error after epoch %i: %f, %f" 
                       %(self.epoch, self.cost, error))
            
            self.epoch += 1
            
        stop = timeit.default_timer()
        self.time = stop - start
        return self
    

    def predict(self, X):
        """
        Arguments:
            X -- input data of size (n_features, n_samples)
        """
    
        # Computes probabilities using forward propagation
        A2, cache = self.forward_propagation(X)
        predictions = A2.argmax(axis=0)
    
        return predictions
    

def mnist_import():
    """
    Returns:
        train_X -- training features (n_samples, n_features)
        train_Y -- training labels (n_samples, 1)
        test_X -- testing features (n_samples, n_features)
        test_Y -- testing labels (n_samples, 1)
    """
    train, valid, test, = pd.read_pickle("mnist.pkl")
    train_X = np.concatenate((train[0], valid[0]))
    train_Y = np.concatenate((train[1], valid[1]))
    test_X = test[0]
    test_Y = test[1]
    train_Y = train_Y[:, np.newaxis]
    test_Y = test_Y[:, np.newaxis]
    
    return train_X, train_Y, test_X, test_Y


def plot_acc_time(accuracy_train, accuracy_test, time, params, x_label):
    fig, ax1 = plt.subplots()
    r = np.arange(len(params))
    
    color = 'tab:red'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('accuracy', color=color)
    ax1.plot(r, accuracy_train, color=color, marker="o")
    ax1.plot(r, accuracy_test, color=color, marker="s")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks([x for x in r])
    ax1.set_xticklabels(params)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:blue'
    ax2.set_ylabel('training time', color=color)  # we already handled the x-label with ax1
    ax2.plot(r, time, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    

def plot_bar(accuracy, params):
    plt.figure()
    bar_width = 0.25
    r1 = np.arange(np.size(accuracy, 1))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    plt.bar(r1, accuracy[0, :], color='#7f6d5f', width=bar_width, edgecolor='white', 
            label='Linear activation')
    plt.bar(r2, accuracy[1, :], color='#557f2d', width=bar_width, edgecolor='white', 
            label='tanh activation')
    plt.bar(r3, accuracy[2, :], color='#2d7f5e', width=bar_width, edgecolor='white', 
            label='relu activation')
    plt.xlabel('Hidden neuron')
    plt.xticks([r + bar_width for r in range(np.size(accuracy, 1))], [x for x in params])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=3)
    plt.show()


def question_1(train_X, train_Y, test_X, test_Y):
    """
    Numerican experiments with different non-linear activation functions, different
    number of neurons and different learning rates
    """
    # Question 1: Organic DL
    # different non-linear activation functions
    activations = ["linear", "tanh", "relu"]
    hidden_neurons = [10, 50, 100, 1000]
    accuracy_train = np.zeros((len(activations), len(hidden_neurons)))
    accuracy_test = np.zeros((len(activations), len(hidden_neurons)))
    idx_1 = 0
    train_Y_tmp = np.argmax(train_Y, axis=0)
    for activation in activations:
        idx_2 = 0
        for hidden_neuron in hidden_neurons:
            params = {"n_h": hidden_neuron,
                      "activation": activation,
                      "mini_batch_size": 512,
                      "lr": 0.1,
                      "optimizer": "GD",
                      "init": "He",
                      "verbose": True
                      }
            model = MLP(**params)
            mlp_model = model.fit(train_X, train_Y)
            
            predictions = MLP.predict(mlp_model, train_X)
            accuracy_train[idx_1, idx_2] = accuracy_score(train_Y_tmp, predictions)
            predictions = MLP.predict(mlp_model, test_X)
            accuracy_test[idx_1, idx_2] = accuracy_score(test_Y, predictions)
            
            idx_2 += 1
        idx_1 += 1
        
    # visualize the results
    plot_bar(accuracy_train, hidden_neurons)
    plot_bar(accuracy_test, hidden_neurons)
    
    # different learning rates
    learning_rates = [0.01, 0.1, 1]
    accuracy_train = np.zeros_like(learning_rates)
    accuracy_test = np.zeros_like(learning_rates)
    time_lr = np.zeros_like(learning_rates)
    idx = 0
    train_Y_tmp = np.argmax(train_Y, axis=0)
    for learning_rate in learning_rates:
        params = {"n_h": 20,
                  "mini_batch_size": 512,
                  "activation": "tanh",
                  "lr": learning_rate,
                  "optimizer": "GD",
                  "init": "He",
                  "verbose": True
                  }
        model = MLP(**params)
        mlp_model = model.fit(train_X, train_Y)
        time_lr[idx] = mlp_model.time
        
        predictions = MLP.predict(mlp_model, train_X)
        accuracy_train[idx] = accuracy_score(train_Y_tmp, predictions)
        predictions = MLP.predict(mlp_model, test_X)
        accuracy_test[idx] = accuracy_score(test_Y, predictions)
        
        idx += 1
        
    plot_acc_time(accuracy_train, accuracy_test, time_lr, learning_rates, "learning rate")
    

def question_3(train_X, train_Y, test_X, test_Y):
    """
    L2 regularization, compare the accuracy and training speed
    """
    Cs = [0, 0.001, 0.005, 0.01]
    accuracy_train = np.zeros_like(Cs)
    accuracy_test = np.zeros_like(Cs)
    time = np.zeros_like(Cs)
    idx = 0
    train_Y_tmp = np.argmax(train_Y, axis=0)
    for C in Cs:
        params = {"n_h": 20,
                  "C": C,
                  "mini_batch_size": 512,
                  "activation": "tanh",
                  "lr": 0.1,
                  "optimizer": "GD",
                  "init": "He",
                  "verbose": True
                  }
        model = MLP(**params)
        mlp_model = model.fit(train_X, train_Y)
        time[idx] = mlp_model.time
        predictions = MLP.predict(mlp_model, test_X)
        accuracy_test[idx] = accuracy_score(test_Y, predictions)
        predictions = MLP.predict(mlp_model, train_X)
        accuracy_train[idx] = accuracy_score(train_Y_tmp, predictions)
        
        idx += 1
    
    plot_acc_time(accuracy_train, accuracy_test, time, Cs, "penalty coefficients")
    

def question_4(train_X, train_Y, test_X, test_Y):
    """
    """
    initializations = ["normal", "zeros", "Xavier", "He"]
    accuracy_train = np.zeros((len(initializations), 1))
    accuracy_test = np.zeros((len(initializations), 1))
    time = np.zeros((len(initializations), 1))
    idx = 0
    train_Y_tmp = np.argmax(train_Y, axis=0)
    for initialization in initializations:
        params = {"n_h": 20,
                  "mini_batch_size": 512,
                  "activation": "tanh",
                  "lr": 0.1,
                  "optimizer": "GD",
                  "init": initialization,
                  "verbose": True
                  }
        model = MLP(**params)
        mlp_model = model.fit(train_X, train_Y)
        time[idx] = mlp_model.time
        predictions = MLP.predict(mlp_model, train_X)
        accuracy_train[idx] = accuracy_score(train_Y_tmp, predictions)
        predictions = MLP.predict(mlp_model, test_X)
        accuracy_test[idx] = accuracy_score(test_Y, predictions)
        
        idx += 1
    
    plot_acc_time(accuracy_train, accuracy_test, time, initializations, "activation function")
    

def question_5(train_X, train_Y, test_X, test_Y):
    """
    """
    optims = ["GD", "ADAM"]
    results = {}
    epoch = np.zeros((len(optims), 1))
    time = np.zeros((len(optims), 1))
    accuracy_train = np.zeros((len(optims), 1))
    accuracy_test = np.zeros((len(optims), 1))
    idx = 0
    train_Y_tmp = np.argmax(train_Y, axis=0)
    for optim in optims:
        params = {"n_h": 20,
                  "mini_batch_size": 512,
                  "activation": "tanh",
                  "lr": 0.1,
                  "optimizer": optim,
                  "init": "He",
                  "verbose": True
                  }
        model = MLP(**params)
        mlp_model = model.fit(train_X, train_Y)
        predictions = MLP.predict(mlp_model, test_X)
        epoch[idx] = mlp_model.epoch
        time[idx] = mlp_model.time
        accuracy_test[idx] = accuracy_score(test_Y, predictions)
        
        predictions = MLP.predict(mlp_model, train_X)
        train_Y_tmp = np.argmax(train_Y, axis=0)
        accuracy_train[idx] = accuracy_score(train_Y_tmp, predictions)
        
        results["epoch"] = epoch
        results["time"] = time
        results["accuracy_train"] = accuracy_train
        results["accuracy_test"] = accuracy_test
        
        idx += 1
        
    return results
    

def question_6(train_X, train_Y, test_X, test_Y):
    """
    """
    ps = [0, 0.1]
    results = {}
    epoch = np.zeros((len(ps), 1))
    time = np.zeros((len(ps), 1))
    accuracy_train = np.zeros((len(ps), 1))
    accuracy_test = np.zeros((len(ps), 1))
    idx = 0
    train_Y_tmp = np.argmax(train_Y, axis=0)
    for p in ps:
        params = {"n_h": 20,
                  "mini_batch_size": 512,
                  "activation": "tanh",
                  "lr": 0.1,
                  "drop_out": p,
                  "optimizer": "GD",
                  "init": "He",
                  "verbose": True
                  }
        model = MLP(**params)
        mlp_model = model.fit(train_X, train_Y)
        predictions = MLP.predict(mlp_model, test_X)
        epoch[idx] = mlp_model.epoch
        time[idx] = mlp_model.time
        accuracy_test[idx] = accuracy_score(test_Y, predictions)
        
        predictions = MLP.predict(mlp_model, train_X)
        accuracy_train[idx] = accuracy_score(train_Y_tmp, predictions)
        
        results["epoch"] = epoch
        results["time"] = time
        results["accuracy_train"] = accuracy_train
        results["accuracy_test"] = accuracy_test
        
        idx += 1
        
    return results
    


if __name__ == "__main__":
    # generate data
    train_X, train_Y, test_X, test_Y = mnist_import()
    train_X = train_X.T
    test_X = test_X.T
    
    train_X = train_X[:, 0:2000]
    train_Y = train_Y[0:2000, :]
    
#    test_X = test_X[:, 0:100]
#    test_Y = test_Y[0:100, :]
    
    # transform to one hot code
    train_Y = enc.fit_transform(train_Y)
    train_Y = train_Y.T
    
    # Question 1
    question_1(train_X, train_Y, test_X, test_Y)
    
#     Question 3
    question_3(train_X, train_Y, test_X, test_Y)
    
    # Question 4
    question_4(train_X, train_Y, test_X, test_Y)
    
    # Question 5
    quest5_results = question_5(train_X, train_Y, test_X, test_Y)
    
    # Question 6
    quest6_results = question_6(train_X, train_Y, test_X, test_Y)
    
    