import numpy as np 
import math

class LogisticRegression:
    def init(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_epochs = num_iterations
        self.weights = None
        self.bias = None


    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def train(self, train_x, train_y):
        n_samples, n_features = train_x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_epochs):
            # get model and pred
            model = np.dot(train_x, self.weights) + self.bias
            y_pred = self.sigmoid(model)

            # weght and loss grads
            dw = (1 / n_samples) * np.dot(train_x.T, (y_pred - train_y))
            db = (1 / n_samples) * np.sum(y_pred - train_y)

            #update w and b
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, test_x):
        # get pred then sigmoid and clip to 0 or 1
        pred = np.dot(test_x, self.weights) + self.bias
        return (self.sigmoid(pred) >= 0.5).astype(int)
    
    def loss(self, y_pred, y_true):
        # cross entropy loss
        n_samples = len(y_true)
        loss = -1 / n_samples * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss