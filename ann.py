import numpy as np

def sigmoid(x, derivative = False):
    if derivative == True: return x * (1 - x)
    else: return 1 / (1 + np.exp(-x))

class ANN():
    
    def __init__(self, shape):
        self.shape = shape     
        self.layers = []
        for layer in range (len(self.shape) - 1):
            self.layers.append(np.random.random((self.shape[layer], self.shape[layer + 1])))
    
    def ff_bp(self, X, y, lr): # Performs forward and backward propagation
        al = [X]
        
        for layer in range (len(self.shape) - 1): # Forward propagation
            X = sigmoid(np.dot(X, self.layers[layer]))
            al.append(X)
        
        self.output = al[-1]
        w_nabla = [] # Back-propagation
        delta = (y - al[-1]) * sigmoid(al[-1], True)
        w_nabla.insert(0, lr * (np.dot(al[-2].T, delta)))
        
        for layer in range (2, len(self.shape)):
            delta = np.dot(delta, self.layers[-layer + 1].T) * sigmoid(al[-layer], True)
            w_nabla.insert(0, lr * (np.dot(al[-layer - 1].T, delta)))
        
        self.layers = np.add(self.layers, w_nabla) # Updating weights with gradients stored in 'w_nabla'

''' Example:
ann = ANN([3, 10, 10, 1])
for epoch in range (50000):
    ann.ff_bp(X = np.array([[0,0,1], [1,1,0], [1,0,0]]), y = np.array([[1, 0, 0]]).T, lr = 0.1)
print(ann.output) '''
