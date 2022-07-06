import numpy as np 


def gradient_desc(Y, X, lr = 0.00001, epoch = 69000):
    W, B = np.random.rand(), np.random.rand()
    N = Y.shape[0]

    for i in range(epoch):
        Y_hat = np.dot(X,W) + B
        C = (2/N) * np.sum(np.power((Y_hat - Y), 2))
        
        dC_dW = (2/N) * np.sum(X * (Y_hat - Y))
        dC_dB = (2/N) * np.sum(Y_hat - Y)
    
        #update weights
        W -= lr * dC_dW
        B -= lr * dC_dB
    
    return B, W

if __name__ == '__main__':
    X = np.arange(200)
    Y = 2 * X 
    print(gradient_desc(Y, X))