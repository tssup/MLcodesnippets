import numpy as np


def gradient_descent(Y, X, lr = 0.010, epochs = 10000):
    R, N = X.shape
    W = np.random.rand(N, 1)
    B = np.random.rand()


    for epoch in range(epochs):
        Y_hat = X @ W + B
        C = (2/N) * np.sum(np.power((Y_hat - Y), 2))
        print(f'error: {C} \n')
        
        for i in range(W.shape[0]):
            #grab each column of X for backprop of weights and reshape for matmul with the error
            col_X = X[:,i].reshape(R, 1)
            #change in error w.r.t. weights
            dC_dW = (2/N) * np.sum(col_X * (Y_hat - Y))
            
            #update weights
            W[i,0] -= lr * dC_dW
       
        #change in error w.r.t bias
        dC_dB = (2/N) * np.sum(Y_hat - Y)
        #update bias
        B -= lr * dC_dB


    return (W, B)
                
if __name__ == '__main__':
    x = np.random.rand(10,1)
    y = 17 * x - 12
    print(gradient_descent(y, x))