import math as math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def fpc(y, yhat):
    print(y.shape)
    print(yhat.shape)

    yhat = np.argmax(yhat, axis=1)
    y = np.argmax(y, axis=1)
    size = len(y)
    num_correct = size - (np.nonzero(y - yhat)[0].shape[0])
    return float(num_correct) / size

def softmax(z):
    # e_x = np.exp(x - np.max(x))
    # return e_x / np.sum(e_x, axis=0) #col-wise
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

def relu(x):
    x[x <= 0] = 0
    return x

def calculateCrossEntropy(y, yhat):
    size = y.shape[1]
    return -(0.1/size) * np.sum(y*np.log(yhat.T))

def forwardProp(X, w1new, w2new, b1new, b2new):
    z1 = w1new.dot(X) + b1new
    z1out = np.copy(z1)
    h1 = relu(z1)
    z2 = w2new.dot(h1) + b2new
    yhat = softmax(z2)
    return yhat, z1out, h1

def backProp(X, yhat, y, w2, z1, h1):
    diff = np.mean((yhat-y), axis=0)
    diff = diff.dot(w2)
    z1t = z1.T
    z1t[z1t <= 0] = 0
    z1t[z1t > 0] = 1

    gt = diff * z1t
    gb1 = gt.T
    gw1 = gb1.dot(X.T)
    gb2 = yhat - y.T 
    gw2 = gb2.dot(h1.T)
    return gb1, gb2, gw1, gw2


def trainNN(trainingFaces, trainingLabels):
    hidden_nodes = 50
    sd1 = 1/math.sqrt(trainingFaces.shape[0])
    sd2 = 1/math.sqrt(hidden_nodes)

    state = np.random.get_state()
    np.random.shuffle(trainingFaces)
    np.random.set_state(state)
    np.random.shuffle(trainingLabels)

    # arrangement = np.arange(trainingFaces.shape[0])
    # np.random.shuffle(arrangement)
    # trainingFaces = trainingFaces[arrangement]
    # trainingLabels = trainingLabels[arrangement]

    X = trainingFaces
    y = trainingLabels

    learning_rate = 0.1
    batch_size = 15
    n = trainingLabels.shape[0]
    num_batches = int(n/batch_size)

    weights = np.random.rand(hidden_nodes, X.shape[0]) * sd1
    weights2 = np.random.rand(10,hidden_nodes) * sd2
    biases  = np.ones(hidden_nodes).reshape((hidden_nodes,1)) * .01
    biases2 = np.ones(10).reshape((10,1))  * .01

    w1old = np.copy(weights)
    w1new = np.copy(weights)
    w2old = np.copy(weights2)
    w2new = np.copy(weights2)
    b1old = np.copy(biases)
    b1new = np.copy(biases)
    b2new = np.copy(biases2)
    b2old = np.copy(biases2)


    for e in range(101):
        curr_index = 0
        for batch in range(num_batches):
            batch_X = X[:,curr_index:curr_index+batch_size]

            batch_yhat, z1, h1 = forwardProp(batch_X, w1new, w2new, b1new, b2new)
            batch_y = y[curr_index:curr_index+batch_size,:]

            gb1, gb2, gw1, gw2 = backProp(batch_X, batch_yhat, batch_y, w2new, z1, h1)
            
            w1old = np.copy(w1new)
            w2old = np.copy(w2new)
            b1old = np.copy(b1new)
            b2old = np.copy(b2new)
            
            w1new = w1old - (learning_rate * gw1)
            w2new = w2old - (learning_rate * gw2)
            b1new = b1old - (learning_rate * gb1)
            b2new = b2old - (learning_rate * gb2)
            curr_index += batch_size

        print(X.shape)
        yhat, z1, h1 = forwardProp(X, w1new, w2new, b1new, b2new)
        pc = fpc(y, yhat)
        cross = 1#calculateCrossEntropy(y,yhat)
        if e >= 0:
            print("***  Epoch " + str(e) + " Statistics  ***")
            print("FPC = " + str(pc))
            print("Cross Entropy = " + str(cross))
            print()

    Z_test = wnew.T.dot(testingFaces.T)
    yhat_test = calculateYhat(Z_test)
    pc_test = fpc(testingLabels.T, yhat_test)
    cross_test = calculateCrossEntropy(testingLabels.T,yhat_test)

    print ("*** Testing Set Results ***")
    print("FPC = " + str(pc_test))
    print("Cross Entropy = " + str(cross_test))


if __name__ == "__main__":
    training_faces = np.load("mnist/mnist_train_images.npy")
    training_labels = np.load("mnist/mnist_train_labels.npy")
    validation_faces = np.load("mnist/mnist_validation_images.npy")
    validation_labels = np.load("mnist/mnist_validation_labels.npy")
    testing_faces = np.load("mnist/mnist_test_images.npy")
    testing_labels = np.load("mnist/mnist_test_labels.npy")

    weights = trainNN(training_faces.T, training_labels)

