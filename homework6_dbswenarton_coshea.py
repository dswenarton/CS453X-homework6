import math as math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def fpc(y, yhat):
    yhat = np.argmax(yhat, axis=1)
    y = np.argmax(y.T, axis=1)
    size = len(y)
    num_correct = size - (np.nonzero(y - yhat)[0].shape[0])
    return float(num_correct) / size

def calculateCrossEntropy(y, yhat):
    size = y.shape[1]
    return -(0.1/size) * np.sum(y*np.log(yhat.T))

def calculateYhat(Z):
    sums = np.sum(np.exp(Z), axis=1)
    return (np.exp(Z).T/sums[None,:])[::-1]

def forwardProp(X, w1new, w2new):
    print("X shape is " + str(X.shape))
    print("w1new shape is " + str(w1new.shape))
    z1 = w1new.T.dot(X)
    h1 = np.copy(z1)
    h1[h1 <= 0] = 0
    #h1[h1 > 0] = 1   # this line makes it relu prime if not commented
    z2 = w2new.dot(h1)

    return calculateYhat(z2), z1

def calcGradient(yhat, y, w2, z1):
    diff = yhat.T - y
    diff = diff.dot(w2)
    z1t = z1.T
    z1t[z1t <= 0] = 0
    z1t[z1t > 0] = 1
    return diff * z1t


def trainNN(trainingFaces, trainingLabels):
    sd = 1/math.sqrt(trainingFaces.shape[1]+1)

    print(trainingFaces.shape)
    print(trainingLabels.shape)

    state = np.random.get_state()
    np.random.shuffle(trainingFaces)
    np.random.set_state(state)
    np.random.shuffle(trainingLabels)

    arrangement = np.arange(trainingFaces.shape[0])
    np.random.shuffle(arrangement)
    trainingFaces = trainingFaces[arrangement]
    trainingLabels = trainingLabels[arrangement]

    trainingFaces = np.vstack((trainingFaces, np.ones(trainingFaces.shape[1])*0.01))
    print(trainingFaces.shape);

    X = trainingFaces
    y = trainingLabels

    learning_rate = 0.1
    nt = 100
    n = trainingLabels.shape[0]
    num_batches = int(n/nt)

    weights = np.random.rand(X.shape[0], 10) * sd
    weights[-1] = 0.01

    w1old = np.copy(weights)
    w1new = np.copy(weights)
    w2old = np.copy(weights)
    w2new = np.copy(weights)


    for e in range(101):
        curr_index = 0
        for batch in range(num_batches):
            batch_X = X[:,curr_index:curr_index+nt]
            batch_yhat, z1 = forwardProp(batch_X, w1new, w2new)
            print("batch yhat size is " + str(batch_yhat.shape))
            batch_y = y[curr_index:curr_index+nt,:]

            gradient = calcGradient(batch_yhat, batch_y, w2new, z1)
            jacobian = gradient.T.dot(batch_X.T)
            print(jacobian.shape)

            wold = np.copy(wnew)
            wnew = wold - (learning_rate * gradient)
            curr_index += nt
        Z = wnew.T.dot(X)
        yhat = calculateYhat(Z)
        pc = fpc(y, yhat)
        cross = calculateCrossEntropy(y,yhat)
        if e >= 80:
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
    #validation_faces = np.load("mnist/mnist_validation_images.npy")
    #validation_labels = np.load("mnist/mnist_validation_labels.npy")
    testing_faces = np.load("mnist/mnist_test_images.npy")
    testing_labels = np.load("mnist/mnist_test_labels.npy")

    weights = trainNN(training_faces.T, training_labels)

