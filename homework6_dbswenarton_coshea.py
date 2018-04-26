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

def trainNN(trainingFaces, trainingLabels):
    print(trainingFaces.shape[1]+1)
    sd = 1/math.sqrt(trainingFaces.shape[1]+1)

    state = np.random.get_state()
    np.random.shuffle(trainingFaces)
    np.random.set_state(state)
    np.random.shuffle(trainingLabels)

    arrangement = np.arange(trainingFaces.shape[0])
    np.random.shuffle(arrangement)
    trainingFaces = trainingFaces[arrangement]
    trainingLabels = trainingLabels[arrangement]

    trainingFaces = np.vstack((trainingFaces.T, np.ones(trainingFaces.shape[0])*0.01))
    print(trainingFaces.shape);

    X = trainingFaces.T
    y = trainingLabels.T

    learning_rate = 0.1
    nt = 100
    n = trainingLabels.shape[0]
    num_batches = int(n/nt)

    weights = np.random.rand(X.shape[1], 10) * sd
    weights[-1] = 0.01

    w1old = np.copy(weights)
    w1new = np.copy(weights)
    w2old = np.copy(weights)
    w2new = np.copy(weights)

    z1 = w1old.T.dot(X.T)
    #h1 = 
    #z2 = w2old.T.dot(h1.T)


    for e in range(101):
        curr_index = 0
        for batch in range(num_batches):
            Z = wnew.T.dot(X[:,curr_index:curr_index+nt])
            batch_yhat = calculateYhat(Z)
            batch_y = y[:,curr_index:curr_index+nt]
            gradient = calculateGradient((X[:, curr_index:curr_index+nt]), batch_y, batch_yhat.T, nt)

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

    weights = trainNN(training_faces, training_labels)

