import math as math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def fPC (y, yhat):
    yhat = np.argmax(yhat, axis=0)
    y = np.argmax(y, axis=0)
    n = float(y.size)
    diff = y - yhat
    diff[diff!=0] = 1 
    num_correct = n - np.sum(diff)
    return num_correct/n

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0) 

def relu(x):
    X= np.copy(x)
    X[X <= 0] = 0
    return X

def calculateCrossEntropy(y, yhat, w2, w1):
    size = y.shape[1]
    return -(1./size) * np.sum(y*np.log(yhat)) 

def forwardProp(X, w1new, w2new, b1new, b2new):
    z1 = w1new.dot(X) + b1new
    h1 = relu(z1)
    z2 = w2new.dot(h1) + b2new
    yhat = softmax(z2)
    return yhat, z1, h1

def backProp(X, yhat, y, w1, w2, z1, h1, batch_size):
    diff = (yhat-y).T
    diff = diff.dot(w2)
    z1t = z1.T
    z1t[z1t <= 0] = 0
    z1t[z1t > 0]  = 1
    gt = diff * z1t

    gb1 = np.mean(gt.T, axis=1)
    gb2 = np.mean(yhat - y, axis=1)
    gb1 = gb1.reshape((len(gb1), 1))
    gb2 = gb2.reshape((len(gb2), 1))
    gw1 = gt.T.dot(X.T)* (1./batch_size)
    gw2 = (yhat - y).dot(h1.T)*(1./batch_size)
    return gb1, gb2, gw1, gw2
    


def updateWeights(X, y, weights, weights2, biases, biases2, hidden_nodes, learning_rate, batch_size, epochs, bl=False):
    n = y.shape[1]
    num_batches = int(n/batch_size)

    w1old = np.copy(weights)
    w1new = np.copy(weights)
    w2old = np.copy(weights2)
    w2new = np.copy(weights2)
    b1old = np.copy(biases)
    b1new = np.copy(biases)
    b2new = np.copy(biases2)
    b2old = np.copy(biases2)
    pc = 0

    for e in range(epochs):
        curr_index = 0
        for batch in range(num_batches):
            batch_X = X[:,curr_index:curr_index+batch_size]
            batch_y = y[:,curr_index:curr_index+batch_size]

            batch_yhat, z1, h1 = forwardProp(batch_X, w1new, w2new, b1new, b2new)
            
            gb1, gb2, gw1, gw2 = backProp(batch_X, batch_yhat, batch_y, w1new, w2new, z1, h1, batch_size)
            
            w1old = np.copy(w1new)
            w2old = np.copy(w2new)
            b1old = np.copy(b1new)
            b2old = np.copy(b2new)
            
            w1new = w1old - (learning_rate * gw1)
            w2new = w2old - (learning_rate * gw2)
            b1new = b1old - (learning_rate * gb1)
            b2new = b2old - (learning_rate * gb2)
            curr_index += batch_size


        yhat, z1, h1 = forwardProp(X, w1new, w2new, b1new, b2new)
        pc = fPC(y, yhat)
        cross = calculateCrossEntropy(y,yhat, w1new, w2new)
        if e >= epochs - 20 and not bl:
            print("*** Epoch " + str(e+1) + " Statistics  ***")
            print("FPC = " + str(pc))
            print("Cross Entropy = " + str(cross))
            print()

    return w1new, w2new, b1new, b2new, pc


def trainNN(trainingFaces, trainingLabels, hidden_nodes=50, learning_rate=.2, batch_size=16, epochs=30, bl=False):
    X = trainingFaces
    y = trainingLabels

    sd1 = 1./math.sqrt(X.shape[0])
    sd2 = 1./math.sqrt(hidden_nodes)

    weights = np.random.randn(hidden_nodes, X.shape[0]) * sd1
    weights2 = np.random.randn(10, hidden_nodes) * sd2
    biases  = np.ones(hidden_nodes).reshape((hidden_nodes,1)) * .01
    biases2 = np.ones(10).reshape((10,1))  * .01

    return updateWeights(X, y, weights, weights2, biases, biases2, hidden_nodes, learning_rate, batch_size, epochs, bl)



def findBestHyperParameters(X, y, w1, w2, b1, b2, val_faces, val_labels):
    w1v = np.copy(w1)
    w2v = np.copy(w2)
    b2v = np.copy(b2)
    b1v = np.copy(b1)

    neurons =  [30, 40, 50]
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    minibatches = [16, 32, 64, 128, 256]
    epochs = [20, 30, 40, 50]

    best_fpc = 0.0
    best_neurons = 0
    best_learning = 0
    best_minibatches = 0
    best_epochs = 0

    for i in range(10):
        print()
        print("*** Validation {} ***".format(i))

        curr_neurons = neurons[int(np.random.rand()*3)]
        curr_learning = learning_rates[int(np.random.rand()*6)]
        curr_minibatches = minibatches[int(np.random.rand()*5)]
        curr_epochs = epochs[int(np.random.rand()*4)]

        print("Neurons = ", curr_neurons)
        print("Learning Rate = ", curr_learning)
        print("Batches = ", curr_minibatches)
        print("Epochs = ", curr_epochs)

        w1 = np.copy(w1v)
        w2 = np.copy(w2v)
        b2 = np.copy(b2v)
        b1 = np.copy(b1v)

        w1v, w2v, b1v, b2v, pc = trainNN(X, y, curr_neurons, curr_learning, curr_minibatches, curr_epochs, bl=True)
        yhat, z1, h1 = forwardProp(val_faces, w1, w2, b1, b2)
        pc = fPC(val_labels, yhat)
        cross = calculateCrossEntropy(val_labels, yhat, w1, w2)

        print("FPC = " + str(pc))
        print("Cross Entropy = " + str(cross))
        print()

        w1 = np.copy(w1v)
        w2 = np.copy(w2v)
        b2 = np.copy(b2v)
        b1 = np.copy(b1v)

        if pc > best_fpc:
            best_fpc = pc
            best_neurons = curr_neurons
            best_learning = curr_learning
            best_minibatches = curr_minibatches
            best_epochs = curr_epochs

            print()
            print("*** Validation Results Update ***")
            print("Improved FPC = ", best_fpc);
            print("Neurons = ", best_neurons)
            print("Learning Rate = ", best_learning)
            print("Batches = ", best_minibatches)
            print("Epochs = ", best_epochs)
            print()

    print()
    print("*** Final Hyperparameters from Validation ***")
    print("fPC = ", best_fpc);
    print("Neurons = ", best_neurons)
    print("Learning Rate = ", best_learning)
    print("Batches = ", best_minibatches)
    print("Epochs = ", best_epochs)
    print()
    print("Retraining with optimized hyperparameters...")
    print()
    w1, w2, b1, b2, pc = trainNN(X, y, best_neurons, best_learning, best_minibatches, best_epochs, bl=False)
    return w1, w2, b1, b2

def test(testingFaces, testingLabels, w1, w2, b1, b2):
    print("Beginning testing...")
    print()
    yhat_test, z1, h1 = forwardProp(testingFaces, w1, w2, b1, b2)
    pc_test = fPC(testingLabels, yhat_test)
    cross_test = calculateCrossEntropy(testingLabels, yhat_test, w1, w2)
    print()
    print ("*** Testing Set Results ***")
    print("FPC = " + str(pc_test))
    print("Cross Entropy = " + str(cross_test))

if __name__ == "__main__":
    training_faces = np.load("mnist_train_images.npy")
    training_labels = np.load("mnist_train_labels.npy")
    validation_faces = np.load("mnist_validation_images.npy")
    validation_labels = np.load("mnist_validation_labels.npy")
    testing_faces = np.load("mnist_test_images.npy")
    testing_labels = np.load("mnist_test_labels.npy")

    rng_state = np.random.get_state()
    np.random.shuffle(training_faces) #randomize both labels and images with same seed by resetting the state
    np.random.set_state(rng_state)
    np.random.shuffle(training_labels)

    np.random.set_state(rng_state)
    np.random.shuffle(validation_faces) #randomize both labels and images with same seed by resetting the state
    np.random.set_state(rng_state)
    np.random.shuffle(validation_labels) 

    w1, w2, b1, b2, pc = trainNN(training_faces.T, training_labels.T)
    w1, w2, b1, b2 = findBestHyperParameters(training_faces.T, training_labels.T, w1, w2, b1, b2, validation_faces.T, validation_labels.T)
    test(testing_faces.T, testing_labels.T, w1, w2, b1, b2)

