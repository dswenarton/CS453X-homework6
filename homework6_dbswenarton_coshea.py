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

def trainNN(training_faces, training_labels):
	state = np.random.get_state()
    np.random.shuffle(trainingFaces)
    np.random.set_state(state)
    np.random.shuffle(trainingLabels)

    arrangement = np.arange(trainingFaces.shape[0])
    np.random.shuffle(arrangement)
    trainingFaces = trainingFaces[arrangement]
    trainingLabels = trainingLabels[arrangement]

    weights = np.random.rand(784, 10) * sd
    wold = np.copy(weights)
    wnew = np.copy(weights)


if __name__ == "__main__":
    training_faces = np.load("mnist_train_images.npy")
	training_labels = np.load("mnist_train_labels.npy")
	validation_faces = np.load("mnist_validation_images.npy")
	validation_labels = np.load("mnist_validation_labels.npy")
    testing_faces = np.load("mnist_test_images.npy")
    testing_labels = np.load("mnist_test_labels.npy")

    weights = trainNN(training_faces, training_labels)

