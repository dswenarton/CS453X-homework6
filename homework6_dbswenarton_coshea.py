import numpy as np
import matplotlib
import matplotlib.pyplot as plt



if __name__ == "__main__":
    training_faces = np.load("mnist_train_images.npy")
	training_labels = np.load("mnist_train_labels.npy")
    testing_faces = np.load("mnist_test_images.npy")
    testing_labels = np.load("mnist_test_labels.npy")