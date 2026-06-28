"""
Loads a trained CNN model and evaluates if on the GTSRB test dataset.
"""

import os

from load_dataset import load_train_data
os.add_dll_directory(r"C:\Users\HP\Vestacka Projekat\build")
os.add_dll_directory(r"C:\msys64\ucrt64\bin")
import sys
sys.path.append(r"C:\Users\HP\Vestacka Projekat\build")

import cpp
import numpy as np
import pandas as pd
import cv2
from model import create_network, evaluate

test_dir = 'dataset/Test'
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CATEGORIES = len(os.listdir('dataset/Train'))

def load_test_data():
    """
    Loads and preprocesses the test images and labels
    
    Returns:
        tuple: Test images and corresponding labels.
    """

    csv_path = os.path.join('dataset/', 'Test.csv')
    df = pd.read_csv(csv_path)
    image_data = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join('dataset/', row['Path'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        image_data.append(img)
        labels.append(row['ClassId'])

    image_data = np.array(image_data, dtype=np.float32)
    image_data = image_data.transpose(0, 3, 1, 2) / 255.0
    labels = np.array(labels, dtype=np.int32)

    return image_data, labels

def main():
    """
    Loads the trained model, evaluates it on the test set,
    and prints the test accuracy.
    """

    net, _, _, dropout = create_network()
    cpp.Serialization.loadNetwork(net, "best_model.bin")

    X_test, y_test = load_test_data()
    dropout.setTraining(False)
    mean = np.mean(X_test, axis=(0, 2, 3), keepdims=True)
    std = np.std(X_test, axis=(0, 2, 3), keepdims=True)
    X_test = (X_test - mean) / std
    test_acc = evaluate(net, X_test, y_test)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

if __name__ == "__main__":
    main()