import os
os.add_dll_directory(r"C:\Users\HP\Vestacka Projekat\build")

import sys
sys.path.append(r"C:\Users\HP\Vestacka Projekat\build")

import cpp
from load_dataset import load_test_data
from model import create_network, evaluate


test_dir = 'dataset/Test'
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CATEGORIES = len(os.listdir('dataset/Train'))

def main():
    net = create_network(NUM_CATEGORIES)
    cpp.Serialization.loadNetwork(net, "best_model.bin")

    X_test, y_test = load_test_data()
    test_acc = evaluate(net, X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()