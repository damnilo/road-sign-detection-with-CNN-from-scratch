import os

from main import load_test_set
from model import build_model


test_dir = 'dataset/Test'
IMG_HEIGHT = 32
IMG_WIDTH = 32
NUM_CATEGORIES = len(os.listdir('dataset/Train'))

def main():
    net = build_model(NUM_CATEGORIES)
    net.load_model('best_model.npz')

    X_test, y_test = load_test_set(test_dir, IMG_HEIGHT, IMG_WIDTH)
    test_loss, test_acc = net.evaluate(X_test, y_test, batch_size=32)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()