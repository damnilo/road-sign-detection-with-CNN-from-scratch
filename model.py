import os
import time

from load_dataset import load_test_data, NUM_CATEGORIES, IMG_WIDTH, IMG_HEIGHT
os.add_dll_directory(r"C:\Users\HP\Vestacka Projekat\build")

import sys
sys.path.append(r"C:\Users\HP\Vestacka Projekat\build")

import numpy as np
import cpp

def numpy_to_tensor(np_array):
    t = cpp.Tensor(list(np_array.shape))
    for idx in np.ndindex(np_array.shape):
        t[idx] = np_array[idx]
    return t

def tensor_to_numpy(tensor):
    shape = tensor.getShape()
    np_array = np.zeros(shape, dtype=np.float32)
    for idx in np.ndindex(*shape):
        np_array[idx] = tensor[idx]
    return np_array

def create_network():
    net = cpp.Network()

    net.addLayer(cpp.Conv2D(3, 16, 3, 1, 1))
    net.addLayer(cpp.ReLU())
    net.addLayer(cpp.MaxPool2D(2, 2))

    net.addLayer(cpp.Conv2D(16, 32, 3, 1, 1))
    net.addLayer(cpp.ReLU())
    net.addLayer(cpp.MaxPool2D(2, 2))

    net.addLayer(cpp.Flatten())
    net.addLayer(cpp.Dense(32 * 8 * 8, 128))
    net.addLayer(cpp.ReLU())
    net.addLayer(cpp.Dense(128, NUM_CATEGORIES))
    net.addLayer(cpp.SoftMax())
    loss = cpp.CrossEntropy()
    optimizer = cpp.SGD(0.01, 0.9)

    return net, loss, optimizer

def train():
    X_train, y_train, X_val, y_val = load_test_data()
    y_train_oh = np.eye(NUM_CATEGORIES)[y_train]
    y_val_oh = np.eye(NUM_CATEGORIES)[y_val]

    net, loss, optimizer = create_network()
    best_acc = 0.0
    
    EPOCHS = 10
    for epoch in range(EPOCHS):
        perm = np.random.permutation(len(X_train))
        total_loss = 0
        t0 = time.time()

        for n, i in enumerate(perm):
            x = numpy_to_tensor(X_train[i:i+1])
            y_true = numpy_to_tensor(y_train_oh[i:i+1])

            y_pred = net.forward(x)
            loss_value = loss.forward(y_pred, y_true)
            total_loss += loss_value

            grad_loss = loss.backward(y_pred, y_true)
            net.backward(grad_loss)
            optimizer.update(net.parameters(), net.gradients())

            if n % 500 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Step {n}/{len(X_train)}, Loss: {total_loss/(n+1):.4f}")

        avg_loss = total_loss / len(X_train)
        acc = evaluate(net, X_val, y_val)
        print(f"Epoch {epoch+1}/{EPOCHS} completed in {time.time() - t0:.2f}s, Average Loss: {avg_loss:.4f}, Validation Accuracy: {acc:.4f}, Time per epoch: {time.time() - t0:.2f}s")

        if acc > best_acc:
            best_acc = acc
            cpp.Serializer.saveNetwork(net, "cpp_model.bin")
            print("Saved trained model to cpp_model.bin")

def evaluate(net, X_val, y_val):
    correct = 0
    for i in range(len(X_val)):
        x = numpy_to_tensor(X_val[i:i+1])
        pred = tensor_to_numpy(net.forward(x))
        if np.argmax(pred) == y_val[i]:
            correct += 1
    accuracy = correct / len(X_val)
    return accuracy

if __name__ == "__main__":
    train()