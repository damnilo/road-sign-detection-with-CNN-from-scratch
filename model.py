import os
import time

from load_dataset import load_test_data, NUM_CATEGORIES, IMG_WIDTH, IMG_HEIGHT
os.add_dll_directory(r"C:\Users\HP\Vestacka Projekat\build")

import sys
sys.path.append(r"C:\Users\HP\Vestacka Projekat\build")

import numpy as np
import cpp

def numpy_to_tensor(np_array):
    return cpp.Tensor(np_array.astype(np.float32))

def tensor_to_numpy(tensor):
    return tensor.to_numpy().astype(np.float32)

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

def clip_gradients(grads, clip_value=1.0):
    for g in grads:
        data = g.to_numpy()
        norm = np.linalg.norm(data)

        if norm > clip_value:
            scale = clip_value / norm

            for i in np.ndindex(data.shape):
                g[i] *= scale

def train():
    X_train, y_train, X_val, y_val = load_test_data()
    y_train_one_hot = np.eye(NUM_CATEGORIES)[y_train]
    y_val_one_hot = np.eye(NUM_CATEGORIES)[y_val]

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    
    mean = np.mean(X_train, axis=(0, 2, 3), keepdims=True)
    std = np.std(X_train, axis=(0, 2, 3), keepdims=True)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    net, loss, optimizer = create_network()

    val_acc = evaluate(net, X_val, y_val)
    params = net.parameters()
    grads = net.gradients()

    best_acc = 0.0

    EPOCHS = 30
    BATCH_SIZE = 64

    learning_rate_decay = 0.95

    for epoch in range(EPOCHS):
        start_time = time.time()
        permutation = np.random.permutation(len(X_train))
        epoch_loss = 0.0
        batches = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            if i % 20 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Batch {i//BATCH_SIZE + 1}/{len(X_train)//BATCH_SIZE}")

            batch_indices = permutation[i:i+BATCH_SIZE]

            X_batch = X_train[batch_indices]
            y_batch = y_train_one_hot[batch_indices]

            X_tensor = numpy_to_tensor(X_batch)
            y_tensor = numpy_to_tensor(y_batch)

            predictions = net.forward(X_tensor)
            loss_value = loss.forward(predictions, y_tensor)
            epoch_loss += loss_value
            batches += 1

            grad_loss = loss.backward()
            net.backward(grad_loss)
            clip_gradients(grads, clip_value=1.0)
            optimizer.update(params, grads)
        print("aaa")            

        avg_loss = epoch_loss / batches
        print("bbb")
        val_acc = evaluate(net, X_val, y_val)
        print("ccc")
        print(f"Epoch {epoch+1}/{EPOCHS}, Validation Accuracy: {val_acc * 100:.2f}%, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            cpp.Serialization.saveNetwork(net, "best_model.bin")

        if best_acc >= 0.96:
            print("Early stopping: Validation accuracy reached 96%")
            break

        optimizer.setLearningRate(optimizer.getLearningRate() * learning_rate_decay)

def evaluate(net, X_val, y_val):
    correct = 0
    batch_size = 64

    for i in range(0, len(X_val), batch_size):
        X_batch = X_val[i:i+batch_size]
        y_batch = y_val[i:i+batch_size]

        X_tensor = numpy_to_tensor(X_batch)
        predictions = tensor_to_numpy(net.forward(X_tensor))        

        labels_batch = np.argmax(predictions, axis=1)
        batch_correct = np.sum(labels_batch == y_batch)

        correct += batch_correct

    return correct / len(X_val)

if __name__ == "__main__":
    train()