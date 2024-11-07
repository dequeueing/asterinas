import numpy as np
import warnings
import os

# 禁用多线程
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# 加载 MNIST 数据集
def load_mnist(path, kind='train'):
    labels_path = f'{path}/{kind}-labels.idx1-ubyte'
    images_path = f'{path}/{kind}-images.idx3-ubyte'
    
    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 28*28)
    
    return images, labels

# 一热编码
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

# ReLU 激活函数及其导数
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Softmax 函数
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / exp_z.sum(axis=1, keepdims=True)

# 初始化参数
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

# 前向传播
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# 反向传播
def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = X.shape[0]
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

# 计算损失函数
def compute_loss(Y, A2):
    m = Y.shape[0]
    log_probs = -np.sum(Y * np.log(A2), axis=1)
    loss = np.sum(log_probs) / m
    return loss

# Mini-batch 创建
def create_mini_batches(X, Y, batch_size):
    m = X.shape[0]
    indices = np.random.permutation(m)
    X_shuffled = X[indices]
    Y_shuffled = Y[indices]
    mini_batches = []
    for i in range(0, m, batch_size):
        X_batch = X_shuffled[i:i + batch_size]
        Y_batch = Y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches

# 数据划分：训练集和验证集
def split_data(X, Y, validation_ratio=0.2):
    m = X.shape[0]
    val_size = int(m * validation_ratio)
    return X[val_size:], Y[val_size:], X[:val_size], Y[:val_size]

# 训练模型 (带 Mini-batch 梯度下降)
def train(X, Y, input_size, hidden_size, output_size, learning_rate, num_epochs, batch_size, target_accuracy=0.7):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    # 划分验证集
    X_train, Y_train, X_val, Y_val = split_data(X, Y, validation_ratio=0.2)

    # calculate the accuracy for randomly generated parameters
    train_accuracy = accuracy(predict(X_train, W1, b1, W2, b2), Y_train)
    val_accuracy = accuracy(predict(X_val, W1, b1, W2, b2), Y_val)
    print(f"For initial parameters, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    
    for epoch in range(num_epochs):
        mini_batches = create_mini_batches(X_train, Y_train, batch_size)
        
        epoch_loss = 0
        for i, (mini_X, mini_Y) in enumerate(mini_batches):
            Z1, A1, Z2, A2 = forward_propagation(mini_X, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward_propagation(mini_X, mini_Y, Z1, A1, Z2, A2, W1, W2)
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            
            # 累积损失
            batch_loss = compute_loss(mini_Y, A2)
            epoch_loss += batch_loss
            
            # 打印每个 mini-batch 的损失
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {batch_loss:.4f}")

        # 计算每个 epoch 的平均损失
        epoch_loss /= len(mini_batches)
        print(f"Epoch {epoch}, Average Loss: {epoch_loss:.4f}")
        
        # 计算训练和验证集的准确率
        train_accuracy = accuracy(predict(X_train, W1, b1, W2, b2), Y_train)
        val_accuracy = accuracy(predict(X_val, W1, b1, W2, b2), Y_val)
        print(f"End of Epoch {epoch}, Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # 如果达到目标准确率则提前终止
        if train_accuracy >= target_accuracy:
            print("Target accuracy reached, stopping training.")
            break

    return W1, b1, W2, b2

# 预测函数
def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(X, W1, b1, W2, b2)
    predictions = np.argmax(A2, axis=1)
    return predictions

# 计算准确率
def accuracy(predictions, labels):
    return np.mean(predictions == np.argmax(labels, axis=1))

# 主程序
if __name__ == "__main__":
    train_images, train_labels = load_mnist('.', kind='train')
    test_images, test_labels = load_mnist('.', kind='t10k')
    
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)
    
    input_size = 784
    hidden_size = 128
    output_size = 10
    learning_rate = 0.1 
    num_epochs = 10       # 增加 epoch 数量
    batch_size = 64
    target_accuracy = 0.7  # 设置目标准确率为 70%
    
    W1, b1, W2, b2 = train(train_images, train_labels, input_size, hidden_size, output_size, learning_rate, num_epochs, batch_size, target_accuracy)
    
    train_preds = predict(train_images, W1, b1, W2, b2)
    test_preds = predict(test_images, W1, b1, W2, b2)
    print("Learning rate: ", learning_rate)
    
    print(f"Final Train accuracy: {accuracy(train_preds, train_labels):.4f}")
    print(f"Final Test accuracy: {accuracy(test_preds, test_labels):.4f}")
