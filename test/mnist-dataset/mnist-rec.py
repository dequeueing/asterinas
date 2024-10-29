import numpy as np
import warnings

# 加载 MNIST 数据集
def load_mnist(path, kind='train'):
    """从本地文件加载 MNIST 数据集"""
    labels_path = f'{path}/{kind}-labels.idx1-ubyte'
    images_path = f'{path}/{kind}-images.idx3-ubyte'
    
    # 读取标签
    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)  # 跳过前8个字节的头信息
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    
    # 读取图像
    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)  # 跳过前16个字节的头信息
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 28*28)
    
    return images, labels

# 一热编码
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

# ReLU激活函数及其导数
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Softmax函数
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # 为数值稳定性做调整
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
    A1 = relu(Z1)  # 使用 ReLU 替换 Sigmoid
    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

# 反向传播
def backward_propagation_origin(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = X.shape[0]
    
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)  # 使用 ReLU 的导数
    dW1 = np.dot(X.T, dZ1) / m    # FIXME:  error happening here!!
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

# 反向传播
def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, W2):
    m = X.shape[0]
    
    dZ2 = A2 - Y
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)  # 使用 ReLU 的导数

    # 捕获 RuntimeWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", RuntimeWarning)
        
        dW1 = np.dot(X.T, dZ1) / m  # 可能会触发溢出警告

        # 检查并打印捕获的 RuntimeWarning
        for warning in w:
            if issubclass(warning.category, RuntimeWarning):
                print("RuntimeWarning 捕获：", warning.message)
                # 打印调试信息
                print("X shape:", X.shape, "dZ1 shape:", dZ1.shape, "m:", m)
                print("X max:", np.max(X), "X min:", np.min(X))
                print("dZ1 max:", np.max(dZ1), "dZ1 min:", np.min(dZ1))

                # print dW1, db1, dW2, db2
                print("dW1:", dW1)
                print("dW2:", dW2)
                print("db2:", db2)

                print("m:", m)

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

# 训练模型 (带 Mini-batch 梯度下降)
def train(X, Y, input_size, hidden_size, output_size, learning_rate, num_epochs, batch_size):
    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(num_epochs):
        mini_batches = create_mini_batches(X, Y, batch_size)
        
        for i, (mini_X, mini_Y) in enumerate(mini_batches):
            Z1, A1, Z2, A2 = forward_propagation(mini_X, W1, b1, W2, b2)
            dW1, db1, dW2, db2 = backward_propagation(mini_X, mini_Y, Z1, A1, Z2, A2, W1, W2)
            
            # 更新参数
            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2
            
            # 打印每个 mini-batch 的损失
            batch_loss = compute_loss(mini_Y, A2)
            print(f"Epoch {epoch}, Batch {i}, Loss: {batch_loss:.4f}")
            print(f"Batch predictions: {np.argmax(A2, axis=1)[:5]}")  # 打印前5个预测

        # 每个 epoch 结束时打印总损失
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        epoch_loss = compute_loss(Y, A2)
        print(f"End of Epoch {epoch}, Total Loss: {epoch_loss:.4f}")

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
    # 从本地文件加载数据
    train_images, train_labels = load_mnist('.', kind='train')
    test_images, test_labels = load_mnist('.', kind='t10k')
    
    # 数据预处理
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)
    
    # 定义超参数
    input_size = 784
    hidden_size = 128
    output_size = 10
    learning_rate = 0.1
    num_epochs = 100  # 降低 epoch 数量
    batch_size = 64   # 使用 mini-batch，64个样本一组
    
    # 训练模型
    W1, b1, W2, b2 = train(train_images, train_labels, input_size, hidden_size, output_size, learning_rate, num_epochs, batch_size)
    
    # 测试模型
    train_preds = predict(train_images, W1, b1, W2, b2)
    test_preds = predict(test_images, W1, b1, W2, b2)
    
    # 输出准确率
    print(f"Train accuracy: {accuracy(train_preds, train_labels)}")
    print(f"Test accuracy: {accuracy(test_preds, test_labels)}")
