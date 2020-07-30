import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)
# y = x * w
# w = w - alpha * d(loss) / d(w)
# d(loss) / d(w) = d/dw * (x*w - y) ^2
#                = 2(xw - y) * x
#                = 2x(xw - y)
def gradient(x, y):
    return 2 * x * (x * w - y)


def train():
    print("predict before training", forward(4))
    global w
    for epoch in range(100):
        for x_val, y_val in zip(x_data, y_data):
            grad = gradient(x_val, y_val)
            w = w - 0.01 * grad
            print("\tgrad: ", x_val, y_val, grad)
            l = loss(x_val, y_val)
        print("progress: ", epoch, "w= ", w, "loss= ", l)
    print("predict after training ", forward(4))
train()
#w_list = []
#mse_list = []
#for w in np.arange(0.0, 4.1, 0.1):
#    print("w= ", w)
#    l_sum = 0
#    for x_val, y_val in zip(x_data, y_data):
#        y_pred = forward(x_val)
#        l = loss(x_val, y_val)
#        l_sum += l
#        print("\t", x_val, y_val, y_pred, l)
#
#    print("MSE= ", l_sum / len(x_data))
#    w_list.append(w)
#    mse_list.append(l_sum / len(x_data))
#
#plt.plot(w_list, mse_list)
#plt.ylabel("Loss")
#plt.xlabel("w")
#plt.show()
