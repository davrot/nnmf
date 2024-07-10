import numpy as np
import matplotlib.pyplot as plt

data = np.load("data_log_cnn_iter20_lr_1.0000e-03_1.0000e-03_1.0000e-03_-_1.0000e-03_0False_1False_2False_3False_klFalse_maxFalse.npy")
plt.loglog(
    data[:, 0],
    100.0 * (1.0 - data[:, 1] / 10000.0),
    "k",
    label="CNN",
)


data = np.load("data_log_nnmf_iter20_lr_1.0000e-03_1.0000e-03_-_1.0000e-02_1.0000e-03_0False_1False_2False_3False_klFalse_maxFalse.npy")
plt.loglog(data[:, 0], 100.0 * (1.0 - data[:, 1] / 10000.0), "b", label="NNMF")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Error [%]")
plt.title("CIFAR10")
plt.show()
