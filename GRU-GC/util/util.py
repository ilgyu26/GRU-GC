import matplotlib.pyplot as plt

import os

def plot(matrix):
    plt.figure(figsize=(8, 3))
    ax = plt.subplot(142)
    ax.matshow(matrix)
    ax.axis('off')

    plt.savefig(os.path.join('./results.png'))