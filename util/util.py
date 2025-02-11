import matplotlib.pyplot as plt

def plot(matrix):
    plt.figure(figsize=(8, 3)) 
    ax = plt.subplot(1, 1, 1)
    ax.matshow(matrix)
    ax.axis('off')

    plt.savefig('results.png')
    plt.close()