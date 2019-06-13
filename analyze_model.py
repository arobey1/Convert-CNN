import torch
import torch.optim as optim
from scipy.linalg import circulant
import numpy as np

from CNN import Net
from train_cnn import create_loaders

BATCH_SIZE = 4

def main():

    model = torch.load('cnn-model.pt')
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # print_model_params(model, optimizer)

    _, test_loader, classes = create_loaders()
    dataiter = iter(test_loader)
    images, labels = dataiter.next()

    weight = model.state_dict()['conv1.weight'].numpy()
    bias = model.state_dict()['conv1.bias'].numpy()

    im = images.numpy().flatten()
    n, k = 32, 5

    # filter = np.arange(9).reshape(3, 3) + 1
    # V = filter_to_bc_matrix(filter, 4)
    # print(V)

    num_rows_per_block = (n - k + 1) ** 2
    W = np.zeros((6 * num_rows_per_block, 3 * n ** 2))
    for i in range(6):
        for j in range(3):
            filter = weight[i, j, :, :]
            V_ij = filter_to_bc_matrix(filter, 32)
            W[i * num_rows_per_block : (i + 1) * num_rows_per_block, j * n ** 2 : (j + 1) * n ** 2] = V_ij

    model.eval()
    result = model(images)
    # print(result.shape)
    # print(result)

    print('\n')
    print('*'*50)
    print('\n')
    print(W.shape)
    print(im.shape)
    print((W @ im).reshape(1, 6, 28, 28))

    print(bias)

def filter_to_bc_matrix(filter_mat, n):
    """Convert """

    k = filter_mat.shape[1]

    num_block_rows = n - k + 1
    rows_per_block_row = n - k + 1
    num_img_pixels = n ** 2

    circ_mats = []
    for row_idx in range(k):
        row = filter_mat[row_idx, :]

        circ_vector = np.concatenate((
            np.array([row[0]]),
            np.zeros(n - k),
            np.flip(row[1:])
        ), axis=0)

        row_circ = circulant(circ_vector)
        top_row_circ = row_circ[0:rows_per_block_row, :]
        circ_mats.append(top_row_circ)

    for i in range(n - k):
        circ_mats.append(np.zeros_like(circ_mats[0]))


    num_rows = num_block_rows * rows_per_block_row
    V = np.zeros((num_rows, num_img_pixels))
    for i in range(num_block_rows):
        row = np.concatenate(tuple(circ_mats), axis=1)
        V[i * rows_per_block_row:(i+1) * rows_per_block_row, :] = row
        circ_mats.insert(0, circ_mats.pop())

    return V

def print_model_params(model, optimizer):

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

if __name__ == '__main__':
    main()
