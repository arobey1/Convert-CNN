import torch

from analyze_model import *
from CNN import Net

FNAME = 'cnn-model.pt'

INPUT_WIDTH, INPUT_HEIGHT = 32, 32
NUM_CHANNELS = 3
KERNEL_SIZE = 5
TEST_BATCH_SIZE = 1

def main():

    model = torch.load(FNAME)
    conv_weights, conv_biases, lin_weights, lin_biases = get_numpy_params(model)

    n_values = [32, 28, 24]
    new_conv_weights = create_conv_weights(conv_weights, n_values)

    _, test_loader, classes = create_loaders()

    false_counter = 0
    for (image, labels) in test_loader:

        output_pytorch = model(image).detach().numpy()

        im = image.numpy().flatten()
        output_manual = run_through_model(new_conv_weights, conv_biases, lin_weights,
                                            lin_biases, n_values, im)

        if np.allclose(output_pytorch, output_manual, rtol=1e-4, atol=1e-4) is not True:
            print(output_pytorch)
            print(output_manual)
            false_counter += 1

    print(false_counter)


def run_through_model(new_conv_weights, conv_biases, lin_weights, lin_biases,
                        n_values, x):
    """Run an input x through the model using the new convolutional weights"""

    for idx, (W, b) in enumerate(zip(new_conv_weights, conv_biases)):
        curr_n = n_values[idx]

        x = conv_layer_as_matrix_op(W, b, x, curr_n, KERNEL_SIZE)
        x = x.flatten()

    end_conv_n = n_values[idx + 1]
    num_linear_layers = len(lin_weights)

    for idx, (W, b) in enumerate(zip(lin_weights, lin_biases)):

        if idx < num_linear_layers - 1:
            x = linear_layer(W, b, x, use_relu=True)
        else:
            x = linear_layer(W, b, x, use_relu=False)

    return x


def create_conv_weights(conv_weights, n_values):
    """Turn convolutional weight tensors into 2D matrices"""

    conv_xform_weights = []
    for idx, conv_w in enumerate(conv_weights):

        curr_n = n_values[idx]
        W = create_W_matrix(curr_n, conv_w)
        conv_xform_weights.append(W)

    return conv_xform_weights


def get_numpy_params(model):
    """Get the parameters of a CNN and store in lists"""

    conv_weights, conv_biases, lin_weights, lin_biases = ([] for _ in range(4))

    for param_tensor in model.state_dict():
        numpy_tensor = model.state_dict()[param_tensor].numpy()
        if 'conv' in param_tensor:
            if 'weight' in param_tensor:
                conv_weights.append(numpy_tensor)
            elif 'bias' in param_tensor:
                conv_biases.append(numpy_tensor)
            else:
                print('ERROR: conv param that is neither a weight nor a bias')

        elif 'fc' in param_tensor:
            if 'weight' in param_tensor:
                lin_weights.append(numpy_tensor)
            elif 'bias' in param_tensor:
                lin_biases.append(numpy_tensor)
            else:
                print('ERROR: linear param that is neither a weight nor a bias')

    return conv_weights, conv_biases, lin_weights, lin_biases


if __name__ == '__main__':
    main()
