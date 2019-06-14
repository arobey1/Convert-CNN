import numpy as np
from scipy.linalg import circulant
from train_cnn import create_loaders


class ConvNet:

    def __init__(self, model, n_values):
        self._model = model
        self.n_values = n_values

        conv_weights, self._conv_biases, self._lin_weights, self._lin_biases = self.__load_numpy_params()

        self.k_values = [W.shape[3] for W in conv_weights]

        self._conv_weights = self.__create_conv_weights(conv_weights)
        # self.__test_similarity()


    @property
    def conv_weights_and_biases():
        return zip(self._conv_weights, self._conv_biases)


    def __load_numpy_params(self):
        """Get the parameters of a CNN and store in lists of numpy arrays

        returns:
            conv_weights: list of np.ndarrays   - convolutional tensors
            conv_biases: list of np.ndarrays    - biases for conv terms
            lin_weights: list of np.ndarrays    - linear weights
            lin_biases: list of np.ndarrays     - biases for linear terms
        """

        conv_weights, conv_biases, lin_weights, lin_biases = ([] for _ in range(4))

        for param_tensor in self._model.state_dict():
            numpy_tensor = self._model.state_dict()[param_tensor].numpy()
            if 'conv' in param_tensor or 'features' in param_tensor:
                if 'weight' in param_tensor:
                    conv_weights.append(numpy_tensor)
                elif 'bias' in param_tensor:
                    conv_biases.append(numpy_tensor)
                else:
                    print('ERROR: conv param that is neither a weight nor a bias')

            elif 'fc' in param_tensor or 'classifier' in param_tensor:
                if 'weight' in param_tensor:
                    lin_weights.append(numpy_tensor)
                elif 'bias' in param_tensor:
                    lin_biases.append(numpy_tensor)
                else:
                    print('ERROR: linear param that is neither a weight nor a bias')

        return conv_weights, conv_biases, lin_weights, lin_biases

    def __create_conv_weights(self, conv_weights):
        """Turn convolutional weight tensors into 2D matrices

        params:
            conv_weights: list of np.ndarrays - convolutional tensors

        returns:
            conv_xform_weights: list of 2D np.ndarrays - 2D matrices that perform
                                                       - the convolution
        """

        conv_xform_weights = []
        for idx, conv_w in enumerate(conv_weights):

            curr_n = self.n_values[idx]
            W = self.__create_W_matrix(curr_n, conv_w)
            conv_xform_weights.append(W)

        return conv_xform_weights

    def __create_W_matrix(self, n, conv_weight):
        """Transfrom convolutional tensor into 2d matrix - we take advantage of the
        isomorphism between convolutions and square matrices

        params:
            n: int                              - input image size
            conv_weight: (i, j, k, k) tensor    - convolutional weight tensor
                                                - i = depth of output stack (number of filters)
                                                - j = depth of input stack (number of channels)
                                                - k = filter size (filter is k x k)

        returns:
            W: (i * (n - k + 1) ** 2, j * n ** 2) - 2D matrix that performs the same
                                                  - operation as a 2D convolution
        """

        # The first two dimensions of the weight are the output and input depth, respectively
        # The third and fourth dimension are equal to the size of the filter -
        # we assume that all filters are square
        output_depth, input_depth, k, _ = conv_weight.shape

        # There are i * j blocks in the output matrix W - each block corresponds to
        # a block matrix V_ij
        num_rows_per_block = (n - k + 1) ** 2

        # initialize W to be an array of all zeros
        W = np.zeros((output_depth * num_rows_per_block, input_depth * n ** 2))
        # W = np.zeros((6 * num_rows_per_block, 3 * n ** 2))

        # i indexes the output feature map stack
        for i in range(output_depth):

            # j indexes the input feature map stack (i.e. number of channels in input image)
            for j in range(input_depth):

                # Get the filter that convolves the i^th output feature map with the
                # j^th input feature map (i.e. channel for the first layer)
                filter = conv_weight[i, j, :, :]

                # Each V_ij is a block circulant (topelitz) matrix - that is, each
                # block of V_ij is a circulant matrix, meaning that each entry appears
                # in each row and column exactly once.  V_ij will have dimensions
                # (num_rows_per_block ** 2, n ** 2)
                V_ij = self.__filter_to_bc_matrix(filter, n)

                # Broadcast V_ij into the i^th block row and the j^th block column of W
                dim0, dim1 = num_rows_per_block, n ** 2
                W[i * dim0 : (i + 1) * dim0, j * dim1 : (j + 1) * dim1] = V_ij

        return W

    def __filter_to_bc_matrix(self, filter_mat, n):
        """Converts a 2D convolutional filter matrix to a block circulant matrix
        that performs a 2D convolution as a linear matrix operation

        params:
            filter_mat: (k, k) matrix   - convolutional (k x k) filter
            n: int                      - size of input image (assume square)

        returns:
            V: ((n - k + 1) ** 2, n ** 2) matrix - block circulant matrix that performs
                                                 - a 2D convolution as a linear matrix
                                                 - operation
        """

        k = filter_mat.shape[1]

        # When an (n,n) image is convoled with a (k,k) filter, the reesulting matrix
        # has dimension (n - k + 1) assuming a stride of 1.  And the total number of
        # pixels in the input image is n ** 2 (assuming the image is square)
        output_im_size, num_img_pixels = n - k + 1, n ** 2

        # number of rows in the output matrix is (n - k + 1) ** 2
        num_rows = output_im_size ** 2

        circ_mats = []

        # loop over the rows of the filter matrix
        for row_idx in range(k):

            # get the current row of the input filter matrix
            row = filter_mat[row_idx, :]

            # create a vector [x_1 0 0 ... 0 x_{k**2} x_{k**2-1} ... x_2] where
            # x_1, ..., x_{k**2} are the values of the flattened kernel matrix
            circ_vector = np.concatenate((
                np.array([row[0]]),
                np.zeros(n - k),
                np.flip(row[1:])
            ), axis=0)

            # create a circulant matrix with this vector
            row_circ = circulant(circ_vector)

            # remove rows that convolve the bottom row with the top row
            top_row_circ = row_circ[0:output_im_size, :]

            # add this matrix to the list of circulant matrices
            circ_mats.append(top_row_circ)

        # add n n - k matrices of zeros to the list (as long as n > k)
        for i in range(n - k):
            circ_mats.append(np.zeros_like(circ_mats[0]))

        # initialize V to be a ((n - k + 1) ** 2, n ** 2) matrix of zeros
        V = np.zeros((num_rows, num_img_pixels))

        # loop over the blocked rows of V - each block has (n - k + 1) rows
        # and there are (n - k + 1) total blocks
        for i in range(output_im_size):

            # concatenate all of the block matrices in circ_mats
            row = np.concatenate(tuple(circ_mats), axis=1)

            # broadcast this block row into V
            V[i * output_im_size:(i+1) * output_im_size, :] = row

            # move the last block matrix of circ_mats to the front for the next iteration
            circ_mats.insert(0, circ_mats.pop())

        return V

    def conv_layer_as_matrix_op(self, W, b, x, n, k):
        """Perform a convolutional operation using the isomorphic weight matrix W

        params:
            W: ((n - k + 1) ** 2, n ** 2) matrix - weight matrix corresponding to conv2d
            b: (i, 1) vector                     - bias vector for conv2d
                                                 - i = number of output channels
            x: (j * n ** 2, 1) vector            - input image serialized into a vector
                                                 - j = number of input channels
            n: int                               - size of input image (assume square)
            k: int                               - k = filter size (assume square)

        returns:
            (i, n - k + 1, n - k + 1) tensor - result of convolution
        """

        i = b.shape[0]

        output_im_size = n - k + 1

        Wx = (W @ x).reshape(i, output_im_size, output_im_size)
        return np.maximum(Wx + b.reshape(i, 1, 1), 0)


    def linear_layer(self, W, b, x, use_relu=True):
        """Pass an input vector x through a fully connected layer defined by a weight
        matrix W and a bias vector b

        params:
            W: (n2, n1) matrix  - weight matrix for a linear layer of a neural network
            b: (n2, 1) vector   - bias vector for alinear layer of a neural network
            x: (n1, 1) vector   - input to the linear layer
            use_relu: bool      - flag to use ReLU activation

        returns:
            (n2, 1) vector representing output of linear layer
        """

        if use_relu is True:
            return np.maximum(W @ x + b, 0)
        return W @ x + b

    def run_through_model(self, x):
        """Run an input x through the model using the new convolutional weights

        params:
            x: (n,n) array - input image

        returns:
            x: (n_classes, 1) vector - output of neural network
        """

        for idx, (W, b) in enumerate(zip(self._conv_weights, self._conv_biases)):
            curr_n = self.n_values[idx]
            curr_k = self.k_values[idx]

            x = self.conv_layer_as_matrix_op(W, b, x, curr_n, curr_k)
            x = x.flatten()

        end_conv_n = self.n_values[idx + 1]
        num_linear_layers = len(self._lin_weights)

        for idx, (W, b) in enumerate(zip(self._lin_weights, self._lin_biases)):

            if idx < num_linear_layers - 1:
                x = self.linear_layer(W, b, x, use_relu=True)
            else:
                x = self.linear_layer(W, b, x, use_relu=False)

        return x

    def __test_similarity(self):
        """Test similarity between CNN and CNN as linear feed-forward network"""

        _, test_loader, _ = create_loaders()

        false_counter = 0
        for (image, labels) in test_loader:

            output_pytorch = self._model(image).detach().numpy()

            im = image.numpy().flatten()
            output_manual = self.run_through_model(im)

            if np.allclose(output_pytorch, output_manual, rtol=1e-4, atol=1e-4) is not True:
                false_counter += 1

        print(f"Number of mistakes: {false_counter}")
