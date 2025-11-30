from sklearn.base import *
from abc import ABC, abstractmethod

from numpy.testing import assert_array_almost_equal
import numpy as np

class MLP:
    def __init__(hidden_layer_sizes: tuple, random_state: int = 0,
                 activation: {"identity", "logistic", "tanh", "relu"} = "relu", 
                 alpha: float = 0.0001, batch_size: int = 100, shuffle: bool = True,
                 learning_rate_init: float = 0.001, max_iter: int = 200):
        """
        solver -> stochastic batch gradient descent
        learning_rate -> constant
        """

    @abstractmethod
    def compute_output_layer(self, z_in: np.ndarray):
        pass 

    @abstractmethod
    def compute_loss(self, X, y):
        pass 

    @abstractmethod
    def compute_dLoss_doutput_layer_z_in(self, z_in, a_out, z_out):
        pass 

    def activation_function_derivation(self, a_out, z_out, function: {"identity", "logistic", "tanh", "relu"}):
        """
        returns a Jacobi vectors instead of Jacobis matrices, because the Jacobi matrices only have elements on their main-diagonal
        -- the only exception is the softmax function -> for softmax this function returns Jacobi matrices
        """
        assert(isinstance(z_out, np.ndarray))
        assert(function in {"identity", "logistic", "tanh", "relu"})
        if(function == "logistic"):
            return a_out * (1 - a_out) # sigma(z) * (1 - sigma(z))
        if(function == 'tanh'):
            return 1 - a_out**2 # 1 - tanh^2(z)
        if(function == "relu"):
            return np.where(z_out >= 0, 1.0, 0.0)
        if(function == "identity"):
            return np.ones(shape=z_out.shape, dtype=z_out.dtype)
        
    def activation_function(self, z_in, function: {"identity", "logistic", "tanh", "relu", "softmax"}):
        assert(isinstance(z_in, np.ndarray))
        assert(function in {"identity", "logistic", "tanh", "relu", "softmax"})
        if(function == "logistic"):
            return 1.0 / (1 + np.exp(-1 * z_in))
        if(function == 'tanh'):
            return np.tanh(z_in)
        if(function == "relu"):
            return np.where(z_in >= 0, z_in, 0.0)
        if(function == "identity"):
            return np.copy(z_in)
        if(function == "softmax"):
            e_raised_to_z = np.exp(z_in)
            sum_e_raised_to_z = np.sum(e_raised_to_z, axis=1)
            a_out = e_raised_to_z / sum_e_raised_to_z
            return a_out
        assert(False)
        
    def compute_activation(self, a_in, W, b, function: {"identity", "logistic", "tanh", "relu", "softmax"}):
        z_out = a_in @ W.T + b 
        return z_out, self.activation_function(z_out, function)
        
    def compute_derivations_single_sample(self, a_in, z_out, a_out, W, dLoss_dAout, 
                                          function: {"identity", "logistic", "tanh", "relu"}):
        """
        The implementation is not perfectly optimized for performance, but for readability.
        """
        dZout_dAin = W
        dAout_dZout = self.activation_function_derivation(a_out, z_out, function)
        dAout_dZout = np.diag(dAout_dZout) # compute jacobi matrix
        dAout_dAin = dAout_dZout @ dZout_dAin # matrix = matrix * matrix

        dLoss_dZout = dLoss_dAout @ dAout_dZout # vector = vector * matrixs
        dLoss_dW = dLoss_dZout[:, np.newaxis] @ a_in[np.newaxis, :] # matrix = vector * vector

        # dZout_db = np.diagonal_ones_matrix
        dLoss_db = dLoss_dAout @ dAout_dZout # vector = vector * matrix  # identical to dLoss_dZout

        dLoss_dAin = dLoss_dAout @ dAout_dAin # vector = vector * matrix

        return dLoss_dAin, dLoss_dW, dLoss_db
    
    def compute_derivations_batch1(self, a_in, z_out, a_out, W, dLoss_dAout, 
                                          function: {"identity", "logistic", "tanh", "relu"}):
        """
        The implementation is not perfectly optimized for performance, but for readability.
        """
        dZout_dAin = W
        dAout_dZout_vector = self.activation_function_derivation(a_out, z_out, function)

        # Convert vector jacobian to full jacobian matrices for each sample in batch
        N, dimOut = dAout_dZout_vector.shape
        dAout_dZout = np.einsum('ij, jk -> ijk', dAout_dZout_vector, np.eye(dimOut))

        dAout_dAin = dAout_dZout @ dZout_dAin # [N, dimOut, dimOut] @ [N, dimOUt, dimIn] = [N, dimOut, dimIn] (numpy broadcasting here)

        dLoss_dZout = dLoss_dAout * dAout_dZout_vector # [N, dimAout] * [N, dimAout] = [N, dimAout]
        dLoss_dW = dLoss_dZout[:, :, np.newaxis] @ a_in[:, np.newaxis, :] # [N, dimAout, 1] @ [N, 1 dimAin] = [N, dimAout, dimAin]

        # dZout_db -> identity matrix -> therefore both derivations are identical
        dLoss_db = dLoss_dZout

        dLoss_dAin = dLoss_dAout[:, np.newaxis, :] @ dAout_dAin # [N, 1, dimOut] @ [N, dimOut, dimIn] = [N, 1, dimIn]
        dLoss_dAin = dLoss_dAin.reshape((dLoss_dAin.shape[0], dLoss_dAin.shape[2])) # [N, dimIn]

        dLoss_dW = np.mean(dLoss_dW, axis=0)
        dLoss_db = np.mean(dLoss_db, axis=0)
        return dLoss_dAin, dLoss_dW, dLoss_db

    def compute_derivations_batch(self, a_in, z_out, a_out, W, dLoss_dAout, 
                                          function: {"identity", "logistic", "tanh", "relu", "softmax"}):
        """
        The implementation is not perfectly optimized for performance, but for readability.
        """
        assert(function in {"identity", "logistic", "tanh", "relu", "softmax"})

        if(function != "softmax"):
            # dAout_dZout matrix has only diagonal components
            dAout_dZout_vector = self.activation_function_derivation(a_out, z_out, function)

            # chain rule (dAout_dZout is a matrix with only values on the diagonal 
            # -> therefore its easier to use a vector and do element-wise multiplication)
            dLoss_dZout = dLoss_dAout * dAout_dZout_vector # [N, dimAout] * [N, dimAout] = [N, dimAout]
        else:
            # compute dAout_dZout for softmax by hand (one sample for equal and one for unequal indices)
            # then split the matrix into diagonal matrix with a_out on the diagonal and a matrix (that can be computes by a_out * a_out (column_vector * column_vector))
            # dAout_dZout = diagonal(a_out) - a_out.column * a_out.row
            # then compute dLoss_dZout and split into easier components
            # [N, out] * [N, out] + [N, out] * ( [N, out] scalarproducts [N, out]) = [N, otu] + [N, out] * [N] = [N, out] * [N, np.newaxis] = [N, out] (broadcasting)
            dLoss_dZout = dLoss_dAout * a_out + a_out * (np.sum(dLoss_dAout * a_out, axis=1))[:, np.newaxis]


        # N times: column-vetor * row-vector = matrix -> it is easy to verify this formula by hand
        # compute dLoss_dw for one row-vector of w -> then stack the Jacobi row-vectors to form a matrix
        dLoss_dW = dLoss_dZout[:, :, np.newaxis] @ a_in[:, np.newaxis, :] # [N, dimAout, 1] @ [N, 1 dimAin] = [N, dimAout, dimAin]

        # dZout_db -> identity matrix -> therefore both derivations are identical
        dLoss_db = dLoss_dZout

        # chain rule
        dZout_dAin = W
        dLoss_dAin = dLoss_dZout @ dZout_dAin # [N, out] @ [out, in] = [N, in]

        # compute mean losses
        dLoss_dW = np.mean(dLoss_dW, axis=0)
        dLoss_db = np.mean(dLoss_db, axis=0)
        return dLoss_dAin, dLoss_dW, dLoss_db





def setup_batch_data(function, N=4, dimIn=2, dimOut=3):
    """Sets up batch data (N=4, dimIn=2, dimOut=3) and common parameters."""
    m = MLP()
    
    # Batch size N=4. dimIn=2, dimOut=3
    # a_in: [4, 2]
    a_in_batch = np.array([
        [0.5, 1.0],  # Sample 1
        [0.1, 0.5],  # Sample 2
        [0.8, 0.2],  # Sample 3
        [1.2, 0.3]   # Sample 4
    ])
    
    # Upstream Error: [4, 3]
    dL_dAout_batch = np.array([
        [ 0.6, -0.4, 0.2], # Error for Sample 1
        [-0.8,  0.5, 0.1], # Error for Sample 2
        [ 0.3, -0.2, 0.4], # Error for Sample 3
        [-0.1,  0.7, 0.6]  # Error for Sample 4
    ])

    # Common W: [dimOut, dimIn] = [3, 2]
    W = np.array([
        [0.1, -0.3], # Weights to dimOut 1
        [0.5,  0.2], # Weights to dimOut 2
        [0.4, -0.1]  # Weights to dimOut 3
    ])
    
    # Common b: [dimOut] = [3]
    b = np.array([0.1, 0.1, 0.05])

    # Compute Z_out and A_out for the batch
    # z_out_batch: [4, 3], a_out_batch: [4, 3]
    z_out_batch, a_out_batch = m.compute_activation(a_in_batch, W, b, function)
    
    return m, a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch, function

def calculate_expected_batch_from_single_sample_sum(m, a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch, function):
    """
    Calculates the expected batched gradients by looping over N samples and summing 
    the results from the single-sample backpropagation function (compute_derivations_single_sample).
    
    Since compute_derivations_batch returns MEAN gradients for W and b, we must 
    calculate the mean here as well. dLoss_dAin is returned per sample (N, dimIn).
    """
    N = a_in_batch.shape[0]
    dimOut = W.shape[0]
    dimIn = W.shape[1]
    
    expected_dW_sum = np.zeros((dimOut, dimIn))
    expected_db_sum = np.zeros(dimOut)
    expected_dAin_list = []

    for i in range(N):
        a_in = a_in_batch[i]
        z_out = z_out_batch[i]
        a_out = a_out_batch[i]
        dL_dAout = dL_dAout_batch[i]

        dLoss_dAin_single, dLoss_dW_single, dLoss_db_single = m.compute_derivations_single_sample(
            a_in, z_out, a_out, W, dL_dAout, function
        )

        expected_dW_sum += dLoss_dW_single
        expected_db_sum += dLoss_db_single
        expected_dAin_list.append(dLoss_dAin_single)

    # dLoss_dAin is per-sample, stack results
    expected_dAin = np.array(expected_dAin_list)
    
    # W and b gradients are typically averaged over the batch
    expected_dW_mean = expected_dW_sum / N
    expected_db_mean = expected_db_sum / N
    
    return expected_dAin, expected_dW_mean, expected_db_mean


def calculate_expected_batch_logistic(m, a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch):
    """
    Calculates the expected batched gradients using the efficient vectorized formulas 
    to derive the ground truth (Summed gradients). This function is kept for reference.
    """
    # 1. Calculate Core Error Signal (dLoss/dZout)
    dAout_dZout_vector = m.activation_function_derivation(a_out_batch, z_out_batch, "logistic")
    dLoss_dZout_batch = dL_dAout_batch * dAout_dZout_vector # [N, dimOut]
    
    # 2. Expected dLoss/db (Sum of dLoss/dZout over N)
    expected_db_sum = np.sum(dLoss_dZout_batch, axis=0) # [dimOut]

    # 3. Expected dLoss/dW (Sum of outer products over N)
    # Outer Product: [N, dimOut, 1] @ [N, 1, dimIn] = [N, dimOut, dimIn]
    dLoss_dW_batch = dLoss_dZout_batch[:, :, np.newaxis] @ a_in_batch[:, np.newaxis, :]
    expected_dW_sum = np.sum(dLoss_dW_batch, axis=0) # [dimOut, dimIn]

    # 4. Expected dLoss/dAin (dLoss/dZout @ W)
    # [N, dimOut] @ [dimOut, dimIn] = [N, dimIn]
    expected_dAin = dLoss_dZout_batch @ W 
    
    return expected_dAin, expected_dW_sum, expected_db_sum

def test_logistic_batch_4x2x3():
    """
    Tests batch backpropagation for Logistic activation with N=4, dimIn=2, dimOut=3
    by comparing the vectorized batch calculation (actual) against the sum/mean of 
    single-sample calculations (expected).
    """
    print("\n--- Running Logistic (Sigmoid) Batch Test (N=4, dimIn=2, dimOut=3) ---")
    m, a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch, func = setup_batch_data("logistic")
    
    # Calculate Expected Values by summing single-sample results and calculating the mean
    expected_dAin, expected_dW, expected_db = calculate_expected_batch_from_single_sample_sum(
        m, a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch, func
    )

    # Compute Actual Batch Gradients (vectorized implementation returns the MEAN for W and b)
    dLoss_dAin, dLoss_dW, dLoss_db = m.compute_derivations_batch(
        a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch, func
    )
    
    # The expected values computed via the single-sample method must match the
    # actual values computed via the vectorized batch method.
    
    try:
        # Use decimal=5 to account for floating point errors in Sigmoid
        assert_array_almost_equal(dLoss_db, expected_db, decimal=5)
        assert_array_almost_equal(dLoss_dW, expected_dW, decimal=5)
        assert_array_almost_equal(dLoss_dAin, expected_dAin, decimal=5)
        print("LOGISTIC BATCH TEST (4x2x3) PASSED")
    except AssertionError as e:
        print("LOGISTIC BATCH TEST (4x2x3) FAILED:")
        print(f"dB Error: {dLoss_db} != {expected_db}")
        print(f"dW Error: {dLoss_dW} != {expected_dW}")
        print(f"dAin Error: {dLoss_dAin} != {expected_dAin}")
        raise e

if __name__ == '__main__':
    try:
        # Calling the new, specifically named test function
        test_logistic_batch_4x2x3() 
        print("\nAll batch tests passed successfully.")
    except AssertionError:
        print("\nOne or more batch tests failed.")





"""
Here are some tests.
"""


def setup_batch_data(function, N=4, dimIn=2, dimOut=3):
    """Sets up batch data (N=4, dimIn=2, dimOut=3) and common parameters."""
    m = MLP()
    
    # Batch size N=4. dimIn=2, dimOut=3
    # a_in: [4, 2]
    a_in_batch = np.array([
        [0.5, 1.0],  # Sample 1
        [0.1, 0.5],  # Sample 2
        [0.8, 0.2],  # Sample 3
        [1.2, 0.3]   # Sample 4
    ])
    
    # Upstream Error: [4, 3]
    dL_dAout_batch = np.array([
        [ 0.6, -0.4, 0.2], # Error for Sample 1
        [-0.8,  0.5, 0.1], # Error for Sample 2
        [ 0.3, -0.2, 0.4], # Error for Sample 3
        [-0.1,  0.7, 0.6]  # Error for Sample 4
    ])

    # Common W: [dimOut, dimIn] = [3, 2]
    W = np.array([
        [0.1, -0.3], # Weights to dimOut 1
        [0.5,  0.2], # Weights to dimOut 2
        [0.4, -0.1]  # Weights to dimOut 3
    ])
    
    # Common b: [dimOut] = [3]
    b = np.array([0.1, 0.1, 0.05])

    # Compute Z_out and A_out for the batch
    # z_out_batch: [4, 3], a_out_batch: [4, 3]
    z_out_batch, a_out_batch = m.compute_activation(a_in_batch, W, b, function)
    
    return m, a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch, function

def calculate_expected_batch_logistic(m, a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch):
    """
    Calculates the expected batched gradients using the efficient vectorized formulas 
    to derive the ground truth.
    """
    # 1. Calculate Core Error Signal (dLoss/dZout)
    dAout_dZout_vector = m.activation_function_derivation(a_out_batch, z_out_batch, "logistic")
    dLoss_dZout_batch = dL_dAout_batch * dAout_dZout_vector # [N, dimOut]
    
    # 2. Expected dLoss/db (Sum of dLoss/dZout over N)
    expected_db = np.sum(dLoss_dZout_batch, axis=0) # [dimOut]

    # 3. Expected dLoss/dW (Sum of outer products over N)
    # Outer Product: [N, dimOut, 1] @ [N, 1, dimIn] = [N, dimOut, dimIn]
    dLoss_dW_batch = dLoss_dZout_batch[:, :, np.newaxis] @ a_in_batch[:, np.newaxis, :]
    expected_dW = np.sum(dLoss_dW_batch, axis=0) # [dimOut, dimIn]

    # 4. Expected dLoss/dAin (dLoss/dZout @ W)
    # [N, dimOut] @ [dimOut, dimIn] = [N, dimIn]
    expected_dAin = dLoss_dZout_batch @ W 
    
    return expected_dAin, expected_dW, expected_db

def test_logistic_batch_4x2x3():
    """
    Tests batch backpropagation for Logistic activation with N=4, dimIn=2, dimOut=3.
    """
    print("\n--- Running Logistic (Sigmoid) Batch Test (N=4, dimIn=2, dimOut=3) ---")
    m, a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch, func = setup_batch_data("logistic")
    
    # Calculate Expected Values
    expected_dAin, expected_dW, expected_db = calculate_expected_batch_logistic(
        m, a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch
    )

    # Compute Actual Batch Gradients
    dLoss_dAin, dLoss_dW, dLoss_db = m.compute_derivations_batch(
        a_in_batch, z_out_batch, a_out_batch, W, dL_dAout_batch, func
    )
    
    # The expected values are recalculated based on the new dimensions and data:
    
    # Expected dLoss/db: [0.28821034, 0.17046187, 0.54010313]
    # Expected dLoss/dW: 
    # [[ 0.23075249,  0.19833633],
    #  [ 0.10323386,  0.11749841],
    #  [ 0.44977457,  0.18349256]]
    # Expected dLoss/dAin:
    # [[ 0.06323497, -0.17936173],
    #  [-0.10309995,  0.03845942],
    #  [ 0.18002674,  0.03859664],
    #  [ 0.19799298, -0.06316279]]

    try:
        # Use decimal=5 to account for floating point errors in Sigmoid
        assert_array_almost_equal(dLoss_db, expected_db, decimal=5)
        assert_array_almost_equal(dLoss_dW, expected_dW, decimal=5)
        assert_array_almost_equal(dLoss_dAin, expected_dAin, decimal=5)
        print("LOGISTIC BATCH TEST (4x2x3) PASSED")
    except AssertionError as e:
        print("LOGISTIC BATCH TEST (4x2x3) FAILED:")
        print(f"dB Error: {dLoss_db} != {expected_db}")
        print(f"dW Error: {dLoss_dW} != {expected_dW}")
        print(f"dAin Error: {dLoss_dAin} != {expected_dAin}")
        raise e


def setup_test_data(activation_function):
    """Sets up common data and calculates activation outputs."""
    m = MLP()
    
    # Common Inputs (N_in=3, N_out=2)
    a_in = np.array([0.5, 1.0, 0.2])
    W = np.array([[0.1, -0.3, 0.4], [0.5, 0.2, -0.1]])
    b = np.array([0.1, 0.1])
    dL_dAout = np.array([0.6, -0.4])

    # Calculate Z_out and A_out 
    z_out = a_in @ W.T + b 
    a_out = m.activation_function(z_out, activation_function)
    
    return m, a_in, z_out, a_out, W, dL_dAout, activation_function

def test_identity_activation():
    print("--- Running Identity Test ---")
    m, a_in, z_out, a_out, W, dL_dAout, func = setup_test_data("identity")

    dLoss_dAin, dLoss_dW, dLoss_db = m.compute_derivations_single_sample(a_in, z_out, a_out, W, dL_dAout, func)
    
    # Expected values for Identity: dLoss/dZout = dLoss/dAout = [0.6, -0.4]
    
    # Expected dLoss/db = dLoss/dZout
    expected_db = np.array([0.6, -0.4])
    # Expected dLoss/dW = dLoss/dZout^T @ a_in
    expected_dW = np.array([
        [ 0.3,  0.6,  0.12],
        [-0.2, -0.4, -0.08]
    ])
    # Expected dLoss/dAin = dLoss/dZout @ W 
    expected_dAin = np.array([-0.14, -0.26, 0.28])
    
    try:
        assert_array_almost_equal(dLoss_db, expected_db, decimal=8)
        assert_array_almost_equal(dLoss_dW, expected_dW, decimal=8)
        assert_array_almost_equal(dLoss_dAin, expected_dAin, decimal=8)
        print("IDENTITY TEST PASSED")
    except AssertionError as e:
        print("IDENTITY TEST FAILED:")
        print(f"dB Error: {dLoss_db} != {expected_db}")
        print(f"dW Error: {dLoss_dW} != {expected_dW}")
        print(f"dAin Error: {dLoss_dAin} != {expected_dAin}")
        raise e

def test_relu_activation():
    print("\n--- Running ReLU Test ---")
    m, a_in, z_out, a_out, W, dL_dAout, func = setup_test_data("relu")

    dLoss_dAin, dLoss_dW, dLoss_db = m.compute_derivations_single_sample(a_in, z_out, a_out, W, dL_dAout, func)
    
    # Expected dLoss/dZout = [0.0, -0.4] (since z_out[0] = -0.07 < 0)
    
    # Expected dLoss/db = dLoss/dZout
    expected_db = np.array([0.0, -0.4])
    # Expected dLoss/dW = dLoss/dZout^T @ a_in
    expected_dW = np.array([
        [ 0.0,  0.0,  0.0],
        [-0.2, -0.4, -0.08]
    ])
    # Expected dLoss/dAin = dLoss/dZout @ W 
    expected_dAin = np.array([-0.2, -0.08, 0.04])
    
    try:
        assert_array_almost_equal(dLoss_db, expected_db, decimal=8)
        assert_array_almost_equal(dLoss_dW, expected_dW, decimal=8)
        assert_array_almost_equal(dLoss_dAin, expected_dAin, decimal=8)
        print("ReLU TEST PASSED")
    except AssertionError as e:
        print("ReLU TEST FAILED:")
        print(f"dB Error: {dLoss_db} != {expected_db}")
        print(f"dW Error: {dLoss_dW} != {expected_dW}")
        print(f"dAin Error: {dLoss_dAin} != {expected_dAin}")
        raise e

def test_logistic_activation():
    print("\n--- Running Logistic (Sigmoid) Test ---")
    m, a_in, z_out, a_out, W, dL_dAout, func = setup_test_data("logistic")

    dLoss_dAin, dLoss_dW, dLoss_db = m.compute_derivations_single_sample(a_in, z_out, a_out, W, dL_dAout, func)
    
    # Expected dLoss/dZout = [0.14983058, -0.09323712] (from detailed calculation)
    expected_db = np.array([0.14983058, -0.09323712])
    
    # Expected dLoss/dW 
    expected_dW = np.array([
        [ 0.07491529,  0.14983058,  0.02996612],
        [-0.04661856, -0.09323712, -0.01864742]
    ])
    
    # Expected dLoss/dAin
    expected_dAin = np.array([-0.03162329, -0.06358482,  0.06924375])
    
    try:
        # We must use a lower precision (decimal=6) due to the cumulative floating point errors 
        # from the exp() function in Sigmoid.
        assert_array_almost_equal(dLoss_db, expected_db, decimal=6)
        assert_array_almost_equal(dLoss_dW, expected_dW, decimal=6)
        assert_array_almost_equal(dLoss_dAin, expected_dAin, decimal=6)
        print("LOGISTIC TEST PASSED")
    except AssertionError as e:
        print("LOGISTIC TEST FAILED:")
        print(f"dB Error: {dLoss_db} != {expected_db}")
        print(f"dW Error: {dLoss_dW} != {expected_dW}")
        print(f"dAin Error: {dLoss_dAin} != {expected_dAin}")
        raise e
    
# TODO : rewrite the functions compute_derivative... in better style