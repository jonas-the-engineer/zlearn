from sklearn.base import *
from abc import ABC, abstractmethod

from numpy.testing import assert_array_almost_equal
import numpy as np
from sklearn.datasets import load_iris, make_classification
from preprocessing import zStandardScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from preprocessing import zLabelBinarizer, ztrain_test_split

class zMLPsuperclass:
    def __init__(self, hidden_layer_sizes: tuple = (1,), random_state: int = 0,
                 activation: {"identity", "logistic", "tanh", "relu"} = "relu", 
                 alpha: float = 0.0001, batch_size: int = 1, shuffle: bool = True,
                 learning_rate_init: float = 0.001, max_iter: int = 200, adam_beta_vel: float = 0.9, adam_beta_smoothness: float = 0.99):
        """
        solver -> adam (only momentum, no division by sqrt(sum(squared gradients exponentially weighted)))
        learning_rate -> constant
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state
        self.activation = activation
        self.alpha = alpha 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.rgen_ = np.random.default_rng(seed=self.random_state)
        self.adam_beta_vel = adam_beta_vel
        self.adam_beta_smoothness = adam_beta_smoothness

    def initialize_weights(self, input_dim, output_dim):
        Ws, bs = [], []
        dim_in = input_dim
        for i, dim_out in enumerate(self.hidden_layer_sizes + (output_dim, )):
            scaling = np.sqrt(1.0 / dim_in)
            if(self.activation == "relu"):
                scaling = np.sqrt(2.0 / dim_in)
            if(self.activation == "logistic"):
                scaling = np.sqrt(1.0 / (dim_in + dim_out))
            W = self.rgen_.standard_normal(size=(dim_out, dim_in)) * scaling
            b = np.zeros(shape=(dim_out,))
            Ws.append(W)
            bs.append(b)

            dim_in = dim_out
        self.Ws_ = Ws
        self.bs_ = bs

    @abstractmethod
    def get_output_activation_function(self):
        pass

    @abstractmethod
    def compute_loss(self, a_out, y):
        pass 

    @abstractmethod
    def compute_dLoss_dAout_output_layer(self, a_out, y):
        pass 

    def get_batches(self, X, y):
        assert(isinstance(X, np.ndarray) and isinstance(y, np.ndarray))
        N_total = X.shape[0]
        # size = int(self.batch_size * N_total)
        size = self.batch_size
        if(self.shuffle):
            permutation = self.rgen_.permutation(N_total)
            X = X[permutation]
            y = y[permutation]
        s = 0
        X_batches = []
        y_batches = []
        while(s < N_total):
            e = s + size
            if(e > N_total):
                break
            X_batches.append(X[s:e])
            y_batches.append(y[s:e])
            s = e
        return X_batches, y_batches
    
    def initialize_adam_parameters(self):
        self.Ws_velocity_ = [np.zeros_like(W) for W in self.Ws_]
        self.Ws_smoothness_ = [np.zeros_like(W) for W in self.Ws_]
        self.bs_velocity_ = [np.zeros_like(b) for b in self.bs_]
        self.bs_smoothness_ = [np.zeros_like(b) for b in self.bs_]


    def fit(self, X, y):
        print("fit with shape X: ", X.shape, " shape y: ", y.shape)
        learning_rate = self.learning_rate_init
        assert(isinstance(X, np.ndarray) and isinstance(y, np.ndarray))
        if(len(y.shape) == 1): 
            y = y[:, np.newaxis]
        input_dim, output_dim = X.shape[1], y.shape[1]
        self.initialize_weights(input_dim, output_dim)
        self.initialize_adam_parameters()
        self.loss_curve_ = []
        
        layer_sizes = self.hidden_layer_sizes + (output_dim,)
        layer_activations = (self.activation,) * len(self.hidden_layer_sizes) + (self.get_output_activation_function(),)

        for iteration in range(self.max_iter):
            X_batches, y_batches = self.get_batches(X, y)
            for X_batch, y_batch in zip(X_batches, y_batches):
                # forward propagation
                a_in = X_batch
                z_outs = []
                a_outs = []
                a_ins = []
                for W, b, activation in zip(self.Ws_, self.bs_, layer_activations):
                    z_out, a_out = self.compute_activation(a_in, W, b, activation)
                    z_outs.append(z_out)
                    a_outs.append(a_out)
                    a_ins.append(a_in)
                    a_in = a_out
                
                # loss computation 
                # loss = self.compute_loss(a_out=a_in, y=y_batch)
                # self.loss_curve_.append(loss)
                
                # backpropagation
                dLoss_dAout_output_layer = self.compute_dLoss_dAout_output_layer(a_out=a_in, y=y_batch)
                dLoss_dAout = dLoss_dAout_output_layer

                zipped_infos = tuple(zip(self.Ws_, self.bs_, self.Ws_velocity_, self.bs_velocity_, self.Ws_smoothness_, self.bs_smoothness_, layer_activations, a_ins, z_outs, a_outs))
                N = y_batch.shape[0]
                for W, b, W_vel, b_vel, W_smth, b_smth, activation, a_in, z_out, a_out in zipped_infos[::-1]:
                    dLoss_dAin, dLoss_dW, dLoss_db = self.compute_derivations_batch(a_in, z_out, a_out, W, dLoss_dAout, activation, y_batch)
                    # compute mean losses
                    dLoss_dW = np.sum(dLoss_dW, axis=0)
                    dLoss_db = np.sum(dLoss_db, axis=0)

                    dW = (dLoss_dW + self.alpha * W) * (1.0 / N)
                    # dW = learning_rate * (- dLoss_dW - self.alpha * W)
                    db = (dLoss_db)

                    W_vel = W_vel * self.adam_beta_vel + (1 - self.adam_beta_vel) * dW
                    b_vel = b_vel * self.adam_beta_vel + (1 - self.adam_beta_vel) * db

                    # removing the (1 - self.adam_beta_vel) bias
                    W_vel_unbiased = W_vel / (1 - self.adam_beta_vel)
                    b_vel_unbiased = b_vel / (1 - self.adam_beta_vel)

                    # compute smoothness
                    W_smth = W_smth * self.adam_beta_smoothness + (1 - self.adam_beta_smoothness) * dW**2
                    b_smth = b_smth * self.adam_beta_smoothness + (1 - self.adam_beta_smoothness) * db**2


                    W += - learning_rate * W_vel_unbiased / (np.sqrt(W_smth) + 1e-12)
                    b += - learning_rate * b_vel_unbiased / (np.sqrt(b_smth) + 1e-12)

                    dLoss_dAout = dLoss_dAin # the ouput of the previous layer is the current input
            
            # a_out = self.predict(X) # the super is only used in derived classes, so it always uses the super of the superclas -> this is NOT a good solution, but a quick one # TODO
            loss = self.compute_loss(X, y)
            self.loss_curve_.append(loss)
        return self

    def predict(self, X):
        assert(isinstance(X, np.ndarray))
        a_in = X
        layer_activations = (self.activation,) * len(self.hidden_layer_sizes) + (self.get_output_activation_function(),)
        for W, b, activation in zip(self.Ws_, self.bs_, layer_activations):
            z_out, a_out = self.compute_activation(a_in, W, b, activation)
            a_in = a_out
        return a_in

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
        
    def compute_pre_activation(self, z_out, function: {"identity", "logistic", "tanh", "relu", "softmax"}):
        assert(isinstance(z_out, np.ndarray))
        assert(function in {"identity", "logistic", "tanh", "relu", "softmax"})
        if(function == "logistic"):
            return 1.0 / (1 + np.exp(-1 * z_out))
        if(function == 'tanh'):
            return np.tanh(z_out)
        if(function == "relu"):
            return np.where(z_out >= 0, z_out, 0.0)
        if(function == "identity"):
            return np.copy(z_out)
        if(function == "softmax"):
            e_raised_to_z = np.exp(z_out)
            sum_e_raised_to_z = np.sum(e_raised_to_z, axis=1)
            a_out = e_raised_to_z / sum_e_raised_to_z[:,np.newaxis]
            return a_out
        
    def compute_activation(self, a_in, W, b, function: {"identity", "logistic", "tanh", "relu", "softmax"}):
        z_out = a_in @ W.T + b 
        return z_out, self.compute_pre_activation(z_out, function)
        
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
    
    def compute_derivations_batch(self, a_in, z_out, a_out, W, dLoss_dAout, 
                                          function: {"identity", "logistic", "tanh", "relu", "softmax"}, y=None):
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
            if(y is not None):    # TODO: use one solution or other not both
                dLoss_dZout = a_out - y


        # N times: column-vetor * row-vector = matrix -> it is easy to verify this formula by hand
        # compute dLoss_dw for one row-vector of w -> then stack the Jacobi row-vectors to form a matrix
        dLoss_dW = dLoss_dZout[:, :, np.newaxis] @ a_in[:, np.newaxis, :] # [N, dimAout, 1] @ [N, 1 dimAin] = [N, dimAout, dimAin]

        # dZout_db -> identity matrix -> therefore both derivations are identical
        dLoss_db = dLoss_dZout

        # chain rule
        dZout_dAin = W
        dLoss_dAin = dLoss_dZout @ dZout_dAin # [N, out] @ [out, in] = [N, in]
        return dLoss_dAin, dLoss_dW, dLoss_db
    
class zMLPRegressor(RegressorMixin, BaseEstimator, zMLPsuperclass):
    def get_output_activation_function(self):
        return "identity"

    def compute_dLoss_dAout_output_layer(self, a_out, y):
        assert(isinstance(a_out, np.ndarray) and isinstance(y, np.ndarray))
        dLoss_dAout = (a_out - y) * 2.0
        return dLoss_dAout
    
    def compute_loss(self, X, y):
        a_out = self.predict(X)
        assert(isinstance(a_out, np.ndarray) and isinstance(y, np.ndarray))
        N = a_out.shape[0]
        Loss = np.mean((a_out - y)**2, axis=(0, 1))
        return Loss
    
class zMLPClassifier(ClassifierMixin, BaseEstimator, zMLPsuperclass):
    def __init__(self, hidden_layer_sizes = (1, ), random_state = 0, activation = "relu", alpha = 0.0001, 
                 batch_size = 1, shuffle = True, learning_rate_init = 0.001, max_iter = 200, adam_beta_vel = 0.9, adam_beta_smoothness = 0.99,
                 output_activation = 'softmax'):
        super().__init__(hidden_layer_sizes, random_state, activation, alpha, batch_size, shuffle, learning_rate_init, max_iter, adam_beta_vel, adam_beta_smoothness)
        self.output_activation = output_activation

    def get_output_activation_function(self):
        return self.output_activation
    
    def fit(self, X, y):
        unique_elements = np.unique(y)
        unique_elements.sort()
        assert(unique_elements[0] == 0 and unique_elements[1] == 1 and len(unique_elements) == 2) # only except one hot encoded targets
        super().fit(X, y)
        return self

    def predict_proba(self, X):
        Y = super().predict(X)
        return Y 
    
    def predict(self, X):
        Y = self.predict_proba(X)
        Y_max = np.max(Y, axis=1)
        Y_onehot = np.where(Y == Y_max[:, np.newaxis], 1, 0)
        return Y_onehot
    
    """
    not used in current implementation
    """
    def compute_dLoss_dAout_output_layer(self, a_out, y):
        assert(isinstance(a_out, np.ndarray) and isinstance(y, np.ndarray))
        dLoss_dAout = (a_out - y) # / (a_out * (1 - a_out))
        return dLoss_dAout
    
    def compute_loss(self, X, y):
        a_out = self.predict_proba(X)
        assert(isinstance(a_out, np.ndarray) and isinstance(y, np.ndarray))
        N = a_out.shape[0]
        log_losses = np.where(a_out > 0.0001, - y * np.log(a_out), 10000)
        log_losses = - y * np.log(a_out)
        Loss = np.mean(log_losses, axis=(0, 1))
        return Loss

    
# the regularization error does not get propagated
def run_classifier_test():
    X, y = make_classification(n_samples=10000, n_features=10, n_classes=2)
    sc = zStandardScaler()
    X = sc.fit_transform(X)
    print("y.shape: ", y.shape)
    X_train, X_test, y_train, y_test = ztrain_test_split(X, y, test_size=0.2, stratify=y)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    lb = zLabelBinarizer()
    y_test = lb.fit_transform(y_test)
    y_train = lb.transform(y_train)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    cl = zMLPClassifier(random_state=2, hidden_layer_sizes=(2,), activation="logistic", batch_size=10, output_activation="softmax", alpha=0.1, max_iter=100, learning_rate_init=0.00001)
    cl.fit(X_train, y_train)
    print(cl.score(X_train, y_train))
    print(cl.score(X_test, y_test))

    import matplotlib.pyplot as plt 
    plt.plot(cl.loss_curve_)
    plt.show()


def run_regressor_test():
    X, y = load_iris(return_X_y=True)
    batch_size = 10
    activation = "identity" # works for logistic, tanh, identity
    learning_rate_init = 0.0001
    alpha = 0
    max_iter = 200
    nn = zMLPRegressor(hidden_layer_sizes=(3, 2), activation=activation, batch_size=batch_size, learning_rate_init=learning_rate_init, alpha=alpha, max_iter=max_iter)
    nn2 = MLPRegressor(hidden_layer_sizes=(3, 2), activation=activation, batch_size="auto", learning_rate_init=10*learning_rate_init, learning_rate="constant",
                          alpha=alpha, max_iter=max_iter, solver="sgd")
    sc = StandardScaler()
    X = sc.fit_transform(X)
    nn.fit(X, y) ; nn2.fit(X, y)
    nn.predict(X) ; nn2.fit(X, y)
    print(nn.score(X, y), nn2.score(X, y))
    
    import matplotlib.pyplot as plt 
    plt.plot(np.arange(len(nn.loss_curve_)), nn.loss_curve_, color="green", label="custom")
    plt.plot(np.arange(len(nn.loss_curve_), step=int(len(nn.loss_curve_)/len(nn2.loss_curve_)))[:len(nn2.loss_curve_)], nn2.loss_curve_, color="red", label="original")
    plt.legend(loc="best")
    plt.show()

    
if __name__ == "__main__":
    run_classifier_test()

# TODO: add get_params and set_params

def setup_batch_data(function, N=4, dimIn=2, dimOut=3):
    """Sets up batch data (N=4, dimIn=2, dimOut=3) and common parameters."""
    m = zMLPsuperclass()
    
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

if __name__ == '__main__' and False:
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
    m = zMLPsuperclass()
    
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
    m = zMLPsuperclass()
    
    # Common Inputs (N_in=3, N_out=2)
    a_in = np.array([0.5, 1.0, 0.2])
    W = np.array([[0.1, -0.3, 0.4], [0.5, 0.2, -0.1]])
    b = np.array([0.1, 0.1])
    dL_dAout = np.array([0.6, -0.4])

    # Calculate Z_out and A_out 
    z_out = a_in @ W.T + b 
    a_out = m.compute_pre_activation(z_out, activation_function)
    
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