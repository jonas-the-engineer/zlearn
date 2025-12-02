from tree import zDecisionTreeClassifier, zDecisionTreeRegressor, zDecisionTreeSuperclass
import numpy as np
from sklearn.base import *
from sklearn.datasets import load_diabetes, make_regression, make_classification
from preprocessing import ztrain_test_split
from abc import ABC, abstractmethod

class zRandomForestSuperclass(ABC):
    def __init__(self, bootstrap_size: float = 0.5, num_features: int = None, random_state: int = 0, num_estimators: int = 10,
                 max_depth: int = None, min_samples_split: int = 2):
        self.bootstrap_size = bootstrap_size
        self.num_features = num_features
        self.random_state = random_state
        self.num_estimators = num_estimators
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self._models = []
        self._bootstrap_features = []
        for X_boot, y_boot, sample_features in self.bootstrap_set_generator(X, y, self.bootstrap_size, self.num_features, 
                                                                            self.random_state, self.num_estimators):
            dt = self.generate_new_tree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            assert(isinstance(dt, zDecisionTreeSuperclass))
            dt.fit(X_boot, y_boot)
            self._bootstrap_features.append(sample_features.tolist())
            self._models.append(dt)

    def bootstrap_set_generator(self, 
        X: np.ndarray, y: np.ndarray,
        bootstrap_size: float = 0.5, num_features: int = None,
        random_state: int = 0,
        num_estimators: int = 10):
        """
        Generates bootstrap samples (X_boot, y_boot) and feature indices (sample_features)
        for use in ensemble methods while maintaining low memory usage via a generator.
        """
        if(num_features is None):
            num_features = X.shape[1]
        assert(bootstrap_size > 0.0 and bootstrap_size <= 1.0)
        assert(num_features <= X.shape[1] and num_features > 0)
        size_of_bootstrap = int(X.shape[0] * bootstrap_size)
        rng = np.random.default_rng(seed=random_state)
        for _ in range(num_estimators):
            sample_indices = rng.integers(low=0, high=X.shape[0], size=size_of_bootstrap)
            all_features_permutation = rng.permutation(X.shape[1])
            sample_features = all_features_permutation[0:num_features]

            X_my_features = X[:, sample_features]

            X_boot, y_boot = X_my_features[sample_indices, :], y[sample_indices]

            yield X_boot, y_boot, sample_features # yield improves memory efficiency

    def predict_all_models(self, X):
        """
        returns predictions from all models in shape [num_estimators, samples]
        """
        predictions = None 
        for model, features in zip(self._models, self._bootstrap_features):
            X_boot = X[:, features]
            assert(isinstance(model, zDecisionTreeSuperclass))
            y_pred = model.predict(X_boot)
            assert(isinstance(y_pred, np.ndarray))
            # y_pred = y_pred.reshape(1, y_pred.shape[0])
            y_pred = y_pred.reshape((1,)+y_pred.shape)
            if(predictions is None):
                predictions = y_pred
            else:
                predictions = np.vstack([predictions, y_pred])
        return predictions
    
    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def generate_new_tree(self, max_depth: int, min_samples_split: int):
        pass

class zRandomForestRegressor(RegressorMixin, BaseEstimator, zRandomForestSuperclass):
    def generate_new_tree(self, max_depth: int, min_samples_split: int):
        dt = zDecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split)
        return dt
    
    def predict(self, X):
        predictions = self.predict_all_models(X)
        avg_predictions = np.mean(predictions, axis=0)
        return avg_predictions
    
class zRandomForestClassifier(ClassifierMixin, BaseEstimator, zRandomForestSuperclass):
    """
    Only useful for classification with more than 2 possible labels, not for multiclass_classification
    """
    def generate_new_tree(self, max_depth, min_samples_split):
        dt = zDecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
        return dt
    
    def predict(self, X):
        # [num_estimators, samples]
        predictions = self.predict_all_models(X)

        # [samples, num_estimators]
        predictions = np.swapaxes(predictions, 0, 1)

        # TODO: vectorize the following

        vote_predictions = np.zeros(shape=predictions.shape[0], dtype=np.int32)
        for i in range(predictions.shape[0]):
            unique_labels, counts = np.unique(predictions[i,:], return_counts=True)
            vote_predictions[i] = unique_labels[np.argmax(counts)]

        return vote_predictions


def test_random_forest_regression():
    X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = ztrain_test_split(X, y, test_size=0.2, random_state=1)
    print(X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    dt = zRandomForestRegressor(max_depth=6, min_samples_split=5, num_estimators=20, bootstrap_size=0.5, random_state=0, num_features=8)
    dt.fit(X_train, y_train)
    print(dt.score(X_train, y_train))
    print(dt.score(X_test, y_test))  # both scores are greater than 0 -> the dt predicts valuable information, but is not a good predictor as a single tree

    y_pred = dt.predict(X)

    sorting = np.argsort(y)
    
    import matplotlib.pyplot as plt 

    plt.plot(np.arange(0, y.shape[0], 1), y[sorting])
    plt.plot(np.arange(0, y.shape[0], 1), y_pred[sorting])

    plt.tight_layout()
    plt.show()

def test_random_forest_classification():
    X, y = make_classification(n_samples=500, n_features=10, n_classes=4, n_informative=5)
    X_train, X_test, y_train, y_test = ztrain_test_split(X, y, test_size=0.2, random_state=1)
    print(X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    dt = zRandomForestClassifier(max_depth=6, min_samples_split=5, num_estimators=20, bootstrap_size=0.5, random_state=0, num_features=8)
    dt.fit(X_train, y_train)
    print(dt.score(X_train, y_train))
    print(dt.score(X_test, y_test))  # both scores are greater than 0 -> the dt predicts valuable information, but is not a good predictor as a single tree

    y_pred = dt.predict(X)

    sorting = np.argsort(y)
    
    import matplotlib.pyplot as plt 

    plt.plot(np.arange(0, y.shape[0], 1), y[sorting])
    plt.plot(np.arange(0, y.shape[0], 1), y_pred[sorting])

    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    test_random_forest_classification()

# TODO : write RandomForestclassifier + move more functionality up to the superclass
# TODO : add set params and get params method
# TODO : add more hyperparameters
# TODO : docstrings