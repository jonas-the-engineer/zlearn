from tree import zDecisionTreeClassifier, zDecisionTreeRegressor
import numpy as np
from sklearn.base import *
from sklearn.datasets import load_diabetes
from preprocessing import ztrain_test_split

class zRandomForestSuperclass():
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

class zRandomForestRegressor(RegressorMixin, BaseEstimator, zRandomForestSuperclass):
    def __init__(self, bootstrap_size: float = 0.5, num_features: int = None, random_state: int = 0, num_estimators: int = 10,
                 max_depth: int = None, min_samples_split: int = 2):
        self.bootstrap_size = bootstrap_size
        self.num_features = num_features
        self.random_state = random_state
        self.num_estimators = num_estimators
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self._regressors = []
        self._bootstrap_features = []
        for X_boot, y_boot, sample_features in self.bootstrap_set_generator(X, y, self.bootstrap_size, self.num_features, 
                                                                            self.random_state, self.num_estimators):
            dt = zDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            dt.fit(X_boot, y_boot)
            self._bootstrap_features.append(sample_features.tolist())
            self._regressors.append(dt)
    
    def predict(self, X):
        predictions = None 
        for regressor, features in zip(self._regressors, self._bootstrap_features):
            X_boot = X[:, features]
            y_pred = regressor.predict(X_boot)
            y_pred = y_pred.reshape(1, y_pred.shape[0])
            if(predictions is None):
                predictions = y_pred
            else:
                predictions = np.vstack([predictions, y_pred])
            # assert(predictions is not None)
        # assert(predictions is not None)
        avg_predictions = np.mean(predictions, axis=0)

        return avg_predictions

if __name__ == "__main__":
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = ztrain_test_split(X, y, test_size=0.2, random_state=1)
    print(X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    rf = zRandomForestRegressor(bootstrap_size=0.7, num_features=None, random_state=1, num_estimators=50, max_depth=5, min_samples_split=15)

    rf.fit(X_train, y_train)
    print(rf.score(X_train, y_train), rf.score(X_test, y_test))
    
# TODO : write RandomForestclassifier + move more functionality up to the superclass
# TODO : add set params and get params method
# TODO : add more hyperparameters
# TODO : docstrings