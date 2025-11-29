from sklearn.base import *
from preprocessing import zLabelEncoder
import numpy as np
from abc import ABC, abstractmethod

if __name__ == "__main__":
    from preprocessing import ztrain_test_split, zStandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris, load_diabetes
    from sklearn.tree import *

class zDecisionTreeSuperclass(ABC):
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split

    def predict(self, X):
        pred_func = lambda x: self.predict_single(x)
        vec_pred_func = np.vectorize(pred_func, signature='(n)->()')
        y_pred = vec_pred_func(X)
        return y_pred
    
    def predict_single(self, x):
        return self.predict_single_rec(x, self.root_node_)

    def predict_single_rec(self, x, root):
        assert(isinstance(root, zDecisionTreeSuperclass.Node))
        we_predict_left = root.get_we_predict_left(x)

        if(we_predict_left):
            if(root.get_left_child() is None):
                return root.prediction_left_
            return self.predict_single_rec(x, root.get_left_child())
        else:
            if(root.get_right_child() is None):
                return root.prediction_right_
            return self.predict_single_rec(x, root.get_right_child())
        
    @abstractmethod
    def child_loss(self, y1, y2):
        pass

    @abstractmethod
    def compute_leaf_prediction(self, y_leaf):
        pass

    def fit(self, X, y):
        """
        This method assumes that the targets are numbers, no labels.
        """
        self.root_node_ = self.fit_rec(X, y, 0)
        return self

    def fit_rec(self, X, y, current_depth):
        best_node = None

        # check termination condition
        if( ((self.max_depth is not None) and current_depth >= self.max_depth) or 
            X.shape[0] < self.min_samples_split or 
            len(np.unique(y)) == 1):

            assert(current_depth != 0)
            return None
        
        current_best_childs_loss = None 
        losses_arr = []

        # iterate over all possible splitting thresholds for all possible features to select the best splitter (best_node)
        for feature_index in range(X.shape[1]):
            # get all possible splitting thresholds
            x = X[:,feature_index]
            x = np.sort(x)
            x_splitting_options = 0.5 * (x[1:] + x[:-1])
            # iterate over all possible splitting thresholds
            for x_splitting in x_splitting_options:
                node = zDecisionTreeSuperclass.Node(
                    decision_feature_index=feature_index,
                    linear_feature_upper_threshold_left_child=x_splitting,
                )
                _, _, y_left, y_right = node.split(X, y)
                childs_loss = y_left.shape[0] * self.child_loss(y_left) + y_right.shape[0] * self.child_loss(y_right)
                losses_arr.append(childs_loss)
                # check if new node is the new optimal node for splitting
                if(current_best_childs_loss is None or childs_loss < current_best_childs_loss):
                    current_best_childs_loss = childs_loss
                    best_node = node

        # ensure proper functionality
        assert(current_best_childs_loss is not None)    
        assert(best_node != None)
        
        # split the data using the best_node
        X_left, X_right, y_left, y_right = best_node.split(X, y)

        if(y_left.shape[0] == 0 or y_right.shape[0] == 0):
            # the best possible split does not split the data -> do not create new node
            return None

        # save the predictions of the best_node
        pred_left, pred_right = self.compute_leaf_prediction(y_left), self.compute_leaf_prediction(y_right)
        best_node.save_child_nodes_predictions(pred_left, pred_right)

        # build the tree doing recursive calls
        current_depth += 1
        
        print("Depth: ", current_depth - 1, " Decision feature ", best_node.decision_feature_index)
    
        best_node.set_left_child(self.fit_rec(X_left, y_left, current_depth))
        best_node.set_right_child(self.fit_rec(X_right, y_right, current_depth))
        return best_node

    class Node():
        def __init__(self, decision_feature_index, linear_feature_upper_threshold_left_child):
            self.decision_feature_index = decision_feature_index 
            self.linear_feature_upper_threshold_left_child = linear_feature_upper_threshold_left_child

            self.left_child = None 
            self.right_child = None

        def get_we_predict_left(self, x):
            return x[self.decision_feature_index] <= self.linear_feature_upper_threshold_left_child

        def split(self, X, y):
            x = X[:, self.decision_feature_index]

            left_indices = x <= self.linear_feature_upper_threshold_left_child
            right_indices = x > self.linear_feature_upper_threshold_left_child

            return X[left_indices], X[right_indices], y[left_indices], y[right_indices]
        
        def save_child_nodes_predictions(self, pred_left, pred_right):
            self.prediction_left_ = pred_left
            self.prediction_right_ = pred_right
        
        def set_left_child(self, left_child):
            self.left_child = left_child
        
        def get_left_child(self):
            return self.left_child
        
        def set_right_child(self, right_child):
            self.right_child = right_child

        def get_right_child(self):
            return self.right_child
        
class zDecisionTreeRegressor(RegressorMixin, BaseEstimator, zDecisionTreeSuperclass):
    def __init__(self, max_depth=None, min_samples_split=2):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)

    def compute_leaf_prediction(self, y_leaf):
        assert(y_leaf.shape[0] > 0)
        return np.mean(y_leaf)
    
    def child_loss(self, y_leaf):
        if(y_leaf.shape[0] == 0):
            return 0.0
        y_pred = np.mean(y_leaf)
        return np.mean((y_pred - y_leaf)**2)
    
class zDecisionTreeClassifier(ClassifierMixin, BaseEstimator, zDecisionTreeSuperclass):
    def __init__(self, max_depth=None, min_samples_split=2, class_weight: str = "balanced"):
        super().__init__(max_depth=max_depth, min_samples_split=min_samples_split)
        self.class_weight = class_weight

    def compute_leaf_prediction(self, y_leaf): 
        assert(y_leaf.shape[0] > 0)
        labels, counts = np.unique(y_leaf, return_counts=True)
        return labels[np.argmax(counts)] # return label with highest count
    
    def child_loss(self, y_leaf): # return gini loss
        # use gini loss
        if(y_leaf.shape[0] == 0):
            return 0.0

        class_counts = [(y_leaf == num_class).sum() for num_class in range(len(self.class_weights_))]
        weighted_class_counts = [weight * count for weight, count in zip(self.class_weights_, class_counts)]
        sum_weighted_counts = sum(weighted_class_counts)
        squared_fractions = [(weighted_count / sum_weighted_counts)**2 for weighted_count in weighted_class_counts]
        return 1 - sum(squared_fractions)

    def fit(self, X, y):
        self.label_encoder_ = zLabelEncoder()
        y_transf = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.class_weights_ = np.array([1.0 for label in self.classes_])
        if(self.class_weight == "balanced"):
            self.class_weights_ = np.array([1.0 / ( (y_transf == class_num).sum() )   for class_num in range(len(self.classes_))])

        super().fit(X, y_transf)
        return self

    def predict(self, X):
        y_pred = super().predict(X)
        y_pred_inverse_transformed = self.label_encoder_.inverse_transform(y_pred)
        return y_pred_inverse_transformed

if __name__ == "__main__":
    dt = zDecisionTreeClassifier(max_depth=5, min_samples_split=2)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = ztrain_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    print(X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    dt.fit(X_train, y_train)
    # dt.print_decision_tree()
    # print(dt.predict(X_train) == y_train)
    print(dt.score(X_train, y_train))
    print(dt.score(X_test, y_test))

    # my own class is compatible with sklearn :)
    # pip = make_pipeline(zStandardScaler(), zDecisionTreeClassifier(max_depth=5, min_samples_split=5))
    pip = zDecisionTreeClassifier2(max_depth=5, min_samples_split=2)
    pip.fit(X_train, y_train)
    print(pip.score(X_train, y_train)) # scores do not change, because feature scaling does not affect decision trees
    print(pip.score(X_test, y_test))

    print("------" * 5)

    X, y = load_diabetes(return_X_y=True)
    
    X_train, X_test, y_train, y_test = ztrain_test_split(X, y, test_size=0.2, random_state=1)
    print(X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    dt = zDecisionTreeRegressor(max_depth=4, min_samples_split=5)
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

# TODO : add set params and get params method
# TODO : add more hyperparameters