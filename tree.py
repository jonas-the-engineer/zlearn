from sklearn.base import *
from preprocessing import zLabelEncoder
import numpy as np

if __name__ == "__main__":
    from preprocessing import ztrain_test_split, zStandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris, load_diabetes
    from sklearn.tree import *

class zDecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2, class_weight='balanced'):
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split
        self.class_weight = class_weight

    def fit(self, X, y):
        self.label_encoder_ = zLabelEncoder()
        y_transf = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.class_weights_ = np.array([1.0 for label in self.classes_])
        if(self.class_weight == "balanced"):
            self.class_weights_ = np.array([1.0 / ( (y_transf == class_num).sum() )   for class_num in range(len(self.classes_))])

        self.root_node_ = self.fit_rec(X, y_transf, 0)
        return self

    def predict(self, X):
        pred_func = lambda x: self.predict_single(x)
        vec_pred_func = np.vectorize(pred_func, signature='(n)->()')
        y_pred = vec_pred_func(X)
        y_pred_inverse_transformed = self.label_encoder_.inverse_transform(y_pred)
        return y_pred_inverse_transformed

    def predict_single(self, x):
        return self.predict_single_rec(x, self.root_node_)

    def predict_single_rec(self, x, root):
        we_predict_left = x[root.decision_feature_index] <= root.linear_feature_upper_threshold_left_child
        
        if(we_predict_left):
            if(root.get_left_child() is None):
                return root.prediction_left_
            return self.predict_single_rec(x, root.get_left_child())
        else:
            if(root.get_right_child() is None):
                return root.prediction_right_
            return self.predict_single_rec(x, root.get_right_child())


    def print_decision_tree(self):
        self.print_decision_tree_rec(self.root_node_, 0)

    def print_decision_tree_rec(self, root, level):
        if(root == None):
            return
        print("    " * level, "Feature: ", root.decision_feature_index, end = "  ")
        print("Type: Linear: ", root.linear_feature_upper_threshold_left_child)

        print()
        print("    " * level, "LEFT")
        self.print_decision_tree_rec(root.get_left_child(), level+1)
        print()
        print("    " * level, "RIGHT")
        self.print_decision_tree_rec(root.get_left_child(), level+1)

    def fit_rec(self, X, y, current_depth):
        best_node = None
        node = None
        if((self.max_depth is not None) and current_depth >= self.max_depth or X.shape[0] < self.min_samples_split or len(np.unique(y)) == 1):
            assert(current_depth != 0)
            return None
        
        current_best_childs_gini_loss = None 
        for feature_index in range(X.shape[1]):
            x = X[:,feature_index]
            x = np.sort(x)
            x_splitting_options = 0.5 * (x[1:] + x[:-1])
            for x_splitting in x_splitting_options:
                node = zDecisionTreeClassifier.Node(
                    class_weights=self.class_weights_,
                    decision_feature_index=feature_index,
                    linear_feature_upper_threshold_left_child=x_splitting,
                )
                childs_gini_loss_sum = node.sum_childs_gini_loss(X, y)
                if(current_best_childs_gini_loss is None or childs_gini_loss_sum < current_best_childs_gini_loss):
                    current_best_childs_gini_loss = childs_gini_loss_sum
                    best_node = node
            
        best_node.calc_and_save_prediction_left_and_prediction_right(X, y)
        assert(best_node != None)
        assert(best_node != None)
        
        current_depth += 1
        X_left, X_right, y_left, y_right = best_node.split(X, y)
        best_node.set_left_child(self.fit_rec(X_left, y_left, current_depth))
        best_node.set_right_child(self.fit_rec(X_right, y_right, current_depth))
        return best_node

    class Node():
        def __init__(self, class_weights, decision_feature_index, linear_feature_upper_threshold_left_child):
            self.decision_feature_index = decision_feature_index 
            self.linear_feature_upper_threshold_left_child = linear_feature_upper_threshold_left_child

            self.class_weights = class_weights

            self.left_child = None 
            self.right_child = None

        def split(self, X, y):
            x = X[:, self.decision_feature_index]
            X_left, X_right, y_left, y_right = None, None, None, None
            left_indices, right_indices = None, None

            left_indices = x <= self.linear_feature_upper_threshold_left_child
            right_indices = x > self.linear_feature_upper_threshold_left_child

            X_left = X[left_indices]
            X_right = X[right_indices]
            y_left = y[left_indices]
            y_right = y[right_indices]
            return X_left, X_right, y_left, y_right

        def gini_loss(self, y):
            if(y.shape[0] == 0):
                return 0.0
            assert(y.shape[0] > 0)
            class_weights = self.class_weights
            class_counts = [(y == num_class).sum() for num_class in range(len(class_weights))]
            weighted_class_counts = [weight * count for weight, count in zip(class_weights, class_counts)]
            sum_weighted_counts = sum(weighted_class_counts)
            assert(sum_weighted_counts > 0)
            squared_fractions = [(weighted_count / sum_weighted_counts)**2 for weighted_count in weighted_class_counts]
            return 1 - sum(squared_fractions)

        def sum_childs_gini_loss(self, X, y):
            X_left, X_right, y_left, y_right = self.split(X, y)
            return y_left.shape[0] * self.gini_loss(y_left) + self.gini_loss(y_right) * y_right.shape[0]
        
        def set_left_child(self, left_child):
            self.left_child = left_child
        
        def get_left_child(self):
            return self.left_child
        
        def set_right_child(self, right_child):
            self.right_child = right_child

        def get_right_child(self):
            return self.right_child
        
        def calc_and_save_prediction_left_and_prediction_right(self, X, y):
            X_left, X_right, y_left, y_right = self.split(X, y)
            
            class_weights = self.class_weights

            y_left_class_counts = [(y_left == num_class).sum() for num_class in range(len(class_weights))]
            y_left_weighted_class_counts = [weight * count for weight, count in zip(class_weights, y_left_class_counts)]

            y_right_class_counts = [(y_right == num_class).sum() for num_class in range(len(class_weights))]
            y_right_weighted_class_counts = [weight * count for weight, count in zip(class_weights, y_right_class_counts)]

            self.prediction_left_ = int(np.argmax(np.array(y_left_weighted_class_counts)))
            self.prediction_right_ = int(np.argmax(np.array(y_right_weighted_class_counts)))

class zDecisionTreeRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, max_depth=None, min_samples_split=2):
        # criterion = "gini"
        # splitter = "best"
        # I do not understand why to use a random state, because of deterministic behavior
        self.max_depth = max_depth 
        self.min_samples_split = min_samples_split

    def fit(self, X, y):
        self.root_node_ = self.fit_rec(X, y, 0)
        return self

    def predict(self, X):
        pred_func = lambda x: self.predict_single(x)
        vec_pred_func = np.vectorize(pred_func, signature='(n)->()')
        y_pred = vec_pred_func(X)
        return y_pred
    
    def predict_single(self, x):
        return self.predict_single_rec(x, self.root_node_)

    def predict_single_rec(self, x, root):
        we_predict_left = x[root.decision_feature_index] <= root.linear_feature_upper_threshold_left_child

        if(we_predict_left):
            if(root.get_left_child() is None):
                return root.prediction_left_
            return self.predict_single_rec(x, root.get_left_child())
        else:
            if(root.get_right_child() is None):
                return root.prediction_right_
            return self.predict_single_rec(x, root.get_right_child())


    def print_decision_tree(self):
        self.print_decision_tree_rec(self.root_node_, 0)

    def print_decision_tree_rec(self, root, level):
        if(root == None):
            return
        print("    " * level, "Feature: ", root.decision_feature_index, end = "  ")
        print("Type: Linear: ", root.linear_feature_upper_threshold_left_child)


        print()
        print("    " * level, "LEFT")
        self.print_decision_tree_rec(root.get_left_child(), level+1)
        print()
        print("    " * level, "RIGHT")
        self.print_decision_tree_rec(root.get_left_child(), level+1)

    def fit_rec(self, X, y, current_depth):
        best_node = None
        node = None
        if((self.max_depth is not None) and current_depth >= self.max_depth or X.shape[0] < self.min_samples_split or len(np.unique(y)) == 1):
            assert(current_depth != 0)
            return None
        
        current_best_childs_gini_loss = None 
        for feature_index in range(X.shape[1]):
            x = X[:,feature_index]
            x = np.sort(x)
            x_splitting_options = 0.5 * (x[1:] + x[:-1])
            for x_splitting in x_splitting_options:
                node = zDecisionTreeRegressor.Node(
                    # curr_feature_is_categorical=False,
                    decision_feature_index=feature_index,
                    # categorical_feature_labels_left_child=None,
                    linear_feature_upper_threshold_left_child=x_splitting,
                )
                childs_gini_loss_sum = node.sum_childs_error(X, y) # DIFF
                if(current_best_childs_gini_loss is None or childs_gini_loss_sum < current_best_childs_gini_loss):
                    current_best_childs_gini_loss = childs_gini_loss_sum
                    best_node = node
            
        best_node.calc_and_save_prediction_left_and_prediction_right(X, y) # DIFF
        assert(best_node != None)
        assert(best_node != None)
        
        current_depth += 1
        X_left, X_right, y_left, y_right = best_node.split(X, y)
        best_node.set_left_child(self.fit_rec(X_left, y_left, current_depth))
        best_node.set_right_child(self.fit_rec(X_right, y_right, current_depth))
        return best_node

    class Node():
        def __init__(self, decision_feature_index, linear_feature_upper_threshold_left_child):
            self.decision_feature_index = decision_feature_index 
            self.linear_feature_upper_threshold_left_child = linear_feature_upper_threshold_left_child

            self.left_child = None 
            self.right_child = None

        def split(self, X, y):
            x = X[:, self.decision_feature_index]
            X_left, X_right, y_left, y_right = None, None, None, None
            left_indices, right_indices = None, None

            left_indices = x <= self.linear_feature_upper_threshold_left_child
            right_indices = x > self.linear_feature_upper_threshold_left_child

            X_left = X[left_indices]
            X_right = X[right_indices]
            y_left = y[left_indices]
            y_right = y[right_indices]
            return X_left, X_right, y_left, y_right

        def mean_squared_error_mean_estimator(self, y): # DIFF
            if(y.shape[0] == 0):
                return 0.0
            assert(y.shape[0] > 0)
            mean = np.mean(y, axis=0)
            squared_errors = (y- mean)**2
            return np.mean(squared_errors)

        def sum_childs_error(self, X, y):  # DIFF
            X_left, X_right, y_left, y_right = self.split(X, y)
            return y_left.shape[0] * self.mean_squared_error_mean_estimator(y_left) + self.mean_squared_error_mean_estimator(y_right) * y_right.shape[0]
        
        def set_left_child(self, left_child):
            self.left_child = left_child
        
        def get_left_child(self):
            return self.left_child
        
        def set_right_child(self, right_child):
            self.right_child = right_child

        def get_right_child(self):
            return self.right_child
        
        def calc_and_save_prediction_left_and_prediction_right(self, X, y):
            X_left, X_right, y_left, y_right = self.split(X, y)

            self.prediction_left_ = np.mean(y_left)
            self.prediction_right_ = np.mean(y_right) 

if __name__ == "__main__":
    dt = zDecisionTreeClassifier(max_depth=100, min_samples_split=2)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = ztrain_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    print(X.shape, y.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    dt.fit(X_train, y_train)
    # dt.print_decision_tree()
    print(dt.predict(X_train) == y_train)
    print(dt.score(X_train, y_train))
    print(dt.score(X_test, y_test))

    # my own class is compatible with sklearn :)
    pip = make_pipeline(zStandardScaler(), zDecisionTreeClassifier(max_depth=5, min_samples_split=5))
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
    
    import matplotlib.pyplot as plt 

    plt.plot(np.arange(0, y.shape[0], 1), y)
    plt.plot(np.arange(0, y.shape[0], 1), y_pred)

    plt.tight_layout()
    plt.show()

# TODO : write a superclass to remove duplicated code
# TODO : add set params and get params method