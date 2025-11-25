from sklearn.base import *
from preprocessing import zLabelEncoder
import numpy as np

if __name__ == "__main__":
    from preprocessing import ztrain_test_split, zStandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import load_iris
    from sklearn.tree import *

def _number_to_binary_list(number: int, limit: int) -> list[int]:
    if not (0 <= number < (2 ** limit)):
        raise ValueError(f"Number {number} is outside the valid range [0, {2**limit - 1}] for limit={limit}.")

    binary_list = [0] * limit
    
    for i in range(limit):
        bit_value = (number >> i) & 1
        
        binary_list[limit - 1 - i] = bit_value
        
    return binary_list

def _all_combinations(label_list):
    """
    print(all_combinatorics(["a", "b", "c"]))
    
    [[], ['c'], ['b'], ['b', 'c'], ['a'], ['a', 'c'], ['a', 'b'], ['a', 'b', 'c']]
    """
    output = []
    num_labels = len(label_list)
    for num_combination in range(2**num_labels):
        part_of_output = []
        binary_list = _number_to_binary_list(num_combination, num_labels)
        for contains, label in zip(binary_list, label_list):
            if(contains == 1):
                part_of_output.append(label)
        output.append(part_of_output)
    return output

class zDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, max_depth=None, min_samples_split=2, class_weight='balanced'):
        # criterion = "gini"
        # splitter = "best"
        # I do not understand why to use a random state, because of deterministic behavior
        self.max_depth = max_depth 
        self.min_samples_split = 2
        self.class_weight = class_weight

    def fit(self, X, y):
        self.label_encoder_ = zLabelEncoder()
        y_transf = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.class_weights_ = np.array([1.0 for label in self.classes_])
        if(self.class_weight == "balanced"):
            self.class_weights_ = np.array([1.0 / ( (y_transf == class_num).sum() )   for class_num in range(len(self.classes_))])

        self.feature_is_categorical_ = [type(x_feat) == str for x_feat in list(X[0,:])]

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
        we_predict_left = True
        if(root.curr_feature_is_categorical):
            we_predict_left = x[root.decision_feature_index] in root.categorical_feature_labels_left_child
        else:
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
        if(self.feature_is_categorical_[root.decision_feature_index]):
            print("Type: Categorical: ", root.categorical_feature_labels_left_child)
        else:
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
        if(current_depth >= self.max_depth or X.shape[0] < self.min_samples_split or len(np.unique(y)) == 1):
            if(current_depth == 0):
                print("ERROR: ", current_depth >= self.max_depth , X.shape[0] < self.min_samples_split ,len(np.unique(y)) == 1)
                print("ERROR: ", current_depth ,self.max_depth , X.shape[0] , self.min_samples_split ,len(np.unique(y)) )
            return None
        
        current_best_childs_gini_loss = None 
        for feature_index, current_feature_categorical in enumerate(self.feature_is_categorical_):
            if(current_feature_categorical):
                label_options = np.unique(X[:,feature_index])
                all_label_combinations = _all_combinations(label_list=label_options)
                for label_combination in all_label_combinations:
                    node = zDecisionTreeClassifier.Node(
                        class_weights=self.class_weights_,
                        curr_feature_is_categorical=True,
                        decision_feature_index=feature_index,
                        categorical_feature_labels_left_child=label_combination,
                        linear_feature_upper_threshold_left_child=0
                    )
                    childs_gini_loss_sum = node.sum_childs_gini_loss(X, y)
                    if(current_best_childs_gini_loss is None or childs_gini_loss_sum < current_best_childs_gini_loss):
                        current_best_childs_gini_loss = childs_gini_loss_sum
                        best_node = node
            else:
                x = X[:,feature_index]
                x = np.sort(x)
                x_splitting_options = 0.5 * (x[1:] + x[:-1])
                for x_splitting in x_splitting_options:
                    node = zDecisionTreeClassifier.Node(
                        class_weights=self.class_weights_,
                        curr_feature_is_categorical=False,
                        decision_feature_index=feature_index,
                        categorical_feature_labels_left_child=None,
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
        def __init__(self, class_weights, curr_feature_is_categorical, 
                     decision_feature_index, categorical_feature_labels_left_child, linear_feature_upper_threshold_left_child):
            self.decision_feature_index = decision_feature_index 
            self.categorical_feature_labels_left_child = categorical_feature_labels_left_child
            self.linear_feature_upper_threshold_left_child = linear_feature_upper_threshold_left_child

            self.class_weights = class_weights
            self.curr_feature_is_categorical = curr_feature_is_categorical

            self.left_child = None 
            self.right_child = None

        def split(self, X, y):
            x = X[:, self.decision_feature_index]
            X_left, X_right, y_left, y_right = None, None, None, None
            left_indices, right_indices = None, None

            if(self.curr_feature_is_categorical):
                left_indices = x in self.categorical_feature_labels_left_child
                right_indices = x not in self.categorical_feature_labels_left_child
            else:
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
    print(pip.score(X_train, y_train))
    print(pip.score(X_test, y_test))