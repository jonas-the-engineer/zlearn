from sklearn.base import *
import numpy as np

class zLabelEncoder(TransformerMixin, BaseEstimator):
    def fit(self, y):
        self.classes_ = np.unique(y)
        return self

    def transform(self, y):
        classes_dict = {key: value for value, key in enumerate(self.classes_)}

        func = lambda orig: classes_dict[orig]
        vec_func = np.vectorize(func)
        y_transformed = vec_func(y)
        return y_transformed 

    def inverse_transform(self, y):
        func = lambda num: self.classes_[num]
        vec_func = np.vectorize(func)
        y_transformed = vec_func(y)
        return y_transformed 
    
class zStandardScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.means_ = np.mean(X, axis=0)
        self.stds_ = np.std(X, axis=0)
        return self 
    
    def transform(self, X):
        X_std = X - self.means_
        X_std = X_std / self.stds_
        return X_std 

    def inverse_transform(self, X_std):
        X = X_std * self.stds_
        X = X + self.means_ 
        return X
    
class zMinMaxScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.means_ = np.mean(X, axis=0)
        self.dividers_ = np.max(X, axis=0) - np.min(X, axis=0)
        return self 
    
    def transform(self, X):
        X_std = X - self.means_
        X_std = X_std / self.dividers_
        return X_std 

    def inverse_transform(self, X_std):
        X = X_std * self.dividers_
        X = X + self.means_ 
        return X
    
class zRobustScaler(OneToOneFeatureMixin, TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.means_ = np.median(X, axis=0)
        self.dividers_ = np.quantile(X, 0.75, axis=0)- np.quantile(X, 0.25, axis=0)
        return self 
    
    def transform(self, X):
        X_robust = X - self.means_
        X_robust = X_robust / self.dividers_
        return X_robust 

    def inverse_transform(self, X_robust):
        X = X_robust * self.dividers_
        X = X + self.means_ 
        return X
    
def ztrain_test_split(*arrays, test_size: float, random_state=0, stratify=None):
    """
    Docstring for train_test_split: shuffle is always true
    
    :param arrays: Description
    :param test_size: Description
    :type test_size: float
    :param random_state: Description
    :param stratify: Description
    """


    def cumulative_sum(list):
        new_list = []
        sum = 0
        for element in list:
            new_list.append(sum)
            sum += element
        return new_list, sum
    
    if(stratify is None):
        stratify = np.ones(shape=(arrays[0].shape[0]))

    rng = np.random.default_rng(seed=random_state)

    # permutation = np.random.permutation(np.arange(0, stratify.shape[0], 1, dtype=np.int32))
    permutation = rng.permutation(stratify.shape[0])
    arrays = [array[permutation] for array in arrays]
    stratify = stratify[permutation]

    sorting_order = np.argsort(stratify)
    stratify = stratify[sorting_order]
    arrays = [array[sorting_order] for array in arrays]

    train_size = 1.0 - test_size
    # train_size_num = train_size * arrays[0].shape[0]
    # test_size_num  = test_size  * arrays[0].shape[0]
    
    unique_y = np.unique(stratify) # I assume that this list is sorted by their first occurence
    unique_y_count = [(stratify == y).sum() for y in unique_y]
    unique_y_test_count = [int(y_count * test_size) for y_count in unique_y_count]
    unique_y_train_count = [int(y_count * train_size) for y_count in unique_y_count]
    unique_y_train_count = [int(y_total - y_test) for y_total, y_test in zip(unique_y_count, unique_y_test_count)]

    test_cum_sum , test_count_sum  = cumulative_sum(unique_y_test_count)
    train_cum_sum, train_count_sum = cumulative_sum(unique_y_train_count)

    # create ready to fill outputs
    tests = []
    trains = []
    for array in arrays:
        shape = array.shape
        train_shape, test_shape = None, None
        if(len(shape) == 1):
            train_shape, test_shape = (train_count_sum,), (test_count_sum,)
        else:
            train_shape, test_shape = (train_count_sum,) + shape[1:], (test_count_sum,) + shape[1:]
        test = np.ndarray(shape=test_shape)
        train = np.ndarray(shape=train_shape)
        tests.append(test)
        trains.append(train)

    # fill the output
    offset = 0
    test_offset = 0
    train_offset = 0
    for num_in_test, num_in_train, num_total in zip(unique_y_test_count, unique_y_train_count, unique_y_count):
        assert(num_in_test + num_in_train == num_total)
        for test, array in zip(tests, arrays):
            test[test_offset:test_offset+num_in_test] = array[offset:offset+num_in_test]
        for train, array in zip(trains, arrays):
            train[train_offset:train_offset+num_in_train] = array[offset+num_in_test:offset+num_in_test+num_in_train]
        offset += num_total
        test_offset += num_in_test
        train_offset += num_in_train

    # post_permutation_test = np.random.permutation(np.arange(0, test_count_sum,  1, dtype=np.int32))
    # post_permutation_train= np.random.permutation(np.arange(0, train_count_sum, 1, dtype=np.int32))

    post_permutation_test = rng.permutation(test_count_sum)
    post_permutation_train = rng.permutation(train_count_sum)

    output = []
    for i in range(len(tests)):
        tests[i] = tests[i][post_permutation_test]
        trains[i] = trains[i][post_permutation_train]
        output.append(trains[i])
        output.append(tests[i])

    return output

if __name__ == "__main__":
    le = zLabelEncoder()
    y = np.array(["A", "B", "A", "C", "B", "A", "A", "A"])
    print(y)
    le.fit(y)
    y_transformed = le.transform(y)
    y_transf_itransf = le.inverse_transform(y_transformed)
    print(y_transformed)
    print(y_transf_itransf)
