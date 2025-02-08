from sklearn.model_selection import train_test_split


class DataSplitter:
    def __init__(self, train_size=0.6, test_size=0.2, val_size=0.2, random_state=42):
        if train_size + test_size + val_size != 1.0:
            raise ValueError(
                "The sum of train, test, and validation proportions must equal 1.0"
            )
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

    def split_data(self, X, y):
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, train_size=self.train_size, random_state=self.random_state
        )
        test_val_ratio = self.test_size / (self.test_size + self.val_size)
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=test_val_ratio, random_state=self.random_state
        )
        return X_train, X_test, X_val, y_train, y_test, y_val
