import abc


class BaseDataset(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_train_dataset(self, train_x, train_labels):
        pass

    @abc.abstractmethod
    def get_test_dataset(self, test_x, train_labels):
        pass