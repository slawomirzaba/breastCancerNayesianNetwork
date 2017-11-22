from pomegranate import BayesianNetwork
from matplotlib import pyplot
from sklearn import preprocessing
from copy import deepcopy
from constants import SIMPLE_ALGORITHMS
from IBayesNetwork import IBayesNetwork
import numpy
import networkx


class SimpleBayesNetwork(IBayesNetwork):
    def __init__(self, feature_names, algorithm_data):
        super().__init__(algorithm_data)
        
        if algorithm_data['name'] not in [value['name'] for value in SIMPLE_ALGORITHMS.values()]:
            raise Exception('Unsupported algorithm in SIMPLE_ALGORITHMS!!')

        self.state_names = deepcopy(feature_names)
        self.state_names.insert(0, "label")
        self.labelEncoder = preprocessing.LabelEncoder()
        self.formatted_labels = None

    def train_bayes(self, x_train, y_train):
        self.__format_labels(y_train)
        X = numpy.concatenate((self.formatted_labels, x_train), axis=1)
        self.model = {
            SIMPLE_ALGORITHMS['chowLiu']['name']: self.__chowLiu_algorithm,
            SIMPLE_ALGORITHMS['naive']['name']: self.__naive_algorithm
        }[self.algorithm_name](X)

    def predict_and_compare(self, x_test, y_test):
        if not self.model:
            raise Exception('Model must be built!!')
        predictions = []

        for sample in x_test:
            mapped_sample = dict(zip(self.state_names[1:], sample))
            beliefs = self.model.predict_proba(mapped_sample, check_input=False)
            graph = dict(zip([state.name for state in self.model.states], beliefs))
            label_probabilities = graph['label'].parameters[0]
            predictions.append(self.labelEncoder.classes_[self.__get_probable_class(label_probabilities)])

        return self.get_compare_results(predictions, y_test)

    def draw_graph(self):
        if not self.model:
            raise Exception('Model must be built!!')

        pyplot.figure()
        self.model.plot()
        pyplot.show()

    def __chowLiu_algorithm(self, X):
        return BayesianNetwork.from_samples(X, algorithm=self.algorithm_name, state_names=self.state_names, root=0)

    def __naive_algorithm(self, X):
        graph = networkx.DiGraph()
        for i in range(1, len(self.state_names)):
            graph.add_edge((0,), (i,))
        return BayesianNetwork.from_samples(X, algorithm=self.algorithm_name, state_names=self.state_names, root=0,
                                            constraint_graph=graph)

    def get_compare_results(self, predictions, correct_results):
        return super().get_compare_results(predictions, correct_results)

    def __get_probable_class(self, probabilites):
        values = list(probabilites.values())
        keys = list(probabilites.keys())
        return int(keys[values.index(max(values))])

    def __format_labels(self, y_train):
        self.formatted_labels = self.labelEncoder.fit_transform(y_train)
        self.formatted_labels = self.formatted_labels.reshape(self.formatted_labels.shape[0], 1)
