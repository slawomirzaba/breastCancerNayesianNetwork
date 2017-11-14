from pomegranate import BayesianNetwork
from matplotlib import pyplot
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from copy import deepcopy
from constants import ALGHORITHM_TYPES
import numpy
import networkx

class BayesNetwork():
    def __init__(self, feature_names, algorithm):
        if algorithm not in ALGHORITHM_TYPES.values():
            raise Exception('Unsupported algorithm!!')

        self.algorithm = algorithm
        self.state_names = deepcopy(feature_names)
        self.state_names.insert(0, "label")
        self.labelEncoder = preprocessing.LabelEncoder()
        self.formatted_labels, self.model = None, None

    def train_and_test_bayes(self, x_train, y_train):
        self.__format_labels(y_train)
        X = numpy.concatenate((self.formatted_labels, x_train), axis=1)
        self.model = {
            ALGHORITHM_TYPES['chowLiu']: self.__chowLiu_algorithm,
            ALGHORITHM_TYPES['naive']: self.__naive_algorithm
        }[self.algorithm](X)

    def predict_and_compare(self, x_test, y_test):
        if not self.model:
            raise Exception('Model must be builded!!')
        predictions = []

        for sample in x_test:
            mapped_sample = dict(zip(self.state_names[1:], sample))
            beliefs = self.model.predict_proba(mapped_sample, check_input=False)
            graph = dict(zip([state.name for state in self.model.states], beliefs))
            label_probabilities = graph['label'].parameters[0]
            predictions.append(self.labelEncoder.classes_[self.__get_probable_class(label_probabilities)])

        return self.__get_compare_results(predictions, y_test)

    def draw_graph(self):
        if not self.model:
            raise Exception('Model must be builded!!')

        pyplot.figure()
        self.model.plot()
        pyplot.show()

    def __chowLiu_algorithm(self, X):
        return BayesianNetwork.from_samples(X, algorithm=self.algorithm, state_names=self.state_names, root=0)

    def __naive_algorithm(self, X):
        graph = networkx.DiGraph()
        for i in range(1, len(self.state_names)):
            graph.add_edge((0,), (i,))
        return BayesianNetwork.from_samples(X, algorithm='exact', state_names=self.state_names, root=0,
                                            constraint_graph=graph)

    def __get_compare_results(self, predictions, correct_results):
        correct_recurrences = correct_no_recurrences = incorrect_recurrences = incorrect_no_recurrences = 0

        for x,y in zip(correct_results, predictions):
            if x=='recurrence-events' and y=='recurrence-events':
                correct_recurrences += 1
            elif x=='no-recurrence-events' and y=='no-recurrence-events':
                correct_no_recurrences += 1
            elif x=='recurrence-events' and y=='no-recurrence-events':
                incorrect_recurrences += 1
            else:
                incorrect_no_recurrences += 1

        return {
            'correct_recurrences': correct_recurrences,
            'correct_no_recurrences': correct_no_recurrences,
            'incorrect_recurrences': incorrect_recurrences,
            'incorrect_no_recurrences': incorrect_no_recurrences
        }

    def __get_probable_class(self, probabilites):
        values = list(probabilites.values())
        keys = list(probabilites.keys())
        return int(keys[values.index(max(values))])

    def __format_labels(self, y_train):
        self.formatted_labels = self.labelEncoder.fit_transform(y_train)
        self.formatted_labels = self.formatted_labels.reshape(self.formatted_labels.shape[0], 1)
