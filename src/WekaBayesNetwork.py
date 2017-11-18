from constants import WEKA_ALGORITHMS
from weka.classifiers import Classifier
import weka.plot.graph as graph

class WekaBayesNetwork():
    def __init__(self, labels, algorithm_name):
        if algorithm_name not in WEKA_ALGORITHMS.values():
            raise Exception('Unsupported algorithm!!')

        self.algorithm_name = algorithm_name
        self.labels = labels
        self.model = None

    def train_bayes(self, train_set):
        self.model = Classifier(classname="weka.classifiers.bayes.net.BayesNetGenerator")
        self.model.options = ['-Q', 'weka.classifiers.bayes.net.search.local.{}'.format(self.algorithm_name)]
        self.model.build_classifier(train_set)

    def predict_and_compare(self, test_set):
        if not self.model:
            raise Exception('Model must be builded!!')

        predictions = []
        y_test = [sample.get_string_value(sample.class_index) for _, sample in enumerate(test_set)]
        for _, inst in enumerate(test_set):
            predictions.append(self.labels[int(self.model.classify_instance(inst))])

        return self.__get_compare_results(predictions, y_test)

    def draw_graph(self):
        if not self.model:
            raise Exception('Model must be builded!!')

        # graph.plot_dot_graph(self.model.graph) drawing graph not working

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
