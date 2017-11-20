from constants import WEKA_ALGORITHMS
from weka.core.classes import from_commandline
from xml_to_dot import parse_xml_to_dot
import weka.plot.graph as plot_graph
import os

class WekaBayesNetwork():
    def __init__(self, labels, algorithm_data):
        self.cmdline = 'weka.classifiers.bayes.BayesNet -D -Q weka.classifiers.bayes.net.search.local.{} -- {} -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5'
        self.cmdline = self.cmdline.format(algorithm_data['name'], algorithm_data['parameters'])
        self.algorithm_name = algorithm_data['name'];
        self.labels = labels
        self.model = None

    def train_bayes(self, train_set):
        self.model = from_commandline(self.cmdline, classname="weka.classifiers.Classifier")
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
        file_name = '{}.dot'.format(self.algorithm_name)
        dot_file_content = parse_xml_to_dot(self.model.graph)
        file = open(file_name, 'w')
        file.write(dot_file_content)
        file.close()
        plot_graph.plot_dot_graph(file_name)
        os.remove(file_name)

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
