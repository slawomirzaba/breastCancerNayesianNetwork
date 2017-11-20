from constants import WEKA_ALGORITHMS
from weka.classifiers import Classifier
from IBayesNetwork import IBayesNetwork
import weka.plot.graph as graph


class WekaBayesNetwork(IBayesNetwork):
    def __init__(self, labels, algorithm_name):
        super().__init__(algorithm_name)
        
        if algorithm_name not in WEKA_ALGORITHMS.values():
            raise Exception('Unsupported algorithm in WEKA_ALGORITHMS!!')

        self.labels = labels

    def train_bayes(self, train_set):
        self.model = Classifier(classname="weka.classifiers.bayes.net.BayesNetGenerator")
        self.model.options = ['-Q', 'weka.classifiers.bayes.net.search.local.{}'.format(self.algorithm_name)]
        self.model.build_classifier(train_set)

    def predict_and_compare(self, test_set):
        if not self.model:
            raise Exception('Model must be built!!')

        predictions = []
        y_test = [sample.get_string_value(sample.class_index) for _, sample in enumerate(test_set)]
        for _, inst in enumerate(test_set):
            predictions.append(self.labels[int(self.model.classify_instance(inst))])

        return self.__get_compare_results(predictions, y_test)

    def draw_graph(self):
        if not self.model:
            raise Exception('Model must be built!!')

        # graph.plot_dot_graph(self.model.graph) drawing graph not working

    def __get_compare_results(self, predictions, correct_results):
        return super().__get_compare_results(predictions, correct_results)
