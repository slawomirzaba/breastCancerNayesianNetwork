from DataProvider import DataProvider
from SimpleBayesNetwork import SimpleBayesNetwork
from WekaBayesNetwork import WekaBayesNetwork
from constants import ALL_ALGORITHMS
import weka.core.jvm as jvm
import traceback


def main():
    data_provider = DataProvider(test_size=0.2)
    jvm.start()
    trainig_data = data_provider.get_weka_training_data() # for weka
    try:
        weka_bayes_network = WekaBayesNetwork(trainig_data['labels'], ALL_ALGORITHMS['k2'])
        weka_bayes_network.train_bayes(trainig_data['train_set'])
        weka_bayes_network.draw_graph()
        results = weka_bayes_network.predict_and_compare(trainig_data['test_set'])

        # trainig_data = data_provider.get_training_data() # for simple
        # chow_liu_bayes_network = SimpleBayesNetwork(trainig_data['feature_names'], ALL_ALGORITHMS['chowLiu'])
        # chow_liu_bayes_network.train_bayes(trainig_data['x_train'], trainig_data['y_train'])
        # chow_liu_bayes_network.draw_graph()
        # results = chow_liu_bayes_network.predict_and_compare(trainig_data['x_test'], trainig_data['y_test'])
     
        print('correct_recurrences:', results['correct_recurrences'])
        print('correct_no_recurrences:', results['correct_no_recurrences'])
        print('incorrect_recurrences:', results['incorrect_recurrences'])
        print('incorrect_no_recurrences:', results['incorrect_no_recurrences'])
    except:
        print('Exception was caught!')
        tb = traceback.format_exc()
        print(tb)

    jvm.stop()

main()
