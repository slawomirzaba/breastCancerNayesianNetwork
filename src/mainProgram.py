from DataProvider import DataProvider
from BayesNetwork import BayesNetwork
from constants import ALGHORITHM_TYPES

def main():
    data_provider = DataProvider(test_size=0.2)
    trainig_data = data_provider.get_training_data()
    chow_liu_bayes_network = BayesNetwork(trainig_data['feature_names'], ALGHORITHM_TYPES['chowLiu'])
    naive_bayes_network = BayesNetwork(trainig_data['feature_names'], ALGHORITHM_TYPES['naive'])

    chow_liu_bayes_network.train_and_test_bayes(trainig_data['x_train'], trainig_data['y_train'])
    results = chow_liu_bayes_network.predict_and_compare(trainig_data['x_test'], trainig_data['y_test'])

    print('correct_recurrences:', results['correct_recurrences'])
    print('correct_no_recurrences:', results['correct_no_recurrences'])
    print('incorrect_recurrences:', results['incorrect_recurrences'])
    print('incorrect_no_recurrences:', results['incorrect_no_recurrences'])
    #naive_bayes_network.train_and_test_bayes(trainig_data['x_train'], trainig_data['y_train'])

    #chow_liu_bayes_network.draw_graph()
    #naive_bayes_network.draw_graph()

main()
