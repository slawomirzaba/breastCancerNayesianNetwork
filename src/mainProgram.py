from DataProvider import DataProvider
from bayesNetwork import train_and_test_bayes

def main():
    data_provider = DataProvider(test_size=0.2)
    trainig_data = data_provider.get_training_data()
    train_and_test_bayes(trainig_data['x_train'], trainig_data['feature_names'])

main()
