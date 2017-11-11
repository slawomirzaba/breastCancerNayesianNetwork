from pomegranate import BayesianNetwork
import matplotlib.pyplot as pyplot

def train_and_test_bayes(X_train, feature_names):
    model = BayesianNetwork.from_samples(X_train, algorithm='chow-liu', state_names=feature_names)
    pyplot.figure(figsize=(20, 8))
    model.plot()
    pyplot.show()
