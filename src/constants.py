import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BREAST_CANCER_FILE_NAME = 'uci-20070111-breast-cancer.csv'
SIMPLE_ALGORITHMS = {
    'chowLiu': 'chow-liu',
    'naive': 'exact'
}
WEKA_ALGORITHMS = {
    'geneticSearch': 'GeneticSearch',
    'hillClimber': 'HillClimber',
    'k2': 'K2',
    'lagdHillClimber': 'LAGDHillClimber',
    'localScoreSearchAlgorithm': 'LocalScoreSearchAlgorithm',
    'repeatedHillClimber': 'RepeatedHillClimber',
    'tabuSearch': 'TabuSearch',
    'simulatedAnnealing': 'SimulatedAnnealing'
}
ALL_ALGORITHMS = {**SIMPLE_ALGORITHMS, **WEKA_ALGORITHMS}
