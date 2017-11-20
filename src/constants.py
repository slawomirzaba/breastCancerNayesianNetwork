import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
BREAST_CANCER_FILE_NAME = 'uci-20070111-breast-cancer.csv'
SIMPLE_ALGORITHMS = {
    'chowLiu': {
        'name': 'chow-liu',
        'parameters': ''
    },
    'naive': {
        'name': 'exact',
        'parameters': ''
    }
}
WEKA_ALGORITHMS = {
    'hillClimber': {
        'name': 'HillClimber',
        'parameters': '-P 1'
    },
    'k2': {
        'name': 'K2',
        'parameters': '-P 1'
    },
    'lagdHillClimber': {
        'name': 'LAGDHillClimber',
        'parameters': '-L 2 -G 5 -P 1'
    },
    'repeatedHillClimber': {
        'name': 'RepeatedHillClimber',
        'parameters': '-U 10 -A 1 -P 1'
    },
    'tabuSearch': {
        'name': 'TabuSearch',
        'parameters': '-L 5 -U 10 -P 1'
    },
    'simulatedAnnealing': {
        'name': 'SimulatedAnnealing',
        'parameters': '-A 10.0 -U 10000 -D 0.999 -R 1'
    }
}
ALL_ALGORITHMS = {**SIMPLE_ALGORITHMS, **WEKA_ALGORITHMS}
