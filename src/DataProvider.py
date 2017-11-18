from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from weka.core.converters import Loader
from weka.filters import Filter
import os
import csv
import constants

class DataProvider:
    def __init__(self, test_size):
        self.test_size = test_size

    def get_training_data(self):
        cancer_bread_data = self.__get_all_breast_cancer_data()
        data = cancer_bread_data['data'];
        target = cancer_bread_data['target'];
        feature_names = cancer_bread_data['feature_names'];
        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=self.test_size)
        return {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names
        }

    def get_weka_training_data(self):
        percentage_of_train_set = str(self.test_size * 100)
        options_for_train_set = ["-P", percentage_of_train_set]
        options_for_test_set = ["-P", percentage_of_train_set, "-V"]
        remove_percentage_filter_class = "weka.filters.unsupervised.instance.RemovePercentage"
        loader = Loader(classname="weka.core.converters.CSVLoader")
        dataset = loader.load_file(os.path.join(constants.BASE_DIR, constants.BREAST_CANCER_FILE_NAME))
        dataset.class_is_last()

        filter_for_train_set = Filter(classname=remove_percentage_filter_class, options=options_for_train_set)
        filter_for_train_set.inputformat(dataset)
        train_set = filter_for_train_set.filter(dataset)

        filter_for_test_set = Filter(classname=remove_percentage_filter_class, options=options_for_test_set)
        filter_for_test_set.inputformat(dataset)
        test_set = filter_for_test_set.filter(dataset)

        return {
            'train_set': train_set,
            'test_set': test_set,
            'labels': dataset.class_attribute.values
        }

    def __get_all_breast_cancer_data(self):
        data = []
        target = []
        bool_map = {'yes': 1, 'no': 0, 'nan': 0}
        with open(os.path.join(constants.BASE_DIR, constants.BREAST_CANCER_FILE_NAME)) as csvfile:
            readCSV = csv.DictReader(csvfile)
            for row in readCSV:
                for key in row:
                    row[key] = row[key].strip("'")
                row['deg-malig'] = int(row['deg-malig'])
                row['node-caps'] = bool_map[row['node-caps']]
                row['irradiat'] = bool_map[row['irradiat']]
                target.append(row['label'])
                row.pop('label')
                data.append(row)

        vectorizer = DictVectorizer(sparse=False)
        data = vectorizer.fit_transform(data, target)
        return {
            "data": data,
            "target": target,
            "feature_names": vectorizer.get_feature_names()
        }
