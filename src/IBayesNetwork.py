from constants import ALL_ALGORITHMS


class IBayesNetwork():

	def __init__(self, algorithm_data):
		if algorithm_data['name'] not in [value['name'] for value in ALL_ALGORITHMS.values()]:
			raise Exception('Unsupported algorithm in ALL_ALGOTITHMS!!')

		self.model = None
		self.algorithm_name = algorithm_data['name']

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