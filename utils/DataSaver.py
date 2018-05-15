import numpy as np
import pickle


class DataSaver:
    def __init__(self,
                 data_names,
                 number_of_participant,
                 trial_length):
        self.data_names = data_names
        self.data_size = [number_of_participant, trial_length]
        self.data = {}
        for data_name in data_names:
            self.data[data_name] = np.zeros(shape=self.data_size)

    def save_data(self, data_name, participant_id, trial_id, data):
        self.data[data_name][participant_id, trial_id] = data

    def save_trials_data(self, data_name, participant_id, trial_data):
        self.data[data_name][participant_id] = trial_data

    def get_data(self, data_name):
        return self.data[data_name]

    def get_trial_data(self, data_name, participant_id):
        return self.data[data_name][participant_id]

    def save_to_file(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from_file(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
