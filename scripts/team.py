

import numpy as np

import constants

class Team(object):

    def __init__(self, row):
        self.id = row[constants.TEAM_CODE_IDX]
        self.row = row
        self.n_games = 1

    def add_row(self, row):
        self.row = np.add(self.row, row)
        self.n_games += 1

    def get_stats(self):
        stats = np.divide(self.row[constants.DATA_START_IDX:], self.n_games)
        return stats

    def is_valid(self):
        return True
