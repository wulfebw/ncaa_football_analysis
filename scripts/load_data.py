
import os
import sys
import file_utils
import numpy as np

import constants

from team import Team
from game import Game

def select_relevant_data(data, labels, desired_labels):
    desired_indices = []
    for idx, label in enumerate(labels):
        if label in desired_labels:
            desired_indices.append(idx)

    idx_data = np.array(data)
    rtn_data = np.zeros((np.shape(data)[0], len(desired_indices)))
    for idx, label_idx in enumerate(desired_indices):
        rtn_data[:, idx] = idx_data[:, label_idx]

    return rtn_data

def load_team_and_game_dicts(data, teams, games, labels):
    for row in data:
        team_id = row[constants.TEAM_CODE_IDX]
        game_id = row[constants.GAME_CODE_IDX]
        
        if team_id in teams:
            teams[team_id].add_row(row)
        else:
            teams[team_id] = Team(row)

        if game_id in games:
            games[game_id].add_row(row)
        else:
            games[game_id] = Game(row)

def load_data_from_year(year_directory, relevant_labels, teams, games):
    team_game_filename = os.path.join(year_directory, constants.TEAM_GAME_STATISTICS_FILENAME)
    team_game_data, labels = file_utils.read_csv_data_file(team_game_filename)
    team_game_data = select_relevant_data(team_game_data, labels, relevant_labels)
    load_team_and_game_dicts(team_game_data, teams, games, labels)

def load_data_from_all_years(relevant_labels):
    teams = dict()
    games = dict()

    years = os.listdir(constants.DATA_PATH)
    for year in years:
        if constants.YEAR_PREFIX in year:
            load_data_from_year(year, relevant_labels, teams, games)

    return teams, games

if __name__ == '__main__':
    relevant_labels = file_utils.read_csv_label_file(constants.LABELS_FILENAME)
    teams, games = load_data_from_all_years(relevant_labels)
    file_utils.write_csv_data_file(constants.OUTPUT_DATA_FILENAME, teams, games, relevant_labels)
