import os
import csv

import constants

def read_csv_label_file(input_filename):
    input_filepath = os.path.join(constants.DATA_PATH, input_filename)
    labels = []
    try:
        with open(input_filepath, 'r') as csvfile:
            labels = csvfile.readline().replace('"', '').replace('\r','').replace('\n','').split(',')
    except IOError as e:
        raise IOError('input filename: {} raised IOError on read'.format(input_filename))
    return labels

def read_csv_data_file(input_filename):
    input_filepath = os.path.join(constants.DATA_PATH, input_filename)
    data = []
    labels = []
    try:
        with open(input_filepath, 'r') as csvfile:
            labels = csvfile.readline().replace('"', '').replace('\r','').replace('\n','').split(',')
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                data.append(map(float, row))
    except IOError as e:
        raise IOError('input filename: {} raised IOError on read'.format(input_filename))
    return data, labels

def write_csv_data_file(output_filename, teams, games, relevant_labels):
    output_filepath = os.path.join(constants.DATA_PATH, output_filename)
    labels = ["Game Code"] + ["Team Id 1"] + ["Team Id 2"] + relevant_labels[2:] + \
                relevant_labels[2:] + ["Team Points 1"] + ["Team Points 2"]
    data = [labels]
    for game_id, game in games.iteritems():
        if not game.is_valid():
            break

        row = []
        row.append(int(game.id))
        row += [int(team_id) for team_id in game.team_ids]

        valid = True
        for team_id in game.team_ids:
            if team_id in teams:
                team = teams[team_id]
                if not team.is_valid():
                    valid = False
                row = row + team.get_stats().tolist()
            else:
                valid = False

        row += [int(point) for point in game.points]

        if valid:
            data.append(row)

    with open(output_filepath, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(data)
