
import constants

class Game(object):
    
    def __init__(self, row):
        self.id = row[constants.GAME_CODE_IDX]
        self.team_ids = []
        self.team_ids.append(row[constants.TEAM_CODE_IDX])
        self.points = []
        self.points.append(row[constants.POINTS_IDX])

    def add_row(self, row):
        self.team_ids.append(row[constants.TEAM_CODE_IDX])
        self.points.append(row[constants.POINTS_IDX])

    def is_valid(self):
        return len(self.team_ids) == 2 and len(self.points) == 2