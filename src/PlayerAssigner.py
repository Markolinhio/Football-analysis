from scipy.optimize import linear_sum_assignment
import numpy as np


class Player:
    def __init__(self, player_id, centroid):
        self.player_id = player_id
        self.centroids = [centroid]
        self.missing_frames = 0


def calculate_centroid(box):
    centroid_x = (box[0] + box[2]) / 2
    centroid_y = (box[1] + box[3]) / 2
    return (centroid_x, centroid_y)


def calculate_distance(box1, box2):
    centroid1 = calculate_centroid(box1)
    centroid2 = calculate_centroid(box2)
    distance = ((centroid1[0] - centroid2[0]) ** 2 +
                (centroid1[1] - centroid2[1]) ** 2) ** 0.5
    return distance


class PlayerTracker:
    def __init__(self):
        self.players_home = []
        self.players_away = []
        self.player_id_counter_home = 1
        self.player_id_counter_away = 20
        self.prev_boxes_home = []
        self.prev_boxes_away = []

    def update(self, boxes_home, boxes_away, threshold=50):
        current_centroids_home = [
            calculate_centroid(box) for box in boxes_home]
        current_centroids_away = [
            calculate_centroid(box) for box in boxes_away]

        if not self.prev_boxes_home:
            self.prev_boxes_home = boxes_home
        if not self.prev_boxes_away:
            self.prev_boxes_away = boxes_away

        if not self.players_home:
            self.players_home = [Player(i, centroid) for i, centroid in enumerate(
                current_centroids_home, self.player_id_counter_home)]
            self.player_id_counter_home += len(self.players_home)
        if not self.players_away:
            self.players_away = [Player(i, centroid) for i, centroid in enumerate(
                current_centroids_away, self.player_id_counter_away)]
            self.player_id_counter_away += len(self.players_away)

        cost_matrix_home = np.zeros(
            (len(self.prev_boxes_home), len(boxes_home)))
        cost_matrix_away = np.zeros(
            (len(self.prev_boxes_away), len(boxes_away)))

        for i, prev_box in enumerate(self.prev_boxes_home):
            for j, current_box in enumerate(boxes_home):
                cost_matrix_home[i, j] = calculate_distance(
                    prev_box, current_box)

        for i, prev_box in enumerate(self.prev_boxes_away):
            for j, current_box in enumerate(boxes_away):
                cost_matrix_away[i, j] = calculate_distance(
                    prev_box, current_box)

        row_ind_home, col_ind_home = linear_sum_assignment(cost_matrix_home)
        row_ind_away, col_ind_away = linear_sum_assignment(cost_matrix_away)

        # Matching players between frames
        for row, col in zip(row_ind_home, col_ind_home):
            if cost_matrix_home[row, col] < threshold:
                self.players_home[row].centroids.append(
                    current_centroids_home[col])

        for row, col in zip(row_ind_away, col_ind_away):
            if cost_matrix_away[row, col] < threshold:
                self.players_away[row].centroids.append(
                    current_centroids_away[col])

        # Update previous boxes
        self.prev_boxes_home = boxes_home
        self.prev_boxes_away = boxes_away

        # Update existing players' positions and handle missing players
        for player in self.players_home:
            if player.player_id in row_ind_home:
                player.missing_frames = 0
            else:
                player.missing_frames += 1
                if player.missing_frames > threshold:
                    self.players_home.remove(player)

        for player in self.players_away:
            if player.player_id in row_ind_away:
                player.missing_frames = 0
            else:
                player.missing_frames += 1
                if player.missing_frames > threshold:
                    self.players_away.remove(player)

        # Assign new IDs to unmatched centroids or new players
        for idx, centroid in enumerate(current_centroids_home):
            if idx not in row_ind_home:
                self.players_home.append(
                    Player(self.player_id_counter_home, centroid))
                self.player_id_counter_home += 1

        for idx, centroid in enumerate(current_centroids_away):
            if idx not in row_ind_away:
                self.players_away.append(
                    Player(self.player_id_counter_away, centroid))
                self.player_id_counter_away += 1

        assigned_ids_home = [player.player_id for player in self.players_home]
        assigned_ids_away = [player.player_id for player in self.players_away]

        return assigned_ids_home, assigned_ids_away
