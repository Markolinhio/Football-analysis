class Player:
    def __init__(self, player_id, centroid):
        self.player_id = player_id
        self.centroids = [centroid]
        self.missing_frames = 0


def calculate_centroid(box):
    centroid_x = (box[0] + box[2]) / 2
    centroid_y = (box[1] + box[3]) / 2
    return (centroid_x, centroid_y)


class PlayerTracker:
    def __init__(self):
        self.players_home = {}
        self.players_away = {}
        self.player_id_counter_home = 0
        self.player_id_counter_away = 0

    def update(self, boxes_home, boxes_away):
        current_centroids_home = [
            self.calculate_centroid(box) for box in boxes_home]

        if not self.players_home:
            for centroid in current_centroids_home:
                self.players_home[self.player_id_counter_home] = Player(
                    self.player_id_counter_home, centroid)
                self.player_id_counter_home += 1
        else:
            for player_id, player in self.players_home.items():
                if player_id < len(current_centroids_home):
                    player.centroids.append(current_centroids_home[player_id])
                    player.missing_frames = 0
                else:
                    player.missing_frames += 1
                    if player.missing_frames > 5:
                        del self.players_home[player_id]

        current_centroids_away = [
            self.calculate_centroid(box) for box in boxes_away]

        if not self.players_away:
            for centroid in current_centroids_away:
                self.players_away[self.player_id_counter_away] = Player(
                    self.player_id_counter_away, centroid)
                self.player_id_counter_away += 1
        else:
            for player_id, player in self.players_away.items():
                if player_id < len(current_centroids_away):
                    player.centroids.append(current_centroids_away[player_id])
                    player.missing_frames = 0
                else:
                    player.missing_frames += 1
                    if player.missing_frames > 5:
                        del self.players_away[player_id]

        return self.players_home, self.players_away
