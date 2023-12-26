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
        self.players_home = []
        self.players_away = []
        self.player_id_counter_home = 1  # Start player IDs for home team from 1
        self.player_id_counter_away = 20  # Start player IDs for away team from 20

    def update(self, boxes_home, boxes_away, threshold=50):
        current_centroids_home = [calculate_centroid(box) for box in boxes_home]
        current_centroids_away = [calculate_centroid(box) for box in boxes_away]

        detected_player_ids_home = set()
        detected_player_ids_away = set()

        if not self.players_home:
            self.players_home = [Player(i, centroid) for i, centroid in enumerate(current_centroids_home, self.player_id_counter_home)]
            self.player_id_counter_home += len(self.players_home)
        if not self.players_away:
            self.players_away = [Player(i, centroid) for i, centroid in enumerate(current_centroids_away, self.player_id_counter_away)]
            self.player_id_counter_away += len(self.players_away)

        for player in self.players_home:
            player.missing_frames += 1

        for player in self.players_away:
            player.missing_frames += 1

        for centroid in current_centroids_home:
            distances = [((player.centroids[-1][0] - centroid[0]) ** 2 +
                          (player.centroids[-1][1] - centroid[1]) ** 2) ** 0.5
                         for player in self.players_home]
            min_distance = min(distances)
            if min_distance < threshold:
                idx = distances.index(min_distance)
                self.players_home[idx].centroids.append(centroid)
                self.players_home[idx].missing_frames = 0
                detected_player_ids_home.add(idx)
            else:
                self.players_home.append(Player(self.player_id_counter_home, centroid))
                self.player_id_counter_home += 1

        for centroid in current_centroids_away:
            distances = [((player.centroids[-1][0] - centroid[0]) ** 2 +
                          (player.centroids[-1][1] - centroid[1]) ** 2) ** 0.5
                         for player in self.players_away]
            min_distance = min(distances)
            if min_distance < 50:
                idx = distances.index(min_distance)
                self.players_away[idx].centroids.append(centroid)
                self.players_away[idx].missing_frames = 0
                detected_player_ids_away.add(idx)
            else:
                self.players_away.append(Player(self.player_id_counter_away, centroid))
                self.player_id_counter_away += 1

        self.players_home = [player for i, player in enumerate(self.players_home) if i in detected_player_ids_home]
        self.players_away = [player for i, player in enumerate(self.players_away) if i in detected_player_ids_away]

        assigned_ids_home = {tuple(player.centroids[-1]): player.player_id for player in self.players_home}
        assigned_ids_away = {tuple(player.centroids[-1]): player.player_id for player in self.players_away}

        return assigned_ids_home, assigned_ids_away
