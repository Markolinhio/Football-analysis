import cv2

class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.tracker = cv2.TrackerCSRT_create()
        self.bbox = None

    def initialize_tracker(self, frame, bbox):
        self.bbox = bbox
        self.tracker.init(frame, tuple(bbox))

    def update_tracker(self, frame):
        ok, new_bbox = self.tracker.update(frame)
        if ok:
            self.bbox = new_bbox
        return ok, self.bbox

class PlayerTracker:
    def __init__(self):
        self.players_home = []
        self.players_away = []
        self.next_away_id = 20

    def assign_players(self, frame, boxes_home, boxes_away):
        for idx, player in enumerate(self.players_home):
            ok, new_bbox = player.update_tracker(frame)
            if not ok:
                self.players_home.pop(idx)

        for idx, player in enumerate(self.players_away):
            ok, new_bbox = player.update_tracker(frame)
            if not ok:
                self.players_away.pop(idx)

        # Create new players or add tracked players not in the list
        for idx, bbox in enumerate(boxes_home):
            if idx >= len(self.players_home):
                player = Player(idx)
                player.initialize_tracker(frame, bbox)
                self.players_home.append(player)

        for idx, bbox in enumerate(boxes_away):
            if idx >= len(self.players_away):
                player = Player(self.next_away_id)
                player.initialize_tracker(frame, bbox)
                self.players_away.append(player)
                self.next_away_id += 1

        # Return assigned IDs
        assigned_ids_home = {player.player_id: player.bbox for player in self.players_home}
        assigned_ids_away = {player.player_id: player.bbox for player in self.players_away}

        return assigned_ids_home, assigned_ids_away