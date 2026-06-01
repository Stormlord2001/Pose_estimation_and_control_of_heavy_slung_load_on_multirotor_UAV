

class MarkerPose:
    def __init__(self, x, y, id):
        self.x = x
        self.y = y
        self.id = id

    def scale_position(self, scale_factor):
        self.x = self.x * scale_factor
        self.y = self.y * scale_factor


