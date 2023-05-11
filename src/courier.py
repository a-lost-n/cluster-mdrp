from src import * 

class Courier():

    def __init__(self,x,y,time):
        self.x = x
        self.y = y
        self.time = time

    # AQUI
    def closest_cluster(self):
        self.g = 2

    def go_to(self, x, y):
        d_x = np.abs(x - self.x)
        d_y = np.abs(y - self.y)
        d = np.sqrt(d_x**2 + d_y**2)
        if d <= DELTA_MINUTES*10:
            self.closest_cluster
