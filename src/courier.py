from src import * 

class Courier():

    def __init__(self,pos, courier_id, cluster_id):
        self.id = courier_id
        self.pos = np.array(pos)
        self.state = "in_cluster"
        self.with_order = False
        self.cluster_id = cluster_id
        self.wait_time = 0
        self.relocation = None

    def assign_order(self, order):
        self.order = order
        self.state = "to_restaurant"

    def display_info(self):
        print("Position:", self.pos)
        print("State:", self.state)
        if self.order is not None:
            print("Order Info:")
            self.order.display_info()
        else:
            print("Cluster ID:", self.cluster_id)
