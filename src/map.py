import gc
from src import *

class Map():

    # El arreglo va a ser de tamaño 3 y va a tener la siguiente forma:
    # [high, mid, low] donde cada número es la cantidad de restaurantes en el mapa
    def __init__(self, restaurant_array=None, grid_size=None, randseed=0, start=1, epochs=1000, start_time = datetime.time(11,0,0),
                clusters=None, couriers=None, restaurants=None, restaurants_tags=None, time=None, filename=None, labels=None, centroids=None):
        self.resTypes = ['high', 'mid', 'low']
        self.start_time = start_time
        if clusters is not None and couriers is not None:
            self.clusters = clusters
            self.couriers = couriers
            self.restaurants = restaurants
            self.restaurants_tags = restaurants_tags
            self.time = time
            self.labels = labels
            self.centroids = centroids
            return
        if randseed != 0:
            seed(randseed)
        self.clusters = []
        self.couriers = []
        self.restaurants = []
        self.restaurants_tags = []
        self.time = start_time
        for resNum, resType in zip(restaurant_array,self.resTypes):
            for _ in range(resNum):
                self.restaurants.append(Restaurant(id=len(self.restaurants),
                                                grid_size=grid_size,
                                                type=resType,
                                                randseed=randint(1,10000)))
                self.restaurants_tags.append(resType)
        self.restaurants = np.array(self.restaurants)
        if filename is not None:
            self.load(filename+".npz")
        else:
            self.init_clusters(start=start, epochs=epochs)
        for i in range(self.centroids.shape[0]):
            self.clusters.append(Cluster(self.restaurants[self.labels == i], self.centroids[i], id=i))


    def copy(self):
        return Map(clusters=copy.deepcopy(self.clusters), couriers=copy.deepcopy(self.couriers), 
                   restaurants=copy.deepcopy(self.restaurants), time=self.time, start_time=self.start_time,
                   labels = self.labels, centroids=self.centroids)


    def init_clusters(self, start, epochs):
        res_positions = []
        for res in self.restaurants:
            res_positions.append(res.pos)
        res_positions = np.array(res_positions)
        self.labels, self.centroids, _ = optimal_kmeans_dist(res_positions.T, start=start, epochs=epochs)

    
    def reset(self):
        self.time = self.start_time
        del self.couriers[:]
        for cluster in self.clusters:
            cluster.reset()
        gc.collect()

    
    def get_state(self, algorithm='DDQN'):
        state_array = []
        if algorithm == 'DDQN':
            done = True
            for cluster in self.clusters:
                cluster_state = cluster.get_state(algorithm=algorithm) 
                state_array.extend(cluster_state)
                done = done and (cluster_state[1] <= cluster_state[0] + cluster_state[2])
            # state_array.extend([self.time.hour])
            return np.array(state_array), done
        elif algorithm == 'Perceptron':
            done = True
            for cluster in self.clusters:
                cluster_state = cluster.get_state(algorithm=algorithm) 
                state_array.extend(cluster_state)
                done = done and (cluster_state[1] > -1)
            return np.array(state_array), done
        else:
            for cluster in self.clusters:
                state_array.append(cluster.get_state(algorithm=algorithm))
        return state_array

    def invoke_courier(self, cluster_id):
        cluster = self.clusters[cluster_id]
        courier = Courier(cluster.centroid, len(self.couriers), cluster.id)
        cluster.queue_courier(courier)
        self.couriers.append(courier)
        return COST_INVOCATION

    def assign_next_courier(self, cluster_id):
        return self.clusters[cluster_id].assign_next_courier()
    
    def get_courier(self, courier_id):
        return self.couriers[courier_id]

    def display_map_clusters(self):
        x_values = []
        y_values = []
        for res in self.restaurants:
            x_values.append(res.pos[0])
            y_values.append(res.pos[1])
        plt.scatter(x_values, y_values,c=self.labels)


    def display_map_types(self):
        x_values = []
        y_values = []
        last_tag = ""
        for res,tag in zip(self.restaurants,self.restaurants_tags):
            if last_tag == "":
                last_tag = tag
            if last_tag != tag:
                plt.scatter(x_values, y_values)
                x_values = []
                y_values = []
            x_values.append(res.pos[0])
            y_values.append(res.pos[1])
            last_tag = tag
        plt.scatter(x_values, y_values)
        plt.legend(self.resTypes)
        plt.show()


    def move_courier(self, courier_id, time, movement=MOVEMENT):
        courier = self.get_courier(courier_id)
        if courier.state == "to_restaurant":
            courier.cluster_id = None
            restaurant = self.restaurants[courier.order.restaurant_id]
            dist = np.linalg.norm(courier.pos - restaurant.pos)
            # Si la distancia es mayor de lo que nos movemos nos quedamos a mitad de camino
            if dist > movement:
                courier.pos = courier.pos + (restaurant.pos - courier.pos) * (dist - movement)/dist
                return 0
            # Si la distancia es menor y aún no llega la orden se espera
            else:
                courier.pos = restaurant.pos
                courier.state = "to_wait_restaurant"
                courier.wait_time = EXPECTED_ATTENDING_RESTAURANT
                return self.move_courier(courier_id=courier_id, time=nextStamp(time, dist/SPEED), movement=float(movement)-dist)

        elif courier.state == "to_wait_restaurant":
            if courier.order.on_time_restaurant <= time:
                if courier.wait_time > movement/SPEED:
                    courier.wait_time -= movement/SPEED
                    return 0
                courier.wait_time = 0
                courier.state = "to_destination"
                courier.with_order = True
                return self.move_courier(courier_id=courier_id, time=time, movement=movement-courier.wait_time*SPEED)
            return 0

        elif courier.state == "to_destination":
            dist = np.linalg.norm(courier.pos - courier.order.destination_pos)
            if dist > movement:
                courier.pos = courier.pos + (courier.order.destination_pos - courier.pos) * (dist - movement)/dist
                return 0
            else:
                courier.pos = courier.order.destination_pos
                courier.state = "to_wait_destination"
                courier.wait_time = EXPECTED_ATTENDING_DESTINATION
                delay = calculateDelay(time, courier.order.on_time_destination, extra_time=dist/SPEED)
                return min(delay, 0) + self.move_courier(courier_id=courier_id, time=time, movement=float(movement)-dist)

        elif courier.state == "to_wait_destination":
            if courier.wait_time > movement/SPEED:
                courier.wait_time -= movement/SPEED
                return 0
            courier.state = "to_cluster"
            courier.with_order = False
            courier.cluster_id = self.closest_cluster(courier.pos)
            self.clusters[courier.cluster_id].set_incoming(courier)
            return self.move_courier(courier_id=courier_id, time=time, movement=movement-courier.wait_time*SPEED)
        
        elif courier.state == "to_cluster":
            cent = self.centroids[courier.cluster_id]
            dist = np.linalg.norm(courier.pos - cent)
            courier.order = None
            if dist > movement:
                courier.pos = courier.pos + (cent - courier.pos) * (dist - movement)/dist
            else:
                courier.pos = cent
                self.clusters[courier.cluster_id].move_to_queue(courier)
                courier.state = "in_cluster"
            return 0
        
        elif courier.state == "in_cluster":
            return 0
        
        elif courier.state == "relocating":
            new_cluster = self.clusters[courier.relocation]
            dist = np.linalg.norm(courier.pos - new_cluster.centroid)
            if dist > movement:
                courier.pos = courier.pos + (new_cluster.centroid - courier.pos) * (dist - movement)/dist
            else:
                courier.pos = new_cluster.centroid
                new_cluster.move_to_queue(courier)
                courier.state = "in_cluster"
            return 0





    def closest_cluster(self, pos):
        min_dist = np.inf
        min_id = -1
        for id in range(len(self.clusters)):
            dist = np.linalg.norm(pos - self.centroids[id])
            if dist < min_dist:
                min_id = id
                min_dist = dist
        return min_id


    def test_produce(self):
        time = datetime.time(11,0,0)
        while time < datetime.time(22,0,0):
            print(time)
            for i in range(len(self.clusters)):
                if(time.minute == 0):
                    self.clusters[i].build_prediction(time)
                self.clusters[i].produce(time)
                print("C",i+1,":",len(self.clusters[i].active_orders))
            print("------")
            time = nextStamp(time)

    def build_prediction(self):
        for cluster in self.clusters:
            cluster.build_prediction(self.time)

    def produce(self):
        for cluster in self.clusters:
            cluster.produce(self.time)

    def pass_time(self):
        reward = 0
        for cluster in self.clusters:
            reward += cluster.drop_expired_orders(self.time)
            cluster.assign_all_orders()
        for courier in self.couriers:
            reward += self.move_courier(courier.id, self.time)
        self.time = nextStamp(self.time)
        if self.time.minute == 0 and self.time.hour != 22:
            self.build_prediction()
        for cluster in self.clusters:
            cluster.produce(self.time)
        done = self.time == datetime.time(22,0,0)
        return reward, done


    def relocate_courier(self, from_cluster, to_cluster):
        from_cluster = self.clusters[from_cluster]
        to_cluster = self.clusters[to_cluster]
        if len(from_cluster.courier_list) == 0:
            return 0
        courier = from_cluster.get_next_courier()
        courier.state = "relocating"
        courier.relocation = to_cluster.id
        to_cluster.set_incoming(courier)
        return -np.linalg.norm(courier.pos - to_cluster.centroid)
        

    def get_empty_clusters(self):
        return [len(c.courier_list)==0 for c in self.clusters]
    
    def save(self, filename):
        np.savez(filename, self.labels, self.centroids)

    def load(self, filename):
        file = np.load(filename)
        self.labels = file['arr_0']
        self.centroids = file['arr_1']
