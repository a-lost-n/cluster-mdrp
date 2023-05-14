from src import *

class Map():

    # El arreglo va a ser de tamaño 3 y va a tener la siguiente forma:
    # [high, mid, low] donde cada número es la cantidad de restaurantes en el mapa
    def __init__(self, restaurant_array, grid_size, randseed=0, start=1, epochs=1000):
        if randseed != 0:
            seed(randseed)
        self.clusters = []
        self.restaurants = []
        self.restaurants_tags = []
        self.resTypes = ['high', 'mid', 'low']
        id = 0
        for resNum, resType in zip(restaurant_array,self.resTypes):
            for _ in range(resNum):
                self.restaurants.append(Restaurant(id=id,
                                                   grid_size=grid_size,
                                                   type=resType,
                                                   randseed=randint(1,10000)))
                self.restaurants_tags.append(resType)
                id += 1
        self.restaurants = np.array(self.restaurants)
        self.init_clusters(start=start, epochs=epochs)
    

    def init_clusters(self, start, epochs):
        res_positions = []
        for res in self.restaurants:
            res_positions.append(res.pos)
        res_positions = np.array(res_positions)
        self.labels, self.centroids, self.inertia = optimal_kmeans_dist(res_positions.T, start=start, epochs=epochs)
        for i in range(self.centroids.shape[0]):
            self.clusters.append(Cluster(self.restaurants[self.labels == i], self.centroids[i], id=i))
    

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


    def move_courier(self, courier, time, movement=MOVEMENT):
        if movement > courier.debt_time:
            movement -= courier.debt_time
            courier.debt_time = 0
        else:
            courier.debt_time -= movement
            return 0
        match courier.state:
            case "to_restaurant":
                courier.cluster_id = -1
                restaurant = self.restaurants[courier.order.restaurant_id]
                dist = np.linalg.norm(courier.pos - restaurant.pos)

                # Si la distancia es mayor de lo que nos movemos nos quedamos a mitad de camino
                if dist > movement:
                    courier.pos += (restaurant.pos - courier.pos) * (dist - movement)/dist

                # Si la distancia es menor y aún no llega la orden se espera
                else:
                    courier.pos = restaurant.pos
                    courier.state = "to_wait_restaurant"

                    # Si la distancia es menor y ya está la orden ir de inmediato al destino
                    if courier.order.on_time_restaurant <= time:
                        print(movement, dist)
                        print(movement - dist)
                        return self.move_courier(courier, time, movement=float(movement)-dist)
                return 0

            case "to_wait_restaurant":
                if courier.order.on_time_restaurant <= time:
                    courier.state = "to_destination"
                    courier.with_order = True
                    if movement <= EXPECTED_ATTENDING_RESTAURANT:
                        courier.debt_time = EXPECTED_ATTENDING_RESTAURANT - movement
                    else:
                        return self.move_courier(courier=courier, time=time, movement=float(movement)-EXPECTED_ATTENDING_RESTAURANT)
                return 0

            case "to_destination":
                dist = np.linalg.norm(courier.pos - courier.order.destination_pos)
                if dist > movement:
                    courier.pos += (courier.order.destination_pos - courier.pos) * (dist - movement)/dist
                else:
                    courier.pos = courier.order.destination_pos
                    courier.state = "to_wait_destination"
                    delay = -calculateDelay(time, dist, courier.order.on_time_destination)
                    return min(delay, 0)
                return 0

            case "to_wait_destination":
                courier.state = "to_cluster"
                courier.with_order = False
                courier.cluster_id = self.closest_cluster(courier.pos)
                if movement <= EXPECTED_ATTENDING_DESTINATION:
                    courier.debt_time = EXPECTED_ATTENDING_DESTINATION - movement
                else:
                    return self.move_courier(courier=courier, time=time, movement=float(movement)-EXPECTED_ATTENDING_DESTINATION)
                return 0
            
            case "to_cluster":
                cent = self.centroids[courier.cluster_id]
                dist = np.linalg.norm(courier.pos - cent)
                courier.order = None
                if dist > movement:
                    courier.pos += (cent - courier.pos) * (dist - movement)/dist
                else:
                    courier.pos = cent
                    self.clusters[courier.cluster_id].queue_courier(courier)
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


    def start(self):
        time = datetime.time(11,0,0)
        while time < datetime.time(22,0,0):
            print(time)
            for i in range(len(self.clusters)):
                if(time.minute == 0):
                    self.clusters[i].build_prediction(time)
                self.clusters[i].get_orders(time)
                print("C",i+1,":",len(self.clusters[i].active_orders))
            print("------")
            time = nextStamp(time)
