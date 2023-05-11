from src import *

class Map():

    # El arreglo va a ser de tamaño 3 y va a tener la siguiente forma:
    # [high, mid, low] donde cada número es la cantidad de restaurantes en el mapa
    def __init__(self, restaurant_array, randseed=0, start=1, epochs=1000):
        if randseed != 0:
            seed(randseed)
        self.clusters = []
        self.restaurants = []
        self.restaurants_tags = []
        self.resTypes = ['high', 'mid', 'low']
        for resNum, resType in zip(restaurant_array,self.resTypes):
            for _ in range(resNum):
                self.restaurants.append(Restaurant(x=randint(0,GRID_SIZE),
                                                   y=randint(0,GRID_SIZE),
                                                   type=resType,
                                                   randseed=randint(1,10000)))
                self.restaurants_tags.append(resType)
        self.restaurants = np.array(self.restaurants)
        self.init_clusters(start=start, epochs=epochs)
    

    def init_clusters(self, start, epochs):
        res_positions = []
        for res in self.restaurants:
            res_positions.append([res.x, res.y])
        res_positions = np.array(res_positions)
        self.labels, self.centroids, self.inertia = optimal_kmeans_dist(res_positions.T, start=start, epochs=epochs)
        for i in range(self.centroids.shape[0]):
            self.clusters.append(Cluster(self.restaurants[self.labels == i], self.centroids[i]))
    

    def display_map_clusters(self):
        x_values = []
        y_values = []
        for res in self.restaurants:
            x_values.append(res.x)
            y_values.append(res.y)
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
            x_values.append(res.x)
            y_values.append(res.y)
            last_tag = tag
        plt.scatter(x_values, y_values)
        plt.legend(self.resTypes)
        plt.show()

    
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
