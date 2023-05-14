from src import *
from src.restaurant import Restaurant

class Cluster():
  
  def __init__(self, restaurant_list, centroid, id):
    self.id = id
    self.restaurants = restaurant_list
    self.centroid = centroid
    self.active_orders = []
    self.courier_list = []
  
  def produce(self, time):
    for restaurant in self.restaurants:
      self.active_orders.extend(restaurant.produce(time))


  def build_prediction(self, time):
    for restaurant in self.restaurants:
      restaurant.build_prediction(time)

  def assign_next_courier(self) -> Courier:
    courier = self.courier_list.pop(0)
    order = self.active_orders.pop(0)
    courier.assign_order(order)
    return courier
  
  def queue_courier(self, courier):
    self.courier_list.append(courier)

  def invoke_courier(self):
    self.courier_list.append(Courier(pos=self.centroid,cluster_id=self.id))

  """
  Hay 3 estados:
  0: Hay un exceso de couriers en el cluster con respecto a la probabilidad de pedidos en el siguiente diferencial de tiempo.
  1: Puede haber la probabilidad de falta de couriers en el siguiente diferencial de tiempo.
  2: Hay un falta de couriers en este instante.
  """
  def set_state(self):
    next_expected = 0
    for restaurant in self.restaurants:
      next_expected += restaurant.expected_orders()