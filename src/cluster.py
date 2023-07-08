from src import *


class Cluster():
  
  def __init__(self, restaurant_list, centroid, id):
    self.id = id
    self.restaurants = restaurant_list
    self.centroid = centroid
    self.active_orders = []
    #TODO: Debería ser un priority queue
    self.courier_list = []
    self.incoming_couriers = []
  
  def produce(self, time):
    for restaurant in self.restaurants:
      self.active_orders.extend(restaurant.produce(time))


  def build_prediction(self, time):
    for restaurant in self.restaurants:
      restaurant.build_prediction(time)

  def get_next_courier(self):
    if len(self.courier_list) == 0:
      return -1
    return self.courier_list.pop(0)
  
  def assign_next_courier(self):
    if len(self.active_orders) == 0 or len(self.courier_list) == 0:
      return -1
    courier = self.get_next_courier()
    order = self.active_orders.pop(0)
    courier.assign_order(order)
    # return courier
  
  def assign_all_orders(self):
    while len(self.active_orders) > 0 and len(self.courier_list) > 0:
      self.assign_next_courier()

  def drop_expired_orders(self, time):
    reward = 0
    for order in self.active_orders:
      if nextStamp(order.order_time, TIME_TO_DROPOUT_ORDER) >= time:
        self.active_orders.remove(order)
        reward += COST_DROPOUT
    return reward

  def queue_courier(self, courier):
    self.courier_list.append(courier)

  def set_incoming(self, courier):
    self.incoming_couriers.append(courier)

  def move_to_queue(self, courier):
    for i in range(len(self.incoming_couriers)):
      if self.incoming_couriers[i].id == courier.id:
        self.incoming_couriers.pop(i)
        self.queue_courier(courier)
        return

  def can_relocate(self):
    return len(self.courier_list) > 0


  def reset(self):
    for restaurant in self.restaurants:
      restaurant.reset()
    del self.courier_list[:]
    del self.active_orders[:]
    del self.incoming_couriers[:]


  def get_state(self):
      return [len(self.courier_list), len(self.active_orders), len(self.incoming_couriers)]
