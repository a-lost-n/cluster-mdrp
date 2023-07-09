from src import *

class Restaurant():

  def __init__(self,id,grid_size,type='mid',randseed=0, x=0, y=0):
    self.id = id
    seed(randseed)
    self.grid_size = grid_size
    self.pos = np.array([randint(0,grid_size),randint(0,grid_size)]) if randseed != 0 else np.array([x,y])
    self.type = type
    self.active_orders = []
    match type:
      case 'low':
        self.mult_m = 0.5
        self.mult_sd = 0.2
      case 'high':
        self.mult_m = 3
        self.mult_sd = 2
      case 'mid':
        self.mult_m = 1
        self.mult_sd = 1


  def produce(self, time):
    orders_size = self.hourly_orders[int(time.minute/DELTA_MINUTES)]
    orders = [Order([randint(0,self.grid_size-1), randint(0,self.grid_size-1)], self.id, time) for _ in range(orders_size)]
    self.active_orders.extend(orders)
    self.hour_order_count += orders_size
    return orders


  def get_mean_sd(self, time):
    if time.hour == 11: m = 3; sd = 1
    elif time.hour == 12: m = 9; sd = 2
    elif time.hour == 13: m = 8; sd = 1.5
    elif time.hour == 14: m = 5; sd = 1.5
    elif time.hour == 15: m = 4; sd = 1.5
    elif time.hour == 16: m = 4; sd = 1
    elif time.hour == 17: m = 6; sd = 2
    elif time.hour == 18: m = 6; sd = 2
    elif time.hour == 19: m = 5; sd = 1.5
    elif time.hour == 20: m = 4; sd = 1
    elif time.hour == 21: m = 3; sd = 1 
    elif time.hour == 22: m = 2; sd = 0.75
    return m, sd


  def build_prediction(self, time):
    self.hour_order_count = 0
    mean, sd = self.get_mean_sd(time)
    orders_size = -1
    while orders_size < 0:
      orders_size = int(gauss(mean * self.mult_m, sd * self.mult_sd)+0.5)

    self.hourly_orders = np.zeros(HOUR_LAPSES, dtype=int)
    for _ in range(orders_size):
      self.hourly_orders[randint(0,HOUR_LAPSES-1)] += 1


  def reset(self):
    del self.active_orders[:]
