from src import *

class Restaurant():

  def __init__(self,id,grid_size,type='mid',randseed=0):
    self.id = id
    if randseed != 0:
      seed(randseed)
    self.grid_size = grid_size
    self.pos = np.array([randint(0,grid_size),randint(0,grid_size)])
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
    #   case 'day':
    #     print('day')
    #   case 'night':
    #     print('night')


  def produce(self, time):
    orders_size = self.hourly_orders[int(time.minute/DELTA_MINUTES)]
    orders = [Order([randint(0,self.grid_size-1), randint(0,self.grid_size-1)], self.id, time) for _ in range(orders_size)]
    self.active_orders.extend(orders)
    self.hour_order_count += orders_size
    return orders


  def get_mean_sd(self, time):
    match time.hour:
      case 11: m = 3; sd = 1
      case 12: m = 9; sd = 2
      case 13: m = 8; sd = 1.5
      case 14: m = 5; sd = 1.5
      case 15: m = 4; sd = 1.5
      case 16: m = 4; sd = 1
      case 17: m = 6; sd = 2
      case 18: m = 6; sd = 2
      case 19: m = 5; sd = 1.5
      case 20: m = 4; sd = 1
      case 21: m = 3; sd = 1 
      case 22: m = 2; sd = 0.75
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


  def expected_orders(self, time, expected_max_deviation=0.8):
    time = nextStamp(time)
    mean, sd = self.get_mean_sd(time)
    mean *= self.mult_m
    sd *= self.mult_sd
    z_value = norm.ppf((expected_max_deviation+1)/2)
    return max(int((mean + z_value * sd - self.hour_order_count)/(HOUR_LAPSES - time.minute/DELTA_MINUTES) + 0.5), 0)

