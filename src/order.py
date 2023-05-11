from src import *

EXPECTED_DELAY = 15

class Order():
  def __init__(self,x,y,time):
    self.x = x
    self.y = y
    self.on_time_delivery = nextStamp(time, next_val=EXPECTED_DELAY)