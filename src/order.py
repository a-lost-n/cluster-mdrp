from src import *



class Order():

  def __init__(self,pos,restaurant_id,time):
    self.restaurant_id = restaurant_id
    self.destination_pos = np.array(pos)
    self.order_time = time
    self.on_time_restaurant = nextStamp(time, next_val=EXPECTED_DELAY_TO_RESTAURANT)
    self.on_time_destination = nextStamp(nextStamp(self.on_time_restaurant,
                                                    next_val=EXPECTED_ATTENDING_RESTAURANT),
                                                    next_val=EXPECTED_DELAY_TO_DESTINATION)
    

  def display_info(self):
    print("Restaurant ID:", self.restaurant_id)
    print("Destination:", self.destination_pos)
    print("Time on Restaurant:", self.on_time_restaurant)
    print("Time on Destination:", self.on_time_destination)