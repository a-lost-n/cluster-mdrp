import numpy as np
import datetime
from random import randint, seed, gauss
from sklearn.cluster import KMeans
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


DELTA_MINUTES = 5
HOUR_LAPSES = int(60/DELTA_MINUTES)
SPEED = 10 # 10 unidades/minuto
MOVEMENT = DELTA_MINUTES*SPEED
EXPECTED_DELAY_TO_RESTAURANT = 15
EXPECTED_ATTENDING_RESTAURANT = 2
EXPECTED_DELAY_TO_DESTINATION = 15
EXPECTED_ATTENDING_DESTINATION = 3
TIME_TO_DROPOUT_ORDER = 30

COST_TRANSLATION_PER_TRAVEL_UNIT = -1
COST_INVOCATION = -300
COST_DELAY_PER_SECOND = -2
COST_DROPOUT = -1000