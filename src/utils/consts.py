import numpy as np
import datetime
import copy
from random import randint, seed, gauss
from sklearn.cluster import KMeans
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


DELTA_MINUTES = 5
HOUR_LAPSES = int(60/DELTA_MINUTES)
SPEED = 10  # 10 unidades/minuto
MOVEMENT = DELTA_MINUTES*SPEED
EXPECTED_DELAY_TO_RESTAURANT = 15
EXPECTED_ATTENDING_RESTAURANT = 2
EXPECTED_DELAY_TO_DESTINATION = 15
EXPECTED_ATTENDING_DESTINATION = 3
TIME_TO_DROPOUT_ORDER = 30

MAX_COST_NORM = 2*60*SPEED
COST_TRANSLATION_PER_TRAVEL_UNIT = 1/MAX_COST_NORM
COST_INVOCATION = 1
COST_DELAY_PER_SECOND = -2/MAX_COST_NORM
COST_DROPOUT = -4000/MAX_COST_NORM
