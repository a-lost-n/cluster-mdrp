import numpy as np
import datetime
from random import randint, seed, gauss
from sklearn.cluster import KMeans
from scipy.stats import norm

GRID_SIZE = 1000
DELTA_MINUTES = 5
HOUR_LAPSES = int(60/DELTA_MINUTES)