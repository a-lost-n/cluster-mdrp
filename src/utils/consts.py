import numpy as np
import datetime
from random import randint, seed, gauss
from sklearn.cluster import KMeans
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

GRID_SIZE = 100
DELTA_MINUTES = 5
HOUR_LAPSES = int(60/DELTA_MINUTES)