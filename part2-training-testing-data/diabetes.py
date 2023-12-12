import numpy as np
import matplotlib.pylot as plt

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = load_diabetes(as_frame=True)
data = data.frame 
print(data) 
x = data[bmi]