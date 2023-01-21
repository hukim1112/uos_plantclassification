import numpy as np
import matplotlib.pyplot as plt

def plot_)


x = df3['scr']
y = df3['recall_gap']

plt.title("acc-gain(%) per scr")
plt.plot(x,y*100,'o')
m, b = np.polyfit(x, y*100, 1)
plt.plot(x, m*x+b)