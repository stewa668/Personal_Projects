import matplotlib.pyplot as plt
from numpy import *

steps = loadtxt("steps.txt")
err = loadtxt("data2.txt")
time = loadtxt("data3.txt")
err = (err[0:9] + err[9:18] + err[18:27] + err[27:36] + err[36:45])/5
time = (time[0:9] + time[9:18] + time[18:27] + time[27:36] + time[36:45])/5
plt.xscale("log")
plt.plot(steps, abs(err))
plt.title("steps vs error")
plt.xlabel("steps")
plt.ylabel("error")
plt.savefig("plot1.png", dpi=300)
plt.yscale("log")
plt.savefig("plot1_1.png", dpi=300)
plt.clf()
plt.xscale("log")
plt.plot(steps, time)
plt.title("steps vs time")
plt.xlabel("steps")
plt.ylabel("time")
plt.savefig("plot2.png", dpi=300)
plt.clf()
