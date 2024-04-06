import matplotlib.pylab as plt
import numpy as np

xmin = 0.0
xmax = 10.0
nx = 500
gamma = 1
x = np.linspace(xmin, xmax, nx)
dx = (xmax-xmin)/nx

y = np.zeros(nx)
y = np.exp(-(np.array(x)-5.0)**2)

tmin = 0.0  
tmax = 10.0 
nt = 1000000
tgrid = np.linspace(tmin,tmax,nt)
dt = (tmax-tmin)/nt

v = np.zeros(nx)
dvdt = np.zeros(nx)


show = True
count = 0
ddx = dx*dx
dvdt2 = dvdt.copy()
for t in tgrid:
    
    #laplacian
    dvdt[1:-2] = gamma*(y[:-3]-2*y[1:-2]+y[2:-1])/ddx  
    y += v*dt
    v += dvdt*dt

print(y)
