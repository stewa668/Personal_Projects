
from math import *

xmin = 0.0
xmax = 10.0
nx = 500
dx = (xmax-xmin)/(nx-1.0)
x = [0.0]*nx
x[0]=0.0
for i in range(1,nx-2):
    x[i]=xmin+i*dx
x[nx-1]=10.0

nt = 100000
tmin = 0.0
tmax = 10.0
dt = (tmax-tmin)/(nt-1.0)
tgrid = [0.0]*nt
tgrid[0]=0.0
for i in range(1,nt-2):
    tgrid[i]=tmin+i*dt
tgrid[nx-1]=10.0


y = [0.0]*nx
v = [0.0]*nx
dvdt = [0.0]*nx
for i in range(0,nx-1):
    y[i] = exp(-(x[i]-5.0)**2)

for t in tgrid:
    for i in range(1,nx-2):
        dvdt[i]=(y[i+1]+y[i-1]-2.0*y[i])/dx/dx
    for i in range(0,nx-1):
        y[i] = y[i] + v[i]*dt
        v[i] = v[i] + dvdt[i]*dt
    
for i in range(0,nx-1):
    print(y[i])

