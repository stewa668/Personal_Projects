from numpy import *
import pickle5 as pk
from mayavi.mlab import *
import mayavi
import os
from IPython.display import clear_output
from tvtk.util.ctf import PiecewiseFunction, ColorTransferFunction
import matplotlib.pyplot as plt

hbar = 1.054571817E-34
elem = 1.602176634E-19
mass = 9.109383701528E-31
coul = 8.987551792314E9
perm = 8.854187812813E-12
bohr = 4*pi*perm*hbar**2/elem**2/mass
hart = hbar**2/mass/bohr**2
1E-15/(hbar/hart)

with open('orb/rho.pk','rb') as infile:
    rho = pk.load(infile)
    
with open('orb/j.pk','rb') as infile:
    J = pk.load(infile)

with open('orb/grid.pk','rb') as infile:
    grid = pk.load(infile)
    
X, Y, Z = meshgrid(grid[0], grid[1], grid[2], indexing='ij')

nsi = loadtxt("DFT_ss_TDCI/y2z/nstate_i.t", skiprows=1)
print(len(nsi))

num_states = int((len(nsi[0])-1)/2)
times = nsi[:,0]
c_r = []
c_i = []
for i in range(num_states):
    c_r.append(list(nsi[:,1+2*i]))
for i in range(num_states):
    c_i.append(list(nsi[:,2+2*i]))
c_r = array(c_r)
c_i = array(c_i)
times_au = times*1E-15/(hbar/hart)

out_path = ''
out_path = os.path.abspath(out_path)
fps = 24
prefix = 'bpy'
ext = '.png'

padding = len(str(len(times)))

vw = (-42.537585270415676,
 89.7402743118783,
 31.046066152956463,
 array([ 0.03573193, -0.57389574, -0.22883734]))

start_step = 0
modulo = 600



options.offscreen = True

f = figure(size=(1920,1080), bgcolor=(1,1,1))

ctf = ColorTransferFunction()
ctf.range = [-.02,.02]

otf = PiecewiseFunction()
otf.add_point(0-.1,0)

otf.add_point(0-.02,0)
otf.add_point(0-.01,1)
otf.add_point(0-0.0001,0)
otf.add_point(0+.0001,0)
otf.add_point(0+.01,1)
otf.add_point(0+.02,0)

otf.add_point(0+.1,0)

f.scene.disable_render = False

source = pipeline.scalar_field(X,Y,Z,rho["0,1"])

vol = pipeline.volume(source, vmin=-.01, vmax=.01, figure=f)

jnorm = sqrt(J["0,1"][0]**2+J["0,1"][1]**2+J["0,1"][2]**2)

qui = quiver3d(X,Y,Z,J["0,1"][0],J["0,1"][1],J["0,1"][2], scalars=jnorm,
                   scale_mode="scalar", transparent=True, figure=f)

for t_i0 in range(len(times[start_step:])):
    
    t_i0 += start_step
    
    t_i = t_i0*modulo

    if t_i0%10==0: print(t_i0, t_i, times[t_i0])

#    f = figure(size=(1920,1080), bgcolor=(1,1,1))

    rho_t = -rho["0,0"]
    J_t = J["0,1"]*0
    for i in range(num_states):
        c = c_r[i,t_i]**2 + c_i[i,t_i]**2
        rho_t += c*rho[str(i)+","+str(i)]
        for j in [x+i+1 for x in range(num_states-i-1)]:
            c = c_r[i,t_i]*c_r[j,t_i] + c_i[i,t_i]*c_i[j,t_i]
            rho_t += 2*c*rho[str(i)+","+str(j)]
            c = c_r[i,t_i]*c_i[j,t_i] - c_r[j,t_i]*c_i[i,t_i]
            J_t += -2*c*J[str(i)+","+str(j)]
			
    #f.scene.disable_render = True

    vol.mlab_source.scalars = rho_t
    #vol.mlab_source.vmin = -.01
    #vol.mlab_source.vmax = .01

    #ctf = ColorTransferFunction()
    #ctf.range = [-.02,.02]
    vol._ctf = ctf
    vol.update_ctf = True

    #otf = PiecewiseFunction()
    #otf.add_point(0-.1,0)
	#
    #otf.add_point(0-.02,0)
    #otf.add_point(0-.01,1)
    #otf.add_point(0-0.0001,0)
    #otf.add_point(0+.0001,0)
    #otf.add_point(0+.01,1)
    #otf.add_point(0+.02,0)
	#
    #otf.add_point(0+.1,0)
    vol._otf = otf
    vol._volume_property.set_scalar_opacity(otf)

    jnorm = sqrt(J_t[0]**2+J_t[1]**2+J_t[2]**2)
    jnorm[ jnorm < .0003 ] = 0

    qui.mlab_source.u = J_t[0]
    qui.mlab_source.v = J_t[1]
    qui.mlab_source.w = J_t[2]

    view(vw[0], vw[1], vw[2], vw[3], roll=0)

    zeros = '0'*(padding - len(str(t_i0)))
    filename = os.path.join(out_path, '{}_{}{}{}'.format(prefix, zeros, t_i0, ext))

    #f.scene.disable_render = False
    #draw(f)
	
    savefig(filename=filename)
    #print(gcf())
    #clf(f)