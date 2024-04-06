import numpy
import pickle
import matplotlib.pyplot as plt
import copy

# Import the required orbkit modules
from orbkit import grid, read, core

from orbkit import detci
from orbkit import main, options

options.quiet = False

# Define Some Constants
eV = 27.211384
Debye = 2.5415800000001716
Ampere = 0.006623619
au2cm = 219474.63

# Set some options
numproc = 8
slice_length = 1e4
ci_readthreshold = 0.0

print('''
==========================================
Reading the Quantum Chemistry Output Files
==========================================
''')
# 1a.
fid_gamess = 'Febpy_CI.out'
fid_molden = 'bpy.molden'
qc = read.main_read(fid_molden,all_mo=True)
for i in range(590-94):
  del qc.mo_spec[-1]
qc,ci = detci.ci_read.main_ci_read(qc,fid_gamess,itype='gamess_MOD',
                                threshold=ci_readthreshold)
ci_pure = copy.deepcopy(ci)

npure = len(ci)
print(npure)

E = []
SocH = [[]]

f = open("Febpy_CI.dat", "r")

SPN_ROT_FLAG = False
SPN_ROT_IND = 1
SPN_ENS_FLAG = False

for line in f.readlines():
    
    if line.strip() ==  "$SPNORB ---   CI   SPIN-ORBIT MATRIX ELEMENTS.":
        SPN_ROT_FLAG = True
        continue
        
    if SPN_ROT_FLAG:
        if line.strip() == "CI   ADIABATIC STATES---":
            SPN_ROT_FLAG = False
            continue
        l = line.replace(" ", "  ").replace("-", " -").replace("E -", "E-").split()
        if len(l) == 6:
            if int(l[0]) != SPN_ROT_IND % 100:
                SPN_ROT_IND += 1
                SocH.append([])
            SocH[-1].extend(l[1:])
        else:
            SocH[-1].extend(l)
    if line.strip() == "CI   SPIN-MIXED STATES---":
        SPN_ENS_FLAG = True
        continue
        
    if SPN_ENS_FLAG:
        if line.strip() == "$END":
            SPN_ENS_FLAG = False
            continue
        l = line.split()
        E.extend(l)

SocH = numpy.array(SocH, float)
SocH = (SocH[:,0::2] + 1j*SocH[:,1::2]) / au2cm

E = numpy.array(E, float) 
E /= au2cm
SocH -= numpy.diag([E[0]]*npure)
E -= E[0]

H_0 = numpy.diag(E)

# Zero of energy set to spin mixed ground state
# Diag of SocH gives spin pure energies

U_s_ev, U_s = numpy.linalg.eigh(SocH)
print(min(numpy.isclose(U_s_ev, E)))


# print('='*80)
# print('ZS: Cleaning CI State Info')
# print('='*80+'\n')
# 
# for s in range(len(ci)):
#   st = ci[s]
#   temp_occ = []
#   temp_coeffs = []
#   for i in range(len(st.coeffs)):
#     if st.occ[i].tolist() in temp_occ:
#       temp_coeffs[temp_occ.index(st.occ[i].tolist())] += st.coeffs[i]
#     else:
#       temp_occ.append(st.occ[i].tolist())
#       temp_coeffs.append(st.coeffs[i])
#   ci[s].coeffs = numpy.array(temp_coeffs)
#   ci[s].occ = numpy.array([numpy.array(x) for x in temp_occ], dtype=numpy.int32)
#   
# 
print('='*80)
print('Preparing All Subsequent Calculations')
print('='*80+'\n')
# 1b.
print('Setting up the grid...')
grid.adjust_to_geo(qc,extend=1.0,step=0.25)
grid.grid_init()    
print(grid.get_grid())

with open("grid.pk","wb") as outfile:
  pickle.dump([grid.x, grid.y, grid.z],outfile,pickle.HIGHEST_PROTOCOL)

print('Computing the Molecular Orbitals and the Derivatives Thereof...\n')
molist = core.rho_compute(qc,
                          calc_mo=True,
                          slice_length=slice_length,
                          drv=[None,'x','y','z','xx','yy','zz'],
                          numproc=numproc)
molistdrv = molist[1:4]                                # \nabla of MOs
molistdrv2 = molist[-3:]                               # \nabla^2 of MOs
molist = molist[0]                                     # MOs

print('''
==========================================
  Starting the detCI@ORBKIT Computations
==========================================
''')


SoI = [x for x in range(20)]
SoI = [0, 3, 4, 7, 8, 10, 12, 18, 19, 22, 25, 27, 30, 31, 34, 38, 40, 45, 48, 50, 52, 53, 54, 55, 57, 59, 63]
SoI.sort()

rho = {}
rho_a = {}
rho_b = {}
j = {}
j_a = {}
j_b = {}


runs = 0

zero,sing,zero_a,sing_a,zero_b,sing_b = detci.occ_check.compare(ci[0],ci[1],True,numproc=numproc)
rho_0 = detci.ci_core.rho(zero,sing,molist,slice_length=slice_length,numproc=numproc)
j_0 = detci.ci_core.jab(zero,sing,molist,molistdrv,                    # Imag(<s1|j|s2>)
                                               slice_length=slice_length,numproc=numproc)
for i1 in SoI:
  for i2 in SoI:
    if i2 < i1:
      continue
    rho[str(i1)+","+str(i2)] = 0.*rho_0
    rho_a[str(i1)+","+str(i2)] = 0.*rho_0
    rho_b[str(i1)+","+str(i2)] = 0.*rho_0
    if i1 != i2:
      j[str(i1)+","+str(i2)] = 0.*j_0
      j_a[str(i1)+","+str(i2)] = 0.*j_0
      j_b[str(i1)+","+str(i2)] = 0.*j_0

for ip1 in range(npure):
  for ip2 in range(npure):
    if ip2 < ip1:
      continue
    if ci[ip1].info["spin"] != ci[ip2].info["spin"]: continue # Check if spin matches
    CALC = False
    for i1 in SoI:
      for i2 in SoI:
        if i2 < i1:
          continue
        c =   U_s[ip1,i1].real * U_s[ip2,i2].real \
            + U_s[ip2,i1].real * U_s[ip1,i2].real \
            + U_s[ip1,i1].imag * U_s[ip2,i2].imag \
            + U_s[ip2,i1].imag * U_s[ip1,i2].imag   # Coefficient of pure density <ip1|ip2> + <ip2|ip1>
        c *= 2                                      # Double to account for <i1|i2> and <i2|i1>
        if ip1 == ip2: c /= 2                       # Account for <ip1|ip1>
        if i1 == i2: c /= 2                         # Account for <i1|i1>
        if abs(c) > 1E-4:
          if CALC == False: 
            print("\n",ip1,ip2)
            zero,sing,zero_a,sing_a,zero_b,sing_b = detci.occ_check.compare(ci[ip1],ci[ip2],True,numproc=numproc)
            pure_rho = detci.ci_core.rho(zero,sing,molist,slice_length=slice_length,numproc=numproc)
            pure_rho_a = detci.ci_core.rho(zero_a,sing_a,molist,slice_length=slice_length,numproc=numproc)
            pure_rho_b = detci.ci_core.rho(zero_b,sing_b,molist,slice_length=slice_length,numproc=numproc)
            if i1 != i2 or ip1 != ip2:
              pure_j = detci.ci_core.jab(zero,sing,molist,molistdrv,                    # Imag(<s1|j|s2>)
                                               slice_length=slice_length,numproc=numproc)
              pure_j_a = detci.ci_core.jab(zero_a,sing_a,molist,molistdrv,                    # Imag(<s1|j|s2>)
                                               slice_length=slice_length,numproc=numproc)
              pure_j_b = detci.ci_core.jab(zero_b,sing_b,molist,molistdrv,                    # Imag(<s1|j|s2>)
                                               slice_length=slice_length,numproc=numproc)
            CALC = True
          print(i1,i2)
          rho[str(i1)+","+str(i2)] += c * pure_rho
          rho_a[str(i1)+","+str(i2)] += c * pure_rho_a
          rho_b[str(i1)+","+str(i2)] += c * pure_rho_b

        if i1 == i2 or ip1 == ip2: continue
        c =   U_s[ip1,i1].real * U_s[ip2,i2].imag \
            + U_s[ip1,i2].real * U_s[ip2,i1].imag \
            - U_s[ip2,i2].real * U_s[ip1,i1].imag \
            - U_s[ip2,i1].real * U_s[ip1,i2].imag   # Coefficient of pure density <ip1|ip2> + <ip2|ip1>
        c *= -2                                      # Double to account for <i1|i2> and <i2|i1>
        if abs(c) > 1E-4:
          if CALC == False:
            print("\n",ip1,ip2)
            zero,sing,zero_a,sing_a,zero_b,sing_b = detci.occ_check.compare(ci[ip1],ci[ip2],True,numproc=numproc)
            pure_rho = detci.ci_core.rho(zero,sing,molist,slice_length=slice_length,numproc=numproc)
            pure_rho_a = detci.ci_core.rho(zero_a,sing_a,molist,slice_length=slice_length,numproc=numproc)
            pure_rho_b = detci.ci_core.rho(zero_b,sing_b,molist,slice_length=slice_length,numproc=numproc)
            if i1 != i2 or ip1 != ip2:
              pure_j = detci.ci_core.jab(zero,sing,molist,molistdrv,                    # Imag(<s1|j|s2>)
                                               slice_length=slice_length,numproc=numproc)
              pure_j_a = detci.ci_core.jab(zero_a,sing_a,molist,molistdrv,                    # Imag(<s1|j|s2>)
                                               slice_length=slice_length,numproc=numproc)
              pure_j_b = detci.ci_core.jab(zero_b,sing_b,molist,molistdrv,                    # Imag(<s1|j|s2>)
                                               slice_length=slice_length,numproc=numproc)
            CALC = True
          print(i1,i2)
          j[str(i1)+","+str(i2)] += c * pure_j
          j_a[str(i1)+","+str(i2)] += c * pure_j_a
          j_b[str(i1)+","+str(i2)] += c * pure_j_b

        
with open("rho.pk","wb") as outfile:
  pickle.dump(rho,outfile,pickle.HIGHEST_PROTOCOL)
with open("rho_a.pk","wb") as outfile:
  pickle.dump(rho_a,outfile,pickle.HIGHEST_PROTOCOL)
with open("rho_b.pk","wb") as outfile:
  pickle.dump(rho_b,outfile,pickle.HIGHEST_PROTOCOL)
with open("j.pk","wb") as outfile:
  pickle.dump(j,outfile,pickle.HIGHEST_PROTOCOL)
with open("j_a.pk","wb") as outfile:
  pickle.dump(j_a,outfile,pickle.HIGHEST_PROTOCOL)
with open("j_b.pk","wb") as outfile:
  pickle.dump(j_b,outfile,pickle.HIGHEST_PROTOCOL)

exit()

