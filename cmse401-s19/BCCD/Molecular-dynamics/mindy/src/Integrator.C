#include <sys/time.h>
#include "PDB.h"
#include "Parameters.h"
#include "Molecule.h"
#include "Mindy.h"
#include "Vector.h"
#include "ComputeBonded.h"
#include "ComputeNonbonded.h"
#include <iostream>

// get the time of day from the system clock, and store it (in seconds)
static double time_of_day(void) {
  struct timeval tm;
  struct timezone tz;

  gettimeofday(&tm, &tz);
  return((double)(tm.tv_sec) + (double)(tm.tv_usec)/1000000.0);
}


void MIN_die(const char *s) {
  printf("%s\n",s);
  exit(1);
}

#define SWITCHDIST 8.5 
#define CUTOFF     10 
#define PAIRLISTDIST 11.5 

int main(int argc, char *argv[]) {
  // argv[1] is the number of steps
  // argv[2] is the pdb file, argv[3] is the psf file, and the rest are
  // parameter files in xplor format.
 
  cout << "Mindy v" << MINDYVERSION << endl;

  if (argc<5) {
    cout << "Usage: " << argv[0] << " nsteps pdb psf paramfile\n";
    return 1;
  }

  int i;
  MinParameters minparams;
  int nsteps = atoi(argv[1]);
  minparams.pdbname = argv[2];
  minparams.psfname = argv[3];
  minparams.prmname = argv[4];
  minparams.switchdist = SWITCHDIST;
  minparams.cutoff     = CUTOFF;
  minparams.pairlistdist = PAIRLISTDIST; 

  PDB pdb(minparams.pdbname);

  Parameters params(minparams.prmname);

  Molecule mol(&params, minparams.psfname);
 
  const int natoms = pdb.num_atoms();
  Vector *pos = new Vector[natoms];
  Vector *vel = new Vector[natoms];
  Vector *f = new Vector[natoms];
  double *imass = new double[natoms];
  memset((void *)vel, 0, natoms*sizeof(Vector));
  for (i=0; i<natoms; i++)
    imass[i] = 1.0/mol.atommass(i);
  pdb.get_all_positions(pos);

  ComputeBonded bonded(&mol, &params);

  ComputeNonbonded nonbonded(&mol, &params, &minparams);
  
  double Ebond, Eangle, Edihedral, Eimproper, Evdw, Eelec;
  
  Ebond = Eangle = Edihedral = Eimproper = Evdw = Eelec = 0;
  //
  // Begin velocity verlet integration
  //
  const double dt = 1.0/TIMEFACTOR;
  double t = 0.0;
  double Ekin, Etot;   
  Ekin = 0.0;
  // Compute forces at time 0
  memset((void *)f, 0, natoms*sizeof(Vector));
  bonded.compute(pos, f, Ebond, Eangle, Edihedral, Eimproper);
  nonbonded.compute(&mol, pos, f, Evdw, Eelec);
  Etot = Ebond + Eangle + Edihedral + Eimproper + Evdw + Eelec + Ekin;
  cout << "t        bond    angle   dihedral   improper   vdw         elec       kinetic  total" << endl;
  cout << t << "     " << Ebond << "    " << Eangle << "  " << Edihedral << "    "
       << Eimproper << "    " << Evdw << "    " << Eelec << "    " << Ekin 
       << "    " <<Etot<<endl; 
  double start = time_of_day();
  for (int i=0; i<nsteps; i++) { 
    int j;
    for (j=0; j<natoms; j++) {
      pos[j] += dt*vel[j] + 0.5*dt*dt*f[j]*imass[j];
      vel[j] += 0.5*dt*f[j]*imass[j];
    }
    
    // Compute forces at time t+dt
    memset((void *)f, 0, natoms*sizeof(Vector));
    bonded.compute(pos, f, Ebond, Eangle, Edihedral, Eimproper);
    nonbonded.compute(&mol, pos, f, Evdw, Eelec);
 
    Ekin = 0; 
    for (j=0; j<natoms; j++) {
      vel[j] += 0.5*dt*f[j]*imass[j];
      Ekin += vel[j]*vel[j]*mol.atommass(j);
    }

    Ekin *= 0.5;
    Etot = Ebond + Eangle + Edihedral + Eimproper + Evdw + Eelec + Ekin;
    t += dt*TIMEFACTOR;

    if (!(i%100))  
      cout << t << "     " << Ebond << "    " << Eangle << "  " << Edihedral << "    " << Eimproper << "    " << Evdw << "    " << Eelec << "    " << Ekin << "    " <<Etot<<endl; 
  }
  double stop = time_of_day();
  cout << t << "     " << Ebond << "    " << Eangle << "  " << Edihedral << "    " << Eimproper << "    " << Evdw << "    " << Eelec << "    " << Ekin << "    " <<Etot<<endl; 
  cout << "time per step = " << (stop-start)/nsteps << endl;

  delete [] f;
  delete [] vel;
  delete [] pos;
  delete [] imass;
  return 0;
}   
