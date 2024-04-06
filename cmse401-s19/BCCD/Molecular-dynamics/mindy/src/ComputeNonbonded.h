
#ifndef COMPUTE_NONBONDED_H__
#define COMPUTE_NONBONDED_H__

class Molecule;
class Parameters;
class Vector;
class LJTable;
struct MinParameters;

struct Pair {
  const Vector *pos1;
  const Vector *pos2;
  double kqq;   // The electrostatic factor
  double vdwA;
  double vdwB;
};

#define COULOMB 332.0636

class ComputeNonbonded {
public:
  ComputeNonbonded(const Molecule *, const Parameters *, const MinParameters *);
  ~ComputeNonbonded();

  void compute(const Molecule *mol, const Vector *pos, 
               Vector *f, double& Evdw, double &Eelec);
private:
  int natoms;

  double cut2;
  double switch2;
  double pair2;
  
  // vdW switching
  double c1, c3;
  LJTable *ljTable;
};

#endif

