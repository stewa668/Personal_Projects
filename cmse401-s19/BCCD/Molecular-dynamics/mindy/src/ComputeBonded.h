#ifndef COMPUTE_BONDED_H__
#define COMPUTE_BONDED_H__

class Molecule;
class Parameters;
class Vector;
struct BondElem;
struct AngleElem;
struct DihedralElem;
struct ImproperElem;

class ComputeBonded {
public:
  // constructor creates the lists of bonds, angles, etc.
  ComputeBonded(const Molecule *, const Parameters *);
  ~ComputeBonded();

  // compute the energy, given the set of coordinates
  void compute(const Vector *coords, Vector *f, double& Ebond, double& Eangle,
               double &Edihedral, double &Eimproper) const;

private:
  int nbonds;
  int nangles;
  int ndihedrals;
  int nimpropers;

  BondElem *bonds;
  AngleElem *angles; 
  DihedralElem *dihedrals;
  ImproperElem *impropers;
   
  void build_bondlist(const Molecule *, const Parameters *);
  void build_anglelist(const Molecule *, const Parameters *);
  void build_dihedrallist(const Molecule *, const Parameters *);
  void build_improperlist(const Molecule *, const Parameters *);

  double compute_bonds(const Vector *, Vector *) const;
  double compute_angles(const Vector *, Vector *) const;
  double compute_dihedrals(const Vector *, Vector *) const;
  double compute_impropers(const Vector *, Vector *) const;
};

#endif
 
