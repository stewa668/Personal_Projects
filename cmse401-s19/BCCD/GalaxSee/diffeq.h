//////////////////////////////////////////////////////////
// GalaxSee (version MPI 0.9)
// Copyright 1997 - 2002
// David A. Joiner and the Shodor Education Foundation
// $Id: diffeq.h,v 1.4 2012/06/27 18:44:08 mmludin08 Exp $
// This file is part of BCCD, an open-source live CD for computational science
// education.
// 
// Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave Joiner, 
//   Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, & Aaron Weeden 

// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//////////////////////////////////////////////////////////


#ifndef _DIFFEQ
#define _DIFFEQ

#include <math.h>
#include <stdlib.h>

#define IMETHOD_EULER 0
#define IMETHOD_IEULER 1
#define IMETHOD_RKUTTA4 2


class diffeq{
  public:
   int neq;
   double *x;
   double time;

  public:
   double * der;
   double * x_old;
   double * der_old;
   double * k1;
   double * k2;
   double * k3;
   double * k4;

  public:
   void init(int j);
   void cleanup();
   void updateEuler(double, void no_name(int,double,double*,double*));
   void updateIEuler(double, void no_name(int,double,double*,double*));
   void updateRKutta4(double, void no_name(int,double,double*,double*));
};

class cart3d{
  public:
   int n;
   double * x;
   double * y;
   double * z;

   void init(int j);

};

class dynamic : public diffeq {
  public:
   int npoints;
   cart3d pos;
   cart3d vel;
   cart3d xder;
   cart3d vder;

   void init(int);
};

void mapPoints(int, int, double *, double *, cart3d *,
      cart3d *, cart3d *, cart3d*);


#endif
