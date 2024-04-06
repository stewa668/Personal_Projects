//////////////////////////////////////////////////////////
// GalaxSee (version MPI 0.9)
// Copyright 1997 - 2002
// David A. Joiner and the Shodor Education Foundation
// $Id: diffeq.cpp,v 1.3 2012/05/01 15:31:46 charliep Exp $
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

#include <math.h>
#include "diffeq.h"


void diffeq::init(int j){
   neq=j;
   x=new double[neq];
   der=new double[neq];
   x_old=new double[neq];
   der_old=new double[neq];
   k1=new double[neq];
   k2=new double[neq];
   k3=new double[neq];
   k4=new double[neq];
   for (int i=0;i<neq;i++) {
      x[i]=0.0;
      der[i]=0.0;
      x_old[i]=0.0;
      der_old[i]=0.0;
      k1[i]=0.0;
      k2[i]=0.0;
      k3[i]=0.0;
      k4[i]=0.0;
   }
   time=0.0;
}

void diffeq::cleanup(){
	delete x;
	delete der;
	delete x_old;
	delete der_old;
	delete k1;
	delete k2;
	delete k3;
	delete k4;
}

void diffeq::updateEuler(double step, void deriv(int,double,double*,double*)){
   deriv(neq,time,x,der);
   for(int i=0;i<neq;i++){
      x[i]=x[i]+step*der[i];
   }
   time=time+step;
}

void diffeq::updateIEuler(double step, void deriv(int,double,double*,double*)){
   for(int i=0;i<neq;i++){
      x_old[i]=x[i];
   }
   deriv(neq,time,x,der);
   for(int i=0;i<neq;i++){
      der_old[i]=der[i];
      x[i]=x[i]+step*der[i];
   }
   deriv(neq,time,x,der);
   for(int i=0;i<neq;i++){
      x[i]=x_old[i]+step*0.5*(der[i]+
         der_old[i]);
   }
   time=time+step;
}

void diffeq::updateRKutta4(double step, void deriv(int,double,double*,double*)){
   for(int i=0;i<neq;i++){
      x_old[i]=x[i];
   }
   double time_old=time;
   deriv(neq,time,x,der);
   for(int i=0;i<neq;i++){
      k1[i]=step*der[i];
      x[i]=x_old[i]+0.5*k1[i];
   }
   time=time_old+step/2.0;
   deriv(neq,time,x,der);
   for(int i=0;i<neq;i++){
      k2[i]=step*der[i];
      x[i]=x_old[i]+0.5*k2[i];
   }
   deriv(neq,time,x,der);
   time=time_old+step;
   for(int i=0;i<neq;i++){
      k3[i]=step*der[i];
      x[i]=x_old[i]+k3[i];
   }
   deriv(neq,time,x,der);
   double con6=1.0/6.0;
   for(int i =0;i<neq;i++){
      k4[i]=step*der[i];
      x[i]=x_old[i]+(k1[i]+2.0*k2[i]+2.0*k3[i]+k4[i])*con6;
   }
}

void cart3d::init(int j){
	n=j;
	x = new double[n];
	y = new double[n];
	z = new double[n];
	for (int i=0;i<j;i++) {
		x[i]=0.0;
		y[i]=0.0;
		z[i]=0.0;
	}
}

void mapPoints(int neq, int ndim, double * x, double * der, cart3d * pos,
      cart3d * vel, cart3d * xder, cart3d* vder){

      div_t check=div(neq,2*ndim);
      int npoints=check.quot;
      pos->x=x;
      pos->y=x+npoints;
      pos->z=x+2*npoints;;
      vel->x=x+3*npoints;;
      vel->y=x+4*npoints;;
      vel->z=x+5*npoints;;
      xder->x=der;
      xder->y=der+npoints;
      xder->z=der+2*npoints;;
      vder->x=der+3*npoints;;
      vder->y=der+4*npoints;;
      vder->z=der+5*npoints;;
}

void dynamic::init(int j){
   npoints=j;
   neq=6*j;
   diffeq::init(neq);
   mapPoints(neq,3,x,der,&pos,&vel,&xder,&vder);
}
