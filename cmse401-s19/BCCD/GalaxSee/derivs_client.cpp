//////////////////////////////////////////////////////////
// GalaxSee (version MPI 0.9)
// Copyright 1997 - 2002
// David A. Joiner and the Shodor Education Foundation
// $Id: derivs_client.cpp,v 1.4 2012/05/30 17:30:27 charliep Exp $
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
#include "modeldata.h"
#include "mpi.h"
#include "mpidata.h"
#include <stdio.h>

extern mpidata g_mpi;


void derivs_client() {

    int npoints;
    int npercpu;
    double gnorm;
    double * mass;
    double * shield_rad;
    double * x;

    // Get information from server
    MPI_Recv(&npoints,1,MPI_INT,0,MPIDATA_PASSNUMBER,
        MPI_COMM_WORLD,&g_mpi.status);    
    MPI_Recv(&npercpu,1,MPI_INT,0,MPIDATA_PASSNUMBERPER,
        MPI_COMM_WORLD,&g_mpi.status);
    MPI_Recv(&gnorm,1,MPI_DOUBLE,0,MPIDATA_PASSGNORM,
        MPI_COMM_WORLD,&g_mpi.status);
    mass = new double[npoints];
    shield_rad = new double[npoints];
    x = new double[npoints*6];
    MPI_Recv(mass,npoints,MPI_DOUBLE,0,MPIDATA_PASSMASS,
        MPI_COMM_WORLD,&g_mpi.status);
    MPI_Recv(shield_rad,npoints,MPI_DOUBLE,0,MPIDATA_PASSSHIELD,
        MPI_COMM_WORLD,&g_mpi.status);
    MPI_Recv(x,npoints*6,MPI_DOUBLE,0,MPIDATA_PASSX,
       MPI_COMM_WORLD,&g_mpi.status);

    // Do stuff and return values
    double * retval = new double[npercpu*6];
    for (int i=0;i<npercpu*6;i++) {retval[i]=0.0;}
    double rad,rad2,dcon,diffx,diffy,diffz;
    int block = npoints/g_mpi.size;
    for(int i=0;i<npercpu;i++){
        int ireal = g_mpi.rank*(block)+i;
        for(int j=0;j<npoints;j++){
          if (ireal!=j) {
            rad2=pow((x[ireal]-x[j]),2)+
                pow((x[npoints+ireal]-x[npoints+j]),2)+
                pow((x[2*npoints+ireal]-x[2*npoints+j]),2);
            rad=sqrt(rad2);
            dcon=gnorm/(rad*rad2);
            diffx=(x[j]-x[ireal])*dcon;
            diffy=(x[npoints+j]-x[npoints+ireal])*dcon;
            diffz=(x[2*npoints+j]-x[2*npoints+ireal])*dcon;

            if (rad>shield_rad[j]) {
                retval[3*npercpu+i]+=diffx*mass[j];
                retval[4*npercpu+i]+=diffy*mass[j];
                retval[5*npercpu+i]+=diffz*mass[j];
            }
          }  
        }
        retval[i]=x[3*npoints+ireal];
        retval[npercpu+i]=x[4*npoints+ireal];
        retval[2*npercpu+i]=x[5*npoints+ireal];
    }
    
    
    MPI_Send(retval,npercpu*6,MPI_DOUBLE,0,
        MPIDATA_DONEDERIVS,MPI_COMM_WORLD);
    
    delete mass;
    delete shield_rad;
    delete x;
    delete retval;
}
