//////////////////////////////////////////////////////////
// GalaxSee (version MPI 0.9)
// Copyright 1997 - 2002
// David A. Joiner and the Shodor Education Foundation
// $Id: modeldata.cpp,v 1.3 2012/05/01 15:31:46 charliep Exp $
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

#include "modeldata.h"
#include <time.h>
#include <stdio.h>

modeldata::modeldata(){
	//set up default galaxy
	default_numstars=512;
	default_shape=SHAPE_SPHERE;
	default_rotatefac=1.0;
	default_scale=SCALE_GALACTIC;
	default_starmass=200.0;
	default_imethod=IMETHOD_IEULER;
	default_fmethod=FMETHOD_DIRECT;
	default_timestep=0.1;
}

void modeldata::init(){
	int j=default_numstars;
	double *massin = new double[j];
	int *colorin = new int[j];
	cart3d posin;
	cart3d velin;
	posin.init(j);
	velin.init(j);

	modeldata::init(j,posin,velin,massin,colorin);
	
}



void modeldata::init(int j,cart3d posin,
					 cart3d velin,double * min,
					 int * cin){
	dynamic::init(j);
	for (int i=0;i<npoints;i++) {
		pos.x[i]=posin.x[i];
		pos.y[i]=posin.y[i];
		pos.z[i]=posin.z[i];
		vel.x[i]=velin.x[i];
		vel.y[i]=velin.y[i];
		vel.z[i]=velin.z[i];
	}
	// Units are MKS.
	msun=1.99e30;
	mearth=5.97e24;
	au=1.50e11;
	gconst=6.673e-11;
	kly=3.00e8*60.0*60.0*24.0*365.25*1.0e3;

	// OK, how should the different possible
	// scales be defined? How about a set of
	// subroutines which determine the scale
	// based upon some default values.

	// Option 1- Solar system scale, mass 
	// is earth mass, distance is AU,
	// time is day.

	// Option 2- Galactic scale, mass is solar
	// mass, distance is kPc, time is year.

	current_scale=default_scale;
	define_scale(current_scale);
	
	mass=min;
	color=cin;
	disp_pos.init(npoints);
	

	// Along with the masses of each star, I want to store
	// the spherical coordinates of each point(DONE), the
	// spherical coordinates of the display, the cartesian
	// coordinates of the display, and the display coordinates.
	// (display coordinates are in x-y, with an array
	// of integers designating ordering (furthest back
	// points are drawn first.

	//initialize phase for viewable coordinates.
	disp_theta=disp_phi=0.0;
	disp_x=new double[npoints];
	disp_y=new double[npoints];
	order_z=new double[npoints];
	disp_offset_x=0.0;
	disp_offset_y=0.0;
	
	disp_order = new int[npoints];


	//box corrdinates
	box_corn_x=new double[8];
	box_corn_y=new double[8];
	box_rear=new bool [8];

	// Model run information
	time_step=default_timestep;
	force_method=default_fmethod;
	int_method=default_imethod;

	//Color coding info for each body
	color_hue  =new double[npoints];
	color_red  =new double[npoints];
	color_green=new double[npoints];
	color_blue =new double[npoints];
	//dot information
	use_cooldot=0;
	point_size=1;
	point_color=0;
	hide_color=new bool [8];
	for (int i=0;i<8;i++) {
		hide_color[i]=false;
	}
	depth_shading=true;
	chroma_depth=false;
	red_shift=false;
	black_background=true;
	//default color information
	default_hue=new double[8];
	default_hue[0]=-1.0;  //white
	default_hue[1]=0.0;   //red
	default_hue[2]=30.0;  //orange
	default_hue[3]=60.0;  //yellow
	default_hue[4]=110.0; //green
	default_hue[5]=230.0; //blue
	default_hue[6]=290.0; //purple
	default_hue[7]=170.0; //cyan

	//rotation info
	rotate_factor=default_rotatefac;
}

void modeldata::new_galaxy(){
	// OK, for a variety of objects of unit mass,
	// pick a random radial distance, and random
	// angles.  Convert to X, Y, and Z. Set velocity to
	// zero.
	time_t ltime;
	::time(&ltime);
    srand( (unsigned)ltime);
	int k=RAND_MAX;

	double b2=box_edge_length/2.0;

	int i,j,l,m;
	double rad;
	double theta;
	double x,y,z,r;
	int ds=1;
	int dr=1;
	int dx=1;
	int dy=1;
	int dz=1;
	switch(default_shape) {
		case SHAPE_2D:
			while (dx*dy<npoints) {
				dx++;
				dy++;
			}
			if (dx*dy>npoints) {
				dx--;
				dy--;
				npoints=dy*dx;
			}
			for (i=0;i<dx;i++) {
				x=-b2+box_edge_length*(double)i/(double)(dx-1);
				for (j=0;j<dy;j++) {
					y=-b2+box_edge_length*(double)j/(double)(dy-1);
					l=i+dx*j;
					z=0.0;
					r = sqrt (x*x+y*y+z*z) ;
					pos.x[l]=x;
					pos.y[l]=y;
					pos.z[l]=z;
				}
			}
		break;
		case SHAPE_3D:
			while (dx*dy*dz<npoints) {
				dx++;
				dy++;
				dz++;
			}
			if (dx*dy*dz>npoints) {
				dx--;
				dy--;
				dz--;
				npoints=dy*dx*dz;
			}
			for (i=0;i<dx;i++) {
				x=-b2+box_edge_length*(double)i/(double)(dx-1);
				for (j=0;j<dy;j++) {
					y=-b2+box_edge_length*(double)j/(double)(dy-1);
					for (m=0;m<dz;m++) {
						z=-b2+box_edge_length*(double)m/(double)(dz-1);
						l=i+dx*j+dx*dx*m;
						r = sqrt (x*x+y*y+z*z) ;
						pos.x[l]=x;
						pos.y[l]=y;
						pos.z[l]=z;
					}
				}
			}
		break;
		case SHAPE_RINGS:
			while (ds*dr<npoints) {
				if(ds==dr) {
					dr*=2;
				} else {
					ds*=2;
				}
			}
			for (i=0;i<ds;i++) {
				theta=(double)i/(double)ds*6.28318;
				for (j=1;j<=dr;j++) {
					l=i+ds*(j-1);
					rad=(double)j/(double)dr*b2;
					z=0.0;
					x=rad*cos(theta);
					y=rad*sin(theta);
					r = sqrt (x*x+y*y+z*z) ;
					pos.x[l]=x;
					pos.y[l]=y;
					pos.z[l]=z;
				}
			}
		break;
		case SHAPE_SPHERE:
			for (i=0;i<npoints;i++) {
				int iflag=1;
				while (iflag!=0) {
					x=b2-box_edge_length*(double)rand()/((double)k);
					y=b2-box_edge_length*(double)rand()/((double)k);
					z=b2-box_edge_length*(double)rand()/((double)k);
					r = sqrt (x*x+y*y+z*z) ;
					if (r<b2) iflag=0;
				}
				pos.x[i]=x;
				pos.y[i]=y;
				pos.z[i]=z;
			}
		break;
		case SHAPE_DISC:
			for (i=0;i<npoints;i++) {
				int iflag=1;
				while (iflag!=0) {
					x=b2-box_edge_length*(double)rand()/((double)k);
					y=b2-box_edge_length*(double)rand()/((double)k);
					z=0.0;
					r = sqrt (x*x+y*y) ;
					if (r<b2) iflag=0;
				}
				pos.x[i]=x;
				pos.y[i]=y;
				pos.z[i]=z;
			}
		break;
		case SHAPE_CUBE:
		default:
			for (i=0;i<npoints;i++) {
				int iflag=1;
				x=b2-box_edge_length*(double)rand()/((double)k);
				y=b2-box_edge_length*(double)rand()/((double)k);
				z=b2-box_edge_length*(double)rand()/((double)k);
				r = sqrt (x*x+y*y+z*z) ;
				pos.x[i]=x;
				pos.y[i]=y;
				pos.z[i]=z;
			}
		break;
	}
	for(i=0;i<npoints;i++) {
		vel.x[i]=0.0;
		vel.y[i]=0.0;
		vel.z[i]=0.0;
		mass[i]=default_starmass;
		color[i]=0;
	}
	spin_galaxy(rotate_factor);
	
}

void modeldata::spin_galaxy(double factor) {
	extern void derivs(int,double,double *, double *);
	double r,a,v,cosine,sine;
	derivs(neq,time_step,x,der);

	for(int i=0;i<npoints;i++) {
		r=sqrt(pos.x[i]*pos.x[i]+pos.y[i]*pos.y[i]);
		if(r>0.0) {
			a=sqrt(vder.x[i]*vder.x[i]+vder.y[i]*vder.y[i]);
			v=sqrt(a*r);
			cosine=pos.x[i]/r;
			sine=pos.y[i]/r;
			vel.x[i]=v*factor*(-sine);
			vel.y[i]=v*factor*cosine;
		}
	}

}


void modeldata::cleanup(){
	diffeq::cleanup();

	delete mass;
	delete color;

	delete disp_x;
	delete disp_y;
	delete order_z;
	
	delete disp_order;

	//box information
	delete box_corn_x;
	delete box_corn_y;
	delete box_rear;

	
	delete color_hue;
	delete color_red;
	delete color_green;
	delete color_blue;
	
	delete default_hue;


}

void modeldata::define_scale(int scale) {
	switch(scale) {
		case SCALE_SOLARSYSTEM:
			mzero= mearth;
			rzero= au;
			tzero= (60.0*60.0*24.0);
			azero= rzero/(tzero* tzero);
			gnorm= gconst* mzero/(azero* rzero* rzero);	
			box_edge_length= 100.0;
			disp_screen=10.0*box_edge_length;
			view_z=0.2*disp_screen;
			disp_scale=1.0/box_edge_length/
				view_z*(disp_screen-box_edge_length/2.0);
		break;
		case SCALE_EARTHSUN:
			mzero= mearth;
			rzero= au;
			tzero= (60.0*60.0*24.0);
			azero= rzero/(tzero* tzero);
			gnorm= gconst* mzero/(azero* rzero* rzero);	
			box_edge_length= 4.0;
			disp_screen=10.0*box_edge_length;
			view_z=0.2*disp_screen;
			disp_scale=1.0/box_edge_length/
				view_z*(disp_screen-box_edge_length/2.0);
			break;
		case SCALE_GALACTIC:
		default:
			mzero= msun;
			rzero= kly;
			tzero= (60.0*60.0*24.0*365.25*1.0e6);
			azero= rzero/(tzero* tzero);
			gnorm= gconst* mzero/(azero* rzero* rzero);	
			box_edge_length= 5.0;
			disp_screen=10.0*box_edge_length;
			view_z=0.2*disp_screen;
			disp_scale=1.0/box_edge_length/
				view_z*(disp_screen-box_edge_length/2.0);
		break;
	}
}

//compute info
void modeldata::compute_info(){
	

//elapsed time
//iterations per second

	// Compute potential energy by pointwise
	// determination of work required
	// to brgin a point in from infinity.

	double pe=0.0;
	double ke=0.0;
	px=0.0;
	py=0.0;
	pz=0.0;
	comx=0.0;
	comy=0.0;
	comz=0.0;
	total_mass=0.0;
	total_energy=0.0;
	double dx,dy,dz,rad2,rad;
	for(int i=0;i<npoints;i++) {
		double potential=0.0;
		for (int j=0;j<i;j++) {
			dx=pos.x[j]-pos.x[i];
			dy=pos.y[j]-pos.y[i];
			dz=pos.z[j]-pos.z[i];
			rad2=dx*dx+dy*dy+dz*dz;
			rad=sqrt(rad2);
			potential+=gnorm*mass[j]/rad;
		}
		pe-=potential*mass[i];
		double v2=(vel.x[i]*vel.x[i]+
			vel.y[i]*vel.y[i]+
			vel.z[i]*vel.z[i]);
		ke+=0.5*mass[i]*v2;
		px+=mass[i]*vel.x[i];
		py+=mass[i]*vel.y[i];
		pz+=mass[i]*vel.z[i];
		comx=mass[i]*pos.x[i];
		comy=mass[i]*pos.y[i];
		comz=mass[i]*pos.z[i];
		total_mass+=mass[i];
	}
	total_energy=pe+ke;
	comx/=total_mass;
	comy/=total_mass;
	comz/=total_mass;

}

double modeldata::calc_depth(double pz, double dz, double a) {
	double rvalue, relpos;

	relpos =pz-dz;
	if (relpos<0.0) {
		rvalue=-(1.0-exp(relpos/a))*0.5+0.5;
	}else{
		rvalue=0.5+(1.0-exp(-relpos/a))*0.5;
	}
	return rvalue;
}

double modeldata::comp_s_rad(double gmin, double tstep) {

	return 5.0*pow(gmin,0.333)*pow(tstep,0.667);
	//return 50.0*pow(gmin,2.0)*pow(tstep,2.0);
}

void modeldata::print_output() {
    printf("%d\n",npoints);
    for (int i=0 ; i < npoints; i++ ) {
        printf("%10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %d\n",
            mass[i],pos.x[i],pos.y[i],pos.z[i],
            vel.x[i]*1000.0,vel.y[i]*1000.0,vel.z[i]*1000.0,
            2);
    }
}
