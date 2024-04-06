///////////////////////////////////////////
// Dart Parameter Space Study
// Copyright 1997-2002
// David A. Joiner and
//   The Shodor Education Foundation, Inc.
// $Id: do_stuff.h,v 1.2 2012/06/27 17:35:49 ibabic09 Exp $
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
///////////////////////////////////////////


#ifndef DO_STUFF
#define DO_STUFF

#define STUFF_IMAGE_WIDTH 600
#define STUFF_IMAGE_HEIGHT 600


#include <X11/Xlib.h> 

#ifdef STAT_KIT
#include "../StatKit/petakit/pkit.h"    // For PetaKit output
#endif


void do_stuff(int rank, int size,int num_average,int num_theta,
        double theta_min, double theta_max,
        int num_rad, double rad_min, double rad_max,
        double sigma, int do_display);
double * create_grid(int num_grid, double grid_min,
        double grid_max);
void setupWindow(int,int,int,double*,int,double*);
void do_output(int rank, int size,int num_rad,double *rad,
				    int num_theta,double *theta,
				    double **result);
void find_maxmin(int num_theta, double * theta);
int xRealToDisplay(double xReal);
int yRealToDisplay(double yReal);
double xDisplayToReal(int xDisplay);
double yDisplayToReal(int yDisplay);
void createPolygons (int num_rad, double * rad, int num_theta, double * theta);
XPoint * createPoly (int n_poly, double rad_min, double rad_max, double theta_min, 
		double theta_max);
				    
		
void ready_output();

#endif
