//////////////////////////////////////////////////////////
// GalaxSee (version MPI 0.9)
// Copyright 1997 - 2002
// David A. Joiner and the Shodor Education Foundation
// $Id: modeldata.h,v 1.4 2012/06/27 18:44:08 mmludin08 Exp $
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

#include "diffeq.h"

#ifndef _MODELDATA
#define _MODELDATA

#define SHAPE_SPHERE 0
#define SHAPE_DISC 1
#define SHAPE_CUBE 2
#define SHAPE_2D 3
#define SHAPE_3D 4
#define SHAPE_RINGS 5

#define SCALE_SOLARSYSTEM 50
#define SCALE_GALACTIC 51
#define SCALE_EARTHSUN 52

#define FMETHOD_DIRECT 0
#define FMETHOD_BARNESHUT 1
#define FMETHOD_PM 2

#ifdef STAT_KIT
#include "../StatKit/petakit/pkit.h"	// For PetaKit output
#endif


class modeldata : public dynamic {
public:
	//Defaults for new galaxies
	int default_numstars;
	int default_shape;
	int default_scale;
	double default_starmass;
	double default_rotatefac;
	int default_imethod;
	int default_fmethod;
	double default_timestep;

	//Methods
	modeldata();
	void define_scale(int);
	void init();
	void init(int,cart3d,cart3d,double *,int *);
	void cleanup();
	void new_galaxy();
	void compute_info();
	void spin_galaxy(double factor);
	static double comp_s_rad(double, double);
	double calc_depth(double,double,double);
	void print_output();

	//constants to be used in problem
	double msun;		//Mass of the sun
	double mearth;		//Mass of the earth
	double au;			//Astronomical unit
	double gconst;		//Gravitational const
	double kly;			//Kilo light year

	// normalization factors
	double mzero;
	double rzero;
	double tzero;
	double azero;
	double gnorm;
	int current_scale;
	double box_edge_length;

	//"extra" information about each dynamic variable
	double *mass;
	int * color;

	//information about the displayed coordinates
	double disp_phi;
	double disp_theta;
	double *disp_x;
	double *disp_y;
	double *order_z;
	double disp_offset_x,disp_offset_y;
	double disp_scale;
	double disp_screen;
	double view_z;
	cart3d disp_pos;
	int disp_num;
	int * disp_order;

	//box information
	double *box_corn_x;
	double *box_corn_y;
	bool *box_rear;

	//model running information
	double time_step;
	int force_method;
	int int_method;

	//color coding information for each body.
	double *color_hue;
	double *color_red;
	double *color_green;
	double *color_blue;
	//point info
	int use_cooldot;
	int point_size;
	int point_color;
	bool depth_shading;
	bool chroma_depth;
	bool red_shift;
	bool black_background;
	bool * hide_color;

	//Default colors
	double *default_hue;

	//rotation info
	double rotate_factor;

	//Galaxy info
	double total_energy;
	double total_mass;
	double px;
	double py;
	double pz;
	double comx;
	double comy;
	double comz;

	
};

#endif
