///////////////////////////////////////////
// Dart Parameter Space Study
// Copyright 1997-2002
// David A. Joiner and
//   The Shodor Education Foundation, Inc.
// $Id: compute.h,v 1.1 2012/05/02 09:53:55 charliep Exp $
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


#include <stdlib.h>
#include <math.h>


double compute_value(double aimRad, double aimTheta, double sigma);

class Throw {
public:
    int score;
    int multiplier;
    
    const static int numSectors=20;
    double * sectorBoundaries;
    int * sectorScores;
    const static int numRings=6;
    double * ringBoundaries;
    
    
    Throw();
    ~Throw();
    void initialize();
    void makeThrowXY(double aimX, double aimY, double sigma);
    void makeThrowRT(double aimRad, double aimTheta, double sigma);
    void scoreThrow(double throwRad, double throwTheta);
    double getTheta(double x, double y);
    double rand_gauss(double mean, double sigma);
    double rand_double(double min, double max);
    
};



