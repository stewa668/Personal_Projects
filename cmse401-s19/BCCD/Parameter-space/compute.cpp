///////////////////////////////////////////
// Dart Parameter Space Study
// Copyright 1997-2002
// David A. Joiner and
//   The Shodor Education Foundation, Inc.
// $Id: compute.cpp,v 1.1 2012/05/02 09:53:55 charliep Exp $
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


#include "compute.h"


double compute_value(double aimRad, double aimTheta, double sigma) {


    Throw * myThrow = new Throw();
    
    int score=0; 
    for (int i=0;i<3; i++) {
        myThrow->makeThrowRT(aimRad,aimTheta,sigma);
        score+=myThrow->score*myThrow->multiplier;
    }
    delete myThrow;
    return (double) score;

}

Throw::Throw() {
    initialize();
}


Throw::~Throw() {
    delete sectorScores;
    delete sectorBoundaries;
    delete ringBoundaries;
}

void Throw::initialize() {
    score = 0;
    multiplier = 0;  
    
    sectorBoundaries = new double[numSectors+1];
    sectorScores = new int[numSectors];
    ringBoundaries = new double[numRings];
    
    sectorBoundaries[0]=-1.727875959;
    sectorBoundaries[1]=-1.413716694;
    sectorBoundaries[2]=-1.099557429;
    sectorBoundaries[3]=-0.785398163;
    sectorBoundaries[4]=-0.471238898;
    sectorBoundaries[5]=-0.157079633;
    sectorBoundaries[6]=0.157079633;
    sectorBoundaries[7]=0.471238898;
    sectorBoundaries[8]=0.785398163;
    sectorBoundaries[9]=1.099557429;
    sectorBoundaries[10]=1.413716694;
    sectorBoundaries[11]=1.727875959;
    sectorBoundaries[12]=2.042035225;
    sectorBoundaries[13]=2.35619449;
    sectorBoundaries[14]=2.670353756;
    sectorBoundaries[15]=2.984513021;
    sectorBoundaries[16]=3.298672286;
    sectorBoundaries[17]=3.612831552;
    sectorBoundaries[18]=3.926990817;
    sectorBoundaries[19]=4.241150082;
    sectorBoundaries[20]=4.555309348;  
    
    sectorScores[0]=3;
    sectorScores[1]=17;
    sectorScores[2]=2;
    sectorScores[3]=15;
    sectorScores[4]=10;
    sectorScores[5]=6;
    sectorScores[6]=13;
    sectorScores[7]=4;
    sectorScores[8]=18;
    sectorScores[9]=1;
    sectorScores[10]=20;
    sectorScores[11]=5;
    sectorScores[12]=12;
    sectorScores[13]=9;
    sectorScores[14]=14;
    sectorScores[15]=11;
    sectorScores[16]=8;
    sectorScores[17]=16;
    sectorScores[18]=7;
    sectorScores[19]=19;
    
    ringBoundaries[0]=0.037037;
    ringBoundaries[1]=0.0925926;
    ringBoundaries[2]=0.583333;
    ringBoundaries[3]=0.629630;
    ringBoundaries[4]=0.953704;
    ringBoundaries[5]=1.0;
}

void Throw::makeThrowXY(double aimX, double aimY, double sigma) {

    double throwRad, throwTheta, throwX, throwY;
    double twopi = 4.0*asin(1.0);
    throwRad = sigma * rand_gauss(0.0,sigma);
    throwTheta = rand_double(0.0,twopi);
    

    throwX = throwRad*cos(throwTheta);
    throwY = throwRad*sin(throwTheta);
        
    throwX = aimX + throwX;
    throwY = aimY + throwY;
        
    throwRad = sqrt(throwX*throwX+throwY*throwY);
    throwTheta = getTheta(throwX,throwY);
    while (throwTheta <
            sectorBoundaries[0]) {
        throwTheta+=twopi;
    }
    while (throwTheta >
            sectorBoundaries[numSectors]) {
        throwTheta-=twopi;
    }
    scoreThrow(throwRad,throwTheta);
}

void Throw::scoreThrow(double rad, double theta) {
        bool done=false;
        score=0;
        multiplier=1;
        for (int i=numSectors; i>0 && !done; i--) {
            if (theta > sectorBoundaries[i]) {
                score = sectorScores[i];
                done = true;
            }
        }
        if (rad<ringBoundaries[0]) {
            multiplier=2;
            score=25;
        } else if (rad < ringBoundaries[1]) {
            score=25;
        } else if (rad < ringBoundaries[2]) {
        } else if (rad < ringBoundaries[3]) {
            multiplier = 3;
        } else if (rad < ringBoundaries[4]) {
        } else if (rad < ringBoundaries[5]) {
            multiplier = 2;
        } else {
            score=0;
        }
    }


double Throw::rand_gauss(double mean, double sigma) {

    double pi = 2.0*asin(1.0);
    double u, r; 
 
    u = (double)rand() / RAND_MAX;
    if (u == 1.0) u = 0.999999999;
 
    r = sigma * sqrt( 2.0 * log( 1.0 / (1.0 - u) ) );
 
    u = (double)rand() / RAND_MAX;
    if (u == 1.0) u = 0.999999999;
 
    return( double ( mean + r * cos(2 * pi * u) ) );
}


double Throw::rand_double(double min, double max) {
    return min+((double)rand()/(double)RAND_MAX)*(max-min);
}



void Throw::makeThrowRT(double aimRad, double aimTheta, double sigma) {
    double aimX = aimRad*cos(aimTheta);
    double aimY = aimRad*sin(aimTheta);
    makeThrowXY(aimX, aimY, sigma);
}

double Throw::getTheta(double x, double y) {
    double pi = 4.0*atan(1.0);
    if (x == 0.0) {
        if (y>=0.0) return pi/2.0;
        else return -pi/2.0;
    } else if (y==0.0) {
        if (x>=0.0) return 0.0;
        else return pi;
    } else {
        if (x>=0.0) return atan(y/x);
        else return atan(y/x)+pi;
    }
}

