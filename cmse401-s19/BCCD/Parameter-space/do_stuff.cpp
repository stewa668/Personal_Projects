///////////////////////////////////////////
// Dart Parameter Space Study
// Copyright 1997-2002
// David A. Joiner and
//   The Shodor Education Foundation, Inc.
// $Id: do_stuff.cpp,v 1.1 2012/05/02 09:53:55 charliep Exp $
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



#include <math.h>
#include <stdio.h>
#include "do_stuff.h"
#include "compute.h"

#include <X11/Xlib.h> 
#include <assert.h>
#include <unistd.h>
#define NIL (0)  

typedef XPoint * XPointArray;
typedef XPointArray * XPoint2DArray;
typedef double * PDouble;


void do_stuff(int rank,int size,int num_average,int num_theta,
        double theta_min,double theta_max,
        int num_rad, double rad_min, double rad_max,
        double sigma, int do_display) {
 
    printf("CPU %d Performing %d steps from %10.3e to %10.3e\n",rank,
        num_theta,theta_min,theta_max);

    double * rad = create_grid(num_rad,rad_min,rad_max);
    double * theta = create_grid(num_theta,theta_min,theta_max);
    double ** result = new PDouble[num_rad];
    for (int i=0;i<num_rad;i++) {
        result[i]=new double[num_theta];
        for (int j=0;j<num_theta;j++) result[i][j]=0.0;
    }
    
    // loop over i, j, and k (rad, theta, average)
    if (do_display==1) setupWindow(STUFF_IMAGE_WIDTH,STUFF_IMAGE_HEIGHT,
	num_rad,rad,num_theta,theta);
        
    // don't redraw every time, otherwise you spend all your time
    // doing graphics
    int draw_interval=10;
    int stop_theta=num_theta-1;
    if (rank==(size-1)) stop_theta=num_theta;
    for (int k=0;k<num_average;k+=draw_interval) {
        double w1 = (double)(k)/(double)(k+draw_interval);
        double w2 = (double)(draw_interval)/(double)(k+draw_interval);
		for (int i=0;i<num_rad;i++) {
			for (int j=0;j<stop_theta;j++) {
			    double sum=0.0;
				for (int l=0;l<draw_interval;l++) {
					sum+=compute_value(rad[i],theta[j],sigma);
				}
				sum/=(double)draw_interval;
				result[i][j]=
				    w1*result[i][j]+w2*sum;
				
			}
		}
		if (do_display==1) do_output(rank,size,num_rad,rad,
		    num_theta,theta,result);
    }
    
    delete rad;
    delete theta;
    for (int i=0;i<num_rad;i++) delete result[i];
    delete result;    
}


// X information, at some point this should be cleaned up so
// that it does not use global variables

// setupWindow modified from the tutorial on
// http://tronche.com/gui/x/xlib-tutorial/
// by Christophe Tronche

Display *dpy;
int blackColor;
int whiteColor;
Window w;
GC gc;
Pixmap buffer;
Colormap theColormap;
int numXGrayscale=50;
XColor Xgrayscale[50];
int n_poly=10;
XPoint *** polygons;
double REAL_X_MAX=0.0;
double REAL_X_MIN=0.0;
double REAL_Y_MAX=0.0;
double REAL_Y_MIN=0.0;
double REAL_X_RANGE=0.0;
double REAL_Y_RANGE=0.0;


void setupWindow(int IMAGE_WIDTH, int IMAGE_HEIGHT, int num_rad, double * rad,
		int num_theta, double * theta) {
      
      // set the scales
      find_maxmin(num_theta,theta);
      createPolygons(num_rad,rad,num_theta,theta);
	
      // Open the display

      dpy = XOpenDisplay(NIL);
      assert(dpy);

      // Get some colors

      blackColor = BlackPixel(dpy, DefaultScreen(dpy));
      whiteColor = WhitePixel(dpy, DefaultScreen(dpy));

      // Create the window

      w = XCreateSimpleWindow(dpy, DefaultRootWindow(dpy), 0, 0, 
				     IMAGE_WIDTH, IMAGE_HEIGHT, 0, blackColor,
				     blackColor);
      buffer = XCreatePixmap(dpy,DefaultRootWindow(dpy),
          IMAGE_WIDTH,IMAGE_HEIGHT,DefaultDepth(dpy,
          DefaultScreen(dpy)));
          
      theColormap = XCreateColormap(dpy, DefaultRootWindow(dpy),
          DefaultVisual(dpy,DefaultScreen(dpy)), AllocNone);
          
      for (int i=0;i<numXGrayscale;i++) {
          int color = (int)((double)i*35535.0/(double)numXGrayscale)+30000;
          Xgrayscale[i].red=color;
          Xgrayscale[i].green=color;
          Xgrayscale[i].blue=color;
          XAllocColor(dpy,theColormap,&(Xgrayscale[i]));
      }

      // We want to get MapNotify events

      XSelectInput(dpy, w, StructureNotifyMask);

      // "Map" the window (that is, make it appear on the screen)

      XMapWindow(dpy, w);

      // Create a "Graphics Context"

      gc = XCreateGC(dpy, w, 0, NIL);

      // Tell the GC we draw using the white color

      XSetForeground(dpy, gc, whiteColor);

      // Wait for the MapNotify event

      for(;;) {
	    XEvent e;
	    XNextEvent(dpy, &e);
	    if (e.type == MapNotify)
		  break;
      }

}

void find_maxmin(int num_theta, double * theta) {
    // what needs to be done? Take the input values, and determine the
    // real max and min. create an arc with the proscribed values.
    // x = r cos theta
    // y = r sin theta
    // rmax = 1

    REAL_X_MAX = 0.0;
    REAL_X_MIN = 0.0;
    REAL_Y_MAX = 0.0;
    REAL_Y_MIN = 0.0;

    for (int i=0;i<num_theta;i++) {
	printf("theta grid %d %lf\n",i,theta[i]);
        double x=cos(theta[i]);
	double y=sin(theta[i]);
	if (x<REAL_X_MIN) REAL_X_MIN=x;
	if (x>REAL_X_MAX) REAL_X_MAX=x;
	if (y<REAL_Y_MIN) REAL_Y_MIN=y;
	if (y>REAL_Y_MAX) REAL_Y_MAX=y;
    }

    REAL_X_RANGE=REAL_X_MAX-REAL_X_MIN;
    REAL_Y_RANGE=REAL_Y_MAX-REAL_Y_MIN;

    printf("Max %lf %lf %lf %lf %lf %lf \n",REAL_X_MAX,REAL_X_MIN,REAL_Y_MAX,REAL_Y_MIN,
        REAL_X_RANGE,REAL_Y_RANGE);
}

int xRealToDisplay(double xReal) {
    return (int)((double)STUFF_IMAGE_WIDTH*(xReal-REAL_X_MIN)/REAL_X_RANGE);
}

int yRealToDisplay(double yReal) {
    return STUFF_IMAGE_HEIGHT-
        (int)((double)STUFF_IMAGE_HEIGHT*(yReal-REAL_Y_MIN)/REAL_Y_RANGE);
}

double xDisplayToReal(int xDisplay) {
    return REAL_X_MIN+(double)xDisplay/(double)STUFF_IMAGE_WIDTH*REAL_X_RANGE;
}

double yDisplayToReal(int yDisplay) {
    return REAL_Y_MIN-(double)yDisplay/(double)STUFF_IMAGE_HEIGHT*REAL_Y_RANGE;
}

// ok, ive got the coordinate routines, and ive gor r/theta points. I need
// to make the display grid. I've got to create polygons "around" those known
// points. What is going to be the best way to do this?
// the inner points should be easy, I need to create and store an array of
// polygons, each of which is an array of points. I should create a polygon class
// and use that. There is something like this already, the XPoint.
//
// Just for fun, start the routine that designs the polygons. 
XPoint * createPoly (int n_poly, double rad_min, double rad_max, double theta_min, 
		double theta_max) {
    XPoint * retval = new XPoint[2*n_poly+1];

    double theta_step = (theta_max-theta_min)/(double)(n_poly-1);
    for (int k=0;k<n_poly;k++) {
        retval[k].x = xRealToDisplay(rad_min*cos(theta_min+k*theta_step));
        retval[k].y = yRealToDisplay(rad_min*sin(theta_min+k*theta_step));
    }
    int j=0;
    for (int k=2*n_poly-1;k>=n_poly;k--) {
        retval[k].x = xRealToDisplay(rad_max*cos(theta_min+j*theta_step));
        retval[k].y = yRealToDisplay(rad_max*sin(theta_min+j*theta_step));
	j++;
    }
    retval[2*n_poly]=retval[0];
    return retval;
}

void createPolygons (int num_rad, double * rad, int num_theta, double * theta) {

    double rad_min,rad_max,theta_min,theta_max;
    polygons = new XPoint2DArray[num_rad];
    for (int i=0;i<num_rad;i++) {
        polygons[i] = new XPointArray[num_theta];
    }

    rad_min=rad[0];
    rad_max=rad[0]+0.5*(rad[1]-rad[0]);
    theta_min=theta[0]-0.5*(theta[1]-theta[0]);
    theta_max=theta[0]+0.5*(theta[1]-theta[0]);
    polygons[0][0] = createPoly(n_poly,rad_min,rad_max,theta_min,theta_max);
    for (int j=1;j<num_theta-1;j++) {
        theta_min = 0.5*(theta[j]+theta[j-1]);
        theta_max = 0.5*(theta[j]+theta[j+1]);
        polygons[0][j] = createPoly(n_poly,rad_min,rad_max,theta_min,theta_max);
    }
    theta_min=theta[num_theta-2]+0.5*(theta[num_theta-1]-theta[num_theta-2]);
    theta_max=theta[num_theta-1]+0.5*(theta[num_theta-1]-theta[num_theta-2]);
    polygons[0][num_theta-1] = createPoly(n_poly,rad_min,rad_max,theta_min,theta_max);
    for (int i=1;i<num_rad-1;i++) {
	rad_min = 0.5*(rad[i]+rad[i-1]);
	rad_max = 0.5*(rad[i]+rad[i+1]);
        theta_min=theta[0]-0.5*(theta[1]-theta[0]);
        theta_max=theta[0]+0.5*(theta[1]-theta[0]);
        polygons[i][0] = createPoly(n_poly,rad_min,rad_max,theta_min,theta_max);
	for (int j=1;j<num_theta-1;j++) {
	    theta_min = 0.5*(theta[j]+theta[j-1]);
	    theta_max = 0.5*(theta[j]+theta[j+1]);
            polygons[i][j] = createPoly(n_poly,rad_min,rad_max,theta_min,theta_max);
	}
        theta_min=theta[num_theta-2]+0.5*(theta[num_theta-1]-theta[num_theta-2]);
        theta_max=theta[num_theta-1]+0.5*(theta[num_theta-1]-theta[num_theta-2]);
        polygons[i][num_theta-1] = createPoly(n_poly,rad_min,rad_max,theta_min,theta_max);
    }
    rad_min=rad[num_rad-2]+0.5*(rad[num_rad-1]-rad[num_rad-2]);
    rad_max=rad[num_rad-1];
    theta_min=theta[0]-0.5*(theta[1]-theta[0]);
    theta_max=theta[0]+0.5*(theta[1]-theta[0]);
    polygons[num_rad-1][0] = createPoly(n_poly,rad_min,rad_max,theta_min,theta_max);
    for (int j=1;j<num_theta-1;j++) {
        theta_min = 0.5*(theta[j]+theta[j-1]);
        theta_max = 0.5*(theta[j]+theta[j+1]);
        polygons[num_rad-1][j] = createPoly(n_poly,rad_min,rad_max,theta_min,theta_max);
    }
    theta_min=theta[num_theta-2]+0.5*(theta[num_theta-1]-theta[num_theta-2]);
    theta_max=theta[num_theta-1]+0.5*(theta[num_theta-1]-theta[num_theta-2]);
    polygons[num_rad-1][num_theta-1] = createPoly(n_poly,rad_min,rad_max,theta_min,theta_max);
}



void do_output(int rank, int size, int num_rad, double * rad,
        int num_theta, double * theta,
        double ** result) {
        
    double rad_min = rad[0];
    double rad_max = rad[num_rad-1];
    double theta_min = theta[0];
    double theta_max = theta[num_theta-1];
    
    int cellWidth, cellHeight;
    int x1, y1, x2, y2;
    
    XSetForeground(dpy, gc, blackColor);
    XFillRectangle(dpy,buffer,gc,0,0,STUFF_IMAGE_WIDTH,STUFF_IMAGE_HEIGHT);
    
    double rad_range = rad[num_rad-1]-rad[0];
    double theta_range = theta[num_theta-1]-theta[0];

    
    for (int j=0;j<num_theta-1;j++) {
        for (int i=0;i<num_rad-1;i++) {
            int color_index = (int)(result[i][j]/60.0*numXGrayscale);
            if (color_index<0) color_index=0;
            else if(color_index>numXGrayscale-1) 
                color_index=numXGrayscale-1;
            XSetForeground(dpy,gc,Xgrayscale[color_index].pixel);
            XFillPolygon(dpy,buffer,gc,polygons[i][j],2*n_poly+1,Complex,CoordModeOrigin);
        }
    }

//    cellWidth = (int)((double)STUFF_IMAGE_WIDTH/(double)size/(double)(num_theta-1)) +1;
//    cellHeight = (int)((double)STUFF_IMAGE_HEIGHT/(double)(num_theta-1)) +1;
//    x2 = 0;
//    for (int j=0;j<num_theta-1;j++) {
//        x1=x2;
//        x2 = (int)((theta[j+1]-theta[0])/theta_range*(double)
//            STUFF_IMAGE_WIDTH/(double)size);
//        y2 = STUFF_IMAGE_HEIGHT;
//        for (int i=0;i<num_rad-1;i++) {
//            y1=STUFF_IMAGE_HEIGHT-
//                (int)((rad[i+1]-rad[0])/rad_range*(double)
//                STUFF_IMAGE_HEIGHT);
//            int color_index = (int)(result[i][j]/60.0*numXGrayscale);
//            if (color_index<0) color_index=0;
//            else if(color_index>numXGrayscale-1) 
//                color_index=numXGrayscale-1;
//            XSetForeground(dpy,gc,Xgrayscale[color_index].pixel);
//            XFillRectangle(dpy,buffer,gc,x1,y1,
//                cellWidth,cellHeight);
//            y2=y1;
//        }
//    }
    
    Throw testThrow;
    
    
    XSetForeground(dpy,gc,blackColor);
    for (int j=0;j<testThrow.numSectors;j++) {
//        double x = testThrow.sectorBoundaries[j];
//        if (x<0.0) x+=4.0*asin(1.0);
//        int x_int = (int)(x/(4.0*asin(1.0))*(double)STUFF_IMAGE_WIDTH);
	int tempx1 = xRealToDisplay(testThrow.ringBoundaries[1]*cos(
	    testThrow.sectorBoundaries[j]));
	int tempy1 = yRealToDisplay(testThrow.ringBoundaries[1]*sin(
	    testThrow.sectorBoundaries[j]));
	int tempx2 = xRealToDisplay(cos(testThrow.sectorBoundaries[j]));
	int tempy2 = yRealToDisplay(sin(testThrow.sectorBoundaries[j]));
        XDrawLine(dpy,buffer,gc,tempx1,tempy1,tempx2,tempy2);
    }
    for (int i=0;i<testThrow.numRings;i++) {
//        double y = testThrow.ringBoundaries[i];
//        int y_int = (int)(y*(double)STUFF_IMAGE_HEIGHT);
//        y_int = STUFF_IMAGE_HEIGHT-y_int;
//        XDrawLine(dpy,buffer,gc,0,y_int,STUFF_IMAGE_WIDTH,y_int);
        double ring = testThrow.ringBoundaries[i];
        int cornerX = xRealToDisplay(-ring);
	int cornerY = yRealToDisplay(ring);
	int width = xRealToDisplay(ring)-xRealToDisplay(-ring);
	int height = yRealToDisplay(-ring)-yRealToDisplay(ring);
        XDrawArc(dpy,buffer,gc,cornerX,cornerY,width,height,0,360*64-1);
    }
    
     XCopyArea(dpy, buffer, w, gc, 0, 0,
         STUFF_IMAGE_WIDTH, STUFF_IMAGE_HEIGHT,  0, 0);
     XFlush(dpy);
}

double * create_grid(int num_grid, double grid_min,
        double grid_max) {
    int num_grid_minus_one = num_grid-1;
    double step = (grid_max-grid_min)/(double)(num_grid-1);
    double * grid = new double[num_grid];
    grid[0] = grid_min;
    for (int i=1;i<num_grid_minus_one;i++) {
        grid[i]=grid_min+(double)i*step;
    }
    grid[num_grid_minus_one]=grid_max;
    return grid;
}
