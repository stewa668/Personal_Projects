//////////////////////////////////////////////////////////
// GalaxSee (version MPI 0.9)
// Copyright 1997 - 2002
// David A. Joiner and the Shodor Education Foundation
// $Id: Gal.cpp,v 1.9 2012/06/27 19:46:45 mmludin08 Exp $
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

// Gal.cpp : Defines the class behaviors for the application.
//

#include "Gal.h"
#include <stdio.h>
#include "modeldata.h"
#include "mpi.h"
#include "mpidata.h"

#ifdef NO_X11
    #define has_x11 0
#else 
    #include <X11/Xlib.h>
    #include <assert.h>
    #include <unistd.h>
    #define NIL (0)
    #define has_x11 1
#endif


modeldata g_dynamic;
mpidata g_mpi;


int main(int argc, char** argv) {
    extern void setup_mpi(int*, char***,mpidata *);
    extern void finalize_mpi();
    extern void run_server(int, char**);
    extern void run_client();

#ifdef STAT_KIT
	startTimer();
#endif

    setup_mpi(&argc,&argv,&g_mpi);
    

    if (g_mpi.rank==0) {
        run_server(argc,argv);
        
        //when the server completes, send an exit message to the clients
        for (int i=1;i<g_mpi.size;i++) {
            MPI_Send(0,0,MPI_INT,
                i,MPIDATA_EXIT_TAG,MPI_COMM_WORLD);
        }
    } else {
        run_client();
    }

    
    finalize_mpi();

#ifdef STAT_KIT
	printStats("GalaxSee",g_mpi.size,"mpi", g_dynamic.default_numstars,"1",0,0);
#endif

}

void run_server(int argc, char ** argv) {
    extern void run_step();
    extern void make_image(int);
    extern void setupWindow(int,int);
    
    // defaults not already defined in g_dynamic
    double t_final = 1.0e3;
    int do_display=1;
    
    // command line arguments
    if (argc > 1) {
        sscanf(argv[1],"%d",&g_dynamic.default_numstars);
 }
    if (argc > 2) {
        sscanf(argv[2],"%lf",&g_dynamic.default_starmass);
    }
    if (argc > 3) {
        sscanf(argv[3],"%lf",&t_final);
    }
    if (argc > 4) {
        sscanf(argv[4],"%d",&do_display);
    }
    if (do_display!=0) do_display=1;
    g_dynamic.default_imethod=IMETHOD_EULER;

    g_dynamic.init();
    g_dynamic.new_galaxy();
        
    double t_step = 8.0;
    
    g_dynamic.time_step = t_step;
    int count=0;
    if (do_display==1) {
        if(has_x11)
            setupWindow(GAL_IMAGE_WIDTH,GAL_IMAGE_HEIGHT);
    }
    while (g_dynamic.time<=t_final) {
        run_step();
        count++;
        if (do_display==1) {
            if(has_x11)
                make_image(g_mpi.rank);
            else
                printf("%lf\n",g_dynamic.time);
        }
    }
    
}

#ifndef NO_X11
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
    int numXGrayscale=10;
    XColor Xgrayscale[10];
    
    void setupWindow(int IMAGE_WIDTH, int IMAGE_HEIGHT) {
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
    
    void make_image(int rank) {
    
        int x1,x2,y1,y2; 
    
        XSetForeground(dpy, gc, blackColor);
        XFillRectangle(dpy,buffer,gc,0,0,GAL_IMAGE_WIDTH,GAL_IMAGE_HEIGHT);
        
        int imheight = GAL_IMAGE_HEIGHT;
        int imwidth = GAL_IMAGE_WIDTH;
       
        // Loop over stars and put a pixel in for each star
        double scale=(double)imheight/5.0;
        double shift=(double)imwidth/4.0;
        for (int i=0 ; i<g_dynamic.npoints; i++) {
            int dispX = (int)(scale*g_dynamic.pos.x[i]+shift);
            int dispY = (int)(scale*g_dynamic.pos.y[i]+shift);
            int dispZ = (int)(scale*g_dynamic.pos.z[i]+shift);
    
            int depthY = (int)((double)dispY/(double)imheight*numXGrayscale);
            int depthZ = (int)((double)dispZ/(double)imheight*numXGrayscale);
            if (depthY>numXGrayscale-1) depthY=numXGrayscale-1;
            if (depthZ>numXGrayscale-1) depthZ=numXGrayscale-1;
    
            if (dispX < imwidth/2) {
                XSetForeground(dpy,gc,Xgrayscale[depthZ].pixel);
                //XDrawPoint(dpy,buffer,gc,dispX,dispY);
                XFillRectangle(dpy,buffer,gc,dispX,dispY,3,3);
            }
            if (dispX > 0) {
                XSetForeground(dpy,gc,Xgrayscale[depthY].pixel);
                //XDrawPoint(dpy,buffer,gc,dispX+imwidth/2,dispZ);
                XFillRectangle(dpy,buffer,gc,dispX+imwidth/2,dispZ,3,3);
            }
        }
    
        
    
        XCopyArea(dpy, buffer, w, gc, 0, 0,
             GAL_IMAGE_WIDTH, GAL_IMAGE_HEIGHT,  0, 0);
         XFlush(dpy);
	  
    }
#endif

void run_step() {
    
    extern void derivs(int,double,double *, double *);

    if(g_dynamic.int_method==IMETHOD_EULER) {
        g_dynamic.updateEuler(g_dynamic.time_step,derivs);
    } else if (g_dynamic.int_method==IMETHOD_IEULER) {
        g_dynamic.updateIEuler(g_dynamic.time_step,derivs);
    } else {
        g_dynamic.updateRKutta4(g_dynamic.time_step,derivs);
    }
}

void run_client() {
    extern void derivs_client();

    int message;
    int tag;
    
    while (1) {
        // wait for a message
        MPI_Recv(&message,1,MPI_INT,0,MPI_ANY_TAG,MPI_COMM_WORLD,
            &g_mpi.status);
        // if the message is to exit, break out of loop.    
        if (g_mpi.status.MPI_TAG==MPIDATA_EXIT_TAG) break;
        if (g_mpi.status.MPI_TAG==MPIDATA_DODERIVS) {
            derivs_client();
        }
        g_mpi.status.MPI_TAG=-1;
    }

}


void setup_mpi(int *argc,char*** argv, mpidata* my_mpi) {
    MPI_Init(argc,argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_mpi->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &my_mpi->size);
}

void finalize_mpi() {
    MPI_Finalize();
}
