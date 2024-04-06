/*
  * $Id: Life.c,v 1.4 2012/06/27 16:26:57 charliep Exp $
  * This file is part of BCCD, an open-source live CD for computational science
  * education.
  * 
  * Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave 
  *   Joiner, Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, & Aaron Weeden 

  * 
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  * 
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  * 
  * You should have received a copy of the GNU General Public License
  * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/


/*******************************************
MPI Life 1.0
Copyright 2002, David Joiner and
  The Shodor Education Foundation, Inc.
Updated 2010, Andrew Fitz Gibbon and
  The Shodor Education Foundation, Inc.

A C implementation of Conway's Game of Life.

To run:
./Life [Rows] [Columns] [Generations] [Display]

See the README included in this directory for
more detailed information.
*******************************************/

#include "Life.h"
#include "Defaults.h" // For Life's constants

int main(int argc, char ** argv) {
	int count;
	struct life_t life;

#ifdef STAT_KIT
	startTimer();
#endif

	init(&life, &argc, &argv);

	for (count = 0; count < life.generations; count++) {
		if (life.do_display)
			do_draw(&life);

		copy_bounds(&life);

		eval_rules(&life);

		update_grid(&life);

		throttle(&life);
	}

	cleanup(&life);

#ifdef STAT_KIT
	printStats("Life",life.size,"mpi",life.ncols * life.nrows, "1.3", 0, 3, "iCOLUMNS", (long long int) life.ncols, "iROWS", (long long int)life.nrows, "iGENERATIONS", (long long int)life.generations);
#endif

	exit(EXIT_SUCCESS);
}
