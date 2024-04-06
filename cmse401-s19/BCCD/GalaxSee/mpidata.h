//////////////////////////////////////////////////////////
// GalaxSee (version MPI 0.9)
// Copyright 1997 - 2002
// David A. Joiner and the Shodor Education Foundation
// $Id: mpidata.h,v 1.4 2012/05/30 17:30:27 charliep Exp $
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

#define MPIDATA_EXIT_TAG 1000
#define MPIDATA_DODERIVS 10
#define MPIDATA_PASSNUMBER 100
#define MPIDATA_PASSNUMBERPER 101
#define MPIDATA_PASSGNORM 110
#define MPIDATA_PASSMASS 120
#define MPIDATA_PASSSHIELD 125
#define MPIDATA_PASSX 130

#define MPIDATA_DONEDERIVS 200

struct mpidata {
    MPI_Status status;
    int rank;
    int size;
};
