#!/usr/bin/perl
# $Id: MPI_TEST_Life.pl,v 1.1 2012/05/02 18:15:21 charliep Exp $

# This file is part of BCCD, an open-source live CD for computational science
# education.
# 
# Copyright (C) 2010 Andrew Fitz Gibbon, Paul Gray, Kevin Hunter, Dave Joiner, 
#   Sam Leeman-Munk, Tom Murphy, Charlie Peck, Skylar Thompson, & Aaron Weeden 

# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#This serves as a template. Just switch the name of what you want to test in for "Life"

use strict;
use File::Basename;
use Readonly;

use lib dirname(__FILE__). "shared_code";

Readonly my $SCRIPT_DIRECTORY => dirname(__FILE__);
use perl_shared;

mpi_test("Life");
