package tests;

# $Id: test_defs.pm,v 1.1 2012/05/02 18:15:20 charliep Exp $

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

use bccd_test_suite;
use vars qw/ %tests /;

%tests = (
		'test' => sub { #Test the test suite
		(my $testdir, my $testname) = @_;
		my $storeOutput = 
		 bccd_test_suite::generate_storeOutput($testdir,$testname);
		system("
		echo \"BCCD\nBCCD\nBCCD\" $storeOutput
		echo \"ok, it works.\" $storeOutput
		")
		},

		'fail' => sub { #Guarantee an error report
		(my $testdir, my $testname) = @_;
		my $storeOutput =
		 bccd_test_suite::generate_storeOutput($testdir,$testname);
		system( "seq 1 100 | sort -R $storeOutput");
		},

		'system' => sub { #Print system information
		(my $testdir, my $testname) = @_;
		my $storeOutput =
		 bccd_test_suite::generate_storeOutput($testdir,$testname);
		system("	
		uname -a $storeOutput
		cat /proc/cpuinfo $storeOutput
		");
		},
	
		'BCCD' => sub { #Test the BCCD's built-in tests
		
		},
			
		'GalaxSee' => sub { #Test GalaxSee
			(my $testdir, my $testname) = @_;
			bccd_test_suite::mpi_test($testdir, $testname);
		},
		
		'Param-Space' => sub{ #Test for Parameter Space Module
		(my $testdir, my $testname) = @_;
		bccd_test_suite::mpi_test($testdir, $testname);
		},

		'Life' => sub { #Test Conway's Game of Life
			(my $testdir, my $testname) = @_;
			bccd_test_suite::mpi_test($testdir, $testname);
		})
