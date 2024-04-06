# $Id: perl_shared.pm,v 1.1 2012/05/02 18:15:21 charliep Exp $

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
use strict;
use Readonly;
use Carp;
use Cwd;

Readonly my $MACHINEFILE => "$ENV{HOME}/machines";

sub mpi_test{
        my ($test_name) = @_;
        my ($rc,$tempDirectory);
        my @cmd;
        my $base_path = getcwd();

####Start test code####
        chdir("../$test_name");
        system('make','clean');
        system('make');
        @cmd = ( 'xvfb-run', '-f', "$ENV{HOME}/.Xauthority",
                 '-l', 'mpirun', '-np', '1',
                 "$test_name");
        print STDERR "Running @cmd for $test_name\n";
        system(@cmd);

	my @machines;
	if (-e $MACHINEFILE){
		open(MACHINES,"<$MACHINEFILE") 
			or croak "could not open machinefile for reading $!";
		@machines = <MACHINES>;
		close MACHINES;
	}

#XXX This function is untested from here on.
        if ($#machines > 0){
               @cmd = ('bccd-syncdir','--ni','.',"$MACHINEFILE");
                print STDERR "Running @cmd for $test_name\n";

                open(my $BCCD_SYNCDIR, '-|', @cmd) or
                        croak "Couldn't run @cmd: $!\n";

                while(my $line = <$BCCD_SYNCDIR>) {
                        if($line =~ m{(/tmp/[\w]+)}) {
                                $tempDirectory = $1;
                        }
                }

                close($BCCD_SYNCDIR);

#get path to temporary directory
                @cmd = ('xvfb-run', '-f', "$ENV{HOME}/.Xauthority",
                        '-l','mpirun','-machinefile',"$MACHINEFILE",
                        '-np','2',"$tempDirectory/$test_name",'2>&1');
                print STDERR "Running @cmd for $test_name\n";

                open(OUTPUT, '-|',  @cmd) or
                    croak "Can't run @cmd: $!\n";
		
		my @output = <OUTPUT>;

		close OUTPUT;
        }
        else{
                print "Machinefile has too few machines: skipping multiprocessor test\n";
        }
####End test code####
        chdir("$base_path");
}

return 1;
