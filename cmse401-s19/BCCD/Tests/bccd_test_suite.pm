# $Id: bccd_test_suite.pm,v 1.1 2012/05/02 18:15:20 charliep Exp $

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

#*******************************************#
#Functions for the Automagic BCCD test suite!
#******************************************#
package bccd_test_suite;
use strict;
use Carp;
use Cwd;
use POSIX;
use Readonly;

Readonly my $SHAREDCODE => ".sharedcode";

# Return lines matching a regex
sub line_match {
    my ( $file, $re ) = @_;
    my $lines;

    open( my $FILE, '<', $file )
      or croak "Can't open $file for reading: $!\n";

    while ( my $line = <$FILE> ) {
        chomp $line;
        if ( $line =~ m{$re} ) {
            push( @{$lines}, $line );
        }
    }

    close($FILE);

    return $lines;
}

#XXX
# Uses command-line find. Perl's built-in find subroutine
# is excessively complex for this simple task
sub find_by_name {
    my ( $directory, $name ) = @_;

    open( PATH, "find $directory -name $name|" )
      or croak "could not find script $!";

    my $path = <PATH>;

    close PATH;
    return $path;
}

#Trim function to remove whitespace
sub trim {
    my ($string) = @_;
    $string =~ s/^\s+//;
    $string =~ s/\s+$//;
    return $string;
}

#Collect meaningful information from the syncdir command
sub get_syncdir {
    my ($tmpoutput) = @_;
    open( my $FILE, '<', $tmpoutput )
      or die "could not open temporary file: $!";
    my $tmp = "";
    while ( $tmp !~ m/\// ) {
        $tmp = readline $FILE;
    }
    my $startindex = index( $tmp, "/", );
    my $endindex = index( $tmp, " ", $startindex );
    my $syncdir = substr( $tmp, $startindex, $endindex - 1 );
    close $FILE;
    return $syncdir;
}

#Get test list from a file
sub read_list {
    my ($path) = @_;
    my @testlist = ();

    open( FILE, '<', $path ) or die "could not open list file: $!";
    @testlist = grep( !/^$|^#/, <FILE> );    #Remove empty
                                             #and comment lines
    close(FILE);

    map ( {s/#.*$//} @testlist );
    map ( {s/\*//} @testlist );
    map ( trim, @testlist );

    return @testlist;
}

sub list_files_recursive {
    my ($directory) = @_;

    open( LIST, "ls -FR1 $directory|" );    #ls ignores files and directories
                                            #starting with '.' be sure to mimic
                                            #this functionality if you change
                                            #this command to not use the shell.

    #cut out directories and blank space
    my @files = grep ( !/\/$|:$|^$/, <LIST> );
    close(LIST);
    chomp @files;

    @files = map { s/\*$//; $_ } @files;

    return @files;
}

return 1;
