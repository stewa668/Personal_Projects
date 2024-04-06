#!/usr/bin/perl

# $Id: gromacs.pl,v 1.1 2012/05/02 18:15:20 charliep Exp $

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

use Bccd;
use Bccd::TestDb qw/open_db insert_results/;
use POSIX;
use Cwd;
use File::Copy;
use Getopt::Long;
use warnings;
use strict;

my $Bccd = new Bccd();
my %opts;
my $stage = $Bccd->get_stage();
my $testname = "gromacs-dppc"; # We test with DPPC
my $svnrev = $Bccd->get_rev();
my $tmpdir;
my $ft;
my @files;

GetOptions(
           'updatedb=s' => \$opts{'updatedb'},
           'moleculedir=s'=> \$opts{'moleculedir'},
           'nocleanup' => \$opts{'nocleanup'}
           );

$ft = new File::Temp();
if( $opts{'nocleanup'} ) {
    $tmpdir = $ft->tempdir('bccd',CLEANUP => 0);
} else {
    $tmpdir = $ft->tempdir('bccd',CLEANUP => 1);
}

$Bccd->redirect_stdio("$tmpdir") if($opts{'updatedb'});

@files = <$opts{'moleculedir'}/*>;

foreach ( @files ) {
    $Bccd->run_test(
                  "fcopy",
                  "",
                  "Copying $_ to $tmpdir.",
                  $_,
                  $tmpdir
                  );
}

$Bccd->run_test(
              "chdir",
              "",
              "cd $tmpdir",
              $tmpdir
              );

$Bccd->run_test(
              "system",
              "",
              "Snarfing hosts",
              "bccd-snarfhosts"
              );

$Bccd->run_test(
              "system",
              "",
              "Preparing molecule w/ grompp.",
              "grompp -np `wc -l $ENV{'HOME'}/machines|awk '{print \$1}'` -shuffle -sort -v"
              );

$Bccd->run_test(
              "system",
              "",
              "Running molecule w/ mdrun.",
              "mpirun -machinefile $ENV{'HOME'}/machines -np `wc -l $ENV{'HOME'}/machines|awk '{print \$1}'` mdrun -np `wc -l $ENV{'HOME'}/machines|awk '{print \$1}'` -machinefile $ENV{'HOME'}/machines -v"
              );

if($opts{'updatedb'}) {
    my $dbh = open_db("$opts{'updatedb'}");
    if( $Bccd->get_passed() < $Bccd->get_total() ) {
        insert_results($testname,$stage,$svnrev,$Bccd->snarf_file("$tmpdir/allout"),"FAILURE",$dbh);
    } else {
        insert_results($testname,$stage,$svnrev,$Bccd->snarf_file("$tmpdir/allout"),"SUCCESS",$dbh);
    }
    $Bccd->close_stdio();
    $dbh->disconnect;
}

exit $Bccd->get_total()-$Bccd->get_passed();
