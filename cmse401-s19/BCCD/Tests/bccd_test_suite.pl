#!/usr/bin/perl

# $Id: bccd_test_suite.pl,v 1.1 2012/05/02 18:15:20 charliep Exp $

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
#***********************************************************#
#BCCD test suite
#Written by Samuel Leeman-Munk
#Runs test scripts and compares with control output in BCCD.
#with --mail, emails differences to bccd-developers@bccd.net
#***********************************************************#

use strict;
use File::Path;      #for rmtree
use Getopt::Long;    #for argument parsing
use MIME::Lite;      #for mailing results
use Readonly;
use Carp;
use POSIX;
use File::Basename;
use IPC::Open3;

Readonly my $SUITE_DIRECTORY => dirname(__FILE__);

use lib dirname(__FILE__);

use bccd_test_suite;

#Constants
use constant TRUE  => 1;
use constant FALSE => 0;
use constant MB    => 1024 * 1024;

Readonly my $DEFAULSE =>
  -1;    #The default false value. Overridden by any user input
Readonly my $DEFAULT_CONTROL_DIR       => "control";
Readonly my $DEFAULT_SCRIPTS_DIRECTORY => 'scripts';
Readonly my $DEFAULT_TEST_DIR          => "test";
Readonly my $DEFAULT_LIST_DIR          => "test_lists";
Readonly my $DEFAULT_SYSTEM_DIR        => "system";
Readonly my $DEFAULT_DIFF_DIR          => "tmp";
Readonly my $DEFAULT_MAILTO =>
  'mobeen.ludin@gmail.com';    #the recipient of the summary email
Readonly my $DEFAULT_MAIL => $DEFAULSE;      #send a summary email?
Readonly my $DEFAULT_LIST => 'all';
Readonly my $SYSTEM       => 'system';

#Global Arguments and Defaults
my $diffdir    = $DEFAULT_DIFF_DIR;          #the directory in which to keep the
                                             #files of differences between
                                             #test and control
my $buildcontrol 		= FALSE;
my $controldir 			= $DEFAULT_CONTROL_DIR;
my $liberation_drive	= FALSE;
my $listdir    			= $DEFAULT_LIST_DIR;
my $listfile   			= $DEFAULT_LIST;
my $mail         		= $DEFAULT_MAIL;
my $mailto       		= $DEFAULT_MAILTO;
my $messy        		= FALSE;
my $scriptdir  			= $DEFAULT_SCRIPTS_DIRECTORY;
my $systemdir  			= $DEFAULT_SYSTEM_DIR;
my $testdir    			= $DEFAULT_TEST_DIR;
my $verbose      		= $DEFAULSE;
my %testargs = ();
my @cmd;

#***MAIN***#
print "Changing Directory to $SUITE_DIRECTORY\n";
chdir $SUITE_DIRECTORY or die "Could not change directory:$!";
GetOptions(
    'controldir|cdir|cd=s' 		=> \$controldir,
    'control|c'            		=> \$buildcontrol,
    'diffdir=s'            		=> \$diffdir,
    'file|f=s'             		=> \$listfile,
    'listdir|l=s'          		=> \$listdir,
    'mail!'                		=> \$mail,
    'mailto|t=s'           		=> \$mailto,
    'messy|m'             		=> \$messy,
    'scriptdir|s=s'       		=> \$scriptdir,
    'systemdir|s=s'        		=> \$systemdir,
    'testdir|d=s'          		=> \$testdir,
    'verbose!'             		=> \$verbose,
	'arg|testarg|scriptarg=s%'	=> \%testargs,
);

if ( $mail == $DEFAULSE ) {
    if ( $mailto ne $DEFAULT_MAILTO ) {
        $mail = TRUE;
    }
    else {
        $mail = FALSE;
    }
}

if ( $listfile ne $DEFAULT_LIST
    && ( $listdir ne $DEFAULT_TEST_DIR || not $listfile =~ m/^\// ) )
{
    $listfile = "$listdir/$listfile";
}

#By default, verbose turns on when mail is turned off
#unless --noverbose is specified, in which a successful
#run of the suite returns no output
if ( ( not $mail ) and ( $verbose == $DEFAULSE ) ) {
    $verbose = TRUE;
}
if ( $verbose == $DEFAULSE ) {
    $verbose = FALSE;
}

#get test list
my @testlist;
print "Running listfile $listfile \n";
if ( $listfile eq 'all' ) {
    @testlist = bccd_test_suite::list_files_recursive($scriptdir);
}
elsif ( -r $listfile ) {
    @testlist = bccd_test_suite::read_list($listfile);
}
else {
    die "Could not read the list file: $!";
}

#Set up to build control directory
if ($buildcontrol) {
    print STDERR "Using control directory: $controldir\n";
    $testdir = $controldir;
}
elsif ( not -e $controldir ) {
    croak
"control directory \"$controldir\" does not exist. Please specify an existing control directory or, if you are using this system as the control system, use option -c to create one";
}

#initialize test result directory
if ( -e $testdir ) {
    rmtree($testdir)
      or croak
      "Could not clear existing directory $testdir for replacement: $!";
}
mkdir($testdir) or croak "Could not make directory $testdir: $!";

if ( -e $systemdir ) {
    rmtree($systemdir)
      or croak
      "Could not clear existing directory $systemdir for replacement: $!";
}
mkdir($systemdir) or croak "Could not make directory $systemdir: $!";

#Run tests and store results
foreach my $test (@testlist) {
    my $test_path = bccd_test_suite::find_by_name( $scriptdir, $test );
	my $args = '';
	my $run_message = '';
	
	chomp $test_path;

	if (defined $testargs{$test}){
		$args = $testargs{$test};
	}


	if ($verbose){
		$run_message = "Running $test, found at $test_path";
		if ($args){
			$run_message .= " using argument string $args\n";
		} else {
			$run_message .= " using no arguments\n";
		}	
	}

    if ( -f $test_path and not -x $test_path ) {
        chmod 0777, "./$test_path"
          or croak "Can't chmod $test_path $!";
    }

    my $system_script = ( index( $test_path, $SYSTEM ) != -1 );

    if ( not $system_script ) {
		print $run_message;
        system("./$test_path &> $testdir/$test.dat $args");
    }
    elsif ( not $buildcontrol ) {
		print $run_message;
        system("./$test_path &> $systemdir/$test.dat $args");
    }

}

my $report;
my $details;
my $date;

#Compare tests to controls
if ( not $buildcontrol ) { #Test comparison is unnecessary when building control
    if ( -e $diffdir ) {
        rmtree $diffdir
          or croak "Could not delete existing temporary directory $diffdir: $!";
    }

    mkdir($diffdir);

    #Prepare a report of the mismatches
    $details = "Details:\n";

    for ( my $i = 0 ; $i < @testlist ; ++$i ) {
        my $test = $testlist[$i];
        if ( -e "$testdir/$test.dat" and -e "$controldir/$test.dat" ) {

            @cmd = ( "diff", "$testdir/$test.dat", "$controldir/$test.dat" );
            if ($verbose) {
                carp "Running @cmd\n";
            }
            open( my $DIFF, '-|', @cmd )
              or croak "Can't run @cmd: $!\n";
            open( my $DIFF_OUT, '>', "$diffdir/$test.diff" )
              or croak "Can't open $diffdir/$test.diff for writing: $!\n";
            while ( my $line = <$DIFF> ) {
                chomp $line;
                print $DIFF_OUT "$line\n";
            }
            close($DIFF);
            close($DIFF_OUT);

            #take first value from wc -l, the number
            my @diff_stat = stat("$diffdir/$test.diff");
            if ( !@diff_stat ) {
                croak "Can't stat $diffdir/$test.diff: $!\n";
            }

            if ( $diff_stat[7] > 0 ) {   # Size of file, will be zero if no diff
                my $lines_added =
                  bccd_test_suite::line_match( "$diffdir/$test.diff", qr{^>} );
                my $lines_missing =
                  bccd_test_suite::line_match( "$diffdir/$test.diff", qr{^<} );
                $report .=
                  "$test had " . ( $#{$lines_added} + 1 ) . " lines added, ";
                $report .= ( $#{$lines_missing} + 1 ) . " missing";
                my $lns_added = join( "\n", @{$lines_added} );
                $details .= "Testname: $test\n $lns_added\n";
            }
            else {
                unlink "$testdir/$test.dat";
            }
        }
    }

    $date = strftime( '%B %d, %Y %T', localtime );
    print "DATE: $date\n";

    if ($verbose) {
        if ($report) {    #if there were any mismatches
            print "$date:\n$report\n\n$details\n";
        }
        else {
            print "$date: No mismatches\n";
        }

    }
}

#Prepare mail
if ($mail) {
    my $type;
    my @attachments;
    my $subject;
    my $text;

    my $version = `bccd-version`;

    if ($buildcontrol) {    #If building control, mail control dir
        @cmd = ( "tar", '-czf', 'control.tgz', $controldir );
        if ($verbose) {
            carp "Running @cmd\n";
        }
        system(@cmd);
        my $rc = WEXITSTATUS($?);
        if ($rc) {
            croak "tar failed!\n";
        }
        $type        = 'multipart/mixed';
        @attachments = ('control.tgz');
        $subject     = 'BCCD Test Control';
        $text        = "BCCD Test Control Data:\n$version";
    }
    elsif ($report) {    #If there is an error report, mail it

        @cmd = ( "tar", '-czf', 'test_results.tgz', $controldir, $testdir,
            $diffdir );
        if ($verbose) {
            carp "Running @cmd\n";
        }
        system(@cmd);
        my $rc = WEXITSTATUS($?);
        if ($rc) {
            croak "tar failed!\n";
        }

        @cmd = ( "tar", '-czf', 'system.tgz', $systemdir );
        if ($verbose) {
            carp "Running @cmd\n";
        }
        system(@cmd);
        $rc = WEXITSTATUS($?);
        if ($rc) {
            croak "tar failed!\n";
        }

        $type        = 'multipart/mixed';
        @attachments = ( 'test_results.tgz', 'system.tgz' );
        $subject     = 'BCCD Test Mismatch';
        $text =
"On $date, the following tests did not match expected values:\n$version\n$report\n$details";

    }
    else {
        @cmd = ( "tar", '-czf', 'system.tgz', $systemdir );
        if ($verbose) {
            carp "Running @cmd\n";
        }
        system(@cmd);
        my $rc = WEXITSTATUS($?);
        if ($rc) {
            croak "tar failed!\n";
        }
        $type        = 'multipart/mixed';
        @attachments = ('system.tgz');
        $subject     = 'BCCD Test Success';
        $text        = "On $date, the BCCD test returned no errors.\n$version";
    }

    my %mail = (
        "From", '<noreply@bccd.net>', "To", "<$mailto>", "Subject", $subject,
        "Type", $type,
    );

    my $msg = MIME::Lite->new(%mail);

    $msg->attach(
        Type => 'TEXT',
        Data => $text,
    );
    foreach my $attachment (@attachments) {
        $msg->attach(
            Type        => 'binary',
            Path        => $attachment,
            Filename    => $attachment,
            Disposition => 'attachment'
        );
    }

    # use Net:SMTP to do the sending
    $msg->send('smtp');

    # Clean up if not told to be messy
    if ( not $messy ) {
        unlink 'test_results.tgz';
        rmtree $testdir;
        rmtree $diffdir;
        rmtree $systemdir;
    }
}


