#!/usr/bin/perl
use strict;
use PlotKit;

use constant MAX_BUFFER=>1024*1024;

#############
##MAIN******#
#############
print "Running...\n";
#Collect arguments
my %globals = PlotKit::getargs();

my $data_file = $globals{'datafile'};
my $gdifile = $globals{'tempgdifile'};

#get output data
open OUTPUT,"<$data_file" or die ("could not open data file $!");
my $heading_row = readline OUTPUT;
my $datum = '';
my $data = '';
do {
	read OUTPUT,$datum,MAX_BUFFER or die ("could not read file $!");
	$data .= $datum;
} while (!eof OUTPUT);
close OUTPUT;

$globals{'data'} = $data;

#Connect headings to column numbers
$heading_row = substr($heading_row,1,length($heading_row)-1);
my @headings = split(/\s+/,$heading_row);
my %heading_hash = ();
my $i;
for ($i = 0; $i <= $#headings; ++$i){
	$heading_hash{$headings[$i]} = $i;
}

if (not defined $heading_hash{$globals{'independent'}}){
	die "|@headings| does not contain \"" . $globals{'independent'} . '"';
}
if (not defined $heading_hash{$globals{'dependent'}}){
	die "|@headings| does not contain \"" . $globals{'dependent'} . '"';
}
if (not defined $heading_hash{$globals{'selector_column'}}){
	die "|@headings| does not contain \"" . $globals{'selector_column'} . '"';

}

#identify columns of independent and dependent variables
my $independent_index = $heading_hash{$globals{'independent'}};
my $dependent_index = $heading_hash{$globals{'dependent'}};
my $tag_index = $heading_hash{$globals{'selector_column'}};

my $split_index;
if ( defined $globals{'split'} ){
	if (defined $heading_hash{$globals{'split'}}){
		$split_index = $heading_hash{$globals{'split'}};
	} else {
		die "|@headings| does not contain \"" . $globals{'split'} . '"';
	}
}


my @sets = PlotKit::build_dataset($tag_index, $independent_index, $dependent_index, $split_index, %globals);

$globals{'sets'} = \@sets;

PlotKit::build_gdi(%globals);

system("gnuplot $gdifile");

unless($globals{'messy'}){
	unlink $gdifile;
	unlink $globals{'tempdata'}; 
}

__END__

=head1 - SYNOPSIS

This script takes stat.pl's text output (or other output in a similar format), performs some simple statistics on it, and produces a graph that can be viewed immediately on a compatible machine, or saved to a .ps file for later viewing. PlotKit is a wrapper to gnuplot, and will not function without it. PlotKit has never been tested on a non-linux system.

If you have any questions, contact leemasa@earlham.edu

=head1 - USAGE

perl PlotKit.pl [arguments] I<tag>

The only mandatory argument of PlotKit.pl is the tag, which must always be the last argument given. The tag identifies the dataset to be plotted and is important because data files often contain multiple sets. Tag may take multiple values, separated by commas. Optional arguments are covered in the "arguments" section below.

=head1 - ARGUMENTS

=head2 Passing Multiple Values to an Argument

Some arguments support multiple values. To pass multiple values to
an argument, separate them with commas. Do not include spaces.

=head2 --help

Show this manual

=head2 --independent I<independent variable name>

Default: threads

The name of the independent variable (x axis)
This should match with the label of a data column in your data file
if the xlabel option is unspecified, the x axis will be the independent variable name.

=head2 --dependent I<dependent variable name>

The name of the dependent variable (y axis)
This should match with the label of a data column in your data file
if the ylabel option is unspecified, the y axis label will be the dependent variable name.

=head2 --split I<split column title>

The column over which to split the data. The data is split into differently colored lines - one for each value in the split column. For instance, with three parallelization styles, --split style will show a separate line for each style.
			
=head2 --selector_column I<column by which to select sets>

Default: tag

This need not be used often. In the unusual case that one might want to select, instead of all the sets with a specific tag, say, all sets with a specific number of cores, one can specify "cores" as the selector column.

=head2 --xlabel I<x axis label> --ylabel I<y axis label>

Labels for the axes. If either value is unspecified, the respective variable name is used. 

=head2 --title I<Plot Title>

Graph Title. Didn't your middle school science teacher ever tell you always to title your graphs?

The default title describes the graph as a comparison between the xlabel value and the ylabel value.

=head2 --errorbars

Enable errorbars to be shown extending to one standard deviation above and below each data point. This is meaningless when there is only one data point per x value.


=head2 --datafile I<datafile path>

Point to your datafile. By default, StatKit will look for output.txt in the directory in which it was run.

=head2 --output I<PostScript file>

The file into which the image of the graph is to be saved.
Right now PlotKit supports only PostScript. If your computer can't handle .ps, try ps2pdf, or for Windows GhostScript http://pages.cs.wisc.edu/~ghost

=head2 --tempgdi I<temporary gnuplot script> --tempdata I<temporary data file>
The names of the temporary in-between files that PlotKit uses to interface with gnuplot. These files are erased at the completion of the program unless --messy is enabled, so these arguments should be seldom used.

=head2 --messy -m

PlotKit tries to be as tidy as possible, cleaning up all of its temporary files at the end of its execution unless this command is specified. Good for debugging or customizing your graph in the gnuplot gdi.

=head1 EXAMPLE

If you've just run stat.pl in its stats folder and want to plot the walltime over threads, use the following command:

perl PlotKit.pl \
--independent threads --dependent walltime \
--datafile stats/output.txt \
--output myRun.ps
myRun 

This will output a file - myRun.ps - containing a graph of the output for the data under the "myRun" tag.
