#!/usr/local/bin/perl
#
# Gives performance data for area-under-curve and GalaxSee
# includes multiprocess support
#
use strict;
use POSIX qw(ceil);
use File::Path;    #for rmtree

#use DBI;            #for connecting to the database
use Cwd;           #to get current path
use Pod::Usage;
use statc;         #various functions
use argbuilder;

#Global Constants
use constant MAX_BUFFER => 1048576;

#Global Variables
my $clean;
my $command_line_template;
my $dbname;
my $finalize;
my $help;
my $hostfile;
my $input_file;
my $mass;
my $max_time;
my $messy;
my $mpirun;
my $nosubmit;
my $path;
my $ppn;
my $ppn_defaulted;
my $problem;
my $proxy_output;
my $ps_func;
my $queue;
my $range;
my $repetitions;
my $scheduler;
my $scriptfile;
my $seed;
my $share_nodes;
my $ssh_key;
my $special_template;
my $table;
my $tag;
my $tempdirectory;
my $templatefile;
my $templates;
my $user;
my $version;
my $working_dir;

my @command_line;
my @functions;
my @problem_sizes;
my @processors;
my @program;
my @steps;
my @styles;

##
##MAIN
##

#Get arguments
get_args();

#This lets us erase the temporary shell scripts
if ($clean) {
    clean();
    exit 0;
}

# If command line template unspecified, find one from the built-in presets or fail
if ( not $command_line_template ) {
    $command_line_template = get_template($problem);
}

# If it doesn't already exist, make the temporary script directory
if ( !-e $tempdirectory ) {
    mkdir $tempdirectory;
}

my $input_template;

if ($input_file) {
    open( INPUT_FILE, "<$input_file" )
      or die "Could not open input file template file $!";
    read( INPUT_FILE, $input_template, MAX_BUFFER )
      or die "Could not read input template file $!";
    close(INPUT_FILE);
}
elsif ( not -t STDIN ) {
    read( STDIN, $input_template, MAX_BUFFER )
      or die "Could not read standard input $!";
}

# This series of nested loops runs a test for each combination of the parameters

my $index = 0;    # Reset the index
for my $style (@styles) {    # For each parallelization
    for my $problem_size (@problem_sizes) {    # For each problem size
        for my $step (@steps) {
            for my $threads (@processors) {    # For each number of threads
                for my $func (@functions) {    # For each function over x
                     # Get the problem size (modified for weak or strong scaling)
                    my $ps =
                      statc::adjust_problem_size( $ps_func, $problem_size,
                        $threads );

                    my %vars = (
                        "function",     \$func,
                        "problem_size", \$ps,
                        "cores",        \$threads,
                        "threads",      \$threads,
                        "processes",    \$threads,
                        "steps",        \$step,
                        "style",        \$style,
                        "hostfile",     \$hostfile,
                    );

                    my $command_line =
                      argbuilder::build_command_line( $command_line_template,
                        %vars );

                    print "COMMAND LINE: $command_line\n";

                # Put together the data into a set of parameters for make_script
                    @program = (
                        $problem,      $style,    $index,
                        $hostfile,     $threads,  $ppn,
                        $command_line, $max_time, $finalize,
                    );

                    $scriptfile =
                      make_script( $tempdirectory, @program, $input_template,
                        $templatefile, $templates, %vars );    # Make the script

                    # Print some information
                    # To identify the current job
                    my $prints = ( join( " ", @program, "\n" ) );
                    print "$prints";

                    if ( -e $tempdirectory . '/' . $scriptfile
                        and not $nosubmit )
                    {

                        # Run the test or submit it to the scheduler
                        run_test( $tempdirectory . '/' . $scriptfile );
                        ++$index;
                    }
                }
            }
        }
    }
}

if ( !$messy ) {
    rmtree($tempdirectory);    #remove the temporary directory
}

##
## Subroutines
##

##************************##
# Collect arguments
# --help for documentation
##************************##
sub get_args {

    use Getopt::Long

      # defaults
      @functions = ();    # The algebraic functions
    @processors  = ();    # The number of processors
    my @process_request;  # The request for processors
    @problem_sizes = ();  # The "problem size":
    @steps         = ();  # the number of steps
    @styles        = ();  # The styles of parallelism to use

    $command_line_template =
      "";    # The template from which the various command lines are built
    $dbname = 'text';    # The database name
    $finalize = '';  # Any additional commands to include after the main command
    $help     = 0;   # open documentation
    $hostfile = '';  # The hostfile
    $input_file   = undef;       # the input file
    $max_time     = '00:30';     # the estimated runtime to give the scheduler
    $messy        = 0;           # Do not delete temporary files
    $mpirun       = 'mpirun';    # where's mpirun?
    $nosubmit     = 0;           # don't run created scripts (for debugging)
    $path         = undef;       # The path to the program folder
    $ppn          = undef;       # Processors per node (dependent on host)
    $problem      = "noname";    # The program to run
    $proxy_output = 0
      ; # Set this to true if the program you're editing is not equipped to receive output
    $ps_func     = 'constant';  # Is the problem size a constant
                                # or related to the number of processors?
    $queue       = 'normal';    # the queue to which to submit jobs
    $range       = undef;       # The domain of the function
    $repetitions = 1;           # The number of repetitions for each combination
    $scheduler   = '';          # Submit to scheduler? (or run right away)
    $share_nodes = 0;           # Share_nodes
    $special_template = '';          # Special user commands before running main
    $ssh_key          = '';          # default ssh_key
    $table            = 'auc';       # The table in the database
    $tag              = '<none>';    # The tag for recordkeeping
    $tempdirectory    = 'statstemp'; # The temporary directory
    $templatefile = 'default.sh';          # The name of the template file
    $templates    = 'script_templates';    #the directory of template files
    $user    = 'petakit';    # The username to be used to access hopper
    $version = "N/A";        # the version of the program
    chomp( $working_dir = `pwd` )
      ;                      # the path to which scheduler output is written
    $working_dir =~ s/ /\ /g;

    # Get Options
    GetOptions(
        'clean|c'                                    => \$clean,
        'command_line_template|cl_temp|cl=s'         => \$command_line_template,
        'database=s'                                 => \$dbname,
        'finalize=s'                                 => \$finalize,
        'function|f=s'                               => \@functions,
        'help'                                       => \$help,
        'hostfile|h=s'                               => \$hostfile,
        'input|input_file|i=s'                       => \$input_file,
        'mass'                                       => \$mass,
        'max-time|mt=s'                              => \$max_time,
        'messy|m'                                    => \$messy,
        'mpirun|mpi_alias=s'                         => \$mpirun,
        'nosubmit'                                   => \$nosubmit,
        'proxy-output'                               => \$proxy_output,
        'path=s'                                     => \$path,
        'ppn|processors_per_node=i'                  => \$ppn,
        'problem_func|pfunc|pf=s'                    => \$ps_func,
        'problem_size|size|ps|stars|segments|n=s'    => \@problem_sizes,
        'problem|program|p=s'                        => \$problem,
        'processes|processors|nodes|threads|cores=s' => \@process_request,
        'queue|q=s'                                  => \$queue,
        'range|x=s'                                  => \$range,
        'repetitions|r=i'                            => \$repetitions,
        'scheduler=s'                                => \$scheduler,
        'seed'                                       => \$seed,
        'share_nodes!'                               => \$share_nodes,
        'special=s'                                  => \$special_template,
        'ssh_key=s'                                  => \$ssh_key,
        'steps=s'                                    => \@steps,
        'style|parallelism=s'                        => \@styles,
        'table=s'                                    => \$table,
        'tag=s'                                      => \$tag,
        'tempdirectory|temp|td=s'                    => \$tempdirectory,
        'templatedir=s'                              => \$templates,
        'template|t=s'                               => \$templatefile,
        'user=s'                                     => \$user,
        'version=s'                                  => \$version,
        'workdir=s'                                  => \$working_dir,
    );

    if ($help) {
        pod2usage( -verbose => 2 );
        exit 0;
    }

    unless ($command_line_template) {
        print "For manual:\nperl stat.pl --help\n";
        exit 1;
    }

    # Corrections - standardize synonyms
    $problem = lc($problem);
    if (   $problem eq 'nbody'
        || $problem eq 'gal'
        || $problem eq 'n-body'
        || $problem eq 'galaxsee' )
    {
        $problem = 'GalaxSee';
    }
    elsif ($problem eq 'area-under-curve'
        || $problem eq 'area-under-a-curve'
        || $problem eq 'riemann-sum'
        || $problem eq 'riemann' )
    {
        $problem = 'area';
    }

    $ps_func = lc($ps_func);
    if ( $ps_func eq 'none' || $ps_func eq 'const' || $ps_func eq 'c' ) {
        $ps_func = 'constant';
    }
    if ( $ps_func eq 'squareroot' || $ps_func eq 'root' || $ps_func eq 'sqrt' )
    {
        $ps_func = 'square_root';
    }
    if ( $ps_func eq 'lin' || $ps_func eq 'n' ) {
        $ps_func = 'linear';
    }
    if ( $ps_func eq 'quad' || $ps_func eq 'n^2' ) {
        $ps_func = 'quadratic';
    }
    for ( my $i = 0 ; $i < @styles ; ++$i ) {
        if ( $styles[$i] eq "shared" ) {
            $styles[$i] = "openmp";
        }
        if ( $styles[$i] eq "distributed" ) {
            $styles[$i] = "mpi";
        }
    }

    unless ( defined $ppn ) {
        $ppn = 4;
    }

    # Other defaults
    if ( not @styles ) {
        @styles = ("_");
    }

    if ( not @functions ) {
        @functions = ('x^2');
    }

    if ( not @problem_sizes ) {
        @problem_sizes = (10000);
    }

    if ( not @steps ) {
        @steps = (30);
    }

    if ( $table eq '' ) {
        if ( $problem eq 'GalaxSee' ) {
            $table = 'GalaxSee';
        }
        elsif ( $problem eq 'area' ) {
            $table = 'area';
        }
    }

    if ( ( $ssh_key eq '' ) and ( $user eq 'petakit' ) ) {
        $ssh_key = "petakit_id";
    }

    # split comma-separated strings
    @styles          = split( /,/, join( ',', @styles ) );
    @functions       = split( /,/, join( ',', @functions ) );
    @process_request = split( /,/, join( ',', @process_request ) );
    @problem_sizes   = split( /,/, join( ',', @problem_sizes ) );
    @steps           = split( /,/, join( ',', @steps ) );

    # Processes - interpolate lists from x1-x2
    @processors = statc::interpolate(@process_request);

    # Remove invalid processor counts
    my @processors_temp;
    foreach my $process_count (@processors) {
        if ( $process_count > 0 ) {
            push( @processors_temp, $process_count );
        }
    }
    @processors = @processors_temp;

    unless (@processors) {
        @processors = (1);
    }
}

##************************##
# Run the test
##************************##
sub run_test {
    my $program = join( " ", @_ );
    for ( my $i = 0 ; $i < $repetitions ; $i++ ) {

        # Submit the script to the appropriate scheduler
        if ( $scheduler eq "pbs" ) {
            `qsub $program`;
        }
        elsif ( $scheduler eq "lsf" ) {
            `bsub < $program`;
        }
        elsif ( $scheduler eq "loadleveler" ) {
            `llsubmit $program`;
        }
        elsif ( $scheduler eq '' ) {
            my $date = substr( localtime(), 4 ); # cut off the day of the week
            my $u = `./$program`;                # run the program, store output
            statc::parse_data( join( ' ', @command_line ),
                $tag, $dbname, $table, $date, $u );
        }
        else {
            print
"ERROR: scheduler not recognized. Supported schedulers include pbs, lsf and loadleveler.";
            exit 1;
        }
    }
}

##***************************##
# Write a temporary script file
##***************************##
sub make_script {
    my $tempdirectory     = shift(@_);
    my $program           = shift(@_);
    my $style             = shift(@_);
    my $index             = shift(@_);
    my $hostfile          = shift(@_);
    my $threads           = shift(@_);
    my $procpernode       = shift(@_);
    my $command_line      = shift(@_);
    my $max_time          = shift(@_);
    my $finalize          = shift(@_);
    my $input_template    = shift(@_);
    my $template_filename = shift(@_);
    my $template_folder   = shift(@_);
    my %vars              = @_;

    my $scriptfile;
    my $nds;
    my $job_type;
    my $class;
    my $prep;
    my $run_id;
    my $formatted_command_line;
    my $special;

    my $extension;
    unless ($template_filename) {
        $extension = "sh";
    }
    else {
        my $extension_index = rindex( $template_filename, '.' );
        $extension = substr( $template_filename, $extension_index + 1 );
    }

    my $suffix;
    my $prefix = '';
    my $main;
    my %scriptvars = (
        "class",       \$class,       "command_line", \$command_line,
        "finalize",    \$finalize,    "hostfile",     \$hostfile,
        "index",       \$index,       "job_type",     \$job_type,
        "main",        \$main,        "max_time",     \$max_time,
        "nds",         \$nds,         "prep",         \$prep,
        "procpernode", \$procpernode, "program",      \$program,
        "queue",       \$queue,       "run_id",       \$run_id,
        "style",       \$style,       "special",      \$special,
        "threads",     \$threads,     "user",         \$user,
        "working_dir", \$working_dir,
    );

    $run_id = $program . '-' . $index;
    print "$run_id\n";

    #initialize scriptfile
    $scriptfile = "$run_id.$extension";

    if ( $procpernode > $threads ) {
        $procpernode = $threads;
    }

    if ( $style eq "openmp" and $threads > $procpernode ) {
        return ();
    }

    if ( $style eq "serial" ) {
        $procpernode = 1;
    }
    if ( $style eq "openmp" or $style eq "serial" ) {
        $nds = 1;
    }
    else {
        $nds = ceil( $threads / $procpernode );
    }
    if ( $style eq 'serial' ) {
        $job_type = 'serial';
        $class    = 'SERIAL';
    }
    else {
        $job_type = 'parallel';
        $class    = 'NORMAL';
    }

    # add --hostfile to mark $hostfile
    if ( defined $hostfile ) {
        $hostfile = '--hostfile ' . $hostfile;
    }

    #initialize $ssh_key
    if ($ssh_key) {
        $ssh_key = "-i $ssh_key";
    }

    #initialize prep
    $prep = "cd $working_dir
      date=\`date +'%Y-%m-%d %H:%M:%S'\` 
      ";

    #initialize $prefix (input command)
    if ($input_template)
    { #if an input file template has been specified, whether through a file or STDIN
        print "\n\n###INPUT TEMPLATE###\n$input_template##############\n\n";
        my $standard_input = build_input_file( $input_template, %vars );
        my $input_filename = $run_id . '.in';
        open( INPUT, ">$tempdirectory/$input_filename" )
          or die "Could not open input file $!";
        print INPUT $standard_input;
        close(INPUT);
        $prefix = "cat $tempdirectory/$input_filename | ";
    }

    #rebuild command line to proxy output (if applicable)
    if ($proxy_output) {
        $formatted_command_line =
          build_output( $command_line, $program, $threads, $style,
            ${ $vars{'problem_size'} }, $version );
    }
    else {
        $formatted_command_line = $command_line;
    }

    #initialize $suffix
    if ( $scheduler ne "" ) {
        if ( $dbname eq "text" ) {
            $suffix =
" | perl parser.pl '$command_line' $dbname $table \"\$date\" '$tag' ";
        }
        else {
            $suffix =
" | ssh $user\@cluster.earlham.edu $ssh_key \"perl ~$user/stats/parser.pl '$command_line' $dbname $table \"\$date\" '$tag'\"";
        }
    }

    #initialize $main
    $main = "$prefix$formatted_command_line$suffix";

    #initialize $special

    $special = argbuilder::build_command_line( $special_template, %scriptvars );

    #build $script
    open( TEMPLATE, "<$template_folder/$template_filename" )
      or die
      "Could not open template file $template_folder/$template_filename $!";

    my $template;
    read( TEMPLATE, $template, 1028 * 1028 )
      or die "Could not read template file $!";

    close(TEMPLATE);

    my $script = argbuilder::build_command_line( $template, %scriptvars );

    open( PROG, ">$tempdirectory/$scriptfile" )
      or die "Could not open scriptfile $!";

    print PROG $script;

    close(PROG);

    print "\n";

    chmod 0765, "$tempdirectory/$scriptfile";

    return $scriptfile;
}

##************************##
# Delete the temporary files
##************************##
sub clean {
    rmtree($tempdirectory);
}

#********************************************************************##
# Build output for programs that aren't equipped to build it themselves
#********************************************************************##
sub build_output {
    my (
        $command_line, $program_name, $threads,
        $architecture, $problem_size, $version
    ) = @_;
    my $output =
"TIME=`(time -p $command_line >> /dev/null) 2>&1 | awk '/^real/ {print \$2}'`
echo \"$statc::RESULTS_BUFFER_BEGIN
PROGRAM		: $program_name
HOSTNAME	: `hostname`
THREADS		: $threads
ARCH		: $architecture
PROBLEM_SIZE	: $problem_size
VERSION		: $version
CPUTIME		: 0
TIME		: \$TIME
$statc::RESULTS_BUFFER_END\"";

    return $output;
}

##*****************************************************************##
# Look for a built-in command line template for the specified program
# If one does not exist, exit with an error
##*****************************************************************##
sub get_template {
    my $problem = shift;
    if ( $problem eq 'GalaxSee' ) {
        @command_line = ('-t $step -m $mass -r $seed --stats $problem_size');
    }
    elsif ( $problem eq 'area' ) {
        @command_line = ('-f $func --range $range $problem_size');
    }
    else {
        die(
"unable to automatically build command line\n- please use --command_line_template to specify a command line"
        );
    }
    return @command_line;
}

##***************************************************************************##
# Take an input template and output an input file string with the respective arguments
##***************************************************************************##
sub build_input_file {
    my @input           = split( /\n/, shift );
    my %vars            = @_;
    my $processed_input = '';
    foreach my $line (@input) {
        chomp $line;
        my $formatted_line = argbuilder::build_command_line( $line, %vars );
        $processed_input .= $formatted_line . "\n";
    }
    return $processed_input;
}

##**************************************************************************##
# Perl trim function to remove whitespace from the start and end of the string
##**************************************************************************##
sub trim() {
    my $string = shift;
    $string =~ s/^\s+//;
    $string =~ s/\s+$//;
    return $string;
}

__END__

For more information contact leemasa@earlham.edu

=head1 - SYNOPSIS

This script is designed to run a given program several times under a range of arguments and collect meaningful data. Most commonly, it is used to analyze how the walltime for a program changes as it is run with more threads or processors.

=head1 - ARGUMENTS

=head2 - Notes on Passing Values to Arguments

=head3 Passing Multiple Values

Some arguments support multiple values over which stat.pl iterates.
To pass multiple values to an argument, separate them with commas.
Do not include spaces.

=head3 Interpolating Numerical Arguments

For supported numeric arguments, There are a few special characters to ease your list creation.
To count from a to b, use a-b (1-4 = 1,2,3,4)
to count by c, use a-(a+c)-b (2-4-8 = 2,4,6,8)

=head3 Examples

=over 8

=item 2,4,16 = 2,4,16

=item 2-4,16 = 2,3,4,16

=item 2,4-16 = 2,4,5,6,7,8,9,10,11,12,13,14,15,16

=item 2-4-16 = 2,4,6,8,10,12,14,16

=back

=head2 --help

Show this manual

=head2 --cl I<command_line_template>

=head3 EXPLANATION

The command line template is the center of any run of stat.pl. It contains not only the program to be run, but the format of the arguments to send to it. With a proper understanding of the command line template a user can evaluate nearly any program with a command line interface.

=head3 USAGE

The string passed to the command line template argument should look exactly like the command you wish to evaluate, with one exception. The arguments you want to change over your various runs are represented by variables instead of the values themselves. A variable is represented by a preceding '$', and matches up with values given to it in additional arguments to PetaKit. PetaKit iterates through all permutations of the arguments given it.

Currently, the following variables are supported:

 cores
 function
 problem_size
 steps
 style
 hostname

Only cores has an effect outside what is defined in the command line template, and only when working with a scheduler.

=head3 EXAMPLE
 
 perl PlotKit.pl -cl 'mpirun -np $cores ./area -s $problem_size'\
 --cores = 1-8\
 --problem_size = 10000,20000\

Runs area 16 times: once for each combination of core number and problem size.
the -s command in this example tells area the number of segments to use in estimating the problem size.

=head2 --template I<filename>, -t I<filename>

Reads a script file B<template>, which follows the same format as the command line template. A few special variables are required. Make sure to include $prep, $special and $main, each on its own line and in that order, but the rest is all for making the scheduler happy. The best way to learn to write one of these templates is to look in the script_templates directory. B<sooner.lsf> would be a good example for starters.

The script file template supports the following variables:

 class      
 command_line
 finalize   
 hostfile   
 index      
 job_type   
 main	 
 max_time   
 nds        
 prep       
 procpernode
 program    
 queue	 
 run_id     
 style      
 special
 threads    
 user 	 
 working_dir

=head3 NOTE

The entire path name is not necessary to access a template in the script_templates directory. Simply use the filename. 

=head2 --special I<special command template>

For particular run-specific commands before $main. Setting particular environment variables, for example. Follows the same template system as command line template.

=head2 --input I<filename>, --input_file I<filename>, -i I<filename>

Reads an input file B<template>, which follows the same format as the command line template, and submits it into the given program's standard input. See --command_line_template.

=head2 --processes I<process_request>, --cores I<process_request>, --threads I<process_request>

The number of processes to use. This maps to a variable in command line template of any of the above three names. Be aware that all three names map to the same variable. When crafting scheduler submission scripts, stat.pl includes information from this argument.

=head2 --function I<string>

A variable whose meaning is defined by the command line template (see --cl).

This argument accepts multiple values.

=head2 --problem_size I<number>

A variable whose meaning is defined by the command line template (see --cl).
Although what it really does is up to you "problem size" refers to the scale of the problem being run.

This argument accepts multiple values.

=head2 --steps I<string>

A variable whose meaning is defined by the command line template (see --cl).
Although what it really does is up to you, "steps" refers to the number of steps over which to run.

This argument accepts multiple values.

=head2 --style I<string>

A variable whose meaning is defined by the command line template (see --cl).
Although what it really does is up to you, "style" refers to the style of parallelization.

This argument accepts multiple values.

=head2 --program I<program_name>, -p I<program_name>

Let stat.pl know what program it's running when not using a stat.pl-compatible program (see --proxy-output)

=head2 --version I<version_number>

Let stat.pl know the version of the program it's running when not using a stat.pl-compatible program (see --proxy-output)

=head2 --problem_func I<[constant,square_root,linear,quadratic]>, -pf I<[constant,square_root,linear,quadratic]> 

For strong, weak, or another type of scaling. Rather than always being the same number, problem size is given to the problem as problem_size*f(threads), where f is the selected function. default is constant.

constant - strong scaling

linear - weak scaling

=head2 --repetitions I<number>

The number of runs for each combination of parameters

=head2 --tag I<tagname> 

The tag for identifying runs. Do not use apostrophes.

default is '<none>'.

=head2 --database I<database_name>

The name of the database. default is "text," for output into a local ASCII database.

=head2 --tempdirectory I<temporary_directory_name>, --temp I<temporary_directory_name>, --td I<temporary_directory_name>

The temporary directory for storing shell scripts and input files

=head3 DEFAULT

statstemp

=head2 --max-time I<hh:mm>, --mt I<hh:mm>

The estimated maximum time the program will run. At the moment this is the same under all parameters, so choose the maximum time of all runs.

=head2 --queue I<queue_name>, -q I<queue_name>

The scheduling queue in which to run.

=head2 --sharenodes

By default, StatKit asks schedulers to make sure it doesn't share nodes with any other processes. This command tells it not to. This should not be necessary, but to make absolutely certain that StatKit insists nodes not be shared --nosharenodes can be used.


=head2 --hostfile I<filename>, -h I<filename>

The path to the file that specifies which computers are to be used for multi-processor parallelism.

=head2 --processors_per_node I<number>, --ppn I<number>

Especially important on schedulers (see 'scheduler'). Specifies the number of cores on each node the program should use.

=head2 --table I<table_name>

The table in the database in which to input information

=head2 --scheduler I<[pbs, loadleveler, lsf]>

The scheduler to use.

=head3 NOTES

The scheduler is the most challenging part of automated benchmarking on arbitrary clusters.
Because each scheduler has its own system and quirks, and each cluster often adds additional
constraints to its scheduling scripts, I'm working on building a system by which users can define
their own scheduler script templates much like the command line template. For the moment, however,
running with an unsupported scheduler will likely require directly modifying the scheduler section
of stat.pl's code.

pbs - Use the Portable Batch Scheduler (BobSCEd, Kraken)
loadleveler - Use LoadLeveler (Note: currently hardcoded to support BigRed)
lsf - Use LSF (Sooner)

=head3 DEFAULT

No scheduler. Run immediately. (ACL, personal computers)

=head2 --sshkey I<filename> 

Pathname of a file containing the sshkey for non-interactive authentication when sending data.

=head3 DEFAULT

If username is petakit and StatKit is accessing a remote database : petakit_id

otherwise empty

=head2 --user I<username>

The username to use to attempt to gain access to hopper and store results in the database. It is important that you have already given whatever system you are using non-interactive access to hopper unders this name or stat.pl will be B<unable to store results>.

=head3 DEFAULT

petakit

=head2 --proxy-output

Instructs StatKit to simulate the statistics output that it collects. StatKit will collect the time of the given command using UNIX's 'time' function, and takes all the other relevant information directly from its command line arguments. The command line arguments to provide this information are --program, --cores, --style, --problem_size, and --version.

=head2 --clean, -c

Cleans residual temporary files from a previous harvest and exits. Only cleans file generated directly by StatKit.

=head2 --messy, -m

Do not clean up automatically after a harvest.

=head2 --nosubmit

Do not run/submit shell scripts. Does not prevent cleanup, so best used with --messy.

=head2 --workdir I<directory_path>

The directory into which StatKit directs scheduler output
