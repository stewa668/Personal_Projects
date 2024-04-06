package statc;

use constant SUPPORTED         => 1;
use constant UNSUPPORTED       => 0;
use constant SECONDS_IN_MINUTE => 60;

use vars qw/$RESULTS_BUFFER_BEGIN $RESULTS_BUFFER_END/;

my $DBI = SUPPORTED;
eval "use DBI; 1" or $DBI = UNSUPPORTED;

require "cluster.pm";

$RESULTS_BUFFER_BEGIN = "!~~~#**BEGIN RESULTS**#~~~!\n";
$RESULTS_BUFFER_END   = "!~~~#**END RESULTS**#~~~!\n";

#take a processor request and convert it into a list of processor counts
#1,4 - run on one processor and then four processors
#1-4 = 1,2,3,4
#1-4-16 = 1,4,8,12,16
sub interpolate {
    my @pp = ();
    for my $proc (@_) {
        my @h;
        if ( index( $proc, '-' ) != -1 ) {
            my @r = split( '-', $proc );
            my $count = @r;
            if ( $count == 2 ) {
                @h = ( $r[0] .. $r[1] );
            }
            elsif ( $count == 3 ) {
                my $a = $r[1] - $r[0];
                $r[0] /= $a;
                $r[2] /= $a;
                @r = ( $r[0] .. $r[2] );
                @h = map { $_ * $a } @r;
            }
            else {
                die
                  "ERROR: $count arguments to processor interpolator. Maximum 3.
				    2-10 -> 2,3,4,5,6,7,8,9,10
				    2-4-10 -> 2,4,6,8,10
				    ";
            }
        }
        else {
            @h = $proc;
        }
        push( @pp, @h );
    }

    return (@pp);
}

#parse the data into the database
sub parse_data {
    my $cl       = shift @_;
    my $tag      = shift @_;
    my $dbname   = shift @_;
    my $table    = shift @_;
    my $date     = shift @_;
    my ($output) = @_;

    #Take only the actual data - cut out everything before and after the buffers
    my $startindex =
      index( $output, $RESULTS_BUFFER_BEGIN ) + length $RESULTS_BUFFER_BEGIN;
    my $endindex = index( $output, $RESULTS_BUFFER_END );

    if ( $endindex == -1 ) {
        die(
"\"$RESULTS_BUFFER_END\" not found. Use --proxy-output to have stat.pl generate output for programs without pkit support
OUTPUT:$output"
        );
    }

    $output = substr( $output, $startindex, $endindex - $startindex );

    print("$cl $tag $dbname $table $date\n");
    print $output;

    $output = trim($output);

    my %data = split( /\s*:\s*|\n/, $output );

    unless ( defined $data{'TIME'} ) {
        print "TIME not defined. Very likely a program error.";
        exit 0;
    }

    if ( $data{'CPUTIME'} > 10**37 ) {  #extreme cpu times confuse the database,
        $data{'CPUTIME'} = 10**37;      #so dumb things down a little bit.
    }    #BTW, cputimes this big are erroneous anyway

    if ( index( $data{'TIME'}, 'm' ) != -1 ) {
        my @time = split( /[ms]/, $data{'TIME'} );
        $data{'TIME'} = ( $time[0] * SECONDS_IN_MINUTE ) + $time[1];
    }

    #Get program part of program name
    @prgm = split( "_", $data{'PROGRAM'} );
    $program = $prgm[0];

    #print to text-based gnuplot table
    if ( $dbname eq "text" ) {
        my $new = 0;
        if ( !-e "output.txt" ) {
            $new = 1;
        }
        $date =~ s/ /_/g;
        open TABLE, ">>output.txt" or die $!;
        if ($new) {
            print TABLE
"\%tag date program walltime cputime architecture problem_size threads command_line\n";
        }
        print TABLE
"$tag $date $program $data{'TIME'} $data{'CPUTIME'} $data{'ARCH'} $data{'PROBLEM_SIZE'} $data{'THREADS'} $cl\n";
        close TABLE;
    }
    else {

        #actual database interaction
        my $dbh =
          DBI->connect("dbi:Pg:database=$dbname;host=cluster.earlham.edu")
          or die "Couldn't connect to database: " . DBI->errstr;
        my $command =
          $dbh->prepare("INSERT INTO $table VALUES(?,?,?,?,?,?,?,?,?,?,?)")
          or die "Couldn't prepare request: " . DBI->errstr;
        $command->execute(
            $date,                 $program,
            $data{'VERSION'},      cluster::getCluster( $data{'HOSTNAME'} ),
            $data{'THREADS'},      $data{'TIME'},
            $data{'CPUTIME'},      $data{'ARCH'},
            $data{'PROBLEM_SIZE'}, $cl,
            $tag
        ) or die "Couldn't execute statement: " . $command->errstr;
    }
}

#adjust the problem size
sub adjust_problem_size {
    ( my $ps_func, my $problem_size, my $processors ) = @_;
    my $ps = $problem_size;
    if ( $ps_func eq 'square_root' ) {
        $ps *= sqrt $processors;
    }
    elsif ( $ps_func eq 'linear' ) {
        $ps *= $processors;
    }
    elsif ( $ps_func eq 'quadratic' ) {
        $ps *= $processors * $processors;
    }
    return $ps;
}

# Perl trim function to remove whitespace from the start and end of the string
sub trim() {
    my $string = shift;
    $string =~ s/^\s+//;
    $string =~ s/\s+$//;
    return $string;
}

return 1;    #this lets perl know the module ran successfully.
