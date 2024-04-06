#!/usr/local/bin/perl
use lib '/cluster/home/petakit/stats';
use statc;

my ( $cl, $dbname, $table, $date, $tag ) = @ARGV;
my @input = <STDIN>;
my $output = join( '', @input );

#command line - chop off all but arguments
@command = split( / /, $cl );
while ( index( $command[0], '-' ) != 0 and @command ) {
    shift @command;
}
$cl = join( ' ', @command );

#parse the data - see statc.pm
statc::parse_data( $cl, $tag, $dbname, $table, $date, $output );
