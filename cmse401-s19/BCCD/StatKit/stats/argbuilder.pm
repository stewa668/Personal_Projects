package argbuilder;
use strict;

sub build_command_line {
    my $cmdline = shift;
    my (%vars)  = @_;
    my @keys    = keys %vars;
    @keys = sort { length $b <=> length $a } @keys;
    my $i;
    for ( $i = 0 ; $i < @keys ; $i++ ) {
        my $variable = $keys[$i];
        my $value    = ${ $vars{$variable} };
        $cmdline =~ s/\$$variable/$value/g;
    }
    return $cmdline;
}

return 1;
