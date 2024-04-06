package cluster;

use strict;

my %name_hash = (
    qr/.*compute.*/, "BobSCEd", qr/acl\d+/,       "ACLs",
    qr/^b\d+/,       "Bazaar",  qr/^[cv]\d+.*/,   "Sooner",
    qr/.*pople.*/,   "Pople",   qr/s\d+c\d+b\d+/, "BigRed",
    qr/.*hopper.*/,  "Hopper"
);

sub getCluster {
    ( my $hostname ) = @_;
    while ( my ( $key, $value ) = each(%name_hash) ) {
        if ( $hostname =~ $key ) {
            return $value;
        }
    }
    return $hostname;
}
return 1;
