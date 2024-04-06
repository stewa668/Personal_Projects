package PlotKit;
use Getopt::Long;

#Boolean constants
use constant FALSE=>0;
use constant TRUE=>1;

#plotkit data columns
use constant TAG=>0;
use constant SPLIT=>1;
use constant INDEPENDENT=>2;
use constant DEPENDENTAVG=>3;
use constant DEPENDENTERROR_MAX=>4;
use constant DEPENDENTERROR_MIN=>5;

#Globals
my @selectors;

# Perl trim function to remove whitespace from the start and end of the string
sub trim()
{
	my $string = shift;
	$string =~ s/^\s+//;
	$string =~ s/\s+$//;
	return $string;
}

#get arguments
sub getargs()
{
	Getopt::Long::Configure ("bundling");
	my %globals = ( 
		#Graph Data Parameters
		"selectors" , \@selectors,
		"independent" , "threads",
		"dependent" , "walltime",
		"selector_column", "tag",
		"split" , undef, 

		#Statistical Flairs
		"errorbars" , FALSE,

		#Labels
		"title" , undef,
		"xlabel" , undef,
		"ylabel" , undef,

		#Data
		"data", undef,
		
		#Source File	
		"datafile" , "output.txt",
		
		#Temporary Files
		"tempdata" , "temp.dat",		
		"tempgdifile" , "temp.gdi",
		"messy" , FALSE,		

		#Output
		"output", "screen",

		);
	
	my $selectors = pop @ARGV;
	@selectors = split(/,/,$selectors);

	GetOptions (
			"independent=s" => \$globals{'independent'}, 
			"dependent=s" => \$globals{'dependent'},
			"split=s" => \$globals{'split'},
			"title=s" => \$globals{'title'},
			"xlabel=s" => \$globals{'xlabel'},
			"ylabel=s" => \$globals{'ylabel'},
			"datafile=s" => \$globals{'datafile'},
			"output|o=s" => \$globals{'output'},
			"tempgdi=s" => \$globals{'tempgdifile'},
			"tempdata=s" => \$globals{'tempdata'},
			"selector_column" => \$globals{'selector_column'},
			"messy|m" => \$globals{'messy'},
			"errorbars|e" => \$globals{'errorbars'},
		) or die "Argument parsing error $!";

	unless (defined $globals{'xlabel'}){
		$globals{'xlabel'} = $globals{'independent'};
	}
	
	unless (defined $globals{'ylabel'}){
		$globals{'ylabel'} = $globals{'dependent'};
	}

	return %globals;
}

sub usage(){
	print "USAGE:
perl PlotKit.pl [options] (tags)\n";
}


#Set up the datafile for splitting over the split data
#and averaging multiple dependent values for independent values
sub build_dataset{
	my $tag_index = shift;
	my $independent_index = shift;
	my $dependent_index = shift;
	my $split_index = shift;
	my %globals =  @_;

	my $data = $globals{'data'};
	my $tempdata = $globals{'tempdata'};
	my $selectorsref = $globals{'selectors'};
	my @selectors = @$selectorsref;

	my @sets;
 
	foreach my $selected (@selectors){
		$data = selectbycolumn($tag_index,$selected,$data);
		unless ($data){
			print "tag \"$selected\" not found";
			exit 1;
		}
		push (@sets, $data);
	}
	
	@sets = stat_repeats(@sets,$tag_index,$split_index,$independent_index,$dependent_index);	
	@sets = organize_data(@sets,SPLIT,INDEPENDENT);
	
	open (DATA,">".$globals{'tempdata'});
		print DATA join ("\n\n\n", @sets);
	close DATA;
	return @sets;	
}

#For each independent value, average together
#multiple dependent values (if applicable)
sub stat_repeats{
	my $dependent_index = pop;
	my $independent_index = pop;
	my $split_index = pop;
	my $tag_index = pop;
	my @sets = @_;

	my @newdata;	
	foreach my $set(@sets){
		my %datapoints;
		foreach my $line(split(/\n\n\n/,$set)){
			foreach my $point(split(/\n/,$set)){
				if ($point){
					my @point = split(/ /,$point);
					my $dependent = $point[$dependent_index];
					my $independent = $point[$independent_index];
					my $tag = $point[$tag_index];

					my $split;
					if (defined $split_index){
						$split = $point[$split_index];
					} else {
						$split = "none";
					}
					my $id = "$tag:$split:$independent";
					if (not defined $datapoints{$id}){
						$datapoints{$id} = "$dependent";
					}else{
						$datapoints{$id} .= ":$dependent";
					}
				}
			}
		}
	
		my @newline; #note that this is a graph line, not a data line
		foreach my $id (keys %datapoints){
			my ($tag,$split,$independent) = split(/:/,$id);
			my @dependents = split(/:/,$datapoints{$id});
			my $element_count = $#dependents;
			for (my $iterate = 0; $iterate <= $element_count; $iterate++)
			{
				my $non_null = shift (@dependents);
				if ($non_null)
				{ push (@dependents, $non_null); }
			}

			if (@dependents){
				my $sum = 0; #sum
					($sum+=$_) for @dependents;
				my $avg = $sum/(@dependents); #average
					my $sqtotal = 0;
				foreach my $item (@dependents) {
					$sqtotal += ($avg-$item)**2;
				}
				my $stddev = ($sqtotal / @dependents)**0.5; #standard deviation
					my $errorhigh = $avg + $stddev;
				my $errorlow = $avg - $stddev;
				#print "$tag $split $independent $avg $errorhigh $errorlow \n";
				push (@newline, "$tag $split $independent $avg $errorhigh $errorlow");
			}
		}
		my $newline = join ("\n",@newline);
		push (@newdata, $newline);
	}

return @newdata;
}

#Compare data by column
sub compare_by_column($$$){
	my ($column,$entry1,$entry2) = @_;
	my @columns1 = split (/\s+/, $entry1);
	my @columns2 = split (/\s+/, $entry2);
	return ($columns1[$column]<=>$columns2[$column]);
}

#Compare data by two columns
sub compare_by_two_columns($$$$){
	my ($column1,$column2,$entry1,$entry2) = @_;
	my $comp = compare_by_column($column1,$entry1,$entry2);
	if ($comp == 0){
		$comp = compare_by_column($column2,$entry1,$entry2);
	}
	if ($comp == 1){
		print "$entry1 > $entry2\n";
	} else {
		print "$entry1 <= $entry2\n";
	}
	return $comp;
}

#split data into multiple lines over split field
#order the data by rising independent variable
sub organize_data{
	my $independent_index = pop;
	my $split_index = pop;
	my @sets = @_;

	my @sets2=();
	foreach my $set(@sets){
		my @entries = split (/\n/,$set);
		@entries = sort {compare_by_column($independent_index,$a,$b)} @entries;
		my %newset;
		foreach my $entry(@entries){
			my @values = split (/\s+/,$entry);
			my $split = $values[$split_index];
			if (defined $split){
				if (not defined $newset{$split}){
					$newset{$split} = $entry;
				}
				else {
					$newset{$split} .= "\n$entry";
				}
			}
		}
		foreach my $key(keys %newset){
			push(@sets2, $newset{$key});
		}
 
		my $i=0;
		do{
			@temp = split (/\s+/,$sets2[$i]);
			$i++;
		}while( not $temp[$split_index] and $i <= $#entries);

		if ($temp[$split_index] and $temp[$split_index] =~ /-?\n*.?\n*/){
			@sets2 = sort {compare_by_column($split_index,$b,$a)} @sets2;
		}
	}

	return @sets2;
}

#build gdi for graphing averaged data
sub build_gdi{
	my %globals = @_;
	my $title = $globals{'title'};
	my $xlabel = $globals{'xlabel'};
	my $ylabel = $globals{'ylabel'};
	my $tempgdifile = $globals{'tempgdifile'};
	my $tempdata = $globals{'tempdata'};
	my $output_file = $globals{'output'};
	my @sets = @{$globals{'sets'}};
	my $gnuplot_independent_index = INDEPENDENT+1;
	my $gnuplot_dependent_index = DEPENDENTAVG+1;
	my $gnuplot_dependent_error_max = DEPENDENTERROR_MAX+1;
	my $gnuplot_dependent_error_min = DEPENDENTERROR_MIN+1;

	my $plot;
	my $i;
	for ($i = 0; $i <= $#sets; ++$i){
		my $suffix;
		my @line = split(/ /,$sets[$i]);
		my $line_title;
		my $errorbars;
		my $style;

		if ($line[SPLIT] eq 'none'){
			$line_title = 'title "' . $line[TAG] . '"';
		} else {
			$line_title = 'title "' . $line[SPLIT] . '"';
		}

		if ($globals{'errorbars'}){
			$errorbars = ":$gnuplot_dependent_error_min:$gnuplot_dependent_error_max";
			$style = "errorlines";
		} else {
			$errorbars = "";
			$style = "linespoints";
		}

		if ($i == $#sets){
			$suffix = "\n";
		} else {
			$suffix = ",\\\n";
		}

		$plot .= "\"$tempdata\" index $i using $gnuplot_independent_index:$gnuplot_dependent_index$errorbars  with $style $line_title $suffix";
	}
	my $settitle = '';
	if (defined $title){
		$settitle = "set title \"$title\"";
	}

	my $output = '';
	my $pause = '';
	if ($output_file eq "screen"){
		$pause = 'pause -1 "hit return to continue..."' 
	} else {
		$output = 
"set terminal postscript enhanced dashed lw 1 \"Helvetica\" 14 
set output \"$output_file\"";
	}
my $gdi = "$settitle
set xlabel \"$xlabel\"
set ylabel \"$ylabel\"
$output
plot $plot
$pause";

	open (GNUPLOT,">$tempgdifile") or die "could not open temporary file $!";
		print GNUPLOT $gdi;
	close GNUPLOT;
}

#select by column
sub selectbycolumn{
	my ($column, $term, $data) = @_;
	my $regex = '^';
	my $i;
	for ($i = 0; $i < $column; ++$i){
		$regex .= '[^\s]+\s+';
	}
	$regex .= "$term";
	
	my @data = split (/\n/,$data);
	my @data2;

	foreach my $line(@data){
		if ($line =~ m/$regex\s/){
			push (@data2,$line);
		}
	}
	$data2 = join("\n",@data2);
	return $data2;	
}

return 1;
