if [ ! -f Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz ]
then
	wget ftp://ftp.ensembl.org/pub/release-95/fasta/caenorhabditis_elegans/dna/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa.gz
else
	echo "file exists"
fi



if [ ! -f Example_Data/DataSet01_Celegans_Paired200Id200Pexp100Cov10E1N0GenomicControl_1.fq.gz ]
then
	cp /mnt/scratch/colbrydi/BWA_Example_Data/DataSet01_Celegans_Paired200Id200Pexp100Cov10E1N0GenomicControl_1.fq.gz .
else
	echo "file exists"
fi

if [ ! -f  DataSet01_Celegans_Paired200Id200Pexp100Cov10E1N0GenomicControl_2.fq.gz ]
then
	cp /mnt/scratch/colbrydi/BWA_Example_Data/DataSet01_Celegans_Paired200Id200Pexp100Cov10E1N0GenomicControl_2.fq.gz .
else
	echo "file exists"
fi
