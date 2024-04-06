OBJS = ComputeBonded.o ComputeNonbonded.o LJTable.o Molecule.o Parameters.o PDB.o PDBData.o strlib.o Integrator.o
BINS = mindy

default:
	$(MAKE) all \
	"ARCH = g++" \
	"CXX = g++" \
	"CCFLAGS = -O3 -ffast-math" \
	"LIBS = -lm"

sun-CC:
	$(MAKE) all \
	"ARCH = sun-CC" \
	"CXX = CC" \
 	"CCFLAGS = -fast -xO5 -native -xarch=v8plusa -fsimple=2" \
	"LIBS = -lm"

alpha-cxx:
	$(MAKE) all \
	"ARCH = alpha-cxx" \
	"CXX = cxx" \
 	"CCFLAGS = -O4 -arch host -tune host -nocleanup" \
	"LIBS = -lm"
	

linux-g++:
	$(MAKE) all \
	"ARCH = linux-g++" \
	"CXX = g++" \
	"CCFLAGS = -O3 -march=pentiumpro -ffast-math" \
	"LIBS = -lm"

linux-KCC:
	$(MAKE) all \
	"ARCH = linux-KCC" \
	"CXX = KCC" \
	"CCFLAGS = -O3 +K3 -Bstatic --backend -march=pentiumpro --backend -ffast-math" \
	"LIBS = -lm"
	 
all: $(BINS) $(OBJS)

.C.o :
	$(CXX) $(CCFLAGS) $(INCS) -c $*.C

mindy: $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCS) $(OBJS) -o mindy_$(ARCH) $(LIBS)

clean :
	rm -f $(OBJS) $(BINS)_*