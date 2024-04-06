# openmp test
## ICER HPC EXAMPLE

This is a small and simple test of using OpenMP on systems with GNU compiler.  The gcc compiler and
openmp libraries are required.  It was developed on Centos 6 but also runs on CentOS 7

The C code by John Burkardt was taken from https://people.sc.fsu.edu/~jburkardt/c_src/prime_openmp/prime_openmp.html 
who has a fantastic collection of example parallel codes in several languages. 

# Running the code

You can compile and run the code with these commands 

    module load gcc  # it's already loaded by default but this ensures do. 
    # compile into an executable with gcc
    gcc -o prime_openmp -fopenmp prime_openmp.c   
    # run it
    ./prime_openmp

This will work with any of the GCC or Intel compilers we have available.    
Note that using parallel threads via the openmp library requires the -fopenmp parameter to gcc.  

# experiments

The number of threads the program uses is controlled by the environment variable.  

    # is it set?
    echo $OPENMP_NUM_THREADS

    # set the variable to 1, run your program and see the time 
    export OPENMP_NUM_THREADS=1
    ./prime_openmp

    # set it to some number > 1 
    echo $OPENMP_NUM_THREADS=4
    ./prime_openmp
   

## full test

See the  script  thread_test.sh to compile and run a test of the program.  You can run it as

    bash thread_test.sh

## exercises : running with a  batch script

1) Could you write a SLURM sbatch script that runs the program once?   Note this program runs on one node, in 1 task, 
but with several thrads (cores).  Submit that to the cluster and see if you can get the same output.     

2) This program, in batch, should use the same number of cpus/threads set with the cores/cpus you set in your sbatch.  
Use the OPENMP variable above to make sure your program users the same number as in your SBATCH options

3) SLURM sets an environment variable that is the number of CPUS per Task that you asked for.   
Could you use this to set the number of threads this program will use, so that it 



