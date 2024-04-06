#!/bin/bash --login
########## Define Resources Needed with SBATCH Lines ##########
 
#SBATCH --time=01:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=1           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem=1G                    # memory required per node - amount of memory (in bytes)
#SBATCH --job-name omp-hello-job       # you can give your job a name for easier identification (same as -J)
#SBATCH -o %x.out
 
########## Command Lines to Run ##########

for i in steps.txt
  /usr/bin/time -f "%e" ./0212int < $i  | cut -d " " -f 2
