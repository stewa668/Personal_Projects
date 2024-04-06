/******************************************
PetaKit header 1.0
Copyright 2010, Samuel Leeman-Munk.

A header for making programs give petakit-format output
******************************************/

#ifdef __cplusplus
#include <iostream>
#else
#include <stdio.h>
#endif

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/time.h>
#include <sys/resource.h>

static const char* BEGIN_BUFFER = "!~~~#**BEGIN RESULTS**#~~~!\n";
static const char* END_BUFFER = "!~~~#**END RESULTS**#~~~!\n";
static const int MAX_HOSTNAME_LENGTH = 1023;

static double petakit_start_time;

double getCPUTime();
//preconditions: none
//postconditions: returns CPU time as a double

double getTime();
//preconditions: none
//postconditions: returns the current time of day in seconds

void startTimer();
//preconditions: none
//postconditions: sets the stat_pl_start_time variable to allow a proper wall_time for printStats

void printStats(const char* program_name, int threads,  const char* architecture, long long int problem_size, const char* version, double cputime, short custom_argc, ...);
//preconditions: required output variables, optional output variables
// startTimer() must have been called at the beginning of the program
//postconditions: prints to screen info readable by stat.pl
//extra output is added as char* NAME  <type> architecture
