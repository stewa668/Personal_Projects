/* Parallelization:  Sieve of Eratosthenes
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * January 2012
 *
 * OpenMP code
 *  -- to run, use ./sieve.serial -n N, where N is the value under which to find
 *     primes.
 *  -- see attached module document for discussion of the code and its algorithm
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    /* Declare variables */
    int N = 16; /* The positive integer under which we are finding primes */
    int sqrtN = 4; /* The square root of N, which is stored in a variable to 
                      avoid making excessive calls to sqrt(N) */
    int c = 2; /* Used to check the next number to be circled */
    int m = 3; /* Used to check the next number to be marked */
    int *list; /* The list of numbers -- if list[x] equals 1, then x is marked. 
                  If list[x] equals 0, then x is unmarked. */
    char next_option = ' '; /* Used for parsing command line arguments */
   
    /* Parse command line arguments -- enter 'man 3 getopt' on a shell to see
       how this works */
    while((next_option = getopt(argc, argv, "n:")) != -1) {
        switch(next_option) {
            case 'n':
                N = atoi(optarg);
                break;
            case '?':
            default:
                fprintf(stderr, "Usage: %s [-n N]\n", argv[0]);
                exit(-1);
        }
    }

    /* Calculate sqrtN */
    sqrtN = (int)sqrt(N);

    /* Allocate memory for list */
    list = (int*)malloc(N * sizeof(int));

    /* Exit if malloc failed */
    if(list == NULL) {
        fprintf(stderr, "Sorry, there was an internal error. Please run again.\n");
        exit(-1);
    }

    /* Run through each number in the list */
#pragma omp parallel for
    for(c = 2; c <= N-1; c++) {

        /* Set each number as unmarked */
        list[c] = 0;
    }

    /* Run through each number in the list up through the square root of N */
    for(c = 2; c <= sqrtN; c++) {

        /* If the number is unmarked */
        if(list[c] == 0) {

            /* Run through each number bigger than c */
#pragma omp parallel for
            for(m = c+1; m <= N-1; m++) {

                /* If m is a multiple of c */
                if(m%c == 0) {

                    /* Mark m */
                    list[m] = 1;
                }
            }
        }
    }

    /* Run through each number in the list */
    for(c = 2; c <= N-1; c++) {

        /* If the number is unmarked */
        if(list[c] == 0) {

            /* The number is prime, print it */
            printf("%d ", c);

        }
    }
    printf("\n");

    /* Deallocate memory for list */
    free(list);

    return 0;
}
