/* Serial and parallel binary tree traversal
 * Part of Parallelization: Binary Tree Traversal, Blue Waters Undergraduate 
 *  Petascale Module
 *
 * Aaron Weeden and Patrick Royal, Shodor Education Foundation, Inc.
 * April 2012
 */

/* Includes */
#include <getopt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tree.h"

#ifdef PARALLEL
#include <mpi.h>
#endif

/* Defined constants */
#define PRE 0
#define IN 1
#define BREADTH 3
#define NO_ASSIGNMENT_YET -1
#define ASSIGNMENT_FOUND -2

/* Functions Declarations */
/* Serial sorts */
void depthfirstsort(struct node * root, int * labels, int * num_labels,
        int sort_type);
void breadthfirstsort(struct node * root, int * labels, int * num_labels,
        int height);
#ifdef PARALLEL
/* Parallel sorts */
void depthfirstparallelsort(struct node * root, int * labels, int * num_labels, 
        int current_height, int rank, int size, int * responsible_rank,
        int * assignment, int * other_labels, int sort_type);
void breadthfirstparallelsort(struct node * root, int * labels,
        int * num_labels, int height, int rank, int size, 
        int * other_labels);
#endif

/* Main function - execution starts here */

int main(int argc, char ** argv) {
    /* Declare variables */
    int ch = 0;
    int usage = 0;
    int sort_type = PRE;
    int height = 3;
    int num_labels = 0;
    int i = 0;
    static struct option longopts[] = {
        { "order",  required_argument, NULL, 'o' },
        { "height", required_argument, NULL, 'h' }
    };
    struct node * root;
    int * labels;

#ifdef PARALLEL
    int rank = 0;
    int size = 1;
    int responsible_rank = 0;
    int assignment = NO_ASSIGNMENT_YET;
    int * other_labels;

    /* Initialize the MPI environment and determine the rank and size */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* We have to have at least 2 processes, exit if we don't */
    if(size < 2) {
        if(rank == 0) fprintf(stderr, "ERROR: Must have at least 2 processes.\n");
        MPI_Finalize();
        exit(-1);
    }

    /* Assume we can only have a size that is a power of 2, plus 1 for Rank 0, 
     *  exit if we don't */
    if(log(size - 1)/log(2) != (int)(log(size - 1)/log(2))) {
        if(rank == 0) fprintf(stderr,
                "ERROR: Number of processes must be 1 + a power of 2 (2, 3, 5, 9, 17, etc).\n");
        MPI_Finalize();
        exit(-1);
    }
#endif

    /* Parse the command line arguments -- See 'man getopt_long' for how 
     *  argument parsing works */
    usage = 0;
    while((ch = getopt_long(argc, argv, "o:h:", longopts, NULL)) != -1) {
        switch(ch) {
            case 'o':
                if(strcmp(optarg, "pre") == 0) {
                    sort_type = PRE;
                } else if(strcmp(optarg, "in") == 0) {
                    sort_type = IN;
                } else if(strcmp(optarg, "breadth") == 0) {
                    sort_type = BREADTH;
                } else {
                    usage = 1;
                }
                break;
            case 'h':
                if((height = atoi(optarg)) < 1) {
                    fprintf(stderr,
                            "Must have a non-negative height\n");
                    usage = 1;
                }
                break;
            default:
                usage = 1;
        }
    }
    /* If there was an error in argument parsing, print a usage message and 
     *  exit */
    if(usage == 1) {
#ifdef PARALLEL
        /* Only have Rank 0 print */
        if(rank == 0)
#endif
            fprintf(stderr, "Usage: %s [-o|--order pre|in|breadth] [-h|--height height]\n",
                    argv[0]);
#ifdef PARALLEL
        /* Finalize the MPI environment */
        MPI_Finalize();
#endif
        exit(-1);
    }

#ifdef PARALLEL
    /* Display an error and exit if more processes exist than work to be done */
    if((size-1) > (int)pow(2, height)) {
        if(rank == 0) fprintf(stderr,
                "ERROR: Number of processes minus 1 must be less than or equal to 2^height.\n");
        MPI_Finalize();
        exit(-1);
    }
#endif

    /* Allocate memory for the list of labels */
    labels = malloc(((int)pow(2, height)-1) * sizeof(int));
#ifdef PARALLEL
    other_labels = malloc(((int)pow(2, height)-1) * sizeof(int));
#endif

    /* Build the tree */
    buildtree(&root, (int)pow(2, height-1), 0, height);

    /* Sort the tree using the parallel sort if we are running parallel and the
     *  serial sort otherwise */
    if(sort_type != BREADTH) {
#ifdef PARALLEL
        depthfirstparallelsort(root, labels, &num_labels, 0, rank, size,
                &responsible_rank, &assignment, other_labels, sort_type);
#else
        depthfirstsort(root, labels, &num_labels, sort_type);
#endif
    } else {
#ifdef PARALLEL
        breadthfirstparallelsort(root, labels, &num_labels, height, rank, size,
                other_labels);
#else
        breadthfirstsort(root, labels, &num_labels, height);
#endif
    }

#ifdef PARALLEL
    /* Only print if we are Rank 0 */
    if(rank == 0) {
#endif
        /* Print the sorted list of labels */
        for(i = 0; i < num_labels; i++) {
            printf("%d ", labels[i]);
        }
        printf("\n");
#ifdef PARALLEL
    }
#endif

    /* Delete the tree and labels list */
    freetree(&root);
    free(labels);
#ifdef PARALLEL
    free(other_labels);
#endif


#ifdef PARALLEL
    /* Clean up the MPI environment */
    MPI_Finalize();
#endif

    /* Exit the main function with success */
    return 0;
}

/* Depth-first serial sort */
void depthfirstsort(struct node * root, int * labels, int * num_labels,
        int sort_type) {
    /* If the current root is not empty */
    if(root != NULL) {
        /* If we are doing pre-order sort */
        if(sort_type == PRE) {
            /* Add the root's label to our list of labels */
            labels[(*num_labels)++] = root->label;
        }
        /* Sort the left subtree */
        depthfirstsort(root->left, labels, num_labels, sort_type);
        /* If we are doing in-order sort */
        if(sort_type == IN) {
            /* Add the root's label to our list of labels */
            labels[(*num_labels)++] = root->label;
        }
        /* Sort the right subtree */
        depthfirstsort(root->right, labels, num_labels, sort_type);
    }
    return;
}

/* Breadth-first serial sort */
void breadthfirstsort(struct node * root, int * labels, int * num_labels,
        int height) {
    /* Declare variables */
    struct stack activenodes;
    struct stack nextnodes;

    /* Initialize stacks and allocate their memory */
    activenodes.size = 0;
    nextnodes.size = 0;
    activenodes.nodes = malloc(((int)pow(2, height) - 1) * sizeof(struct node));
    nextnodes.nodes = malloc(((int)pow(2, height) - 1) * sizeof(struct node));

    /* Add the root to the active nodes stack */
    push(&activenodes, root);
    /* Run until the top of the active nodes stack is empty */
    while(peek(activenodes) != NULL) {
        /* Run until the top of the active nodes stack is empty */
        while(peek(activenodes) != NULL) {
            /* Assign the top of the active stack to the current node and pop 
             *  the stack */
            struct node * current = pop(&activenodes);
            /* Add the current node's label to the list of labels */
            labels[(*num_labels)++] = current->label;
            /* Add the current node's children to the next nodes
             *  stack */
            push(&nextnodes, current->left);
            push(&nextnodes, current->right);
        }
        /* Run until the next nodes stack is empty */
        while(peek(nextnodes) != NULL) {
            /* Move the top of the next nodes stack to the top of
             *  the active nodes stack */
            push(&activenodes, pop(&nextnodes));
        }
    }

    /* Free the stacks */
    free(nextnodes.nodes);
    free(activenodes.nodes);

    return;
}

/* Depth-first parallel sort */
#ifdef PARALLEL
void depthfirstparallelsort(struct node * root, int * labels, int * num_labels, 
        int current_height, int rank, int size, int * responsible_rank,
        int * assignment, int * other_labels, int sort_type) {
    /* Declare variables */
    int other_num_labels = 0;
    int i = 0;

    /* If the current root is not empty */
    if(root != NULL) {
        /* If we are the Rank 0 process */
        if(rank == 0) {
            /* If we are within the parent part of the tree */
            if(size-1 > (int)pow(2, current_height)) {
                /* If we are doing pre-order sort */
                if(sort_type == PRE) {
                    /* Add the current root's label to our list of labels */
                    labels[(*num_labels)++] = root->label;
                }
                /* Continue sorting in the left sub-tree */
                depthfirstparallelsort(root->left, labels, num_labels,
                        current_height + 1, rank, size, responsible_rank, 
                        assignment, other_labels, sort_type);
                /* If we are doing in-order sort */
                if(sort_type == IN) {
                    /* Add the current root's label to our list of labels */
                    labels[(*num_labels)++] = root->label;
                }
                /* Continute sorting in the right sub-tree */
                depthfirstparallelsort(root->right, labels, num_labels,
                        current_height + 1, rank, size, responsible_rank, 
                        assignment, other_labels, sort_type);
                /* If we are not within the parent part of the tree */
            } else {
                /* Figure out which process is responsible for the current
                 *  root */
                (*responsible_rank)++;
                /* Send the current root's label as an assignment to that 
                 *  process */
                MPI_Send(&(root->label), 1, MPI_INT, *responsible_rank, 0,
                        MPI_COMM_WORLD);
                /* Receive the label count from the worker process */
                MPI_Recv(&other_num_labels, 1, MPI_INT, *responsible_rank, 0, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                /* Receive the label list from the worker process */
                MPI_Recv(other_labels, other_num_labels, MPI_INT,
                        *responsible_rank, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);
                /* Add the labels to Rank 0's label list */
                for(i = 0; i < other_num_labels; i++) {
                    labels[(*num_labels)++] = other_labels[i];
                }
            }
            /* If we are not the Rank 0 process */
        } else {
            /* If we haven't received our assignment yet */
            if(*assignment == NO_ASSIGNMENT_YET) {
                /* Receive our assignment from Rank 0 */
                MPI_Recv(assignment, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);
                /* Look for the assignment in the left subtree */
                depthfirstparallelsort(root->left, labels, num_labels,
                        current_height+1, rank, size, responsible_rank,
                        assignment, other_labels, sort_type);
                /* If we didn't find the assignment in the left subtree */
                if(*assignment != ASSIGNMENT_FOUND) {
                    /* Look for the assignment in the right subtree */
                    depthfirstparallelsort(root->right, labels, num_labels,
                            current_height+1, rank, size, responsible_rank,
                            assignment, other_labels, sort_type);
                }
                /* If we have received our assignment */
            } else {
                /* If the current root is not our assignment */
                if(root->label != *assignment) {
                    /* Keep searching in the left subtree */
                    depthfirstparallelsort(root->left, labels, num_labels,
                            current_height+1, rank, size, responsible_rank, 
                            assignment, other_labels, sort_type);
                    /* If we didn't find the assignment in the left subtree */
                    if(*assignment != ASSIGNMENT_FOUND) {
                        /* Look for the assignment in the right subtree */
                        depthfirstparallelsort(root->right, labels, num_labels,
                                current_height+1, rank, size, responsible_rank, 
                                assignment, other_labels, sort_type);
                    }
                    /* If the current root is our assignment */
                } else {
                    /* Mark the assignment as found */
                    *assignment = ASSIGNMENT_FOUND;
                    /* If we are doing pre-order sort */
                    if(sort_type == PRE) {
                        /* Add the root's label to our list of labels */
                        labels[(*num_labels)++] = root->label;
                    }
                    /* Sort the left subtree using the serial sort */
                    depthfirstsort(root->left, labels, num_labels, sort_type);
                    /* If we are doing in-order sort */
                    if(sort_type == IN) {
                        /* Add the root's label to our list of labels */
                        labels[(*num_labels)++] = root->label;
                    }
                    /* Sort the right subtree using the serial sort */
                    depthfirstsort(root->right, labels, num_labels, sort_type);
                    /* Send the label count back to Rank 0 */
                    MPI_Send(num_labels, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                    /* Send our sorted list of labels back to Rank 0 */
                    MPI_Send(labels, *num_labels, MPI_INT, 0, 0,
                            MPI_COMM_WORLD);
                }
            }
        }
    }
    return;
}

/* Breadth-first parallel sort */
void breadthfirstparallelsort(struct node * root, int * labels,
        int * num_labels, int height, int rank, int size, 
        int * other_labels) {
    /* Declare variables */
    struct stack activenodes;
    struct stack nextnodes;
    int current_height = 0;
    int responsible_rank = size - 1;
    int assignment = 0;
    int i = 0;
    int j = 0;
    int k = 0;

    /* Initialize stacks and allocate their memory */
    activenodes.size = 0;
    nextnodes.size = 0;
    activenodes.nodes = malloc(((int)pow(2, height) - 1) * sizeof(struct node));
    nextnodes.nodes = malloc(((int)pow(2, height) - 1) * sizeof(struct node));

    /* If we are rank 0 */
    if(rank == 0) {
        /* If the root is within the parent of the tree */
        if(size-1 > (int)pow(2, current_height)) {
            /* Add the root to the active nodes stack */
            push(&activenodes, root);
            /* Run until the top of the active nodes stack is empty */
            while(peek(activenodes) != NULL) {
                /* Run until the top of the active nodes stack is empty */
                while(peek(activenodes) != NULL) {
                    /* Assign the top of the active stack to the current node 
                     *  and pop the stack */
                    struct node * current = pop(&activenodes);
                    /* Add the current node's label to the list of labels */
                    labels[(*num_labels)++] = current->label;
                    /* Add the current node's children to the next nodes
                     *  stack */
                    push(&nextnodes, current->left);
                    push(&nextnodes, current->right);
                    /* Move on to the next row of the tree */
                    current_height++;
                }
                /* Run until the next nodes stack is empty */
                while(peek(nextnodes) != NULL) {
                    /* If the next row of the tree is within the parent of the
                     *  tree */
                    if(size-1 > (int)pow(2, current_height)) {
                        /* Move the top of the next nodes stack to the top of
                         *  the active nodes stack */
                        push(&activenodes, pop(&nextnodes));
                    /* If the next row of the tree is not within the parent of
                     *  the tree */
                    } else {
                        /* Assign the top of the next nodes stack to the next 
                         *  process */
                        MPI_Send(&(pop(&nextnodes)->label), 1, MPI_INT,
                                responsible_rank, 0, MPI_COMM_WORLD);
                        responsible_rank--;
                    }
                }
            }
        }
        /* Receive each row */
        for(i = 0; i < height - (current_height - 1); i++) {
            for(j = 1; j < size; j++) {
                /* Receive each worker's label list */
                MPI_Recv(other_labels, (int)pow(2, i), MPI_INT, j, 0, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                /* Add the label list to the root's label list */
                for(k = 0; k < (int)pow(2, i); k++) {
                    labels[(*num_labels)++] = other_labels[k];
                }
            }
        }
    /* If we are not Rank 0 */
    } else {
        /* Receive assignment from Rank 0 */
        MPI_Recv(&assignment, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        /* Add the root to the active nodes stack */
        push(&activenodes, root);
        /* Run until the top of the active nodes stack is empty */
        while(peek(activenodes) != NULL) {
            /* Run until the top of the active nodes stack is empty */
            while(peek(activenodes) != NULL) {
                /* Assign the top of the stack to the current node and pop 
                 *  the stack */
                struct node * current = pop(&activenodes);
                /* If we just found the assignment */
                if(current->label == assignment) {
                    /* Pop everything from both stacks */
                    while(peek(activenodes) != NULL) {
                        pop(&activenodes);
                    }
                    while(peek(nextnodes) != NULL) {
                        pop(&nextnodes);
                    }
                    /* Set the assignment as found */
                    assignment = ASSIGNMENT_FOUND;
                }
                /* If we have ever found the assignment */
                if(assignment == ASSIGNMENT_FOUND) {
                    /* Add the current node's label to the list of labels */
                    labels[(*num_labels)++] = current->label;
                }
                /* Add the current node's children to the next nodes
                 *  stack */
                push(&nextnodes, current->left);
                push(&nextnodes, current->right);
            }
            /* If we have found the assignment */
            if(assignment == ASSIGNMENT_FOUND) {
                /* Add 1 to the height counter */
                current_height++;
                /* Send the newly-added nodes to Rank 0 */
                MPI_Send(labels, (int)pow(2, current_height-1), MPI_INT, 0, 0,
                        MPI_COMM_WORLD);
                /* Reset the label list */
                (*num_labels) = 0;
            }
            /* Run until the next nodes stack is empty */
            while(peek(nextnodes) != NULL) {
                /* Move the top of the next nodes stack to the top of
                 *  the active nodes stack */
                push(&activenodes, pop(&nextnodes));
            }
        }
    }

    /* Free the stacks */
    free(nextnodes.nodes);
    free(activenodes.nodes);

    return;
}
#endif
