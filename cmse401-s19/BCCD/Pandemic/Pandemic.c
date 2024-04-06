/* Parallelization: Infectious Disease
 * By Aaron Weeden, Shodor Education Foundation, Inc.
 * November 2011
 *
 * Parallel code -- MPI for distributed memory (processes), OpenMP for shared
 *  memory (threads). "Hybrid" uses both (in which case each MPI process can 
 *  spawn OpenMP threads).
 *
 * Parts corresponding to the module's algorithm are indicated by comments that
 *  begin with ALG I:, ALG I.A:, ALG I.A.1:, etc.
 *
 * Note on naming scheme:  Variables that begin with "our" are private to
 *  processes and shared by threads ("our" is from the perspective of the
 *  threads).  Variables that begin with "my" are private to threads (again,
 *  "my" from the perspective of threads). */

#include <assert.h> /* for assert */
#include <stdio.h> /* printf */
#include <stdlib.h> /* malloc, free, and various others */
#include <time.h> /* time is used to seed the random number generator */
#include <unistd.h> /* random, getopt, some others */
#include <X11/Xlib.h> /* X display */

#ifdef _MPI
#include <mpi.h> /* MPI_Allgather, MPI_Init, MPI_Comm_rank, MPI_Comm_size */
#endif

/* States of people -- all people are one of these 4 states */
/* These are const char because they are displayed as ASCII if TEXT_DISPLAY 
 *  is enabled */
const char INFECTED = 'X';
const char IMMUNE = 'I';
const char SUSCEPTIBLE = 'o';
const char DEAD = ' ';

#ifdef X_DISPLAY
const int PIXEL_WIDTH_PER_PERSON = 10;
const int PIXEL_HEIGHT_PER_PERSON = 10;
#endif

/* PROGRAM EXECUTION BEGINS HERE */
int main(int argc, char** argv)
{
    /** Declare variables **/
    /* People */
    int total_number_of_people = 50;
    int total_num_initially_infected = 1;
    int total_num_infected = 1;
    int our_number_of_people = 50;
    int our_person1 = 0;
    int our_current_infected_person = 0;
    int our_num_initially_infected = 1;
    int our_num_infected = 0;
    int our_current_location_x = 0;
    int our_current_location_y = 0;
    int our_num_susceptible = 0;
    int our_num_immune = 0;
    int our_num_dead = 0;
    int my_current_person_id = 0;
    int my_num_infected_nearby = 0;
    int my_person2 = 0;

    /* Environment */
    int environment_width = 30;
    int environment_height = 30;

    /* Disease */
    int infection_radius = 3;
    int duration_of_disease = 50;
    int contagiousness_factor = 30;
    int deadliness_factor = 30;
#ifdef SHOW_RESULTS
    double our_num_infections = 0.0;
    double our_num_infection_attempts = 0.0;
    double our_num_deaths = 0.0;
    double our_num_recovery_attempts = 0.0;
#endif

    /* Time */
    int total_number_of_days = 250;
    int our_current_day = 0;
    int microseconds_per_day = 100000;
    
    /* Movement */
    int my_x_move_direction = 0; 
    int my_y_move_direction = 0;

    /* Distributed Memory Information */
    int total_number_of_processes = 1;
    int our_rank = 0;
#ifdef _MPI
    int current_rank = 0;
    int current_displ = 0;
#endif

    /* getopt */
    int c = 0;

    /* Integer arrays, a.k.a. integer pointers */
    int *x_locations;
    int *y_locations;
    int *our_x_locations;
    int *our_y_locations;
    int *our_infected_x_locations;
    int *our_infected_y_locations;
    int *their_infected_x_locations;
    int *their_infected_y_locations;
    int *our_num_days_infected;
    int *recvcounts;
    int *displs;

    /* Character arrays, a.k.a. character pointers */
    char *states;
    char *our_states;

#ifdef TEXT_DISPLAY
    /* Array of character arrays, a.k.a. array of character pointers, for text
     *  display */
    char **environment;
#endif

#ifdef X_DISPLAY
    /* Declare X-related variables */
    Display *display;
    Window window;
    int screen;
    Atom delete_window;
    GC gc;
    XColor infected_color;
    XColor immune_color;
    XColor susceptible_color;
    XColor dead_color;
    Colormap colormap;
    char red[] = "#FF0000";
    char green[] = "#00FF00";
    char black[] = "#000000";
    char white[] = "#FFFFFF";
#endif

#ifdef _MPI
    /* Each process initializes the distributed memory environment */
    MPI_Init(&argc, &argv);
#endif

    /* ALG I: Each process determines its rank and the total number of processes     */
#ifdef _MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &our_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_number_of_processes);
#else
    our_rank = 0;
    total_number_of_processes = 1;
#endif

    /* ALG II: Each process is given the parameters of the simulation */
    /* Get command line options -- this follows the idiom presented in the
     *  getopt man page (enter 'man 3 getopt' on the shell for more) */
    while((c = getopt(argc, argv, "n:i:w:h:t:T:c:d:D:m:")) != -1)
    {
        switch(c)
        {
            case 'n':
                total_number_of_people = atoi(optarg);
                break;
            case 'i':
                total_num_initially_infected = atoi(optarg);
                break;
            case 'w':
                environment_width = atoi(optarg);
                break;
            case 'h':
                environment_height = atoi(optarg);
                break;
            case 't':
                total_number_of_days = atoi(optarg);
                break;
            case 'T':
                duration_of_disease = atoi(optarg);
                break;
            case 'c':
                contagiousness_factor = atoi(optarg);
                break;
            case 'd':
                infection_radius = atoi(optarg);
                break;
            case 'D':
                deadliness_factor = atoi(optarg);
                break;
            case 'm':
                microseconds_per_day = atoi(optarg);
                break;
                /* If the user entered "-?" or an unrecognized option, we need 
                 *  to print a usage message before exiting. */
            case '?':
            default:
                fprintf(stderr, "Usage: ");
#ifdef _MPI
                fprintf(stderr, "mpirun -np total_number_of_processes ");
#endif
                fprintf(stderr, "%s [-n total_number_of_people][-i total_num_initially_infected][-w environment_width][-h environment_height][-t total_number_of_days][-T duration_of_disease][-c contagiousness_factor][-d infection_radius][-D deadliness_factor][-m microseconds_per_day]\n", argv[0]);
                exit(-1);
        }
    }
    argc -= optind;
    argv += optind;

    /* ALG III: Each process makes sure that the total number of initially 
     *  infected people is less than the total number of people */
    if(total_num_initially_infected > total_number_of_people)
    {
        fprintf(stderr, "ERROR: initial number of infected (%d) must be less than total number of people (%d)\n", total_num_initially_infected, 
                total_number_of_people);
        exit(-1);
    }

    /* ALG IV: Each process determines the number of people for which it is 
     *  responsible */
    our_number_of_people = total_number_of_people / total_number_of_processes;

    /* ALG V: The last process is responsible for the remainder */
    if(our_rank == total_number_of_processes - 1)
    {
        our_number_of_people += total_number_of_people % total_number_of_processes;
    }

    /* ALG VI: Each process determines the number of initially infected people 
     *  for which it is responsible */
    our_num_initially_infected = total_num_initially_infected 
        / total_number_of_processes;

    /* ALG VII: The last process is responsible for the remainder */
    if(our_rank == total_number_of_processes - 1)
    {
        our_num_initially_infected += total_num_initially_infected 
            % total_number_of_processes;
    }
    
    /* Allocate the arrays */
    x_locations = (int*)malloc(total_number_of_people * sizeof(int));
    y_locations = (int*)malloc(total_number_of_people * sizeof(int));
    our_x_locations = (int*)malloc(our_number_of_people * sizeof(int));
    our_y_locations = (int*)malloc(our_number_of_people * sizeof(int));
    our_infected_x_locations = (int*)malloc(our_number_of_people * sizeof(int));
    our_infected_y_locations = (int*)malloc(our_number_of_people * sizeof(int));
    their_infected_x_locations = (int*)malloc(total_number_of_people 
            * sizeof(int));
    their_infected_y_locations = (int*)malloc(total_number_of_people 
            * sizeof(int));
    our_num_days_infected = (int*)malloc(our_number_of_people * sizeof(int));
    recvcounts = (int*)malloc(total_number_of_processes * sizeof(int));
    displs = (int*)malloc(total_number_of_processes * sizeof(int));
    states = (char*)malloc(total_number_of_people * sizeof(char));
    our_states = (char*)malloc(our_number_of_people * sizeof(char));

#ifdef TEXT_DISPLAY
    environment = (char**)malloc(environment_width * environment_height
            * sizeof(char*));
    for(our_current_location_x = 0;
            our_current_location_x <= environment_width - 1;
            our_current_location_x++)
    {
        environment[our_current_location_x] = (char*)malloc(environment_height
                * sizeof(char));
    }
#endif

    /* ALG VIII: Each process seeds the random number generator based on the
     *  current time */
    srandom(time(NULL));

    /* ALG IX: Each process spawns threads to set the states of the initially 
     *  infected people and set the count of its infected people */
#ifdef OMP 
#pragma omp parallel for private(my_current_person_id) \
    reduction(+:our_num_infected)
#endif
    for(my_current_person_id = 0; my_current_person_id 
            <= our_num_initially_infected - 1; my_current_person_id++)
    {
        our_states[my_current_person_id] = INFECTED;
        our_num_infected++;
    }

    /* ALG X: Each process spawns threads to set the states of the rest of its 
     *  people and set the count of its susceptible people */
#ifdef OMP 
#pragma omp parallel for private(my_current_person_id) \
    reduction(+:our_num_susceptible)
#endif
    for(my_current_person_id = our_num_initially_infected; 
            my_current_person_id <= our_number_of_people - 1; 
            my_current_person_id++)
    {
        our_states[my_current_person_id] = SUSCEPTIBLE;
        our_num_susceptible++;
    }

    /* ALG XI: Each process spawns threads to set random x and y locations for 
     *  each of its people */
#ifdef OMP 
#pragma omp parallel for private(my_current_person_id)
#endif
    for(my_current_person_id = 0;
            my_current_person_id <= our_number_of_people - 1; 
            my_current_person_id++)
    {
        our_x_locations[my_current_person_id] = random() % environment_width;
        our_y_locations[my_current_person_id] = random() % environment_height;
    }

    /* ALG XII: Each process spawns threads to initialize the number of days 
     *  infected of each of its people to 0 */
#ifdef OMP 
#pragma omp parallel for private(my_current_person_id)
#endif
    for(my_current_person_id = 0;
            my_current_person_id <= our_number_of_people - 1;
            my_current_person_id++)
    {
        our_num_days_infected[my_current_person_id] = 0;
    }

    /* ALG XIII: Rank 0 initializes the graphics display */
#ifdef X_DISPLAY
    if(our_rank == 0)
    {
        /* Initialize the X Windows Environment
         * This all comes from 
         *   http://en.wikibooks.org/wiki/X_Window_Programming/XLib
         *   http://tronche.com/gui/x/xlib-tutorial
         *   http://user.xmission.com/~georgeps/documentation/tutorials/
         *      Xlib_Beginner.html
         */

        /* Open a connection to the X server */
        display = XOpenDisplay(NULL);
        if(display == NULL)
        {
            fprintf(stderr, "Error: could not open X display\n");
        }
        screen = DefaultScreen(display);
        window = XCreateSimpleWindow(display, RootWindow(display, screen),
                0, 0, environment_width * PIXEL_WIDTH_PER_PERSON, 
                environment_height * PIXEL_HEIGHT_PER_PERSON, 1,
                BlackPixel(display, screen), WhitePixel(display, screen));
        delete_window = XInternAtom(display, "WM_DELETE_WINDOW", 0);
        XSetWMProtocols(display, window, &delete_window, 1);
        XSelectInput(display, window, ExposureMask | KeyPressMask);
        XMapWindow(display, window);
        colormap = DefaultColormap(display, 0);
        gc = XCreateGC(display, window, 0, 0);
        XParseColor(display, colormap, red, &infected_color);
        XParseColor(display, colormap, green, &immune_color);
        XParseColor(display, colormap, white, &dead_color);
        XParseColor(display, colormap, black, &susceptible_color);
        XAllocColor(display, colormap, &infected_color);
        XAllocColor(display, colormap, &immune_color);
        XAllocColor(display, colormap, &susceptible_color);
        XAllocColor(display, colormap, &dead_color);
    }
#endif

    /* ALG XIV: Each process starts a loop to run the simulation for the
     *  specified number of days */
    for(our_current_day = 0; our_current_day <= total_number_of_days - 1; 
            our_current_day++)
    {
        /* ALG XIV.A: Each process determines its infected x locations and 
         *  infected y locations */
        our_current_infected_person = 0;
        for(our_person1 = 0; our_person1 <= our_number_of_people - 1;
                our_person1++)
        {
            if(our_states[our_person1] == INFECTED)
            {
                our_infected_x_locations[our_current_infected_person] =
                    our_x_locations[our_person1];
                our_infected_y_locations[our_current_infected_person] =
                    our_y_locations[our_person1];
                our_current_infected_person++;
            }
        }
#ifdef _MPI
        /* ALG XIV.B: Each process sends its count of infected people to all the
         *  other processes and receives their counts */
        MPI_Allgather(&our_num_infected, 1, MPI_INT, recvcounts, 1, 
                MPI_INT, MPI_COMM_WORLD);

        total_num_infected = 0;
        for(current_rank = 0; current_rank <= total_number_of_processes - 1;
                current_rank++)
        {
            total_num_infected += recvcounts[current_rank];
        }

        /* Set up the displacements in the receive buffer (see the man page for 
         *  MPI_Allgatherv) */
        current_displ = 0;
        for(current_rank = 0; current_rank <= total_number_of_processes - 1;
                current_rank++)
        {
            displs[current_rank] = current_displ;
            current_displ += recvcounts[current_rank];
        }

        /* ALG XIV.C: Each process sends the x locations of its infected people 
         *  to all the other processes and receives the x locations of their 
         *  infected people */
        MPI_Allgatherv(our_infected_x_locations, our_num_infected, MPI_INT, 
                their_infected_x_locations, recvcounts, displs, 
                MPI_INT, MPI_COMM_WORLD);
        
        /* ALG XIV.D: Each process sends the y locations of its infected people 
         *  to all the other processes and receives the y locations of their 
         *  infected people */
        MPI_Allgatherv(our_infected_y_locations, our_num_infected, MPI_INT, 
                their_infected_y_locations, recvcounts, displs, 
                MPI_INT, MPI_COMM_WORLD);
#else
        total_num_infected = our_num_infected;
        for(my_current_person_id = 0;
                my_current_person_id <= total_num_infected - 1;
                my_current_person_id++)
        {
            their_infected_x_locations[my_current_person_id] = 
                our_infected_x_locations[my_current_person_id];
            their_infected_y_locations[my_current_person_id] =
                our_infected_y_locations[my_current_person_id];
        }
#endif

#if defined(X_DISPLAY) || defined(TEXT_DISPLAY)
        /* ALG XIV.E: If display is enabled, Rank 0 gathers the states, x 
         *  locations, and y locations of the people for which each process is 
         *  responsible */
#ifdef _MPI
        /* Set up the receive counts and displacements in the receive buffer 
         *  (see the man page for MPI_Gatherv) */
        current_displ = 0;
        for(current_rank = 0; current_rank <= total_number_of_processes - 1;
                current_rank++)
        {
            displs[current_rank] = current_displ;
            recvcounts[current_rank] = total_number_of_people
                / total_number_of_processes;
            if(current_rank == total_number_of_processes - 1)
            {
                recvcounts[current_rank] += total_number_of_people
                    % total_number_of_processes;
            }
            current_displ += recvcounts[current_rank];
        }

        MPI_Gatherv(our_states, our_number_of_people, MPI_CHAR, states,
                recvcounts, displs, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Gatherv(our_x_locations, our_number_of_people, MPI_INT, x_locations,
                recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(our_y_locations, our_number_of_people, MPI_INT, y_locations,
                recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
#else
#ifdef OMP 
#pragma omp parallel for private(my_current_person_id)
#endif
        for(my_current_person_id = 0; my_current_person_id 
                <= total_number_of_people - 1; my_current_person_id++)
        {
            states[my_current_person_id] = our_states[my_current_person_id];
            x_locations[my_current_person_id] 
                = our_x_locations[my_current_person_id];
            y_locations[my_current_person_id] 
                = our_y_locations[my_current_person_id];
        }
#endif
#endif

        /* ALG XIV.F: If display is enabled, Rank 0 displays a graphic of the 
         *  current day */
#ifdef X_DISPLAY
        if(our_rank == 0)
        {
            XClearWindow(display, window);
            for(my_current_person_id = 0; my_current_person_id 
                    <= total_number_of_people - 1; my_current_person_id++)
            {
                if(states[my_current_person_id] == INFECTED)
                {
                    XSetForeground(display, gc, infected_color.pixel);
                }
                else if(states[my_current_person_id] == IMMUNE)
                {
                    XSetForeground(display, gc, immune_color.pixel);
                }
                else if(states[my_current_person_id] == SUSCEPTIBLE)
                {
                    XSetForeground(display, gc, susceptible_color.pixel);
                }
                else if(states[my_current_person_id] == DEAD)
                {
                    XSetForeground(display, gc, dead_color.pixel);
                }
                else
                {
                    fprintf(stderr, "ERROR: person %d has state '%c'\n",
                            my_current_person_id, states[my_current_person_id]);
                    exit(-1);
                }
                XFillRectangle(display, window, gc,
                        x_locations[my_current_person_id] 
                        * PIXEL_WIDTH_PER_PERSON, 
                        y_locations[my_current_person_id]
                        * PIXEL_HEIGHT_PER_PERSON, 
                        PIXEL_WIDTH_PER_PERSON, 
                        PIXEL_HEIGHT_PER_PERSON);
            }
            XFlush(display);
        }
#endif
#ifdef TEXT_DISPLAY
        if(our_rank == 0)
        {
            for(our_current_location_y = 0; 
                    our_current_location_y <= environment_height - 1;
                    our_current_location_y++)
            {
                for(our_current_location_x = 0; our_current_location_x 
                        <= environment_width - 1; our_current_location_x++)
                {
                    environment[our_current_location_x][our_current_location_y] 
                        = ' ';
                }
            }

            for(my_current_person_id = 0; 
                    my_current_person_id <= total_number_of_people - 1;
                    my_current_person_id++)
            {
                environment[x_locations[my_current_person_id]]
                    [y_locations[my_current_person_id]] = 
                    states[my_current_person_id];
            }

            printf("----------------------\n");
            for(our_current_location_y = 0;
                    our_current_location_y <= environment_height - 1;
                    our_current_location_y++)
            {
                for(our_current_location_x = 0; our_current_location_x 
                        <= environment_width - 1; our_current_location_x++)
                {
                    printf("%c", environment[our_current_location_x]
                            [our_current_location_y]);
                }
                printf("\n");
            }
        }
#endif

#if defined(X_DISPLAY) || defined(TEXT_DISPLAY)
        /* Wait between frames of animation */
        usleep(microseconds_per_day);
#endif

        /* ALG XIV.G: For each of the process’s people, each process spawns 
         *  threads to do the following */
#ifdef OMP 
#pragma omp parallel for private(my_current_person_id, my_x_move_direction, \
        my_y_move_direction)
#endif
        for(my_current_person_id = 0; my_current_person_id 
                <= our_number_of_people - 1; my_current_person_id++)
        {
            /* ALG XIV.G.1: If the person is not dead, then */
            if(our_states[my_current_person_id] != DEAD)
            {
                /* ALG XIV.G.1.a: The thread randomly picks whether the person 
                 *  moves left or right or does not move in the x dimension */
                my_x_move_direction = (random() % 3) - 1;

                /* ALG XIV.G.1.b: The thread randomly picks whether the person 
                 *  moves up or down or does not move in the y dimension */
                my_y_move_direction = (random() % 3) - 1;

                /* ALG XIV.G.1.c: If the person will remain in the bounds of the
                 *  environment after moving, then */
                if((our_x_locations[my_current_person_id] 
                            + my_x_move_direction >= 0)
                        && (our_x_locations[my_current_person_id] 
                            + my_x_move_direction < environment_width)
                        && (our_y_locations[my_current_person_id] 
                            + my_y_move_direction >= 0)
                        && (our_y_locations[my_current_person_id] 
                            + my_y_move_direction < environment_height))
                {
                    /* ALG XIV.G.i: The thread moves the person */
                    our_x_locations[my_current_person_id] 
                        += my_x_move_direction;
                    our_y_locations[my_current_person_id] 
                        += my_y_move_direction;
                }
            }
        }

        /* ALG XIV.H: For each of the process’s people, each process spawns 
         *  threads to do the following */
#ifdef OMP 
#pragma omp parallel for private(my_current_person_id, my_num_infected_nearby, \
        my_person2) reduction(+:our_num_infection_attempts) \
        reduction(+:our_num_infected) reduction(+:our_num_susceptible) \
        reduction(+:our_num_infections)
#endif
        for(my_current_person_id = 0; my_current_person_id 
                <= our_number_of_people - 1; my_current_person_id++)
        {
            /* ALG XIV.H.1: If the person is susceptible, then */
            if(our_states[my_current_person_id] == SUSCEPTIBLE)
            {
                /* ALG XIV.H.1.a: For each of the infected people (received
                 *  earlier from all processes) or until the number of infected 
                 *  people nearby is 1, the thread does the following */
                my_num_infected_nearby = 0;
                for(my_person2 = 0; my_person2 <= total_num_infected - 1
                        && my_num_infected_nearby < 1; my_person2++)
                {
                    /* ALG XIV.H.1.a.i: If person 1 is within the infection 
                     *  radius, then */
                    if((our_x_locations[my_current_person_id] 
                                > their_infected_x_locations[my_person2]
                                - infection_radius)
                            && (our_x_locations[my_current_person_id] 
                                < their_infected_x_locations[my_person2] 
                                + infection_radius)
                            && (our_y_locations[my_current_person_id]
                                > their_infected_y_locations[my_person2] 
                                - infection_radius)
                            && (our_y_locations[my_current_person_id]
                                < their_infected_y_locations[my_person2] 
                                + infection_radius))
                    {
                        /* ALG XIV.H.1.a.i.1: The thread increments the number 
                         *  of infected people nearby */
                        my_num_infected_nearby++;
                    }
                }

#ifdef SHOW_RESULTS
                if(my_num_infected_nearby >= 1)
                    our_num_infection_attempts++;
#endif

                /* ALG XIV.H.1.b: If there is at least one infected person 
                 *  nearby, and a random number less than 100 is less than or
                 *  equal to the contagiousness factor, then */
                if(my_num_infected_nearby >= 1 && (random() % 100) 
                        <= contagiousness_factor)
                {
                    /* ALG XIV.H.1.b.i: The thread changes person1’s state to 
                     *  infected */
                    our_states[my_current_person_id] = INFECTED;

                    /* ALG XIV.H.1.b.ii: The thread updates the counters */
                    our_num_infected++;
                    our_num_susceptible--;

#ifdef SHOW_RESULTS
                    our_num_infections++;
#endif
                }
            }
        }

        /* ALG XIV.I: For each of the process’s people, each process spawns 
         *  threads to do the following */
#ifdef OMP 
#pragma omp parallel for private(my_current_person_id) \
        reduction(+:our_num_recovery_attempts) reduction(+:our_num_dead) \
        reduction(+:our_num_infected) reduction(+:our_num_deaths) \
        reduction(+:our_num_immune)
#endif
        for(my_current_person_id = 0; my_current_person_id 
                <= our_number_of_people - 1; my_current_person_id++)
        {
            /* ALG XIV.I.1: If the person is infected and has been for the full 
             *  duration of the disease, then */
            if(our_states[my_current_person_id] == INFECTED
                    && our_num_days_infected[my_current_person_id] 
                    == duration_of_disease)
            {
#ifdef SHOW_RESULTS
                our_num_recovery_attempts++;
#endif
                /* ALG XIV.I.a: If a random number less than 100 is less than 
                 *  the deadliness factor, then */
                if((random() % 100) < deadliness_factor)
                {
                    /* ALG XIV.I.a.i: The thread changes the person’s state to 
                     *  dead */
                    our_states[my_current_person_id] = DEAD;

                    /* ALG XIV.I.a.ii: The thread updates the counters */
                    our_num_dead++;
                    our_num_infected--;

#ifdef SHOW_RESULTS
                our_num_deaths++;
#endif
                }
                /* ALG XIV.I.b: Otherwise, */
                else
                {
                    /* ALG XIV.I.b.i: The thread changes the person’s state to 
                     *  immune */
                    our_states[my_current_person_id] = IMMUNE;

                    /* ALG XIV.I.b.ii: The thread updates the counters */
                    our_num_immune++;
                    our_num_infected--;
                }
            }
        }

        /* ALG XIV.J: For each of the process’s people, each process spawns 
         *  threads to do the following */
#ifdef OMP 
#pragma omp parallel for private(my_current_person_id)
#endif
        for(my_current_person_id = 0; my_current_person_id 
                <= our_number_of_people - 1; my_current_person_id++)
        {
            /* ALG XIV.J.1: If the person is infected, then */
            if(our_states[my_current_person_id] == INFECTED)
            {
                /* ALG XIV.J.1.a: Increment the number of days the person has 
                 *  been infected */
                our_num_days_infected[my_current_person_id]++;
            }
        }
    }

    /* ALG XV: If X display is enabled, then Rank 0 destroys the X Window and 
     *  closes the display */
#ifdef X_DISPLAY
    if(our_rank == 0)
    {
        XDestroyWindow(display, window);
        XCloseDisplay(display);
    }
#endif

#ifdef SHOW_RESULTS
    printf("Rank %d final counts: %d susceptible, %d infected, %d immune, \
%d dead\nRank %d actual contagiousness: %f\nRank %d actual deadliness: \
%f\n", our_rank, our_num_susceptible, our_num_infected, our_num_immune, 
        our_num_dead, our_rank, 100.0 * (our_num_infections / 
        (our_num_infection_attempts == 0 ? 1 : our_num_infection_attempts)),
        our_rank, 100.0 * (our_num_deaths / (our_num_recovery_attempts == 0 ? 1 
            : our_num_recovery_attempts)));
#endif

    /* Deallocate the arrays -- we have finished using the memory, so now we
     *  "free" it back to the heap */
#ifdef TEXT_DISPLAY 
    for(our_current_location_x = environment_width - 1; 
            our_current_location_x >= 0; our_current_location_x--)
    {
        free(environment[our_current_location_x]);
    }
    free(environment);
#endif
    free(our_states);
    free(states);
    free(displs);
    free(recvcounts);
    free(our_num_days_infected);
    free(their_infected_y_locations);
    free(their_infected_x_locations);
    free(our_infected_y_locations);
    free(our_infected_x_locations);
    free(our_y_locations);
    free(our_x_locations);
    free(y_locations);
    free(x_locations);

#ifdef _MPI
    /* MPI execution is finished; no MPI calls are allowed after this */
    MPI_Finalize();
#endif

    /* The program has finished executing successfully */
    return 0;
}
