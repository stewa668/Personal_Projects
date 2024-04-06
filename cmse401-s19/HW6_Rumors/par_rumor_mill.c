#include <stdio.h>          
#include <stdlib.h>
#include <png.h>
#include <mpi.h>

void writeworld(char * filename, char ** my_world, int sz_x, int sz_y) {

   int width = sz_x+2;
   int height = sz_y+2;

   FILE *fp = fopen(filename, "wb");
   if(!fp) abort();

   png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
   if (!png) abort();

   png_infop info = png_create_info_struct(png);
   if (!info) abort();

   if (setjmp(png_jmpbuf(png))) abort();

   png_init_io(png, fp);

   png_set_IHDR(
     png,
     info,
     width, height,
     8,
     PNG_COLOR_TYPE_RGB,
     PNG_INTERLACE_NONE,
     PNG_COMPRESSION_TYPE_DEFAULT,
     PNG_FILTER_TYPE_DEFAULT
   );
   png_write_info(png, info);

   png_bytep row = (png_bytep) malloc(3 * width * sizeof(png_byte));
   //printf("writing RGB %d, %d\n",width, height);

   int full_size_with_boarders = (sz_x+2)*(sz_y+2);

   for(int r=0;r <height;r++) {
     for(int i=0;i < width;i++){
        switch (my_world[r][i]){
  	 case 1:
              row[(i*3)+0] = 255;
              row[(i*3)+1] = 255;
              row[(i*3)+2] = 255; 
              break;
	 case 2:
              row[(i*3)+0] = 255;
              row[(i*3)+1] = 0;
              row[(i*3)+2] = 0;
              break;
         case 3:
              row[(i*3)+0] = 0;
              row[(i*3)+1] = 255;
              row[(i*3)+2] = 0;
              break;
         case 4:
              row[(i*3)+0] = 0;
              row[(i*3)+1] = 0;
              row[(i*3)+2] = 255;
              break;
         case 5:
              row[(i*3)+0] = 255;
              row[(i*3)+1] = 255;
              row[(i*3)+2] = 0;
              break;
         case 6:
              row[(i*3)+0] = 255;
              row[(i*3)+1] = 0;
              row[(i*3)+2] = 255;
              break;
         case 7:
              row[(i*3)+0] = 0;
              row[(i*3)+1] = 255;
              row[(i*3)+2] = 255;
              break;
         case 8:
              row[(i*3)+0] = 233;
              row[(i*3)+1] = 131;
              row[(i*3)+2] = 0;
              break;
	 default:
              row[(i*3)+0] = 0;
              row[(i*3)+1] = 0;
              row[(i*3)+2] = 0;
              break;
      }
    }
    png_write_row(png, row);
   }
  png_write_end(png, NULL);
 // printf("done writing file\n");

  free(row);
  fclose(fp);
}

int main(int argc, char **argv)
{  
   MPI_Status status;
   MPI_Init(&argc, &argv);
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   srand(0);
   
   //Simulation Parameters
double total = 0;
struct timeval tot0, tot1;
gettimeofday(&tot0,0);
   int XX = 2000;
   int YY = 1000;
   int sz_x = XX;
   int sz_y = YY/size;
   int L = YY%size ;
   if (rank < L) sz_y += 1 ;
   char filename[sizeof "./images/mpi00000.png"];
   int img_count = 0;

   float pop_prob = 0.75;
   float rumor_prob = 0.25;
   int num_loops = 1000;
   int NUM_RUMORS = 7;

   //Allocate space for my world
   int full_size_with_boarders = (sz_x+2)*(sz_y+2);
   char * world_mem = (char *) malloc(2*full_size_with_boarders*sizeof(char));
   for (int i=0; i<full_size_with_boarders; i++)
   	world_mem[i] = 0; 

   char ** my_world[2];
   my_world[0] = (char **) malloc((sz_y+2) * sizeof(char*));
   my_world[1] = (char **) malloc((sz_y+2) * sizeof(char*));
   for (int which=0; which < 2; which++) {
      for (int r=0; r<(sz_y+2); r++)
         my_world[which][r] = &world_mem[(r*(sz_x+2))+which*full_size_with_boarders];
   }

   //Allocate space for all world
   int FULL_size_with_boarders = (XX+2)*(YY+2);
   char * worlds_mem = (char *) malloc(FULL_size_with_boarders*sizeof(char));
   for (int i=0; i<FULL_size_with_boarders; i++)
        worlds_mem[i] = 0;

   char ** the_world;
   the_world = (char **) malloc((YY+2) * sizeof(char*));
   for (int r=0; r<(YY+2); r++)
      the_world[r] = &worlds_mem[(r*(XX+2))];

   //if (rank == 0){
   //printf("After Allocate\n");
   //}
   int which = 0;
   //Inicialize Random World
   if (rank==0){
   for (int r=0; r<YY+2; r++)
   	for (int c=0; c<XX+2; c++)
	{
	   float rd = ((float) rand())/(float)RAND_MAX;
           if (rd < pop_prob) 
        	the_world[r][c] = 1;
 	}
  // printf("After Inicialize\n");
   
   //Pick Rumor Starting location
   for (int u; u < NUM_RUMORS; u++) {
      int r = (int) ((float)rand())/(float)RAND_MAX*(YY+2);
      int c = (int) ((float)rand())/(float)RAND_MAX*(XX+2);
      the_world[r][c] = u+2; 
    //  printf("Starting a Rumor %d, %d = %d\n",r,c,u+2); 
   } 
   writeworld("start.png", my_world[which], sz_x, sz_y);
  // printf("After Start location picked \n");
   }
   
   if (rank==0){
     for (int i=1; i<(sz_y+1); i++) my_world[which][i] = the_world[i];
     if (L > 0){
     for (int r=1; r<(L); r++) {
       for (int i=0; i<(sz_y); i++){
     //    printf("s%d\n", 1+r*(sz_y)+i);
         MPI_Send(the_world[1+r*(sz_y)+i], sz_x, MPI_BYTE, r, 3, MPI_COMM_WORLD);}
     }
     for (int r=L; r<size; r++) {
       for (int i=0; i<(sz_y-1); i++){
     //    printf("s%d\n", 1+(1000%size)*(sz_y) + (r-(1000%size))*(sz_y-1) + i);
         MPI_Send(the_world[1+L*(sz_y) + (r-L)*(sz_y-1) + i], sz_x, MPI_BYTE, r, 3, MPI_COMM_WORLD);}
     }}
else{for (int r=L+1; r<size; r++) {
       for (int i=0; i<(sz_y); i++){
     //    printf("r%d\n", 1+(1000%size)*(sz_y) + r*(sz_y) + i);
         MPI_Send(the_world[1+L*(sz_y) + r*(sz_y) + i], sz_x, MPI_BYTE, r, 3, MPI_COMM_WORLD);}}
     }

   }
   else {
     for (int i=1; i<sz_y+1; i++){
       MPI_Recv(my_world[which][i], sz_x, MPI_BYTE, 0, 3, MPI_COMM_WORLD, &status);
     }
   }
   //Main Time loop
   for(int t=0; t<num_loops;t++) {

        //Communicate Edges
        
        if (rank == 0) {
          MPI_Send(my_world[which][sz_y], sz_x, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
          MPI_Recv(my_world[which][0], sz_x, MPI_BYTE, size-1 , 0, MPI_COMM_WORLD, &status);
	}


        else {
          MPI_Recv(my_world[which][0], sz_x, MPI_BYTE, (rank-1), 0, MPI_COMM_WORLD, &status);
          MPI_Send(my_world[which][sz_y], sz_x, MPI_BYTE, (rank+1)%size, 0, MPI_COMM_WORLD);
        }
 
        if (rank == 0) {
          MPI_Send(my_world[which][1], sz_x, MPI_BYTE, size-1 , 1, MPI_COMM_WORLD);
          MPI_Recv(my_world[which][sz_y+1], sz_x, MPI_BYTE, 1 , 1, MPI_COMM_WORLD, &status);
        }


        else {
          MPI_Recv(my_world[which][sz_y+1], sz_x, MPI_BYTE, (rank+1)%size , 1, MPI_COMM_WORLD, &status);
          MPI_Send(my_world[which][1], sz_x, MPI_BYTE, (rank-1), 1, MPI_COMM_WORLD);
        }

        //Loop Edges
   	for (int r=1; r<sz_y+1; r++){
           my_world[which][r][0] = my_world[which][r][sz_x];
           my_world[which][r][sz_x+1] = my_world[which][r][1];
        }
             
//   	if (rank == 0) printf("Step %d\n",t);
        int rumor_counts[NUM_RUMORS+2];
   	for (int r=1; r<sz_y+1; r++) {
           for (int c=1; c<sz_x+1; c++)
	   {
	     my_world[!which][r][c] = my_world[which][r][c];
	     if (my_world[which][r][c] >0) { 
		for(int n=0;n<NUM_RUMORS+2;n++)
                   rumor_counts[n] = 0;
                rumor_counts[my_world[which][r-1][c]]++;
                rumor_counts[my_world[which][r+1][c]]++;
                rumor_counts[my_world[which][r][c-1]]++; 
                rumor_counts[my_world[which][r][c+1]]++;

		float rd = ((float) rand())/(float)RAND_MAX; 
		float my_prob = 0;
		for(int n=2;n<NUM_RUMORS+2;n++) {
		   if (rumor_counts[n] > 0) {
			my_prob += rumor_prob*rumor_counts[n];
	           	if(rd <= my_prob) {
		      		my_world[!which][r][c] = n;
 		      		break;
	           	}
		   } 
		}
             }
	   }
        }	
	which = !which;
	if(t%10==0) {       
	   //Send everything back to master for saving.
	   if (rank == 0){
           if (L == 0){
           for (int r=1; r<size; r++) {
             for (int i=0; i<(sz_y); i++){
               MPI_Recv(the_world[1+L*(sz_y) + r*(sz_y) + i], sz_x, MPI_BYTE, r, 4, MPI_COMM_WORLD, &status);}}
           }
           else{
           for (int r=1; r<L; r++) {
             for (int i=0; i<(sz_y); i++){
               MPI_Recv(the_world[1+ r*(sz_y) + i], sz_x, MPI_BYTE, r, 4, MPI_COMM_WORLD, &status);}}
           for (int r=L; r<size; r++) {
             for (int i=0; i<(sz_y-1); i++){
               MPI_Recv(the_world[1+L*(sz_y) + (r-L)*(sz_y-1) + i], sz_x, MPI_BYTE, r, 4, MPI_COMM_WORLD, &status);}}
           }
           sprintf(filename, "./images/mpi%05d.png", img_count);
   	   writeworld(filename, the_world, XX, YY);
           }
           else {
             for (int i=1; i<sz_y+1; i++){
               MPI_Send(my_world[which][i], sz_x, MPI_BYTE, 0, 4, MPI_COMM_WORLD);
             }
           }
           img_count++;
	}
   } 

   //Write out output image using 1D serial pointer
   //writeworld("end.png", world_mem, sz_x, sz_y);
   //if (rank == 0) printf("After Loop\n");
   free(my_world[0]);
   free(my_world[1]);
   //if (rank == 0) printf("After Clean up\n");
if (rank == 0) gettimeofday(&tot1,0);
total = (tot1.tv_sec-tot0.tv_sec)*1000000 + tot1.tv_usec-tot0.tv_usec;
if (rank == 0) printf("%f,\n ", total/1000000);
   MPI_Finalize();
   return(0);
}

