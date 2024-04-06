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
   printf("writing RGB %d, %d\n",width, height);

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
  printf("done writing file\n");

  free(row);
  fclose(fp);
}

int main(int argc, char **argv)
{
   int rank, size;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Rank (ID) of this process
   MPI_Comm_size(MPI_COMM_WORLD, &size); // Total size of MPI job
   MPI_Status status;
   
   srand(0);
   
   //Simulation Parameters
   int world_x = 500;
   int world_y = 1000;

   if (world_y%size != 0) {
      printf("ERROR, I'm lazy and cant figure out edge cases\n");
      return(1);
   }
   
   int sz_x = 500;
   int sz_y = world_y/size;
   char filename[sizeof "./images/file00000.png"];
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
   printf("After Allocate\n");

   char * full_world_mem;
   char ** full_world;
   if(rank==0) {
      int full_world_size_with_boarders = (world_x+2)*(world_y+2);
      full_world_mem = (char *) malloc(2*full_world_size_with_boarders*sizeof(char));
      for (int i=0; i<full_world_size_with_boarders; i++)
           full_world_mem[i] = 0;

      full_world = (char **) malloc((world_y+2) * sizeof(char*));
      for (int r=0; r<(world_y+2); r++)
          full_world[r] = &full_world_mem[(r*(sz_x+2))];

      printf("After world Allocate\n");


   }

   int which = 0;
  
   //Pick Rumor Starting location
   for (int u; u < NUM_RUMORS; u++) {
      int r = (int) ((float)rand())/(float)RAND_MAX*(world_y);
      int c = (int) ((float)rand())/(float)RAND_MAX*(world_x);
      if ( r > (world_y/size)*rank && r < (world_y/size)*(rank+1)) {
	 int my_r = r - ((world_y/size)*rank);
         my_world[which][my_r+1][c+1] = u+2;
         printf("Starting a Rumor %d, %d = %d\n",r,c,u+2);
      }
   }
   //writeworld("start.png", my_world[which], sz_x, sz_y);
   printf("After Start location picked \n");
   srand(rank+1);

   //Inicialize Random World
   for (int r=0; r<sz_y+2; r++)
   	for (int c=0; c<sz_x+2; c++)
	{
	   float rd = ((float) rand())/(float)RAND_MAX;
           if (rd < pop_prob && my_world[which][r][c] == 0) 
        	my_world[which][r][c] = 1;
 	}
   printf("After Inicialize\n");
  

   int up = rank-1;
   int down = rank+1;

   if (up < 0) 
      up = size-1;

   if (down > size-1)
      down = 0;

   int up_tag = 1;
   int down_tag = 2;
   int output_tag = 3;

   printf("Me %d, up %d, down %d\n",rank,up,down);
 
   //Main Time loop
   for(int t=0; t<num_loops;t++) {

        //printf("%d sending up\n",rank);
        //Communicate Edges
        // Send up
        if (rank == 0) {
	   MPI_Recv(my_world[which][sz_y+1],sz_x+2,MPI_BYTE,down,up_tag,MPI_COMM_WORLD,&status);
	   MPI_Send(my_world[which][1],sz_x+2,MPI_BYTE,up,up_tag,MPI_COMM_WORLD);
	} else {
	   MPI_Send(my_world[which][1],sz_x+2,MPI_BYTE,up,up_tag,MPI_COMM_WORLD);
	   MPI_Recv(my_world[which][sz_y+1],sz_x+2,MPI_BYTE,down,up_tag,MPI_COMM_WORLD,&status);
	}
        //printf("done sending up\n");	

	// Send down
	if (rank == size-1) {
	   MPI_Recv(my_world[which][0],sz_x+2,MPI_BYTE,up,down_tag,MPI_COMM_WORLD,&status);
	   MPI_Send(my_world[which][sz_y],sz_x+2,MPI_BYTE,down,down_tag,MPI_COMM_WORLD);
	} else {
	   MPI_Send(my_world[which][sz_y],sz_x+2,MPI_BYTE,down,down_tag,MPI_COMM_WORLD);
	   MPI_Recv(my_world[which][0],sz_x+2,MPI_BYTE,up,down_tag,MPI_COMM_WORLD,&status);
	}
        //printf("done sending down\n");	

        //Loop left/right 
        //for (int c=1; c<sz_x+1; c++){
        //   my_world[which][0][c] = my_world[which][sz_y][c];
        //   my_world[which][sz_y+1][c] = my_world[which][1][c];
        //}
   	for (int r=1; r<sz_y+1; r++){
           my_world[which][r][0] = my_world[which][r][sz_x];
           my_world[which][r][sz_x+1] = my_world[which][r][1];
        }
             
   	//printf("Step %d\n",t);
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
	      			//printf("."); 
		      		my_world[!which][r][c] = n;
 		      		break;
	           	}
		   } 
		}
             }
	   }
        }	
        //printf("%d done with main loop\n", rank);
        //printf("%d with Barrier\n", rank);
        if(t%10==0) {       
             //printf("%d printing output\n",rank); 
             if (rank == 0) {
   		for (int r=0; r<(sz_y+1); r++)
   		   for (int c=0; c<(sz_x+1); c++)
        	   	full_world[r][c] = my_world[which][r][c];
		//printf("done with mycopy\n");
                for (int r=1; r<size;r++) {
                   MPI_Recv(full_world[(r*sz_y)+1],(sz_x+2)*(sz_y),MPI_BYTE,r,output_tag,MPI_COMM_WORLD,&status);
		   //printf("Sucessfully recived %d \n",r);
                }
	        //Send everything back to master for saving.
                sprintf(filename, "./images/file%05d.png", img_count);
		//printf("saving %s \n",filename);
                writeworld(filename, full_world, world_x, world_y);
                img_count++;
             } else {
                MPI_Send(my_world[which][1],(sz_x+2)*(sz_y),MPI_BYTE,0,output_tag,MPI_COMM_WORLD);
             }
          }
	MPI_Barrier(MPI_COMM_WORLD);
	which = !which;
   } 

   //Write out output image using 1D serial pointer
   //writeworld("end.png", world_mem, sz_x, sz_y);
   printf("After Loop\n");
   free(my_world[0]);
   free(my_world[1]);
   printf("After Clean up\n");
   MPI_Finalize();
   return(0);
}

