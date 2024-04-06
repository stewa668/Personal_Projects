// $Id: cpumon_v1.0.cpp 4299 2013-09-08 01:21:19Z skylar $
//
// Author: Zach Goodwyn (University of Mary Washington)
// Adviser: David Toth (University of Mary Washington - dtoth@umw.edu)
// CPU Monitoring Software
//
// This code uses ideas from GalaxSee and from 
// http://tronche.com/gui/x/xlib-tutorial/

#include<mpi.h>
#include<sys/types.h>
#include<iostream>
#include<X11/Xlib.h>
#include<X11/Xutil.h>
#include<X11/Xresource.h>
#include<X11/Xos.h>
#include<X11/Xatom.h>
#include<X11/keysym.h>
#include<stdio.h>
#include<string.h>
#include<unistd.h>
#include<stdlib.h>
#include<cstdlib>
#include<ctime>
#include<sstream>
#include<fstream>

using namespace std;

//global X11 objects
Display * d;
Window w;
XEvent e;
int s;


//this class is responsible for each graph that displays and updates the graph for each node
class CPUGraph
{
private:

	//the x and y coordinates that every aspect of the graph depends on
	int graph_x;
	int graph_y;

	//the title above the graph
	char* graph_Title;

	const static int NUM_VALUES = 5;
	
	//the current loads for the nodes cores
	int core_One_Current;
	int core_Two_Current;
	
	//the arrays for the saved load values
	int* core_One_Saved;
	int* core_Two_Saved;
	int title_Size;

public:
	
	CPUGraph(string title, int x_Bound, int y_Bound,int node_ID);
	void GraphExpose();
	void InputCoreA(int load);
	void InputCoreB(int load);
	void RedrawGraph();
	int node_Number;

};

//the constructor for the graph class, only responsible for memory allocation and variable initialization
//@Param_One: string title, the title displayed above the graph
//@param_Two: int x_Bound, the upper left x coordinate for the graph
//@Param_Three: int y_Bound, the upper left y coordinate for the graph
CPUGraph::CPUGraph(string title, int x_Bound, int y_Bound, int node_ID)
{
	node_Number = node_ID;
	graph_x = x_Bound;
	graph_y = y_Bound;

	//alocate memory for the core loads, holds on to the five latest values
	core_One_Saved = new int[NUM_VALUES];
	core_Two_Saved = new int[NUM_VALUES];

	//allocate memory for the title string
	graph_Title = new char[title.size()];
	title_Size = title.size();
	//initialize values in loads arrays to -1, flags to not print these values
	for(int index = 0; index < NUM_VALUES; index++)
	{
		core_One_Saved[index] = -1;
		core_Two_Saved[index] = -1;
	}
	//sets the graphs private title variable to the one entered into the constructor
	for(int index = 0; index < title.size(); index++)
	{
		graph_Title[index] = title[index]; 
	}

	
}

//function called to draw the empty graphs on the x window when the expose event is triggered
//no params
void CPUGraph::GraphExpose()
{
	char* graph_Id;

        stringstream temp_Test;
        temp_Test << node_Number;
        string number = temp_Test.str();

        int test_Size = number.size();
        graph_Id = new char[test_Size];
        for(int index = 0; index < test_Size; index++)
        {
                graph_Id[index] = number[index];
        }
	char* real_Title = new char[5 + test_Size];
	for(int index = 0; index < 5; index ++)
	{
		real_Title[index] = graph_Title[index];
	}
	for(int index = 5; index < (5+test_Size); index++)
	{
		real_Title[index] = graph_Id[index-5];
	}

	

	//string to be printed beneath the each graph (label)
	char* cpu_Core_One_Label = "Core One:";
	char* cpu_Core_Two_Label = "Core Two:";

	XDrawLine(d,w,DefaultGC(d,s),graph_x,graph_y,graph_x,(graph_y + 150));
	XDrawLine(d,w,DefaultGC(d,s),graph_x,graph_y,(graph_x + 150),graph_y);
	XDrawLine(d,w,DefaultGC(d,s),(graph_x + 150),graph_y,(graph_x + 150),(graph_y + 150));
	XDrawLine(d,w,DefaultGC(d,s),graph_x,(graph_y + 150),(graph_x + 150),(graph_y + 150));

	XDrawString(d,w,DefaultGC(d,s),(graph_x + 25),(graph_y - 10),real_Title, (5+test_Size));

	XDrawString(d,w,DefaultGC(d,s),(graph_x + 25),(graph_y + 165),cpu_Core_One_Label,strlen(cpu_Core_One_Label));
	XDrawString(d,w,DefaultGC(d,s),(graph_x + 25),(graph_y + 185),cpu_Core_Two_Label,strlen(cpu_Core_Two_Label));
	
}

//function used to input core ones current load
//@Param_One: int load, current core load, each later load is then pushed further down the array
void CPUGraph::InputCoreA(int load)
{
		core_One_Current = load;
		core_One_Saved[4] = core_One_Saved[3];
		core_One_Saved[3] = core_One_Saved[2];
		core_One_Saved[2] = core_One_Saved[1];
		core_One_Saved[1] = core_One_Saved[0];
		core_One_Saved[0] = load;
		
}

//fuction used to input cure twos current load
//@Param_Two: int load, current core load, each later load is then pushed further down the array
void CPUGraph::InputCoreB(int load)
{
		core_Two_Current = load;
		core_Two_Saved[4] = core_Two_Saved[3];
		core_Two_Saved[3] = core_Two_Saved[2];
		core_Two_Saved[2] = core_Two_Saved[1];
		core_Two_Saved[1] = core_Two_Saved[0];
		core_Two_Saved[0] = load;
		
}

//function called to update the graph after the the loads are entered, first converts the load int
//into a char* and puts it into a new variable so it can be printed onto the x window, then updates that
//value on underneath the graph near its respective label, lastly it runs a for loop that draw the lines
//and points the represent the 5 most current loads for their respective cores. If the found core load is
// a -1 it is not represented
void CPUGraph::RedrawGraph()
{

	//creating the blue and red color GCs to differentiate the 2 cores on each node
	GC red_gc;
	GC blue_gc;

	XColor red_col;
        XColor blue_col;

        Colormap colormap;

        char red[] = "#FF0000";
        char blue[] = "#4169E1";

        colormap = DefaultColormap(d,0);
        red_gc = XCreateGC(d,w,0,0);
        blue_gc = XCreateGC(d,w,0,0);


        XParseColor(d,colormap,red,&red_col);
        XAllocColor(d,colormap,&red_col);

        XParseColor(d,colormap,blue,&blue_col);
        XAllocColor(d,colormap,&blue_col);


        XSetForeground(d,red_gc,red_col.pixel);
        XSetForeground(d,blue_gc,blue_col.pixel);	

	//conversion from an int to a char* using stringstream objects
	//lately i have found this to be costly on resources and i am
	//currently building a new function to deal with it
	char* temp_Load_One;
	char* temp_Load_Two;
	
	stringstream temp_One;
	stringstream temp_Two;
	temp_One << core_One_Saved[0];
	temp_Two << core_Two_Saved[0];

	string cpuOne = temp_One.str();
	string cpuTwo = temp_Two.str();

	int one_Size = cpuOne.size();
	temp_Load_One = new char[one_Size];
	
	int two_Size = cpuTwo.size();
	temp_Load_Two = new char[two_Size];

	char* test;

	stringstream temp_Test;
	temp_Test << node_Number;
	string number = temp_Test.str();

	int test_Size = number.size();
	test = new char[test_Size];
	for(int index = 0; index < test_Size; index++)
	{
		test[index] = number[index];
	}


	for(int index = 0; index < one_Size; index++)
	{
		temp_Load_One[index] = cpuOne[index];
	}
	for(int index = 0; index < two_Size; index++)
	{
		temp_Load_Two[index] = cpuTwo[index];
	}
	
	//clears area for new data to be printed
	XClearArea(d,w,(graph_x + 80),(graph_y + 151),50,50,0);
	XClearArea(d,w,(graph_x + 1),(graph_y +1),149,149,0);
	XFlush(d);
	

	
	//updates the numerical representation for the current load
	XDrawString(d,w,red_gc,(graph_x + 80),(graph_y + 166),temp_Load_One,one_Size);
	XDrawString(d,w,blue_gc,(graph_x + 80),(graph_y + 185),temp_Load_Two,two_Size);


	//for loop that draws the current load graph using the five latest loads
	//also draws the lines and the numerical values for the loads in blue and
	//red
	for(int inc = 0; inc < NUM_VALUES; inc++)
			{
				if(core_One_Saved[inc] != -1)
				{
					if(inc == 0)
					{
						XDrawPoint(d,w,DefaultGC(d,s),(graph_x + 150),((graph_y + 150) - core_One_Saved[inc]));

						XDrawPoint(d,w,DefaultGC(d,s),(graph_x + 150),((graph_y + 150) - core_Two_Saved[inc]));
					}
					if(inc == 1)
					{
						XDrawPoint(d,w,DefaultGC(d,s),(graph_x + 113),((graph_y + 150) - core_One_Saved[inc]));
						XDrawLine(d,w,red_gc,(graph_x + 113),((graph_y + 149) - core_One_Saved[inc]),(graph_x + 150),((graph_y + 149) - core_One_Saved[inc-1]));

						XDrawPoint(d,w,DefaultGC(d,s),(graph_x + 113),((graph_y + 150) - core_Two_Saved[inc]));
						XDrawLine(d,w,blue_gc,(graph_x + 113),((graph_y + 149) - core_Two_Saved[inc]),(graph_x + 150),((graph_y + 149) - core_Two_Saved[inc-1]));
					}
					if(inc == 2)
					{
						XDrawPoint(d,w,DefaultGC(d,s),(graph_x + 76),((graph_y + 150) - core_One_Saved[inc]));
						XDrawLine(d,w,red_gc,(graph_x + 76),((graph_y + 149) - core_One_Saved[inc]),(graph_x + 113),((graph_y + 149) - core_One_Saved[inc-1]));

						XDrawPoint(d,w,DefaultGC(d,s),(graph_x + 76),((graph_y + 150) - core_Two_Saved[inc]));
						XDrawLine(d,w,blue_gc,(graph_x + 76),((graph_y + 149) - core_Two_Saved[inc]),(graph_x + 113),((graph_y + 149) - core_Two_Saved[inc-1]));
					}
					if(inc == 3)
					{
						XDrawPoint(d,w,DefaultGC(d,s),(graph_x + 39),((graph_y + 150) - core_One_Saved[inc]));
						XDrawLine(d,w,red_gc,(graph_x + 39),((graph_y + 149) - core_One_Saved[inc]),(graph_x + 76),((graph_y + 149) - core_One_Saved[inc-1]));

						XDrawPoint(d,w,DefaultGC(d,s),(graph_x + 39),((graph_y + 150) - core_Two_Saved[inc]));
						XDrawLine(d,w,blue_gc,(graph_x + 39),((graph_y + 149) - core_Two_Saved[inc]),(graph_x + 76),((graph_y + 149) - core_Two_Saved[inc-1]));
					}
					if(inc == 4)
					{
						XDrawPoint(d,w,DefaultGC(d,s),(graph_x + 2),((graph_y + 150) - core_One_Saved[inc]));
						XDrawLine(d,w,red_gc,(graph_x + 2),((graph_y + 149) - core_One_Saved[inc]),(graph_x + 39),((graph_y + 149) - core_One_Saved[inc-1]));

						XDrawPoint(d,w,DefaultGC(d,s),(graph_x + 2),((graph_y + 150) - core_Two_Saved[inc]));
						XDrawLine(d,w,blue_gc,(graph_x + 2),((graph_y + 149) - core_Two_Saved[inc]),(graph_x + 39),((graph_y + 149) - core_Two_Saved[inc-1]));
					}
				}
			}

}

//function used to add the elements of an array
//@Param_One: int numbers, the numbers to be added
//@Param_Two: int size. the size of the array
int sumArray(int numbers[], int size)
{
	int sum = 0;
	for(int index = 0; index < size; index++)
	{

		sum = sum + numbers[index];
	}

	return sum;


}
//used to test the composition of arrays belonging to each node
/*
void printArray(int numbers[], int size, int rank)
{
	//cout<<"Rank: "<<rank<<"   ";
	for(int index = 0; index < size; index++)
	{
	//	cout<<numbers[index]<<" ";
	}
	//cout<<endl;


}
*/

//Main function, holds the x11 while loop, and calls the subrutines to manipulate the x11 window
//responsible for getting the information from each node
int main(int argc, char* argv[])
{


	int myid, numprocs, i;
	//array that holds the cpu loads values for each mpi_gather call
	int hostCpuValues[18];

	//array that holds each nodes own values on its repective memory
	//[0] = node id, [1] = core one load, [2] = core two load
	int nodeCpuValues[3];

	for(int index = 0; index < 12; index++)
	{
		hostCpuValues[index] = 0;

	}
	nodeCpuValues[0] = 0;
	nodeCpuValues[1] = 0;
	
	//printf("Initializing MPI...\n");	
	if (MPI_Init(&argc,&argv) == MPI_SUCCESS)
	{
		printf("Successfully initialized MPI\n");
	}
	else 
	{
		printf("Failed to intialize MPI\n");
	}

	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	//if (myid == 0)
	{
	//	printf("Got MPI process rank and number of MPI processes.\nHitting a barrier.\n");
	}

	//MPI_Barrier(MPI_COMM_WORLD);

	//if (myid == 0)
	{
	//	printf("Passed MPI barrier\n");
	}

	unsigned seed =  time(0);
	srand(seed);

	if (myid == 0) // Only the root MPI process is going to control the XDisplay
        {
          //      printf("setting XDisplay\n");
                d = XOpenDisplay(NULL);
				s = DefaultScreen(d);
				w = XCreateSimpleWindow(d,RootWindow(d,s),10,10,600,600,1,BlackPixel(d,s),WhitePixel(d,s));
        }

		//fstream to get the host name of each system (unfinished)
        //gets the hostname but does not doing anything with it
        //my idea is to add a public char*  variable to the graph class to host the hostname(implemented in 2.1)
        //the graph title for each graph will be replaced with this
        //then i will bea able to give the new loads to their respective graph nodes by comparing
        //this new string to the one i recieve each time(implemented in 2.1)
        ifstream hostname("/etc/hostname");

        if(!hostname)
        {
                cout<<"an error had occured"<<endl;
        }


		//this section gets each node id and "names" each graph with it
        string node_Name;
        string node_String = "xxx";
        int node_Number;
        hostname>>node_Name;
        node_String[0] = node_Name[4];
        node_String[1] = node_Name[5];
        node_String[2] = node_Name[6];

        stringstream buffer(node_String);
        buffer >> node_Number;

	int test_Number[1];
	test_Number[0] = node_Number;

  //      cout<<"rank: "<<myid<<" "<<"node name: "<<test_Number[0]<<endl;
        
        hostname.close();
	
	int host_Names[6];
	for(int index = 0; index < 6; index++)
	{
		host_Names[index] = 0;

	}
	
	MPI_Gather(test_Number, 1, MPI_INT, host_Names,1,MPI_INT,0,MPI_COMM_WORLD);

	

	//this is where each graph is initiallized with its name and location on the x window
        CPUGraph node_One("Node One", 50, 50,host_Names[0]);
        CPUGraph node_Two("Node Two", 225,50,host_Names[1]);
        CPUGraph node_Three("Node Three", 400, 50,host_Names[2]);
        CPUGraph node_Four("Node Four", 50, 300,host_Names[3]);
        CPUGraph node_Five("Node Five", 225, 300,host_Names[4]);
        CPUGraph node_Six("Node Six", 400, 300,host_Names[5]);
	
	CPUGraph graph_List[6] = {node_One,node_Two,node_Three,node_Four,node_Five,node_Six};
        if (myid == 0)
        {
          //      printf("Created graphs\n");
        }


	
	//arrays responsible for holding the data after each /proc/stat file read
	int core_One_Total_Jiffies[10];
	int core_One_Work_Jiffies[3];
	int core_Two_Total_Jiffies[10];
	int core_Two_Work_Jiffies[3];
	string file_Input;
	
	//create variables for the cpu load file stream
	

	int core_One_Total_Jiffies_1;
	int core_One_Work_Jiffies_1;

	int core_Two_Total_Jiffies_1;
	int core_Two_Work_Jiffies_1;

	int core_One_Total_Jiffies_2;
	int core_One_Work_Jiffies_2;

	int core_Two_Total_Jiffies_2;
	int core_Two_Work_Jiffies_2;

	int core_One_Work_Over_Period;
	int core_One_Total_Over_Period;

	int core_Two_Work_Over_Period;
	int core_Two_Total_Over_Period;

	if (myid == 0) // Only the root MPI process is going to control the XDisplay
	{
		if(d == NULL)
		{

			fprintf(stderr, "Cannot open display");
			exit(1);
		}

		XSelectInput(d,w, ExposureMask|KeyPressMask);

		XMapWindow(d,w);
	}
	//int accumulator = 0;
	ifstream cpu_Data;

//	printf("starting while(1) loop...\n");
	while(1)
	{

		
		if (myid == 0) // Only the root MPI process is going to control the XDisplay
		{
			//printf("Root process is drawing empty graphs\n");
			XNextEvent(d, &e);
		
			//this is where the window is first opened and the intial graphs are drawn
			if(e.type == Expose)
			{	
			
				node_One.GraphExpose();
				node_Two.GraphExpose();
				node_Three.GraphExpose();
				node_Four.GraphExpose();
				node_Five.GraphExpose();
				node_Six.GraphExpose();
				XFlush(d);
		
			
			
			}
			
			if(e.type == KeyPress)
			{

				break;
			}
		}
		
		//this loop is the body of the program, it gets the load values and updates the graph objects each second
		while(1)
		{
			//printf("All nodes getting CPU info from the slash proc slash stat file\n");
			if(myid == 0)
			{
			node_One.GraphExpose();
                        node_Two.GraphExpose();
                        node_Three.GraphExpose();
                        node_Four.GraphExpose();
                        node_Five.GraphExpose();
                        node_Six.GraphExpose();
                        XFlush(d);
			}			
			cpu_Data.open("/proc/stat");

			if(!cpu_Data)
			{
				cout<<"an error has occured"<<endl;
				exit(1);
			}

			//this for loop is called to skip the first line with coorlates to the total cpu stats
			for(int index = 0; index < 11; index++)
			{
				cpu_Data>>file_Input;
			}

			//gets the data for the first core
			for(int index = 0; index < 11; index++)
			{
				cpu_Data>>file_Input;
				if((index > 0) && (index < 4))
				{
					stringstream convert_One(file_Input);
					stringstream convert_Two(file_Input);
					convert_One >> core_One_Total_Jiffies[index-1];
					convert_Two >> core_One_Work_Jiffies[index-1];
				}
				else if(index > 0)
				{
					stringstream convert_Three(file_Input);
					convert_Three >> core_One_Total_Jiffies[index-1];
				}					

			}
			//gets the data for the second core
			for(int index = 0; index < 11; index++)
                        {
                                cpu_Data>>file_Input;
                                if((index > 0) && (index < 4))
                                {
                                        stringstream convert_One(file_Input);
										stringstream convert_Two(file_Input);
                                        convert_One >> core_Two_Total_Jiffies[index-1];
                                        convert_Two >> core_Two_Work_Jiffies[index-1];
                                }
                                else if(index > 0)
                                {
                                        stringstream convert_Three(file_Input);
                                        convert_Three >> core_Two_Total_Jiffies[index-1];
                                } 

                        }
			
			cpu_Data.close();

			//sums the data
			core_One_Total_Jiffies_1 = sumArray(core_One_Total_Jiffies, 10);
			core_One_Work_Jiffies_1 = sumArray(core_One_Work_Jiffies, 3);

			core_Two_Total_Jiffies_1 = sumArray(core_Two_Total_Jiffies, 10);
			core_Two_Work_Jiffies_1 = sumArray(core_Two_Work_Jiffies, 3);

			//printf("Finished getting CPU data...Going to sleep...\n");

			sleep(1);
			

			//reads the file the second time with the same mechanics 
			cpu_Data.open("/proc/stat");

                        if(!cpu_Data)
                        {
                                cout<<"an error has occured"<<endl;
                                exit(1);
                        }
                        for(int index = 0; index < 11; index++)
						{
							cpu_Data>>file_Input;

						}
                        for(int index = 0; index < 11; index++)
                        {
                                cpu_Data>>file_Input;
                                if((index > 0) && (index < 4))
                                {
                                        stringstream convert_One(file_Input);
										stringstream convert_Two(file_Input);
                                        convert_One >> core_One_Total_Jiffies[index-1];
                                        convert_Two >> core_One_Work_Jiffies[index-1];
                                }
                                else if(index > 0)
                                {
                                        stringstream convert_Three(file_Input);
                                        convert_Three >> core_One_Total_Jiffies[index-1];
                                }

                        }

                        for(int index = 0; index < 11; index++)
                        {
                                cpu_Data>>file_Input;
                                if((index > 0) && (index < 4))
                                {
                                        stringstream convert_One(file_Input);
										stringstream convert_Two(file_Input);
                                        convert_One >> core_Two_Total_Jiffies[index-1];
                                        convert_Two >> core_Two_Work_Jiffies[index-1];
                                }
                                else if(index > 0)
                                {
                                        stringstream convert_Three(file_Input);
                                        convert_Three >> core_Two_Total_Jiffies[index-1];
                                }

                        }

                        cpu_Data.close();
                        core_One_Total_Jiffies_2 = sumArray(core_One_Total_Jiffies, 10);
                        core_One_Work_Jiffies_2 = sumArray(core_One_Work_Jiffies, 3);

                        core_Two_Total_Jiffies_2 = sumArray(core_Two_Total_Jiffies, 10);
                        core_Two_Work_Jiffies_2 = sumArray(core_Two_Work_Jiffies, 3);
			
			//calculations for the finished core loads
			core_One_Work_Over_Period = core_One_Work_Jiffies_2 - core_One_Work_Jiffies_1;
			core_One_Total_Over_Period = core_One_Total_Jiffies_2 - core_One_Total_Jiffies_1;

			core_Two_Work_Over_Period = core_Two_Work_Jiffies_2 - core_Two_Work_Jiffies_1;
			core_Two_Total_Over_Period = core_Two_Total_Jiffies_2 - core_Two_Total_Jiffies_1;
	
			
			//converts the amount of jiffies over one second to core load and puts that value in an array to be sent to the host
			nodeCpuValues[0] =(((double) (core_One_Work_Over_Period))/ core_One_Total_Over_Period)*100;
			nodeCpuValues[1] =(((double) (core_Two_Work_Over_Period))/ core_Two_Total_Over_Period)*100;
			
			//gets the node id so the host node knows where each load came from
			hostname.open("/etc/hostname");

        		if(!hostname)
        		{
                		cout<<"an error had occured"<<endl;
        		}

        		string node_Name;
        		string node_String = "xxx";
        		int node_Number;
        		hostname>>node_Name;
        		node_String[0] = node_Name[4];
        		node_String[1] = node_Name[5];
        		node_String[2] = node_Name[6];

        		stringstream buffer(node_String);
        		buffer >> node_Number;

        		hostname.close();
			nodeCpuValues[2] = node_Number;


	
			//gathers the data from each node and updates the graph objects
			MPI_Gather(nodeCpuValues, 3, MPI_INT, hostCpuValues,3 , MPI_INT, 0, MPI_COMM_WORLD);
		
//			printf("Completed gather\n");
			
			if (myid == 0) // Only the root MPI process is going to control the XDisplay
			{

				//matches each load value with its respective graph on the gui
				for(int outer = 2; outer < 18; outer = outer + 3)
				{
					for(int index = 0; index < 6; index++)
					{
						if(hostCpuValues[outer] == graph_List[index].node_Number)
						{
							graph_List[index].InputCoreA(hostCpuValues[outer-2]);
							graph_List[index].InputCoreB(hostCpuValues[outer-1]);
						}
					}


				}
				//tells the xserver to update the graphs with the new information that was added
				node_One.RedrawGraph();
				node_Two.RedrawGraph();
				node_Three.RedrawGraph();
				node_Four.RedrawGraph();
				node_Five.RedrawGraph();
				node_Six.RedrawGraph();

				XFlush(d);
				
			}			

		}
	}
	MPI_Finalize();

}
