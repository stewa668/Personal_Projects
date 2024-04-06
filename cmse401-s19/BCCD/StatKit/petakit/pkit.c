#include "pkit.h"
#include <unistd.h>

double getCPUTime()
{
    struct rusage ru;
    getrusage(RUSAGE_SELF,&ru);
    struct timeval utim = ru.ru_utime;
    struct timeval stim = ru.ru_stime;
    double cputime = (double) utim.tv_sec + (double) utim.tv_usec / 1000000.0;
    cputime += (double) stim.tv_sec + (double) stim.tv_usec / 1000000.0;
    return cputime;
}

double getTime()
{
	double time;
        struct timeval stime;
        gettimeofday(&stime,NULL);
        time = stime.tv_usec;
        time /= 1000000.0;
        time += stime.tv_sec;
	return time;
}

void startTimer()
{
	petakit_start_time = getTime();
}

//extra output is added as char* NAME  <type> architecture
void printStats( const char* program, int threads, const char* architecture, long long int problem_size, const char* version, double cputime, short custom_argc, ...){
	va_list args;
	double time = getTime() - petakit_start_time;
	short i;
	char* valueS;
	long long int valueI;
	long double valueD;
	char hostname[MAX_HOSTNAME_LENGTH];

	gethostname(hostname,MAX_HOSTNAME_LENGTH);

#ifdef __cplusplus
	std::cout << BEGIN_BUFFER
		<< "PROGRAM\t\t\t: " <<program<<"\n"
		<< "HOSTNAME\t\t: " <<hostname<<"\n"
		<< "THREADS\t\t\t: " <<threads<<"\n"
		<< "ARCH\t\t\t: " <<architecture<<"\n"
		<< "PROBLEM_SIZE\t\t: " << problem_size << "\n"
		<< "VERSION\t\t\t: " <<version<<"\n"
		<< "CPUTIME\t\t\t: " <<cputime<<"\n"
		<< "TIME\t\t\t: " <<time<<"\n";
	
	va_start(args,custom_argc);
	for (i = 0; i < custom_argc; i++){
		char* tag = va_arg(args,char*);
		char type = tag[0];
		char* label = (char*) malloc (sizeof (tag)-1);
		memcpy(label,tag+1,strlen(tag));
		switch (type) {
		case 's': /*string*/
			valueS = va_arg(args,char*);
			std::cout << label <<"\t\t: " << valueS <<"\n";
			break;
		case 'i': /*integer*/
			valueI = va_arg(args,long long int);
			std::cout << label <<"\t\t: " << valueI <<"\n";
			break;
		case 'd': /*decimal*/
			valueD = va_arg(args,long double);
			std::cout << label <<"\t\t: " << valueD <<"\n";
			break;
		}
		free (label);
	}
	std::cout << END_BUFFER;

#else
	fprintf(stdout, "%s", BEGIN_BUFFER);
	fprintf(stdout, "%-20s: %s\n", "PROGRAM", program);
	fprintf(stdout, "%-20s: %s\n", "HOSTNAME", hostname);
	fprintf(stdout, "%-20s: %d\n", "THREADS", threads);
	fprintf(stdout, "%-20s: %s\n", "ARCH", architecture);
	fprintf(stdout, "%-20s: %lld\n", "PROBLEM_SIZE", problem_size);
	fprintf(stdout, "%-20s: %s\n", "VERSION", version);
	fprintf(stdout, "%-20s: %.8g\n", "CPUTIME",cputime);
	fprintf(stdout, "%-20s: %.4g\n", "TIME", time);
	
	va_start(args,custom_argc);
	for (i = 0; i < custom_argc; i++){
		char* tag = va_arg(args,char*);
		char type = tag[0];
		char* label = (char*) malloc (sizeof (tag)-1);
		memcpy(label,tag+1,strlen(tag));
		switch (type) {
		case 's': /*string*/
			valueS = va_arg(args,char*);
			fprintf(stdout, "%-20s: %s\n", label, valueS);
			break;
		case 'i': /*integer*/
			valueI = va_arg(args,long long int);
			fprintf(stdout, "%-20s: %lld\n", label, valueI);
			break;
		case 'd': /*decimal*/
			valueD = va_arg(args,long double);
			fprintf(stdout, "%-20s: %.10Lg\n", label, valueD);
			break;
		}
		free (label);
	}
	
	fprintf(stdout, "%s", END_BUFFER);
#endif
	return;
}

#ifdef TEST
int main(){
	startTimer();
	sleep(1);
	printStats("program","hostname",4,"architecture",40000,"1.1",2.2, 3, "sMALADIES", "malapropism", "iTERRAFORMING",(long long int) 21, "dMARKET_SHARE", (long double) 2.2);
}
#endif
