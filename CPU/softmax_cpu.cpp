#include <stdio.h>
#include <unistd.h>
#include <getopt.h>
#include <string>

string PRECISION = "FP16";
int N 100;
int REPS 10000;


int main(int argc, char *argv[])
{
    int opt;
    int option_index = 0;

    // Define the long options array
    static struct option long_options[] = {
        {"precision", required_argument, 0, 'p'},
        {"n",         required_argument, 0, 'n'},
        {"reps",      required_argument, 0, 'r'},
        {0, 0, 0} // Sentinel to mark the end of the array
    };

    // getopt_long takes the short options string as the third argument.
    // Include short equivalents (like -f, -p, -n, -r) if desired.
    while((opt = getopt_long(argc, argv, "f:p:n:r:", long_options, &option_index)) != -1)
    {
        switch(opt)
        {
            case 'p': // Triggered by --precision or -p
                printf("option precision: %s\n", optarg);
		PRECISION = 
                break;
            case 'n': // Triggered by --n or -n
                printf("option n: %s\n", optarg);
                break;
            case 'r': // Triggered by --reps or -r
                printf("option reps: %s\n", optarg);
                break;
            case '?':
                printf("unknown option\n");
                break;
            default:
		printf("no arguments were given\n");
                break;
        }
    }


//softmax();
    return 0;
}
