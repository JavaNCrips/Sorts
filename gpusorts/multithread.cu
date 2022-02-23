#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
using namespace std;

/**********************************************************
* **********************************************************
* error checking stufff
***********************************************************
***********************************************************/
// Enable this for error checking
#define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )
inline void __cudaSafeCall(cudaError err,
    const char* file, const int line)
{
#ifdef CUDA_CHECK_ERROR
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
    do
    {
        if (cudaSuccess != err)
        {
            fprintf(stderr,
                "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
            exit(-1);
        }
    } while (0);
#pragma warning( pop )
#endif // CUDA_CHECK_ERROR
    return;
}
inline void __cudaCheckError(const char* file, const int line)
{
#ifdef CUDA_CHECK_ERROR
#pragma warning( push )
#pragma warning( disable: 4127 ) // Prevent warning on do-while(0);
    do
    {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr,
                "cudaCheckError() failed at %s:%i : %s.\n",
                file, line, cudaGetErrorString(err));
            exit(-1);
        }
        // More careful checking. However, this will affect performance.
        // Comment if not needed.
        err = cudaThreadSynchronize();
        if (cudaSuccess != err)
        {
            fprintf(stderr,
                "cudaCheckError() with sync failed at %s:%i : %s.\n",
                file, line, cudaGetErrorString(err));
            exit(-1);
        }
    } while (0);
#pragma warning( pop )
#endif // CUDA_CHECK_ERROR
    return;
}
/***************************************************************
* **************************************************************
* end of error checking stuff
****************************************************************
***************************************************************/
// function takes an array pointer, and the number of rows and cols in the array, and
// allocates and intializes the array to a bunch of random numbers
// Note that this function creates a 1D array that is a flattened 2D array
// to access data item data[i][j], you must can use data[(i*rows) + j]
int* makeRandArray(const int size, const int seed) {
    srand(seed);
    int* array = new int[size];
    for (int i = 0; i < size; i++) {
        array[i] = std::rand() % 1000000;
    }
    return array;
}

void printArray(int* array, int size)
{
    for (int i = 0; i < size; i++)
        cout << array[i] << endl;
}

//*******************************//
// your kernel here!!!!!!!!!!!!!!!!!
//*******************************//

//function found in and deduced from: https://github.com/master-hpc/mp-generic-bubble-sort
// device function that swaps two integers of an array if called 
__device__ void swap(int* a, int* b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

//function found in and deduced from: https://github.com/master-hpc/mp-generic-bubble-sort
__global__ void bubbleSort(int* array, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    // Algorithm works as follows:
    // [1] The elements of the array are considered as pairs (array[i] and array[i+1]
    // [2] 1. Iteration: i is even, so the pairs will look like: {array[0], array[1]}, {array[2], array[3]}, ... 
    // [3] Each independent pair will be sorted ascendantly in its own thread (if input / 2 < max threads even simultanously)
    // [4] 2. Iteration: i is odd, so the pairs will be like: {array[1], array[2]}, {array[3], array[4]}, ...
    // [5] After n (= the number of elements of the array) even/odd iterations the array will be sorted (only if 1/2 size of the array < max Threads of devide)
    // [6] elements without neighbours won't be considered in each iteration (for example the first element in the odd iterations)
    for (int i = 0; i < n; i++)
    {
        int offset = i % 2;
        int left = 2 * id + offset; //left index
        int right = left + 1; //right index

        if (right < n)
        {
            if (array[left] > array[right])
            {
                swap(&array[left], &array[right]);
            }
        }      
        __syncthreads(); //
    }
}

int main(int argc, char* argv[])
{
    int* array; // the pointer to the array of rands
    int size, seed; // values for the size of the array
    bool printSorted = false;
    // and the seed for generating
    // random numbers
    // check the command line args
    if (argc < 4) {
        std::cerr << "usage: "
            << argv[0]
            << " [amount of random nums to generate] [seed value for rand]"
            << " [1 to print sorted array, 0 otherwise]"
            << std::endl;
        exit(-1);
    }
    // convert cstrings to ints
    {
        std::stringstream ss1(argv[1]);
        ss1 >> size;
    }
    {
        std::stringstream ss1(argv[2]);
        ss1 >> seed;
    }
    {
        int sortPrint;
        std::stringstream ss1(argv[3]);
        ss1 >> sortPrint;
        if (sortPrint == 1)
            printSorted = true;
    }
    // get the random numbers
    array = makeRandArray(size, seed);
    /***********************************
    * create a cuda timer to time execution
    **********************************/
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal, 0);
    /***********************************
    * end of cuda timer creation
    **********************************/
    /////////////////////////////////////////////////////////////////////
/////////////////////// YOUR CODE HERE ///////////////////////
/////////////////////////////////////////////////////////////////////

    // 1) allocate device memory
    int* d_array; //device array
    CudaSafeCall(cudaMalloc(&d_array, size * sizeof(int)));
    CudaCheckError();
    CudaSafeCall(cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice));
    CudaCheckError();

    //2) set up the grid and block sizes
    dim3 grdDim; 
    dim3 blkDim;

    //before setting setting the grid and block size, we need to determine the maximum values
    //Cuda offers built in functions for determining the properties of the divice(s) on the system (source: http://tdesell.cs.und.edu/lectures/cuda_2.pdf on page 28)
    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0); // 0 = device count, so this code only works with 1 gpu installed
    int maxThreads = dev_prop.maxThreadsDim[0]; // dev_prop.maxThreadsDim[0] = maximum block x dimension
    if (size / 2 < maxThreads)
    {
        blkDim = dim3(size / 2, 1, 1);
        grdDim = dim3(1, 1, 1); // threads < 1024 fit in one block
        //in a parallel bubblesort each thread can sort an individual pair.
        //hence we best use size / 2 number of threads which leads to O(n) time complexity in optimal case
    }
    else // if 1/2 of the input size is greater than maximum threads / block of the device we set the number to its maximum
    {
        blkDim = dim3(maxThreads, 1, 1); // dev_prop.maxThreadsDim[0] is typically and on the hardware tested 1024
        grdDim = dim3(ceil((size / (2.0 * maxThreads)) ), 1, 1); // and distribute the threads on several blocks
        /*
        *** HOW TO FIND A GOOD SIZE FOR THE GRID? ***
        We need minimum 1/2 n threads. (n = input size)
        Total threads = threads * blocks
        total threads = 1/2 n
        1/2 n = threads * blocks => blocks = n / (2 x threads)
        ceiling function because we need an integer that is greater than the equations results floating-point number
        */
    }

    //// 3) call kernel functions
    bubbleSort << < grdDim, blkDim >> > (d_array, size);
    CudaCheckError();

    //cudaDeviceSynchronize();
    //CudaCheckError();

    // 4) get result back from GPU
    CudaSafeCall(cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));
    CudaCheckError();

    /*
    * You need to implement your kernel as a function at the top of this file.
    * Here you must
    * 1) allocate device memory
    * 2) set up the grid and block sizes
    * 3) call your kenrnel
    * 4) get the result back from the GPU
    *
    *
    * to use the error checking code, wrap any cudamalloc functions as follows:
    * CudaSafeCall( cudaMalloc( &pointer_to_a_device_pointer,
    * length_of_array * sizeof( int ) ) );
    * Also, place the following function call immediately after you call your kernel
    * ( or after any other cuda call that you think might be causing an error )
    * CudaCheckError();
    */
    /***********************************
    * Stop and destroy the cuda timer
    **********************************/
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    /***********************************
    * end of cuda timer destruction
    **********************************/

    std::cerr << "Total time in seconds: "
        << timeTotal / 1000.0 << std::endl;
    if (printSorted)
    {
        printArray(array, size);
    }
    	
    // Release memory
    cudaFree(d_array); // on device
    free(array); // on host\

    return 0;
}