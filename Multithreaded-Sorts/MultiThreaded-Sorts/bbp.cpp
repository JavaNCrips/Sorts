/*
Project 1:  Tour d’Algorithms: OpenMP Qualifier
Part:       Bubble sort parallel (bbp)
Course:     CS-516 Computer Architecture
Instructor: Dr. McKenny
Name:      Brandon Hudson
Semester:   Fall 2021
*/

#include <math.h>
#include <omp.h>

#include <cstdlib>
#include <iostream>
#include <sstream>

// create an array of length size of random numbers
// returns a pointer to the array
// seed: seeds the random number generator

int* randNumArray(const int size, const int seed);
void parallel_bubble_sort(int* array, int size);
void insertion_sort(int* array, int array_size);
void swapping(int& a, int& b);
void display(int* array, int size);

int* randNumArray(const int size, const int seed) {
    srand(seed);
    int* array = new int[size];
    for (int i = 0; i < size; i++) {
        array[i] = std::rand() % 1000000;
    }
    return array;
}

int main(int argc, char** argv) {
    int* array;      // the poitner to the array of rands
    int size, seed;  // values for the size of the array
    // and the seed for generating
    // random numbers
    // check the command line args
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
            << " [amount of random nums to generate] [seed value for rand]"
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
    // get the random numbers
    array = randNumArray(size, seed);
    // **************************
    // **************************
    // **************************

    //Debugging: Display sorted array
    std::cout << "Unsorted:" << std::endl;
    //display(array, size); //display whole array
    display(array, 5); //display first five elements

    double start, stop, totalTime;
    start = 0;
    stop = 0;
    totalTime = 0;
    start = omp_get_wtime();

    // display(array, size);

#pragma omp parallel
    { parallel_bubble_sort(array, size); }

    // Now each thread's subset is sorted
    // e.g. for 5 threads, thread 0 has sorted every 5th element
    // array[0] <= array[5] <= array[10] and so on
    // However, the array as a whole is not fully sorted, so we perform insertion
    // sort to finalize things (Insertion sort performs near-linear on
    // nearly-sorted data)

    insertion_sort(array, size);
    // display(array, size);

    stop = omp_get_wtime();
    totalTime = stop - start;

    //Debugging: Display sorted array
    std::cout << "Sorted with Bubblesort parallel :" << std::endl;
    //display(array, size); //display whole array
    display(array, 5); //display first five elements

    printf("\nThis is the total time: %f\n", totalTime);

    // delete the heap memory
    delete[] array;
}

// Run by each thread. Bubble sort is performed over a subset of array elements
// determined by thread number.
void parallel_bubble_sort(int* array, int array_size) {
    int num_threads = omp_get_num_threads();
    int thread_num = omp_get_thread_num();
    int subset_size =
        std::ceil(float(array_size) / num_threads);  // Divide work evenly
    int* subset_indices = new int[subset_size];

    for (int i = 0; i < subset_size; i++) {
        subset_indices[i] = (num_threads * i) + thread_num;
        // Interleaved accesses with (almost) equal workload for all threads
        // e.g. with 10 threads, each thread gets every tenth item, offset by its
        // thread num
    }

    if (subset_indices[subset_size - 1] > (array_size - 1))
        subset_size--;  // Some threads may try to access past the end of the array
                        // on their last item This would be illegal and nobody wants
                        // the police to get involved So we decrement subset_size

      // Perform bubble sort over the subset of the array
      // subset_indices is used to map from this standard bubble sort implementation
      // to non-contiguous array elements
    for (int i = 0; i < subset_size; i++) {
        int swaps = 0;  // flag to detect any swap is there or not
        for (int j = 0; j < subset_size - i - 1; j++) {
            if (array[subset_indices[j]] > array[subset_indices[j + 1]])
                // when the current item is bigger than next
            {
                swapping(array[subset_indices[j]], array[subset_indices[j + 1]]);
                swaps = 1;  // set swap flag
            }
        }
        if (!swaps) {
            break;  // No swap in this pass, so array is sorted
        }
    }
}

void swapping(int& a, int& b)  // swap the content of a and b
{
    int temp;
    temp = a;
    a = b;
    b = temp;
}

void display(int* array, int size) {
    for (int i = 0; i < size; i++) {
        std::cout << i + 1 << " " << array[i] << std::endl;
    }
}


// Insertion sort
// A sorted list is maintained at the front of the array
// The first item in the unsorted list is swapped with adjacent items until it
// is in its proper place in the sorted array
void insertion_sort(int* array, int array_size) {
    for (int i = 0; i < array_size; i++) {
        int j = i;
        while ((j > 0) && (array[j - 1] > array[j])) {
            swapping(array[j], array[j - 1]);
            j--;
        }
    }
}
