/*
Project 1:  Tour d’Algorithms: OpenMP Qualifier
Part:       STL sort algorithm for reference
Course:     CS-516 Computer Architecture
Instructor: Dr. McKenny
Group:      Brandon Hudson, Sam Schrader, Jan-Niklas Harders
Semester:   Fall 2021
*/

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <omp.h>

void display(int* array, int size);
int compare_ints(const void* a, const void* b);
int* randNumArray(const int size, const int seed);

// create an array of length size of random numbers
// returns a pointer to the array
// seed: seeds the random number generator
int* randNumArray(const int size, const int seed)
{
    srand(seed);
    int* array = new int[size];
    for (int i = 0; i < size; i++)
    {
        array[i] = std::rand() % 1000000;
    }
    return array;
}

int main(int argc, char** argv)
{
    int* array; // the poitner to the array of rands
    int size, seed; // values for the size of the array
    // and the seed for generating
    // random numbers
    // check the command line args
    if (argc < 3)
    {
        std::cerr << "usage: "
            << argv[0]
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


    std::cout <<"Unsorted array:" << std::endl;
    //display(array, size); //display whole array
    display(array, 5); //display first five elements

    //Variables for tracking the time
    double start, stop, totalTime;
    start = 0;
    stop = 0;
    totalTime = 0;
    start = omp_get_wtime();

    //add stl sort algorithm here
    qsort(array, size, sizeof(int), compare_ints);

    stop = omp_get_wtime();
    totalTime = stop - start;

    std::cout << "Sorted array:" << std::endl;
    //display(array, size); //display whole array
    display(array, 5); //display first five elements

    printf("\nThis is the total time: %f\n", totalTime);

    // delete the heap memory
    delete[] array;
}

// Comparison function for cstdlib sort
// Adapted from cppreference.com/w/c/algorithm/qsort (documentation on cstdlib sort)
int compare_ints(const void* a, const void* b)
{
    int arg1 = *(const int*)a; // Cast to const int pointer then dereference to cast to int
    int arg2 = *(const int*)b;

    if (arg1 < arg2) return -1;
    if (arg1 > arg2) return 1;
    return 0;
}

void display(int* array, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << i + 1 << ". Value: " << array[i] << std::endl;
    }
}