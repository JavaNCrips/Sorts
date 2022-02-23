/*
Project 1:  Tour d’Algorithms: OpenMP Qualifier
Part:       Bubble sort serial (bbs)
Course:     CS-516 Computer Architecture
Instructor: Dr. McKenny
Name:      Brandon Hudson
Semester:   Fall 2021
*/

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <omp.h>

void bubbleSort(int* array, int size);
void swapping(int& a, int& b);
void display(int* array, int size);
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

    //Debugging: Display unsorted array
    std::cout << "Unsorted Array:" << std::endl;
    //display(array, size); //display whole array
    display(array, 5); //display first five elements

    //Variables for tracking the time
    double start, stop, totalTime;
    start = 0;
    stop = 0;
    totalTime = 0;
    start = omp_get_wtime();

    bubbleSort(array, size);

    stop = omp_get_wtime();
    totalTime = stop - start;

    //Debugging: Display sorted array
    std::cout << "Sorted with Bubblesort serial:" << std::endl;
    //display(array, size); //display whole array
    display(array, 5); //display first five elements

    printf("\nThis is the total time: %f\n", totalTime);

    // delete the heap memory
    delete[] array;
}

void bubbleSort(int* array, int size) 
{
    for (int i = 0; i < size; i++)
    {
        int swaps = 0; //flag to detect any swap is there or not
        for (int j = 0; j < size - i - 1; j++)
        {
            if (array[j] > array[j + 1]) //when the current item is bigger than next
            { 
                swapping(array[j], array[j + 1]);
                swaps = 1; //set swap flag
            }
        }
        if (!swaps)
        {
            break; // No swap in this pass, so array is sorted
        }
    }
}

void swapping(int& a, int& b) //swap the content of a and b
{      
    int temp;
    temp = a;
    a = b;
    b = temp;
}

void display(int* array, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << i+1 << ". Value: " << array[i] << std::endl;
    }
}
