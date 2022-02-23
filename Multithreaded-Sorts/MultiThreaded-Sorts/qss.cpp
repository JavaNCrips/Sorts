/*
Project 1:  Tour d’Algorithms: OpenMP Qualifier
Part:       Quick sort serial (qss)
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
int partition(int arr[], int low, int high);
void quickSort(int arr[], int low, int high);
void swap(int* num1, int* num2);
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

    ////Debugging: Display unsorted sorted
    std::cout << "Unsorted array: " << std::endl;
    //display(array, size); //display whole array
    display(array, 5); //display first five elements

    //Variables for tracking the time
    double start, stop, totalTime;
    start = 0;
    stop = 0;
    totalTime = 0;
    start = omp_get_wtime();

    quickSort(array, 0, size-1);

    stop = omp_get_wtime();
    totalTime = stop - start;

    ////Debugging: Display sorted array 
    std::cout << "Sorted with quick sort serial: " << std::endl;
    //display(array, size); //display whole array
    display(array, 5); //display first five elements

    printf("\nThis is the total time: %f\n", totalTime);

    // delete the heap memory
    delete[] array;
}

void display(int* array, int size)
{
    for (int i = 0; i < size; i++)
    {
        std::cout << i + 1 << ". Value: " << array[i] << std::endl;
    }
}

void swap(int* a, int* b) {
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high)
{
    int pivot = arr[high];    // pivot
    int i = (low - 1);  // Index of smaller element

    for (int j = low; j <= high - 1; j++)
    {
        // If current element is smaller than or
        // equal to pivot
        if (arr[j] <= pivot)
        {
            i++;    // increment index of smaller element
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

/* The main function that implements QuickSort
 arr[] --> Array to be sorted,
  low  --> Starting index,
  high  --> Ending index */
void quickSort(int arr[], int low, int high)
{
    if (low < high)
    {
        /* pi is partitioning index, arr[p] is now
           at right place */
        int pi = partition(arr, low, high);

        // Separately sort elements before
        // partition and after partition
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}