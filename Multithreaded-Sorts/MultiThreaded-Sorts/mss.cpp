/*
Project 1:  Tour d’Algorithms: OpenMP Qualifier
Part:       Merge sort serial (mss)
Course:     CS-516 Computer Architecture
Instructor: Dr. McKenny
Group:      Brandon Hudson, Sam Schrader, Jan-Niklas Harders
Semester:   Fall 2021
*/

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <omp.h>

int* randNumArray(const int size, const int seed);
void merge(int* array, int left, int middle, int right);
void mSort(int* array, int left, int right);
void display(int* array, int size);


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

	//Debugging Deisplay unsorted array
	std::cout << "Unsorted Array:" << std::endl;
	//display(array, size); //display whole array
	display(array, 5); //display first five elements

	//Variables for tracking the time
	double start, stop, totalTime;
	start = 0;
	stop = 0;
	totalTime = 0;
	start = omp_get_wtime();

	mSort(array, 0, size - 1);

	stop = omp_get_wtime();
	totalTime = stop - start;

	//Display sorted Array
	std::cout << "Sorted with serial merge sort: " << std::endl;
	//display(array, size); //display whole array
	display(array, 5); //display first five elements

	printf("\nThis is the total time: %f\n", totalTime);

	// delete the heap memory
	delete[] array;
}

void merge(int* array, int left, int middle, int right)
{
	// Create temp arrays for sorting
	// The merge sort algorithm can be done in-place, but it is much trickier to do efficiently than simply creating "workspace" arrays

	int leftSize = middle - left + 1; // Left array contains the elements between array[left] and array[middle], inclusive.
	int rightSize = right - middle; // Right array contains the elements between array[middle + 1] and array[right], inclusive.

	int* leftArr = new int[leftSize];
	int* rightArr = new int[rightSize];

	std::copy(&array[left], &array[middle + 1], leftArr);
	std::copy(&array[middle + 1], &array[right + 1], rightArr);

	// Merge subarrays -- we iterate over leftArr and rightArr, placing the next largest/equal item into the appropriate position in array
	int i = 0; // Index for accessing leftArr
	int j = 0; // Index for accessing rightArr
	int k = left; // Index for placing into array -- start at beginning of left subarray

	while ((i < leftSize) && (j < rightSize)) // While we have not finished iterating over either array
	{
		if (leftArr[i] <= rightArr[j])
		{
			// Current item in left subarray is <= current item in right subarray, so we place it in the next "open" spot in array
			array[k] = leftArr[i];
			// Move to next item in left subarray
			i++;
		}
		else
		{
			// Current item in right subarray is > current item in left subarray, so we place it in the next "open" spot in array
			array[k] = rightArr[j];
			// Move to next item in right subarray
			j++;
		}
		// Move to next "open" spot in array
		k++;
	}

	// Finish iterating through the array that still has items to copy over
	while (i < leftSize)
	{
		array[k] = leftArr[i];
		i++;
		k++;
	}

	while (j < rightSize)
	{
		array[k] = rightArr[j];
		j++;
		k++;
	}
}

// merge sort implementation called by mergeSort()
// Defined here so we can time the whole operation inside mergeSort
// Recursively called on subarrays of the whole array by changing the left and right index boundaries.
// Once we hit the base case (single-item arrays), we recursively merge these sorted arrays together with merge().
void mSort(int* array, int left, int right)
{
	// Array is not a single number
	if (left < right)
	{
		int middle = (left + right) / 2; // Integer division rounds down (floor)

		mSort(array, left, middle); // Sort left half of subarray
		mSort(array, middle + 1, right); // Sort right half of subarray

		merge(array, left, middle, right); // Merge sorted halves 
	}

}

void display(int* array, int size)
{
	for (int i = 0; i < size; i++)
	{
		std::cout << i + 1 << ". Value: " << array[i] << std::endl;
	}
}