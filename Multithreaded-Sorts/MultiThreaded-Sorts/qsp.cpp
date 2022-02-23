/*
Project 1:  Tour d’Algorithms: OpenMP Qualifier
Part:       Quick sort parallel (qsp)
Course:     CS-516 Computer Architecture
Instructor: Dr. McKenny
Name:      Brandon Hudson
Semester:   Fall 2021
*/

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>


void swap(int* num1, int* num2);
void quickSort(int arr[], int low, int high);
int partition(int arr[], int low, int high);
int parallel_partition(int arr[], int low, int high);
void parallel_quickSort(int arr[], int low, int high);
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

	double start, stop, totalTime;

	////Debugging: Display unsorted sorted
	std::cout << "Unsorted array: " << std::endl;
	//display(array, size); //display whole array
	display(array, 5); //display first five elements

	start = 0;
	stop = 0;
	totalTime = 0;
	start = omp_get_wtime();

	parallel_quickSort(array, 0, size - 1);

	stop = omp_get_wtime();

	totalTime = (stop - start);

	////Debugging: Display sorted array 
	std::cout << "Sorted with quick sort parallel: " << std::endl;
	//display(array, size); //display whole array
	display(array, 5); //display first five elements

	printf("\nThis is the Total Time(parallel): %f", totalTime);

	return 0;
}

void swap(int* num1, int* num2)
{
	int temp = *num1;
	*num1 = *num2;
	*num2 = temp;
}

int parallel_partition(int arr[], int low, int high)
{
	int pivot = arr[high];    // pivot
	int i = (low - 1);  // Index of smaller element

#pragma omp parallel for
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
void parallel_quickSort(int arr[], int low, int high)
{
	if (low < high)
	{
		/* pi is partitioning index, arr[p] is now
		   at right place */
		int pi = partition(arr, low, high);

		// Separately sort elements before
		// partition and after partition
#pragma omp parallel sections
		{
#pragma omp section
			quickSort(arr, low, pi - 1);
#pragma omp section
			quickSort(arr, pi + 1, high);

		}
	}
}

void display(int* array, int size)
{
	for (int i = 0; i < size; i++)
	{
		std::cout << i + 1 << ". Value: " << array[i] << std::endl;
	}
}

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
