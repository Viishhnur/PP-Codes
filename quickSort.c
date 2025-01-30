#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define MAX_DEPTH 5  // Adjust depth for recursion control

__device__ int partition(int* array, int left, int right) {
    int pivot = array[right];
    int i = left - 1;

    for (int j = left; j < right; j++) {
        if (array[j] <= pivot) {
            i++;
            // Swap elements
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }
    // Swap pivot into correct position
    int temp = array[i + 1];
    array[i + 1] = array[right];
    array[right] = temp;

    return i + 1;
}

__global__ void quicksort(int* array, int left, int right, int depth) {
    if (depth <= 0 || left >= right) {
        // Use insertion sort for small chunks or deep recursion
        for (int i = left + 1; i <= right; i++) {
            int j = i;
            while (j > left && array[j - 1] > array[j]) {
                int temp = array[j];
                array[j] = array[j - 1];
                array[j - 1] = temp;
                j--;
            }
        }
        return;
    }

    int pivotIndex = partition(array, left, right);

    // Launch recursive kernel calls
    if (left < pivotIndex) {
        quicksort<<<1, 1>>>(array, left, pivotIndex - 1, depth - 1);
    }
    if (pivotIndex < right) {
        quicksort<<<1, 1>>>(array, pivotIndex + 1, right, depth - 1);
    }
}

int main() {
    const int arraySize = 1000;  
    const int arrayBytes = arraySize * sizeof(int);
    
    int* hostArray = new int[arraySize];
    int* deviceArray;

    // Initialize array with random values
    for (int i = 0; i < arraySize; i++) {
        hostArray[i] = rand() % 1000;
    }

    // Allocate memory on GPU and copy data
    cudaMalloc((void**)&deviceArray, arrayBytes);
    cudaMemcpy(deviceArray, hostArray, arrayBytes, cudaMemcpyHostToDevice);

    // Launch QuickSort kernel
    quicksort<<<1, 1>>>(deviceArray, 0, arraySize - 1, MAX_DEPTH);
    cudaDeviceSynchronize();

    // Copy sorted array back to host
    cudaMemcpy(hostArray, deviceArray, arrayBytes, cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] hostArray;
    cudaFree(deviceArray);

    return 0;
}
