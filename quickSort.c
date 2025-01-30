#include <stdio.h>
#include <cuda_runtime.h>

__device__ void swap(int *arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}

// Lomuto partition scheme
__device__ int partition(int *arr, int low, int high) {
    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(arr, i, j);
        }
    }

    swap(arr, i + 1, high);
    return i + 1;
}

// QuickSort kernel
__global__ void quickSort(int *arr, int low, int high, int depth) {
    if (depth <= 0 || low >= high) {
        return;
    }

    int pivotIdx = partition(arr, low, high);

    // Launch new CUDA threads for recursion
    if (low < pivotIdx) {
        quickSort<<<1, 1>>>(arr, low, pivotIdx - 1, depth - 1);
    }
    if (pivotIdx < high) {
        quickSort<<<1, 1>>>(arr, pivotIdx + 1, high, depth - 1);
    }
}

// Host function to launch CUDA kernel
void sortArray(int *arr, int n) {
    int *d_arr;
    size_t size = n * sizeof(int);

    cudaMalloc((void **)&d_arr, size);
    cudaMemcpy(d_arr, arr, size, cudaMemcpyHostToDevice);

    quickSort<<<1, 1>>>(d_arr, 0, n - 1, 10); // Launch with recursion depth 10
    cudaDeviceSynchronize();

    cudaMemcpy(arr, d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
}

int main() {
    int n;
    scanf("%d", &n);
    int arr[n];

    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    sortArray(arr, n);

    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    
    return 0;
}
