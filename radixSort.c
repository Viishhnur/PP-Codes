#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

int main() {
    const int size = 1024;

    // Generate random data on the host
    thrust::host_vector<int> hostData(size);
    for (int i = 0; i < size; ++i) {
        hostData[i] = rand() % 1000; // Generate random numbers
    }

    // Transfer data to the device
    thrust::device_vector<int> deviceData = hostData;

    // Use Thrust to perform radix sort on the device
    thrust::sort(deviceData.begin(), deviceData.end());

    // Transfer sorted data back to the host
    thrust::copy(deviceData.begin(), deviceData.end(), hostData.begin());

    // Print sorted data (optional)
    std::cout << "Sorted Data:\n";
    for (int i = 0; i < size; ++i) {
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
