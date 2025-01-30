% % cu
#include <stdio.h>
#include <cuda_runtime.h>

        __global__ void
        add(int *d, int *e, int *f)
{
    int x = blockIdx.x;
    int y = blockIdx.y;
    int id = gridDim.x * y + x;
    f[id] = d[id] + e[id];
}
int main()
{
    int a[2][3] = {{1, 2, 3}, {4, 5, 6}}, b[2][3] = {{1, 2, 3}, {4, 5, 6}}, c[2][3], *d, *e, *f;
    cudaMalloc((void **)&d, 6 * sizeof(int));
    cudaMalloc((void **)&e, 6 * sizeof(int));
    cudaMalloc((void **)&f, 6 * sizeof(int));
    cudaMemcpy(d, &a, 6 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(e, &b, 6 * sizeof(int), cudaMemcpyHostToDevice);
    dim3 grid(3, 2);
    add<<<grid, 1>>>(d, e, f);
    cudaMemcpy(&c, f, 6 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            printf("%d\t", c[i][j]);
    printf("\n");
    cudaFree(d);
    cudaFree(e);
    cudaFree(f);
    cudaDeviceSynchronize();
    return 0;
}