#include <stdio.h>

__global__ void Fase1() {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    unsigned int x = (unsigned int)(tid + bid * blockDim.x);
    x = (x ^ (x >> 11)) * 2654435761u;
    // 2654435761 golden ratio constant
    x ^= x >> 15;
    unsigned int brillo = x % 10u;
    printf("Galaxia %d - Estrella %d - brillo %d!\n", bid, tid, brillo);
    
}
#define NUM_GALAXIAS 2
#define ESTRELLAS_POR_GALAXIA 10
#define STRIDE 512
int main() {
    // // # ==== fase 1 ===== #
    printf("Fase 1: Generacion de galaxias y estrellas\n");
    Fase1<<<NUM_GALAXIAS, ESTRELLAS_POR_GALAXIA>>>();
    cudaDeviceSynchronize();

    return 0;
}   