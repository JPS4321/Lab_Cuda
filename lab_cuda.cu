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

__global__ void Fase2(unsigned int* brillos, int estrellasPorGalaxia) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx =  bid * blockDim.x + tid;
    // Buffer for output
    unsigned int x = (unsigned int)(tid + bid * blockDim.x);
    x = (x ^ (x >> 11)) * 2654435761u;
    x ^= x >> 15;
    unsigned int brillo = x % 10u;
    __syncthreads();

    brillos[idx] = brillo;  
//     if (tid == 0){
//         printf("# === Galaxia %d === #\n", bid);
//     }
//     printf("\tEstrella %d - brillo %d!\n", tid, brillo);
}

__global__ void Fase3(unsigned int* brillos, float* brillos_prom, int estrellasPorGalaxia) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx =  bid * blockDim.x + tid;
    // Buffer for output
    unsigned int x = (unsigned int)(tid + bid * blockDim.x);
    x = (x ^ (x >> 11)) * 2654435761u;
    x ^= x >> 15;
    unsigned int brillo = x % 10u;

    __shared__ unsigned int tem_brillos[256];  // máximo 256 hilos por bloque
    tem_brillos[tid] = brillo;
    brillos[idx] = brillo;

    __syncthreads();
    if(tid == 0){
        unsigned int suma = 0;
        for (int i = 0; i < blockDim.x; i++) {
            suma += tem_brillos[i];
        }

        float avg = (float)suma / blockDim.x;
        brillos_prom[bid] = avg;
    }

}


#define NUM_GALAXIAS 2
#define ESTRELLAS_POR_GALAXIA 10
#define STRIDE 512
int main() {
    
    // // # ==== fase 1 ===== #
    // printf("Fase 1: Generacion de galaxias y estrellas\n");
    // Fase1<<<NUM_GALAXIAS, ESTRELLAS_POR_GALAXIA>>>();
    // cudaDeviceSynchronize();


    // // # ==== fase 2 ===== #
    // printf("\n\nFase 2: Generacion de galaxias y estrellas\n");

    // // Alocacion de memoria
    // unsigned int* d_brillos, * h_brillos;
    // cudaMalloc(
    //     &d_brillos, 
    //     NUM_GALAXIAS * ESTRELLAS_POR_GALAXIA * sizeof(unsigned int)
    // );
    // h_brillos = (unsigned int*)malloc(NUM_GALAXIAS * ESTRELLAS_POR_GALAXIA * sizeof(unsigned int));

    // // Llamada a kernel
    // Fase2<<<NUM_GALAXIAS,  ESTRELLAS_POR_GALAXIA>>>(d_brillos, ESTRELLAS_POR_GALAXIA);

    // // Almacenamiento de resultados
    // cudaMemcpy(h_brillos, d_brillos, NUM_GALAXIAS * ESTRELLAS_POR_GALAXIA * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // // Impresion de resultados
    // for (int b = 0; b < NUM_GALAXIAS; b++) {
    //     printf("# === Galaxia %d completa === #\n", b);
    //     for (int t = 0; t < ESTRELLAS_POR_GALAXIA; t++) {
    //         printf("\tEstrella %d - Brillo %d\n", t, h_brillos[b * ESTRELLAS_POR_GALAXIA + t]);
    //     }
    // }
    // // Liberacion de memoria
    // cudaFree(d_brillos);
    // free(h_brillos);


    // # ==== fase 3 ===== #

    unsigned int* d_brillos;
    float* d_brillo_promedio;

    unsigned int* h_brillos;
    float* h_brillo_promedio;

    size_t totalBrillos = NUM_GALAXIAS * ESTRELLAS_POR_GALAXIA * sizeof(unsigned int);
    cudaMalloc(&d_brillos, totalBrillos);
    h_brillos = (unsigned int*)malloc(totalBrillos);

    size_t totalProm = NUM_GALAXIAS * sizeof(float);
    cudaMalloc(&d_brillo_promedio, totalProm);
    h_brillo_promedio = (float*)malloc(totalProm);

    // launch kernel with shared memory = estrellasPorGalaxia * sizeof(unsigned int)
    Fase3<<<NUM_GALAXIAS, ESTRELLAS_POR_GALAXIA,
            ESTRELLAS_POR_GALAXIA * sizeof(float)>>>(
        d_brillos,
        d_brillo_promedio,
        ESTRELLAS_POR_GALAXIA
    );

    cudaDeviceSynchronize();

    // copy results
    cudaMemcpy(h_brillos, d_brillos, totalBrillos, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_brillo_promedio, d_brillo_promedio, totalProm, cudaMemcpyDeviceToHost);

    // print
    for (int b = 0; b < NUM_GALAXIAS; b++) {
        printf("# === Galaxia %d | Brillo Promedio %.2f === #\n", b, h_brillo_promedio[b]);
        for (int t = 0; t < ESTRELLAS_POR_GALAXIA; t++) {
            printf("\tEstrella %d - Brillo %d\n",
                t, h_brillos[b * ESTRELLAS_POR_GALAXIA + t]);
        }
    }

    // free
    cudaFree(d_brillos);
    cudaFree(d_brillo_promedio);
    free(h_brillos);
    free(h_brillo_promedio);


    return 0;
}   