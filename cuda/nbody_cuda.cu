/*
Welliton Slaviero
Matrícula: 178342

Meu dispositivo: {
    CPU: AMD Ryzen 5 5600X 6-Core Processor,
    Disco: KINGSTON SNV2S1000G, Capacidade: 932 GB,
    Memória: 16384 MB, Velocidade: 2400 MHz,
    GPU: NVIDIA GeForce RTX 3060 Ti,
    Informações CUDA do meu GPU: {
        Quantidade de devices: 1
        Device 0:
        Nome do device: NVIDIA GeForce RTX 3060 Ti
        Warp Size: 32
        Número máximo de threads por bloco: 1024
        Número máximo de threads por bloco por dimensão (X, Y, Z): (1024, 1024, 64)
        Número máximo de threads por grid por dimensão (X, Y, Z): (2147483647, 65535, 65535)
        Quantidade de multiprocessadores: 38
        Número máximo de threads por multiprocessador: 1536
    }
}
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <time.h>

#define MODULUS 2147483647
#define MULTIPLIER 48271
#define DEFAULT 123456789

static long seed = DEFAULT;

double Random(void) {
    const long Q = MODULUS / MULTIPLIER;
    const long R = MODULUS % MULTIPLIER;
    long t;

    t = MULTIPLIER * (seed % Q) - R * (seed / Q);
    if (t > 0)
        seed = t;
    else
        seed = t + MODULUS;
    return ((double)seed / MODULUS);
}

typedef struct {
    double x, y, z;
    double mass;
} Particle;

typedef struct {
    double xold, yold, zold;
    double fx, fy, fz;
} ParticleV;

void InitParticles( Particle[], ParticleV [], int );
__global__ void computeForcesKernel(Particle*, ParticleV*, double*, int);
double ComputeNewPos( Particle [], ParticleV [], int, double);

// Calcular ms
double diff_ms(struct timespec start, struct timespec end) {
    return (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_nsec - start.tv_nsec) / 1e6;
}

// "MESTRE"
int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Uso: %s <numero_de_particulas> <numero_de_passos>\n", argv[0]);
        return 1;
    }

    int npart = atoi(argv[1]); // Número total de partículas
    int cnt = atoi(argv[2]); // Número de vezes em loop (passos)

    // Inicio da contagem de tempo
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Alocação de memória na CPU
    Particle *particles = (Particle *)malloc(sizeof(Particle) *npart); // Partículas
    ParticleV *pv = (ParticleV *)malloc(sizeof(ParticleV) *npart); // Velocidade das Partículas
    double *forces = (double *)malloc(sizeof(double) *npart); // Para armazenar os resultados calculados por computeForces

    InitParticles(particles, pv, npart); // Inicializa as partículas
    cudaSetDevice(0); // Para selecionar a GPU

    // Alocação de memória na GPU
    Particle *device_particles; cudaMalloc(&device_particles, sizeof(Particle) *npart); // Partículas
    ParticleV *device_pv; cudaMalloc(&device_pv, sizeof(ParticleV) *npart); // Velocidade das Partículas
    double *device_forces; cudaMalloc(&device_forces, sizeof(double) *npart); // Para armazenar os resultados calculados por computeForces

    // Primeira cópia dos dados da CPU para a GPU
    cudaMemcpy(device_particles, particles, sizeof(Particle) *npart, cudaMemcpyHostToDevice);
    cudaMemcpy(device_pv, pv, sizeof(ParticleV) *npart, cudaMemcpyHostToDevice);

    dim3 block(256); // Threads por bloco
    dim3 grid(128); // Blocos por grid

    // Inicia o listener de eventos do CUDA
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    float total_gpu_time_ms = 0.0f;

    while (cnt--) {
        // Marca o início do kernel
        cudaEventRecord(ev_start, 0);

        computeForcesKernel<<<grid,block>>>(device_particles, device_pv, device_forces, npart); // Computa forças em paralelo (kernel)
        cudaDeviceSynchronize(); // Para sincronizar o kernel com a GPU

        // marca fim e sincroniza
        cudaEventRecord(ev_stop, 0);
        cudaEventSynchronize(ev_stop);

        // Acumula o tempo gasto no kernel
        float ms;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        total_gpu_time_ms += ms;

        // Para copiar os dados da GPU para a CPU
        cudaMemcpy(forces, device_forces, sizeof(double) *npart, cudaMemcpyDeviceToHost);
        cudaMemcpy(pv, device_pv, sizeof(ParticleV) *npart, cudaMemcpyDeviceToHost);

        // Percorre os resultados calculados por computeForces para encontrar o maior valor
        double max_f;
        for (int i = 0; i < npart; i++) {
            if (forces[i] > max_f) {
                max_f = forces[i];
            }
        }

        ComputeNewPos(particles, pv, npart, max_f); // Atualiza posições sequencialmente

        // Para copiar os dados da CPU para a GPU
        cudaMemcpy(device_particles, particles, sizeof(Particle) *npart, cudaMemcpyHostToDevice);
        cudaMemcpy(device_pv, pv, sizeof(ParticleV) *npart, cudaMemcpyHostToDevice);
    }

    // Mede tempo do host e calcula os totais
    clock_gettime(CLOCK_MONOTONIC, &end);
    double total_time_ms = diff_ms(start, end);
    double total_cpu_ms = total_time_ms - total_gpu_time_ms;

    // Imprime tempos
    fprintf(stdout, "Tempos de Execução:\n", npart, cnt);
    fprintf(stdout, "----------------------------------------\n");
    fprintf(stdout, "Tempo total (host): %.3f ms\n", total_time_ms);
    fprintf(stdout, "Tempo paralelo (GPU kernel): %.3f ms\n", total_gpu_time_ms);
    fprintf(stdout, "Tempo sequencial (host): %.3f ms\n", total_cpu_ms);
    fprintf(stdout, "----------------------------------------\n");

    // Recupera posições finais
    cudaMemcpy(particles, device_particles, sizeof(Particle) *npart, cudaMemcpyDeviceToHost);
    for (int i = 0; i < npart; i++)
        fprintf(stdout, "%.5lf %.5lf %.5lf\n", particles[i].x, particles[i].y, particles[i].z);

    cudaFree(device_particles);
    cudaFree(device_pv);
    cudaFree(device_forces);
    free(particles);
    free(pv);
    free(forces);

    return 0;
}

void InitParticles(Particle particles[], ParticleV pv[], int npart) {
    for (int i = 0; i < npart; i++) {
        particles[i].x = Random();
        particles[i].y = Random();
        particles[i].z = Random();
        particles[i].mass = 1.0;
        pv[i].xold = particles[i].x;
        pv[i].yold = particles[i].y;
        pv[i].zold = particles[i].z;
        pv[i].fx = 0;
        pv[i].fy = 0;
        pv[i].fz = 0;
    }
}

// "ESCRAVO"
__global__ void computeForcesKernel(Particle *particles, ParticleV *pv, double *forces, int npart) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npart) return;

    double xi, yi, rx, ry, mj, r, fx, fy, rmin;
    rmin = 100.0;
    xi = particles[idx].x;
    yi = particles[idx].y;
    fx = 0.0;
    fy = 0.0;

    for (int j = 0; j < npart; j++) {
        rx = xi - particles[j].x;
        ry = yi - particles[j].y;
        mj = particles[j].mass;
        r = rx * rx + ry * ry;

        if (r == 0.0) continue;
        if (r < rmin) rmin = r;

        r = r * sqrt(r);

        fx -= mj * rx / r;
        fy -= mj * ry / r;
    }

    pv[idx].fx += fx;
    pv[idx].fy += fy;

    fx = sqrt(fx * fx + fy * fy) / rmin;
    forces[idx] = fx;
}

double ComputeNewPos( Particle particles[], ParticleV pv[], int npart, double max_f)
{
    int i;
    double a0, a1, a2;
    static double dt_old = 0.001, dt = 0.001;
    double dt_new;
    a0	 = 2.0 / (dt * (dt + dt_old));
    a2	 = 2.0 / (dt_old * (dt + dt_old));
    a1	 = -(a0 + a2);
    for (i=0; i<npart; i++) {
        double xi, yi;
        xi = particles[i].x;
        yi = particles[i].y;
        particles[i].x = (pv[i].fx - a1 * xi - a2 * pv[i].xold) / a0;
        particles[i].y = (pv[i].fy - a1 * yi - a2 * pv[i].yold) / a0;
        pv[i].xold = xi;
        pv[i].yold = yi;
        pv[i].fx = 0;
        pv[i].fy = 0;
    }
    dt_new = 1.0/sqrt(max_f);
    if (dt_new < 1.0e-6) dt_new = 1.0e-6;
    if (dt_new < dt) {
        dt_old = dt;
        dt = dt_new;
    } else if (dt_new > 4.0 * dt) {
        dt_old = dt;
        dt *= 2.0;
    }
    return dt_old;
}
