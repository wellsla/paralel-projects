/*
Nome do Aluno: Welliton Slaviero
Matrícula: 178342

Ambiente de execução:
    CPU: AMD Ryzen 5 5600X 6-Core Processor
    Disco: KINGSTON SNV2S1000G
    Capacidade: 932 GB
    Memória: 16GB DDR4 2400MHz
    GPU: NVIDIA RTX 3060 Ti (38 MPs, 4864 cores) 8GB GDDR6 448 GB/s bandwidth
    Sistema: Windows 11 + WSL2 Ubuntu 22.04
    CUDA: Versão 12.6
    Informações CUDA do GPU:
        - Quantidade de devices: 1
        - Device 0:
        - Nome do device: NVIDIA GeForce RTX 3060 Ti
        - Warp Size: 32
        - Número máximo de threads por bloco: 1024
        - Número máximo de threads por bloco por dimensão (X, Y, Z): (1024, 1024, 64)
        - Número máximo de threads por grid por dimensão (X, Y, Z): (2147483647, 65535,65535)
        - Quantidade de multiprocessadores: 38
        - Número máximo de threads por multiprocessador: 1536
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define MODULUS 2147483647
#define MULTIPLIER 48271
#define DEFAULT 123456789

static long seed = DEFAULT;

double Random(void) {
    const long Q = MODULUS / MULTIPLIER;
    const long R = MODULUS % MULTIPLIER;
    long t = MULTIPLIER * (seed % Q) - R * (seed / Q);
    seed = (t > 0 ? t : t + MODULUS);
    return ((double)seed / MODULUS);
}

typedef struct {
    double x, y, z, mass;
} Particle;

typedef struct {
    double xold, yold, zold;
    double fx, fy, fz;
} ParticleV;

__global__ void slaveKernel(Particle*, Particle*, ParticleV*, double*, int);
double masterSlaveComputeForces(Particle*, ParticleV*, int);
void InitParticles(Particle[], ParticleV[], int);
double ComputeNewPos(Particle[], ParticleV[], int, double);

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Input correto: %s <NUMERO_DE_PARTICULAS> <NUMERO_DE_PASSOS>\n", argv[0]);
        return 1;
    }

    int npart = atoi(argv[1]);
    int cnt = atoi(argv[2]);

    Particle *particles;
    ParticleV *pv;
    double sim_t;

    particles = (Particle *) malloc(sizeof(Particle)*npart);
    pv = (ParticleV *) malloc(sizeof(ParticleV)*npart);

    InitParticles(particles, pv, npart);
    sim_t = 0.0;

    cudaSetDevice(0);

    while (cnt--) {
        double max_f;
        max_f = masterSlaveComputeForces(particles, pv, npart);
        sim_t += ComputeNewPos(particles, pv, npart, max_f);
    }

    for (int i = 0; i < npart; i++) {
        fprintf(stdout, "%.5lf %.5lf %.5lf\n", particles[i].x, particles[i].y, particles[i].z);
    }

    free(particles);
    free(pv);
    return 0;
}

// KERNEL "ESCRAVO"
__global__ void slaveKernel(Particle *myparticles, Particle *others, ParticleV *pv, double *forces, int npart) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npart) return;

    int j;
    double xi, yi, rx, ry, mj, r, fx, fy, rmin;

    rmin = 100.0;
    xi = myparticles[i].x;
    yi = myparticles[i].y;
    fx = 0.0;
    fy = 0.0;

    for (j = 0; j < npart; j++) {
        rx = xi - others[j].x;
        ry = yi - others[j].y;
        mj = others[j].mass;
        r = rx * rx + ry * ry;
        if (r == 0.0) continue;
        if (r < rmin) rmin = r;
        r = r * sqrt(r);
        fx -= mj * rx / r;
        fy -= mj * ry / r;
    }

    pv[i].fx += fx;
    pv[i].fy += fy;
    fx = sqrt(fx*fx + fy*fy)/rmin;
    forces[i] = fx;
}

// FUNÇÃO "MESTRE"
double masterSlaveComputeForces(Particle *particles, ParticleV *pv, int npart) {
    static Particle *d_particles = NULL;
    static ParticleV *d_pv = NULL;
    static double *d_forces = NULL;
    static double *h_forces = NULL;
    static int allocated_npart = 0;
    static cudaEvent_t start, stop;
    static int first_call = 1;

    if (first_call) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        first_call = 0;
    }

    if (d_particles == NULL || allocated_npart != npart) {
        if (d_particles) {
            cudaFree(d_particles);
            cudaFree(d_pv);
            cudaFree(d_forces);
            free(h_forces);
        }
        cudaMalloc(&d_particles, npart * sizeof(Particle));
        cudaMalloc(&d_pv, npart * sizeof(ParticleV));
        cudaMalloc(&d_forces, npart * sizeof(double));
        h_forces = (double*)malloc(npart * sizeof(double));
        allocated_npart = npart;
    }

    cudaMemcpy(d_particles, particles, npart * sizeof(Particle), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pv, pv, npart * sizeof(ParticleV), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (npart + threadsPerBlock - 1) / threadsPerBlock;

    slaveKernel<<<blocksPerGrid, threadsPerBlock>>>(d_particles, d_particles, d_pv, d_forces, npart);

    cudaDeviceSynchronize();

    cudaMemcpy(pv, d_pv, npart * sizeof(ParticleV), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_forces, d_forces, npart * sizeof(double), cudaMemcpyDeviceToHost);

    double max_f = 0.0;
    for (int i = 0; i < npart; i++) {
        if (h_forces[i] > max_f) max_f = h_forces[i];
    }

    return max_f;
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

double ComputeNewPos(Particle particles[], ParticleV pv[], int npart, double max_f) {
    int i;
    double a0, a1, a2;
    static double dt_old = 0.001, dt = 0.001;
    double dt_new;

    a0 = 2.0 / (dt * (dt + dt_old));
    a2 = 2.0 / (dt_old * (dt + dt_old));
    a1 = -(a0 + a2);

    for (i = 0; i < npart; i++) {
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