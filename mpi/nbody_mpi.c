#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>

#define MODULUS    2147483647
#define MULTIPLIER 48271
#define DEFAULT    123456789

static long seed = DEFAULT;

double Random(void) {
    const long Q = MODULUS / MULTIPLIER;
    const long R = MODULUS % MULTIPLIER;
    long t = MULTIPLIER * (seed % Q) - R * (seed / Q);
    seed = (t > 0) ? t : t + MODULUS;
    return ((double) seed / MODULUS);
}

typedef struct {
    double x, y, z;
    double mass;
} Particle;

typedef struct {
    double xold, yold, zold;
    double fx, fy, fz;
} ParticleV;

void InitParticles(Particle[], ParticleV[], int);
double ComputeForces(Particle[], Particle[], ParticleV[], int, int, int);
double ComputeNewPos(Particle[], ParticleV[], int, double);

int main(int argc, char** argv) {
    int rank, size;
    char hostname[256];
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    gethostname(hostname, 256);

    if (argc != 3) {
        if (rank == 0) {
            printf("Uso: %s <numero_particulas> <passos>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int npart = atoi(argv[1]);
    int steps = atoi(argv[2]);

    if (npart % size != 0) {
        if (rank == 0) {
            printf("Número de partículas deve ser múltiplo do número de processos.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int local_n = npart / size;
    Particle* all_particles = malloc(sizeof(Particle) * npart);
    ParticleV* all_pv = malloc(sizeof(ParticleV) * npart);

    if (rank == 0) {
        InitParticles(all_particles, all_pv, npart);
        printf("[Rank %d - %s] Partículas inicializadas\n", rank, hostname);
    }

    MPI_Bcast(all_particles, npart * sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_pv, npart * sizeof(ParticleV), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Início da contagem de tempo paralela
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    double sim_t = 0.0;
    for (int step = 0; step < steps; step++) {
        double local_max_f = ComputeForces(all_particles, all_particles, all_pv, npart,
                                           rank * local_n, (rank + 1) * local_n);

        double global_max_f;
        MPI_Allreduce(&local_max_f, &global_max_f, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        ComputeNewPos(all_particles, all_pv, npart, global_max_f);
        MPI_Bcast(all_particles, npart * sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // Fim da contagem de tempo
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    double exec_time = end_time - start_time;

    // Impressão final (Rank 0)
    if (rank == 0) {
        for (int i = 0; i < npart; i++) {
            printf("%.5lf %.5lf %.5lf\n", all_particles[i].x, all_particles[i].y, all_particles[i].z);
        }

        // Cálculo dos indicadores
        double tempo_sequencial = 550.0;  // Dado do enunciado
        double speedup = tempo_sequencial / exec_time;
        double eficiencia = speedup / size;
        double custo = exec_time * size;
        double grao = tempo_sequencial / size;
        double escalabilidade = speedup / size;

        printf("\n=== Indicadores de Desempenho ===\n");
        printf("Tempo de execução paralelo: %.3f s\n", exec_time);
        printf("Tempo de execução sequencial (referência): %.3f s\n", tempo_sequencial);
        printf("Speedup: %.2f\n", speedup);
        printf("Eficiência: %.2f%%\n", eficiencia * 100);
        printf("Custo computacional: %.2f\n", custo);
        printf("Grão: %.2f\n", grao);
        printf("Escalabilidade: %.2f\n", escalabilidade);
        printf("Facilidade de programação: Média (uso direto de MPI e sincronização com barreiras)\n");
    }

    free(all_particles);
    free(all_pv);
    MPI_Finalize();
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
        pv[i].fx = pv[i].fy = pv[i].fz = 0;
    }
}

double ComputeForces(Particle myp[], Particle others[], ParticleV pv[], int npart, int start, int end) {
    double max_f = 0.0;
    for (int i = start; i < end; i++) {
        double fx = 0.0, fy = 0.0, xi = myp[i].x, yi = myp[i].y;
        double rmin = 100.0;
        for (int j = 0; j < npart; j++) {
            double rx = xi - others[j].x;
            double ry = yi - others[j].y;
            double mj = others[j].mass;
            double r = rx * rx + ry * ry;
            if (r == 0.0) continue;
            if (r < rmin) rmin = r;
            r = r * sqrt(r);
            fx -= mj * rx / r;
            fy -= mj * ry / r;
        }
        pv[i].fx += fx;
        pv[i].fy += fy;
        double f = sqrt(fx * fx + fy * fy) / rmin;
        if (f > max_f) {
            max_f = f;
        }
    }
    return max_f;
}

double ComputeNewPos(Particle p[], ParticleV pv[], int npart, double max_f) {
    static double dt_old = 0.001, dt = 0.001;
    double a0 = 2.0 / (dt * (dt + dt_old));
    double a2 = 2.0 / (dt_old * (dt + dt_old));
    double a1 = -(a0 + a2);

    for (int i = 0; i < npart; i++) {
        double xi = p[i].x;
        double yi = p[i].y;
        p[i].x = (pv[i].fx - a1 * xi - a2 * pv[i].xold) / a0;
        p[i].y = (pv[i].fy - a1 * yi - a2 * pv[i].yold) / a0;
        pv[i].xold = xi;
        pv[i].yold = yi;
        pv[i].fx = pv[i].fy = 0;
    }

    double dt_new = 1.0 / sqrt(max_f);
    if (dt_new < 1.0e-6) {
        dt_new = 1.0e-6;
    }
    if (dt_new < dt) {
        dt_old = dt;
        dt = dt_new;
    } else if (dt_new > 4.0 * dt) {
        dt_old = dt;
        dt *= 2.0;
    }

    return dt_old;
}
