/*
* Nome do Aluno: Welliton Slaviero
* Matrícula: 178342
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/time.h>

#define MODULUS 2147483647
#define MULTIPLIER 48271
#define DEFAULT 123456789

typedef struct { double x, y, z, mass; } Particle;
typedef struct { double xold, yold, zold, fx, fy, fz; } ParticleV;

static long seed = DEFAULT;
double Random(void) {
    const long Q = MODULUS / MULTIPLIER;
    const long R = MODULUS % MULTIPLIER;
    long t = MULTIPLIER*(seed % Q) - R*(seed / Q);
    seed = (t > 0 ? t : t + MODULUS);
    return ((double)seed / MODULUS);
}

void InitParticles(Particle p[], ParticleV pv[], int n) { 
    for (int i = 0; i < n; i++) {
        p[i].x = Random();
        p[i].y = Random();
        p[i].z = Random();
        p[i].mass = 1.0;
        pv[i].xold = p[i].x;
        pv[i].yold = p[i].y;
        pv[i].zold = p[i].z;
        pv[i].fx = pv[i].fy = pv[i].fz = 0.0;
    }
}

double ComputeForces(Particle local_p[], Particle all_p[], ParticleV local_pv[], int local_n, int total_n)
{
    double max_f = 0.0;
    for (int i = 0; i < local_n; i++) {
        double xi = local_p[i].x, yi = local_p[i].y;
        double fx = 0.0, fy = 0.0, rmin = 100.0;
        for (int j = 0; j < total_n; j++) {
            double rx = xi - all_p[j].x;
            double ry = yi - all_p[j].y;
            double r2 = rx*rx + ry*ry;
            if (r2 == 0.0) continue;
            if (r2 < rmin) rmin = r2;
            double r = r2 * sqrt(r2);
            fx -= all_p[j].mass * rx / r;
            fy -= all_p[j].mass * ry / r;
        }
        local_pv[i].fx += fx;
        local_pv[i].fy += fy;
        double f = sqrt(fx*fx + fy*fy) / rmin;
        if (f > max_f) max_f = f;
    }
    return max_f;
}

double ComputeNewPos(Particle p[], ParticleV pv[], int n, double max_f)
{
    static double dt_old = 0.001, dt = 0.001;
    double a0 = 2.0 / (dt * (dt + dt_old));
    double a2 = 2.0 / (dt_old * (dt + dt_old));
    double a1 = -(a0 + a2);

    for (int i = 0; i < n; i++) {
        double xi = p[i].x, yi = p[i].y;
        p[i].x = (pv[i].fx - a1*xi - a2*pv[i].xold) / a0;
        p[i].y = (pv[i].fy - a1*yi - a2*pv[i].yold) / a0;
        pv[i].xold = xi;
        pv[i].yold = yi;
        pv[i].fx = pv[i].fy = 0.0;
    }

    double dt_new = 1.0 / sqrt(max_f);
    if (dt_new < 1e-6) dt_new = 1e-6;
    if (dt_new < dt) {
        dt_old = dt;
        dt = dt_new;
    } else if (dt_new > 4.0 * dt) {
        dt_old = dt;
        dt *= 2.0;
    }
    return dt_old;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, procs, tag = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    FILE *fp = NULL;
    if (rank == 0) {
        fp = fopen("output.txt", "w");
    } else {
        fp = fopen("output.txt", "a");
    }
    if (!fp) {
        printf("Não foi possível abrir output.txt\n");
        MPI_Finalize();
        return 1;
    }

    int npart, steps;
    if (rank == 0) {
        if (argc < 3) {
            printf("Uso: %s <npart> <steps>\n", argv[0]);
            MPI_Finalize();
            return 1;
        }
        npart = atoi(argv[1]);
        steps = atoi(argv[2]);
        printf("[%s](Processo: %d) -> Lendo npart=%d steps=%d...\n", hostname, rank, npart, steps);
        for (int d = 1; d < procs; d++) {
            MPI_Send(&npart, 1, MPI_INT, d, tag, MPI_COMM_WORLD);
            MPI_Send(&steps, 1, MPI_INT, d, tag, MPI_COMM_WORLD);
            printf("[%s](Processo: %d) -> Enviou npart/steps para o processo (%d)...\n", hostname, rank, d);
        }
    } else {
        MPI_Recv(&npart,  1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&steps,  1, MPI_INT, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[%s](Processo: %d) -> Recebeu npart=%d steps=%d do processo (0) mestre...\n", hostname, rank, npart, steps);
    }

    Particle  *particles = malloc(npart * sizeof(Particle));
    ParticleV *pv = malloc(npart * sizeof(ParticleV));
    if (!particles || !pv) {
        MPI_Finalize();
        return 2;
    }

    if (rank == 0) {
        InitParticles(particles, pv, npart);
        printf("[%s](Processo: %d) -> Inicializou partículas...\n", hostname, rank);
        for (int d = 1; d < procs; d++) {
            MPI_Send(particles, npart * sizeof(Particle), MPI_BYTE, d, tag, MPI_COMM_WORLD);
            MPI_Send(pv, npart * sizeof(ParticleV), MPI_BYTE, d, tag, MPI_COMM_WORLD);
            printf("[%s](Processo: %d) -> Enviou o estado inicial para o processo (%d)...\n", hostname, rank, d);
        }
    } else {
        MPI_Recv(particles, npart * sizeof(Particle), MPI_BYTE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(pv, npart * sizeof(ParticleV), MPI_BYTE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[%s](Processo: %d) -> Recebeu o estado inicial do processo (0) mestre...\n", hostname, rank);
    }

    int base = npart / procs;
    int rem = npart % procs;
    int start = rank * base + (rank < rem ? rank : rem);
    int local_n = base + (rank < rem ? 1 : 0);
    
    double t0 = MPI_Wtime();

    for (int step = 0; step < steps; step++) {
        if (rank == 0) {
            for (int d = 1; d < procs; d++) {
                MPI_Send(particles, npart * sizeof(Particle), MPI_BYTE, d, tag, MPI_COMM_WORLD);
                MPI_Send(pv, npart * sizeof(ParticleV), MPI_BYTE, d, tag, MPI_COMM_WORLD);
                printf("[%s](Processo: %d) -> Enviou o estado do %dº passo para o processo (%d)...\n", hostname, rank, step, d);
            }
        } else {
            MPI_Recv(particles, npart * sizeof(Particle), MPI_BYTE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(pv, npart * sizeof(ParticleV), MPI_BYTE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("[%s](Processo: %d) -> Recebeu o estado de %dº passo do processo (0) mestre...\n", hostname, rank, step);
        }

        double local_max = ComputeForces(&particles[start], particles, &pv[start], local_n, npart);
        printf("[%s](Processo: %d) -> Passo %d local_max=%.6f\n", hostname, rank, step, local_max);

        double global_max;
        if (rank == 0) {
            global_max = local_max;
            for (int s = 1; s < procs; s++) {
                double tmp;
                MPI_Recv(&tmp, 1, MPI_DOUBLE, s, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                printf("[%s](Processo: %d) -> Recebeu local_max=%.6f do processo (%d)...\n", hostname, rank, tmp, s);
                if (tmp > global_max) global_max = tmp;
            }           
            for (int d = 1; d < procs; d++) {
                MPI_Send(&global_max, 1, MPI_DOUBLE, d, tag, MPI_COMM_WORLD);
                printf("[%s](Processo: %d) -> Enviou global_max=%.6f para o processo (%d)...\n", hostname, rank, global_max, d);
            }
        } else {
            MPI_Send(&local_max, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
            MPI_Recv(&global_max, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("[%s](Processo: %d) -> Recebeu global_max=%.6f do processo (0) mestre...\n", hostname, rank, global_max);
        }

        if (rank == 0) {
            for (int s = 1; s < procs; s++) {
                int st = s * base + (s < rem ? s : rem);
                int ln = base + (s < rem ? 1 : 0);
                MPI_Recv(&pv[st], ln * sizeof(ParticleV), MPI_BYTE, s, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            MPI_Send(&pv[start], local_n * sizeof(ParticleV), MPI_BYTE, 0, tag, MPI_COMM_WORLD);
        }
 
        if (rank == 0) {
            ComputeNewPos(particles, pv, npart, global_max);
        }
    }

    double t1 = MPI_Wtime();

    if (rank == 0) {  
        double t_par = t1 - t0, t_seq = 426.116928;

        printf("\n=======================================\n");
        printf("\nMétricas de desempenho:\n");
        printf("\nTempo de execução paralelo = %.6f s\n", t_par);
        printf("Tempo de execução sequencial = %.6f s\n", t_seq);
        printf("Speedup = %.6f\n", t_seq/t_par);
        printf("Eficiência = %.6f\n", t_seq/(t_par*procs));
        printf("Custo computacional = %.6f s-process\n", t_par * procs);

        for (int i = 0; i < npart; i++) {
            fprintf(fp, "%.5lf %.5lf %.5lf\n", particles[i].x, particles[i].y, particles[i].z);
        }
    }

    fclose(fp);
    free(particles);
    free(pv);
    MPI_Finalize();
    return 0;
}


