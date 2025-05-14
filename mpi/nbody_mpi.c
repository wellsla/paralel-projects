/*
 * nbody_mpi_dc.c
 *
 * Paralelização do N-Body com MPI usando Divisão & Conquista (SPMD)
 * Somente utiliza chamadas MPI já vistas em aula (Init, Rank/Size, Send/Recv, Reduce, Barrier).
 * Mantém exatamente o mesmo I/O sequencial do enunciado :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include <netdb.h>
#include <arpa/inet.h>

/* pRNG fornecido pelo professor */
#define MODULUS    2147483647
#define MULTIPLIER 48271
#define DEFAULT    123456789

static long seed = DEFAULT;
double Random(void) {
    const long Q = MODULUS / MULTIPLIER;
    const long R = MODULUS % MULTIPLIER;
    long t = MULTIPLIER*(seed%Q) - R*(seed/Q);
    seed = (t>0 ? t : t+MODULUS);
    return ((double)seed / MODULUS);
}

typedef struct {
    double x,y,z;
    double mass;
} Particle;
typedef struct {
    double xold,yold,zold;
    double fx,fy,fz;
} ParticleV;

/* Protótipos das funções sequenciais já fornecidas */
void InitParticles( Particle p[], ParticleV pv[], int n );
double ComputeForces( Particle local_p[], Particle all_p[],
                      ParticleV local_pv[], int local_n );
double ComputeNewPos( Particle p[], ParticleV pv[], int n, double max_f );

int main(int argc, char **argv) {
    int rank, procs;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    /* Descobrir e imprimir IP do host */
    char hostname[256];
    gethostname(hostname,256);
    struct hostent *he = gethostbyname(hostname);
    char *ip = inet_ntoa(*(struct in_addr*)he->h_addr_list[0]);
    printf("[PROC %d/%d] Host IP: %s\n", rank, procs, ip);

    /* Root lê N e S */
    int npart, steps;
    if (rank==0) {
        if (scanf("%d",&npart)!=1 || scanf("%d",&steps)!=1) {
            fprintf(stderr,"Erro na leitura de input\n");
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
    /* Envia N e S para todos */
    MPI_Bcast(&npart, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Aloca arrays */
    Particle  *particles = malloc(sizeof(Particle)*npart);
    ParticleV *pv        = malloc(sizeof(ParticleV)*npart);
    if (!particles||!pv) MPI_Abort(MPI_COMM_WORLD,2);

    /* Root inicializa partículas */
    if (rank==0) {
        InitParticles(particles,pv,npart);
    }
    /* Sincroniza e replica estado inicial */
    MPI_Bcast(particles, npart*sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pv,        npart*sizeof(ParticleV), MPI_BYTE, 0, MPI_COMM_WORLD);

    /* Cálculo de fatias para divisão */
    int base = npart / procs;
    int rem  = npart % procs;
    int start = rank*base + (rank<rem ? rank : rem);
    int local_n = base + (rank<rem ? 1 : 0);
    int end = start + local_n - 1;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Loop de simulação */
    for(int step=0; step<steps; step++) {
        /* Broadcast do estado atual */
        if (rank==0) {
            /* root já tem particles/pv atualizados */
        }
        MPI_Bcast(particles, npart*sizeof(Particle),  MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Bcast(pv,        npart*sizeof(ParticleV), MPI_BYTE, 0, MPI_COMM_WORLD);

        printf("[PROC %d] Computando forças para particulas [%d..%d]\n",
               rank, start, end);

        /* Cada processo calcula forças na sua fatia */
        double local_max = ComputeForces(
            &particles[start], particles, &pv[start], local_n
        );

        /* Reduz para obter força máxima global */
        double global_max;
        MPI_Reduce(&local_max, &global_max, 1,
                   MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        /* Root coleta forças locais (campo pv) de cada escravo */
        if (rank==0) {
            for(int p=1; p<procs; p++) {
                int s = p*base + (p<rem ? p : rem);
                int ln = base + (p<rem ? 1 : 0);
                printf("[ROOT] Recebendo forças de PROC %d [%d..%d]\n",
                       p, s, s+ln-1);
                MPI_Recv(&pv[s], ln*sizeof(ParticleV), MPI_BYTE,
                         p, /*tag=*/99, MPI_COMM_WORLD, &status);
            }
            /* Root também já atualizou pv em sua fatia */
            /* Agora atualiza posições em todo o vetor */
            ComputeNewPos(particles,pv,npart,global_max);
        } else {
            /* Escravo envia somente sua fatia de pv (forças) */
            printf("[PROC %d] Enviando forças calculadas [%d..%d]\n",
                   rank, start, end);
            MPI_Send(&pv[start], local_n*sizeof(ParticleV),
                     MPI_BYTE, 0, /*tag=*/99, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    double t1 = MPI_Wtime();
    if (rank==0) {
        double t_par = t1 - t0;
        /* Impressão final das posições — idêntico ao sequencial */
        for(int i=0;i<npart;i++){
            printf("%.5lf %.5lf %.5lf\n",
                   particles[i].x,
                   particles[i].y,
                   particles[i].z);
        }
        /* Métricas de desempenho */
        printf("=== Métricas de desempenho ===\n");
        printf("Tempo de execução paralelo: %.6f s\n", t_par);
        printf("Tempo de execução sequencial: [execute nbody sequencial]\n");
        printf("Eficiência        : speedup / p = T_seq / (T_par * %d)\n", procs);
        printf("Custo computacional: T_par * %d = %.6f s-process\n", procs, t_par*procs);
        printf("Granularidade     : razão Cálculo/Comunicação (a medir)\n");
        printf("Escalabilidade    : (análise qualitativa)\n");
        printf("Facilidade Prog.  : (média de complexidade de implementação)\n");
    }

    MPI_Finalize();
    free(particles);
    free(pv);
    return 0;
}

/* ——— Funções fornecidas pelo enunciado ——— */

void InitParticles( Particle particles[], ParticleV pv[], int npart ) {
    for(int i=0;i<npart;i++){
        particles[i].x    = Random();
        particles[i].y    = Random();
        particles[i].z    = Random();
        particles[i].mass = 1.0;
        pv[i].xold = particles[i].x;
        pv[i].yold = particles[i].y;
        pv[i].zold = particles[i].z;
        pv[i].fx = pv[i].fy = pv[i].fz = 0.0;
    }
}

double ComputeForces( Particle myp[], Particle allp[],
                      ParticleV pv[], int n ) {
    double max_f = 0.0;
    for(int i=0;i<n;i++){
        double xi = myp[i].x, yi = myp[i].y;
        double fx=0, fy=0, rmin=1e6;
        for(int j=0;j<n; j++){
            double rx = xi - allp[j].x;
            double ry = yi - allp[j].y;
            double r2 = rx*rx + ry*ry;
            if (r2==0) continue;
            if (r2<rmin) rmin=r2;
            double r = r2*sqrt(r2);
            fx -= allp[j].mass * rx / r;
            fy -= allp[j].mass * ry / r;
        }
        pv[i].fx += fx;
        pv[i].fy += fy;
        double f = sqrt(fx*fx + fy*fy)/rmin;
        if (f>max_f) max_f=f;
    }
    return max_f;
}

double ComputeNewPos( Particle particles[], ParticleV pv[],
                      int npart, double max_f) {
    static double dt_old = 0.001, dt = 0.001;
    double a0 = 2.0/(dt*(dt+dt_old));
    double a2 = 2.0/(dt_old*(dt+dt_old));
    double a1 = -(a0 + a2);
    for(int i=0;i<npart;i++){
        double xi = particles[i].x, yi = particles[i].y;
        particles[i].x = (pv[i].fx - a1*xi - a2*pv[i].xold)/a0;
        particles[i].y = (pv[i].fy - a1*yi - a2*pv[i].yold)/a0;
        pv[i].xold = xi;  pv[i].yold = yi;
        pv[i].fx = pv[i].fy = 0.0;
    }
    double dt_new = 1.0/sqrt(max_f);
    if (dt_new<1e-6) dt_new=1e-6;
    if (dt_new<dt)        { dt_old=dt; dt=dt_new; }
    else if (dt_new>4*dt) { dt_old=dt; dt*=2.0; }
    return dt_old;
}
