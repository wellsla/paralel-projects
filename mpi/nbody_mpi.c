/*
 * nbody_mpi_dc_output.c
 *
 * Paralelização do N-Body com MPI usando Divisão & Conquista (SPMD)
 * Outputs escritos em arquivo “output-N.txt”, onde N = número de processos.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include <math.h>
 #include <mpi.h>
 #include <unistd.h>
 #include <netdb.h>
 #include <arpa/inet.h>
 
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
 
 typedef struct { double x,y,z,mass; } Particle;
 typedef struct { double xold,yold,zold,fx,fy,fz; } ParticleV;
 
 /* Protótipos das funções sequenciais fornecidas */
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
 
     /* Monta nome do arquivo de saída */
     char fname[32];
     //snprintf(fname, sizeof(fname), "output-%d.txt", procs);
     snprintf(fname, sizeof(fname), "output-teste-de-output.txt");
 
     /* Rank 0 cria/trunca o arquivo; depois todos abrem em append */
     MPI_Barrier(MPI_COMM_WORLD);
     if (rank == 0) {
         FILE *f = fopen(fname, "w");
         if (!f) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD,1); }
         fclose(f);
     }
     MPI_Barrier(MPI_COMM_WORLD);
 
     FILE *fp = fopen(fname, "a");
     if (!fp) { perror("fopen"); MPI_Abort(MPI_COMM_WORLD,2); }
 
     /* Descobre hostname e IP */
     char hostname[256];
     gethostname(hostname,256);
     struct hostent *he = gethostbyname(hostname);
     char *ip = inet_ntoa(*(struct in_addr*)he->h_addr_list[0]);
     fprintf(fp, "[PROC %d/%d] Hostname: %s, IP: %s\n",
             rank, procs, hostname, ip);
 
     /* Root lê N e S e broadcast */
     int npart, steps;
     if (rank == 0) {
         if (scanf("%d",&npart)!=1 || scanf("%d",&steps)!=1) {
             fprintf(stderr,"Erro na leitura de input\n");
             MPI_Abort(MPI_COMM_WORLD,3);
         }
     }
     MPI_Bcast(&npart, 1, MPI_INT, 0, MPI_COMM_WORLD);
     MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
 
     /* Alocação */
     Particle  *particles = malloc(sizeof(Particle)*npart);
     ParticleV *pv        = malloc(sizeof(ParticleV)*npart);
     if (!particles||!pv) MPI_Abort(MPI_COMM_WORLD,4);
 
     if (rank==0) InitParticles(particles,pv,npart);
 
     MPI_Bcast(particles, npart*sizeof(Particle),  MPI_BYTE, 0, MPI_COMM_WORLD);
     MPI_Bcast(pv,        npart*sizeof(ParticleV), MPI_BYTE, 0, MPI_COMM_WORLD);
 
     /* Cálculo de fatias */
     int base = npart / procs, rem = npart % procs;
     int start = rank*base + (rank<rem ? rank : rem);
     int local_n = base + (rank<rem ? 1 : 0);
     int end = start + local_n - 1;
 
     MPI_Barrier(MPI_COMM_WORLD);
     double t0 = MPI_Wtime();
 
     /* Loop de simulação */
     for(int step=0; step<steps; step++) {
         MPI_Bcast(particles, npart*sizeof(Particle),  MPI_BYTE, 0, MPI_COMM_WORLD);
         MPI_Bcast(pv,        npart*sizeof(ParticleV), MPI_BYTE, 0, MPI_COMM_WORLD);
 
         fprintf(fp, "[PROC %d] Recebeu partículas[%d..%d], calculando forças\n",
                 rank, start, end);
         double local_max = ComputeForces(
             &particles[start], particles, &pv[start], local_n
         );
         fprintf(fp, "[PROC %d] Enviando força máxima local=%.6f ao ROOT\n",
                 rank, local_max);
 
         double global_max;
         MPI_Reduce(&local_max, &global_max, 1,
                    MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
 
         if (rank==0) {
             fprintf(fp, "[ROOT] Força máxima global=%.6f\n", global_max);
             for(int p=1; p<procs; p++) {
                 int s = p*base + (p<rem ? p : rem);
                 int ln = base + (p<rem ? 1 : 0);
                 fprintf(fp, "[ROOT] Recebendo forças de PROC %d [%d..%d]\n",
                         p, s, s+ln-1);
                 MPI_Recv(&pv[s], ln*sizeof(ParticleV), MPI_BYTE,
                          p, 99, MPI_COMM_WORLD, &status);
             }
             ComputeNewPos(particles,pv,npart,global_max);
         } else {
             fprintf(fp, "[PROC %d] Enviando forças calculadas [%d..%d]\n",
                     rank, start, end);
             MPI_Send(&pv[start], local_n*sizeof(ParticleV),
                      MPI_BYTE, 0, 99, MPI_COMM_WORLD);
         }
         MPI_Barrier(MPI_COMM_WORLD);
     }
 
     double t1 = MPI_Wtime();
 
     if (rank==0) {
         double t_par = t1 - t0;
         /* Métricas de desempenho */
         fprintf(fp, "\n=== Métricas de desempenho ===\n");
         fprintf(fp, "Tempo de execução paralelo: %.6f s\n", t_par);
         fprintf(fp, "Tempo de execução sequencial: [execute o binário sequencial]\n");
         fprintf(fp, "Eficiência        : T_seq / (T_par * %d)\n", procs);
         fprintf(fp, "Custo computacional: T_par * %d = %.6f s-process\n", procs, t_par*procs);
         fprintf(fp, "Granularidade     : (razão Cálculo/Comunicação)\n");
         fprintf(fp, "Escalabilidade    : (analise qualitativa)\n");
         fprintf(fp, "Facilidade Prog.  : (nível de complexidade)\n");
 
         /* Espaçamento antes da matriz */
         fprintf(fp, "\n\n");
 
         /* Saída padrão do N-Body (matriz de posições) */
         for(int i=0; i<npart; i++) {
             fprintf(fp, "%.5lf %.5lf %.5lf\n",
                     particles[i].x,
                     particles[i].y,
                     particles[i].z);
         }
     }
 
     fclose(fp);
     MPI_Finalize();
     free(particles);
     free(pv);
     return 0;
 }
 
 /* ——— Funções sequenciais originais ——— */
 
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
         for(int j=0;j<n;j++){
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
         pv[i].xold = xi; pv[i].yold = yi;
         pv[i].fx = pv[i].fy = 0.0;
     }
     double dt_new = 1.0/sqrt(max_f);
     if (dt_new<1e-6) dt_new=1e-6;
     if (dt_new<dt)        { dt_old=dt; dt=dt_new; }
     else if (dt_new>4*dt) { dt_old=dt; dt*=2.0; }
     return dt_old;
 }
 