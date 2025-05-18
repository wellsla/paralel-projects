#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h> // Adicionado para medir o tempo

#define MODULUS 2147483647
#define MULTIPLIER 48271
#define DEFAULT 123456789

static long seed = DEFAULT;

double Random(void)
{
  const long Q = MODULUS / MULTIPLIER;
  const long R = MODULUS % MULTIPLIER;
  long t;

  t = MULTIPLIER * (seed % Q) - R * (seed / Q);
  if (t > 0) 
    seed = t;
  else 
    seed = t + MODULUS;
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

void InitParticles( Particle[], ParticleV [], int );
double ComputeForces( Particle [], Particle [], ParticleV [], int );
double ComputeNewPos( Particle [], ParticleV [], int, double);

int main()
{
    clock_t start_time, end_time;
    double elapsed_time;

    start_time = clock();  // Início da medição

    Particle  * particles;
    ParticleV * pv;
    int npart, cnt;
    double sim_t;
    int tmp;

    tmp = scanf("%d", &npart);
    tmp = scanf("%d", &cnt);

    particles = (Particle *) malloc(sizeof(Particle) * npart);
    pv = (ParticleV *) malloc(sizeof(ParticleV) * npart);

    InitParticles(particles, pv, npart);
    sim_t = 0.0;

    while (cnt--) {
        double max_f;
        max_f = ComputeForces(particles, particles, pv, npart);
        sim_t += ComputeNewPos(particles, pv, npart, max_f);
    }

    for (int i = 0; i < npart; i++) {
        printf("%.5lf %.5lf %.5lf\n", particles[i].x, particles[i].y, particles[i].z);
    }

    end_time = clock();
    elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nTempo de execução: %.6f segundos\n", elapsed_time);

    free(particles);
    free(pv);

    return 0;
}
