#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define MODULUS    2147483647
#define MULTIPLIER 48271
#define DEFAULT    123456789

static long seed = DEFAULT;

double Random(void){
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

void InitParticles(Particle particles[], ParticleV pv[], int npart){
    int i;
    for (i=0; i<npart; i++) {
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

double ComputeForces(Particle myparticles[], Particle others[], ParticleV pv[], int npart){
  double max_f = 0.0;
  for (int i=0; i<npart; i++) {
    double xi = myparticles[i].x, yi = myparticles[i].y;
    double fx = 0.0, fy = 0.0, rmin = 100.0;
    for (int j=0; j<npart; j++) {
      double rx = xi - others[j].x, ry = yi - others[j].y;
      double mj = others[j].mass, r = rx * rx + ry * ry;
      if (r == 0.0) continue;
      if (r < rmin) rmin = r;
      r  = r * sqrt(r);
      fx -= mj * rx / r;
      fy -= mj * ry / r;
    }
    pv[i].fx += fx;
    pv[i].fy += fy;
    fx = sqrt(fx*fx + fy*fy)/rmin;
    if (fx > max_f) max_f = fx;
  }
  return max_f;
}

double ComputeNewPos(Particle particles[], ParticleV pv[], int npart, double max_f){
  static double dt_old = 0.001, dt = 0.001;
  double dt_new, a0 = 2.0 / (dt * (dt + dt_old));
  double a2 = 2.0 / (dt_old * (dt + dt_old)), a1 = -(a0 + a2);

  for (int i=0; i<npart; i++) {
    double xi = particles[i].x, yi = particles[i].y;
    particles[i].x = (pv[i].fx - a1 * xi - a2 * pv[i].xold) / a0;
    particles[i].y = (pv[i].fy - a1 * yi - a2 * pv[i].yold) / a0;
    pv[i].xold = xi; pv[i].yold = yi;
    pv[i].fx = pv[i].fy = 0;
  }
  dt_new = 1.0/sqrt(max_f);
  if (dt_new < 1.0e-6) dt_new = 1.0e-6;
  if (dt_new < dt) {
    dt_old = dt; dt = dt_new;
  } else if (dt_new > 4.0 * dt) {
    dt_old = dt; dt *= 2.0;
  }
  return dt_old;
}

int main(int argc, char **argv){
  int rank, size, npart, steps, local_npart;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 3) {
    if (rank == 0) printf("Uso: %s <npart> <steps>\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  npart = atoi(argv[1]);
  steps = atoi(argv[2]);
  local_npart = npart / size;

  Particle *particles = malloc(sizeof(Particle)*npart);
  Particle *local_particles = malloc(sizeof(Particle)*local_npart);
  ParticleV *pv = malloc(sizeof(ParticleV)*local_npart);

  if(rank == 0){
    ParticleV *tmp_pv = malloc(sizeof(ParticleV)*npart);
    InitParticles(particles, tmp_pv, npart);
  }

  MPI_Scatter(particles, local_npart*sizeof(Particle), MPI_BYTE, local_particles, local_npart*sizeof(Particle), MPI_BYTE, 0, MPI_COMM_WORLD);

  for(int step=0; step<steps; step++){
    double max_f = ComputeForces(local_particles, particles, pv, local_npart);
    ComputeNewPos(local_particles, pv, local_npart, max_f);
    MPI_Allgather(local_particles, local_npart*sizeof(Particle), MPI_BYTE, particles, local_npart*sizeof(Particle), MPI_BYTE, MPI_COMM_WORLD);
  }

  if(rank == 0){
    for(int i=0; i<npart; i++){
      printf("%.5lf %.5lf %.5lf\n", particles[i].x, particles[i].y, particles[i].z);
    }
  }

  free(particles); free(local_particles); free(pv);
  MPI_Finalize();
  return 0;
}
