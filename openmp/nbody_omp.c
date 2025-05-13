/*

Discente: Welliton Slaviero & Matrícula: 178342

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <omp.h>

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

void InitParticles(Particle particles[], ParticleV pv[], int npart);
double ComputeForces(Particle myparticles[], Particle others[], ParticleV pv[], int npart);
double ComputeNewPos(Particle particles[], ParticleV pv[], int npart, double max_f);
double ComputeForces_omp(Particle myparticles[], Particle others[], ParticleV pv[], int npart);
double ComputeNewPos_omp(Particle particles[], ParticleV pv[], int npart, double max_f);

int main(int argc, char *argv[]) {
  Particle *particles;
  ParticleV *pv;
  int npart, i, j;
  int cnt, initial_cnt;
  double sim_t_s = 0.0, sim_t_p = 0.0;
  double start_time, end_time, duration_s, duration_p;
  int num_threads = 0;

  if (argc != 4) {
    printf("Uso: %s <Número de Threads> <Número de Partículas> <Número de Repetições>\n", argv[0]);   
    return 0;
  }

  num_threads = atoi(argv[1]);
  npart = atoi(argv[2]);
  initial_cnt = atoi(argv[3]);
  if (num_threads == 2 || num_threads == 4 || num_threads == 8 || num_threads == 16) {
    omp_set_num_threads(num_threads);    
  } else {
    printf("Número de threads precisa ser 2, 4, 8 ou 16\n");    
    return 0;
  }
  if (npart <= 0 || npart > 32768) {
    printf("Número de partículas precisa estar entre 1 e 32768\n");   
    return 0;
  }
  if (initial_cnt <= 0 || initial_cnt > 100) {
    printf("Número de repetições precisa estar entre 1 e 100\n");    
    return 0;
  }

  FILE *fp = fopen("output.txt", "w");
  if (fp == NULL) {
    printf("Erro ao abrir o arquivo de output para escrita\n");
    return 0;
  }

  particles = (Particle *)malloc(sizeof(Particle) * npart);
  pv = (ParticleV *)malloc(sizeof(ParticleV) * npart);

  seed = DEFAULT;
  InitParticles(particles, pv, npart);
  cnt = initial_cnt;

  start_time = omp_get_wtime();

  while (cnt--) {
    double max_f;
    max_f = ComputeForces(particles, particles, pv, npart);
    sim_t_s += ComputeNewPos(particles, pv, npart, max_f);
  }

  end_time = omp_get_wtime();
  duration_s = end_time - start_time;

  fprintf(fp, "Análise Sequencial:\n");
  fprintf(fp,"- Duração: %.6f segundos\n\n", duration_s);
  
  seed = DEFAULT;
  InitParticles(particles, pv, npart);
  cnt = initial_cnt;

  start_time = omp_get_wtime();

  while (cnt--) {
    double max_f;   
    max_f = ComputeForces_omp(particles, particles, pv, npart);
    sim_t_p += ComputeNewPos_omp(particles, pv, npart, max_f);
  }

  end_time = omp_get_wtime();
  duration_p = end_time - start_time;
  fprintf(fp, "Análise Paralela:\n");
  fprintf(fp,"- Duração utilizando %d threads: %.6f segundos\n\n", num_threads, duration_p);    

  if (duration_p > 0) { 
    fprintf(fp, "Speedup: %.4f\n\n", duration_s / duration_p);
  }

  fprintf(fp, "Coordenadas de %d partícula(s) em %d repetição(ões):\n", npart, initial_cnt);
  for (int i = 0; i < npart; i++) {
    fprintf(fp, "%.5lf %.5lf %.5lf\n", particles[i].x, particles[i].y, particles[i].z);
  }

  fclose(fp);

  printf("Resultados salvos em output.txt\n");
  printf("Execução finalizada\n");

  return 0;
}

void InitParticles(Particle particles[], ParticleV pv[], int npart) 
{
  int i;
  for (i = 0; i < npart; i++) {
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

double ComputeForces(Particle myparticles[], Particle others[], ParticleV pv[], int npart) 
{
  double max_f = 0.0;
  int i;
  for (i = 0; i < npart; i++) {
    int j;
    double xi, yi, mi, rx, ry, mj, r, fx, fy, rmin;
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
    double current_force_magnitude = sqrt(fx * fx + fy * fy) / rmin;
    if (current_force_magnitude > max_f) max_f = current_force_magnitude;
  }
  return max_f;
}

double ComputeForces_omp(Particle myparticles[], Particle others[], ParticleV pv[], int npart) 
{
  double max_f = 0.0;
  int i;
  #pragma omp parallel for reduction(max : max_f)
  for (i = 0; i < npart; i++) {
    int j;
    double xi, yi, mi, rx, ry, mj, r, fx, fy, rmin;
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
    double current_force_magnitude = sqrt(fx * fx + fy * fy) / rmin;
    if (current_force_magnitude > max_f) max_f = current_force_magnitude;
  }
  return max_f;
}

double ComputeNewPos(Particle particles[], ParticleV pv[], int npart, double max_f) 
{
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
  dt_new = 1.0 / sqrt(max_f);
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

double ComputeNewPos_omp(Particle particles[], ParticleV pv[], int npart, double max_f) 
{
  int i;
  double a0, a1, a2;
  static double dt_old = 0.001, dt = 0.001;
  double dt_new; 
  a0 = 2.0 / (dt * (dt + dt_old));
  a2 = 2.0 / (dt_old * (dt + dt_old));
  a1 = -(a0 + a2);
  #pragma omp parallel for
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
  dt_new = 1.0 / sqrt(max_f);
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