## Código normal ##

Compilar:
```bash
  gcc nbody.c -o nbody -lm
````

Arquivo input.txt:

```
16384
100
```

Executar:

```bash
  ./nbody < input.txt
```

## CUDA ##

Compilar:

```bash
  nvcc nbody_cuda.cu -o nbody_cuda --fmad=false -lm
```

Executar:

```bash
  ./nbody_cuda 16384 100
```