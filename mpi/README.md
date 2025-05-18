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

## MPI ##

Compilar:

```bash
  mpicc nbody_mpi.c -o nbody_mpi -lm
```

Executar:

```bash
  mpirun –np 4 nbody_mpi
```

### Arquivo hosts.txt com apenas uma máquina: ###

```text
192.168.14.201 slots=4
ou
192.168.14.201 slots=8
ou
192.168.14.201 slots=12
...
```

### Arquivo hosts.txt com várias máquinas (4 núcleos cada): ###

```text
192.168.14.201
192.168.14.202
192.168.14.203
...
```

### Executar com hosts.txt: ###

```bash
  mpirun -np 12 --hostfile hosts.txt nbody_mpi
```

## Observação importante ##

Foi tentada a execução distribuída em múltiplas máquinas no laboratório LCI, utilizando os IPs abaixo:

```
192.168.14.201
192.168.14.205
192.168.14.207
192.168.14.209
```

Entretanto, ocorreu o seguinte erro durante a tentativa:

```bash
178342@lci-16-1-lnx:~/Projects/paralel-projects/mpi$ mpirun -np 12 --hostfile hosts.txt /mpi/178342/nbody_mpi 16384 100
--------------------------------------------------------------------------
mpirun was unable to launch the specified application as it could not access
or execute an executable:

Executable: /mpi/178342/nbody_mpi
Node: 192.168.14.207

while attempting to start process rank 8.
--------------------------------------------------------------------------
8 total processes failed to start
```