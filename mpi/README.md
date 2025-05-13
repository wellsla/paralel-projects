## Código normal ##
Compilar:
```bash
  gcc arquivo.c -o arquivo -lm
```
Executar:
```bash
  ./arquivo
```
## Compilar MPI ##
Compilar:
```bash
  mpicc arquivo.c -o arquivo -lm
```
Executar:
```bash
  mpirun –np 4 arquivo
```
Arquivo hosts.txt com apenas uma máquina:
```text
192.168.9.1 slots=4
ou
192.168.9.1 slots=8
ou
192.168.9.1 slots=12
...
```
Arquivo hosts.txt com várais máquinas (4 núcleos cada):
```text
192.168.9.1
192.168.9.2
192.168.9.3
...
```
Executar com hosts.txt:
```bash
  mpirun -np 12 --hostfile hosts.txt arquivo
```