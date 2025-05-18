EXEC = nbody_mpi
SRC = $(EXEC).c
MPICC = mpicc
CFLAGS = -lm
NP ?= 4
ARGS = 16384 100

all: $(EXEC)

# COMPILAR
$(EXEC): $(SRC)
	$(MPICC) $(SRC) -o $(EXEC) $(CFLAGS)

# EXECUTAR COM QUANTIDADE DE PROCESSOS
run: $(EXEC)
	mpirun -np $(NP) ./$(EXEC) $(ARGS)

# EXECUTAR COM HOSTFILE
hostrun: $(EXEC)
	mpirun -np $(NP) --hostfile hosts.txt ./$(EXEC) $(ARGS)

# TIRAR O EXECUTÁVEL
clean:
	rm -f $(EXEC)