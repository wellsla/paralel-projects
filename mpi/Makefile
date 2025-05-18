# Nome do executável
EXEC = nbody_mpi

# Compilador MPI
MPICC = mpicc

# Flags de compilação
CFLAGS = -Wall -lm

# Arquivo fonte principal
SRC = $(EXEC).c

# Comando padrão (executado ao rodar apenas 'make')
all: $(EXEC)

# Regra para compilar
$(EXEC): $(SRC)
	$(MPICC) $(SRC) -o $(EXEC) $(CFLAGS)

# Limpeza de arquivos gerados
clean:
	rm -f $(EXEC)
