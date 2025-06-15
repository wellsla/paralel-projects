## Nome do Aluno: Welliton Slaviero  ##
### Matrícula: 178342 ###
	  
### Ambiente de execução: ###

    CPU: AMD Ryzen 5 5600X 6-Core Processor
    Disco: KINGSTON SNV2S1000G
    Capacidade: 932 GB
    Memória: 16GB DDR4 2400MHz
    GPU: NVIDIA RTX 3060 Ti (38 MPs, 4864 cores) 8GB GDDR6 448 GB/s bandwidth
    Sistema: Windows 11 + WSL2 Ubuntu 22.04
    CUDA: Versão 12.6
    Informações CUDA do GPU:       
        - Quantidade de devices: 1        
        - Device 0:        
        - Nome do device: NVIDIA GeForce RTX 3060 Ti        
        - Warp Size: 32        
        - Número máximo de threads por bloco: 1024        
        - Número máximo de threads por bloco por dimensão (X, Y, Z): (1024, 1024, 64)        
        - Número máximo de threads por grid por dimensão (X, Y, Z): (2147483647, 65535,65535) 
        - Quantidade de multiprocessadores: 38       
        - Número máximo de threads por multiprocessador: 1536
### Implementação: ###

	Algoritmo: Mestre-Escravo adaptado para CUDA  
	Mestre (CPU): criação e distribuição de tarefas, coleta e agregação de resultados  
	Escravos (GPU Threads): processamento de tarefas individuais  
	Comunicação: transferência estruturada de tarefas e resultados  
	Balanceamento: divisão automática de carga entre workers  
	Sincronização: barreira entre fases de distribuição e coleta  

### Compilar: ###

```bash
  nvcc nbody_cuda.cu -o nbody_cuda --fmad=false -lm
```

### Executar: ###

```bash
  ./nbody_cuda 16384 100
```

### Makefile: ###

```bash
  make help
```