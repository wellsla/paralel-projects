PARAMS := \
    16384:100:v1 16384:100:v2 16384:100:v3 16384:100:v4 16384:100:v5 \
    5000:400:v1 5000:400:v2 5000:400:v3 5000:400:v4 5000:400:v5 \
    1200:200:v1 1200:200:v2 1200:200:v3 1200:200:v4 1200:200:v5 \
    600:100:v1 600:100:v2 600:100:v3 600:100:v4 600:100:v5 \
    300:50:v1 300:50:v2 300:50:v3 300:50:v4 300:50:v5 \
    4:10:v1 4:10:v2 4:10:v3 4:10:v4 4:10:v5

ifdef CUDA
    PROGRAM := nbody_cuda
    SOURCE := nbody_cuda.cu
    COMPILER := nvcc
    COMPILE_FLAGS := --fmad=false -lm
    TYPE := CUDA
    OUTDIR := out_cuda
else
    PROGRAM := nbody
    SOURCE := nbody.c
    COMPILER := gcc
    COMPILE_FLAGS := -lm
    TYPE := CPU
    OUTDIR := out_cpu
endif

.DEFAULT_GOAL := help

.PHONY: run-all
run-all: compile $(OUTDIR)
	@echo "Iniciando execuções $(TYPE)..."
	@for param in $(PARAMS); do \
		N=$$(echo $$param | cut -d':' -f1); \
		P=$$(echo $$param | cut -d':' -f2); \
		V=$$(echo $$param | cut -d':' -f3); \
		echo "Executando $(TYPE): N=$$N, P=$$P, V=$$V"; \
		./$(PROGRAM) $$N $$P > $(OUTDIR)/particulas_$${N}_passos_$${P}_$${V}.out; \
		echo "Concluído"; \
	done
	@echo "Todas as execuções $(TYPE) foram concluídas!"

.PHONY: run
run: compile $(OUTDIR)
	@if [ -z "$(N)" ] || [ -z "$(P)" ] || [ -z "$(V)" ]; then \
		echo "Erro: Use 'make run N=valor P=valor V=valor [CUDA=1]'"; \
		echo "Exemplos:"; \
		echo "  make run N=1000 P=500 V=v1            # Versão CPU"; \
		echo "  make run N=1000 P=500 V=v1 CUDA=1     # Versão CUDA"; \
		exit 1; \
	fi
	@echo "Executando versão $(TYPE) com N=$(N), P=$(P), V=$(V)"
	@./$(PROGRAM) $(N) $(P) > $(OUTDIR)/particulas_$(N)_passos_$(P)_$(V).out
	@echo "✓ Arquivo gerado: $(OUTDIR)/particulas_$(N)_passos_$(P)_$(V).out"

.PHONY: compile
compile: $(PROGRAM)

$(PROGRAM): $(SOURCE)
	@echo "Compilando versão $(TYPE)..."
	@$(COMPILER) $(SOURCE) -o $(PROGRAM) $(COMPILE_FLAGS)
	@echo "✓ Compilação $(TYPE) concluída"

$(OUTDIR):
	@mkdir -p $(OUTDIR)

.PHONY: build-cpu
build-cpu:
	@echo "Compilando versão CPU..."
	@gcc nbody.c -o nbody -lm
	@echo "✓ nbody (CPU) compilado com sucesso"

.PHONY: build-cuda
build-cuda:
	@echo "Compilando versão CUDA..."
	@nvcc nbody_cuda.cu -o nbody_cuda --fmad=false -lm
	@echo "✓ nbody_cuda (CUDA) compilado com sucesso"

.PHONY: build-both
build-both: build-cpu build-cuda
	@echo "✓ Ambas as versões compiladas"

.PHONY: clean
clean:
	@echo "Removendo arquivos..."
	@rm -f nbody nbody_cuda
	@rm -rf out_cpu out_cuda
	@echo "✓ Limpeza concluída"

.PHONY: clean-output
clean-output:
	@echo "Removendo arquivos de saída..."
	@rm -rf out_cpu out_cuda
	@echo "✓ Arquivos de saída removidos"

.PHONY: clean-bin
clean-bin:
	@echo "Removendo executáveis..."
	@rm -f nbody nbody_cuda
	@echo "✓ Executáveis removidos"

.PHONY: show-params
show-params:
	@echo "Parâmetros definidos ($(words $(PARAMS)) execuções):"
	@for param in $(PARAMS); do \
		N=$$(echo $$param | cut -d':' -f1); \
		P=$$(echo $$param | cut -d':' -f2); \
		V=$$(echo $$param | cut -d':' -f3); \
		echo "  N=$$N, P=$$P, V=$$V -> $(OUTDIR)/particulas_$${N}_passos_$${P}_$${V}.out"; \
	done

.PHONY: help
help:
	@echo "Makefile para simulação N-Body (CPU ou CUDA)"
	@echo ""
	@echo "Targets principais:"
	@echo "  run-all         - Compila e executa todas as configurações"
	@echo "  run             - Executa uma configuração específica"
	@echo ""
	@echo "Targets de compilação:"
	@echo "  build-cpu       - Compila apenas a versão CPU"
	@echo "  build-cuda      - Compila apenas a versão CUDA"
	@echo "  build-both      - Compila ambas as versões"
	@echo ""
	@echo "Targets de limpeza:"
	@echo "  clean           - Remove executáveis e pastas de saída"
	@echo "  clean-output    - Remove apenas pastas de saída"
	@echo "  clean-bin       - Remove apenas executáveis"
	@echo ""
	@echo "Targets utilitários:"
	@echo "  show-params     - Mostra lista de execução atual"
	@echo "  help            - Mostra esta ajuda"
	@echo ""
	@echo "Exemplos de uso:"
	@echo "  make run-all                       # CPU -> out_cpu/"
	@echo "  make run-all CUDA=1                # CUDA -> out_cuda/"
	@echo "  make run N=1000 P=500 V=v1         # CPU - execução única (N=particulas, P=passos, V=versão/índice de execução)"
	@echo "  make run N=1000 P=500 V=v1 CUDA=1  # CUDA - execução única (N=particulas, P=passos, V=versão/índice de execução)"
	@echo "  make show-params                   # Ver lista de execução atual"