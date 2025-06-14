# Defina seus parâmetros aqui (formato: N:P:V separados por espaço)
PARAMS := \
    16384:100:v1 \
    16384:100:v2 \
    16384:100:v3 \
    16384:100:v4 \
    16384:100:v5 \
    5000:400:v1 \
    5000:400:v2 \
    5000:400:v3 \
    5000:400:v4 \
    5000:400:v5 \
    1200:200:v1 \
    1200:200:v2 \
    1200:200:v3 \
    1200:200:v4 \
    1200:200:v5 \
    600:100:v1 \
    600:100:v2 \
    600:100:v3 \
    600:100:v4 \
    600:100:v5 \
    300:50:v1 \
    300:50:v2 \
    300:50:v3 \
    300:50:v4 \
    300:50:v5 \
    4:10:v1 \
    4:10:v2 \
    4:10:v3 \
    4:10:v4 \
    4:10:v5

# Configuração condicional CPU vs CUDA
ifdef CUDA
    PROGRAM := ./nbody_cuda
    SOURCE := nbody_cuda.cu
    COMPILER := nvcc
    COMPILE_FLAGS := --fmad=false -lm
    TYPE := CUDA
    OUTDIR := out_cuda
else
    PROGRAM := ./nbody
    SOURCE := nbody.c
    COMPILER := gcc
    COMPILE_FLAGS := -lm
    TYPE := CPU
    OUTDIR := out_cpu
endif

# Função para extrair N, P, V de cada parâmetro
get_n = $(word 1,$(subst :, ,$(1)))
get_p = $(word 2,$(subst :, ,$(1)))
get_v = $(word 3,$(subst :, ,$(1)))

# Cria lista de arquivos de saída baseada nos parâmetros (apenas para referência)
OUTPUTS := $(foreach param,$(PARAMS),$(OUTDIR)/particulas_$(call get_n,$(param))_passos_$(call get_p,$(param))_$(call get_v,$(param)).out)

# Target principal - compila e executa todas as simulações sequencialmente
.PHONY: run-all
run-all: compile $(OUTDIR)
	@echo "Iniciando execuções $(TYPE)..."
	@for param in $(PARAMS); do \
		N=`echo $$param | cut -d':' -f1`; \
		P=`echo $$param | cut -d':' -f2`; \
		V=`echo $$param | cut -d':' -f3`; \
		echo "Executando $(TYPE): N=$$N, P=$$P, V=$$V"; \
		$(PROGRAM) $$N $$P > $(OUTDIR)/particulas_$${N}_passos_$${P}_$${V}.out; \
		echo "Concluido"; \
	done
	@echo "Todas as execuções $(TYPE) foram concluídas!"

# Target para compilar o programa
.PHONY: compile
compile: $(PROGRAM)

$(PROGRAM): $(SOURCE)
	@echo "Compilando versão $(TYPE)..."
	@echo "Comando: $(COMPILER) $(SOURCE) -o $(PROGRAM) $(COMPILE_FLAGS)"
	@$(COMPILER) $(SOURCE) -o $(PROGRAM) $(COMPILE_FLAGS)
	@echo "✓ Compilação $(TYPE) concluída"
	@echo "---"

# Cria o diretório de saída
$(OUTDIR):
	@mkdir -p $(OUTDIR)

# Regra genérica para execução individual (para o target 'run')
$(OUTDIR)/%.out: $(PROGRAM) | $(OUTDIR)
	@echo "Executando $(TYPE): $(notdir $@)"
	@FILENAME="$(notdir $@)"; \
	N=$(echo $FILENAME | sed 's/particulas_\([^_]*\)_passos_\([^_]*\)_\([^.]*\)\.out/\1/'); \
	P=$(echo $FILENAME | sed 's/particulas_\([^_]*\)_passos_\([^_]*\)_\([^.]*\)\.out/\2/'); \
	V=$(echo $FILENAME | sed 's/particulas_\([^_]*\)_passos_\([^_]*\)_\([^.]*\)\.out/\3/'); \
	echo "Parâmetros: N=$N, P=$P, V=$V"; \
	$(if $(CUDA),$(PROGRAM) $N $P > $@,$(PROGRAM) $N $P > $@)
	@echo "✓ Concluído: $@"

# Target para executar arquivo específico por nome
.PHONY: run-file
run-file: compile
	@if [ -z "$(FILE)" ]; then \
		echo "Erro: Use 'make run-file FILE=nome_do_arquivo.out [CUDA=1]'"; \
		echo "Exemplo: make run-file FILE=particulas_1000_passos_500_v1.out"; \
		exit 1; \
	fi
	@echo "Gerando arquivo específico: $(OUTDIR)/$(FILE)"
	@mkdir -p $(OUTDIR)
	@$(MAKE) $(OUTDIR)/$(FILE)
.PHONY: run
# Target para executar com parâmetros específicos
.PHONY: run
run: compile
	@if [ -z "$(N)" ] || [ -z "$(P)" ] || [ -z "$(V)" ]; then \
		echo "Erro: Use 'make run N=valor P=valor V=valor [CUDA=1]'"; \
		echo "Exemplos:"; \
		echo "  make run N=1000 P=500 V=v1            # Versão CPU"; \
		echo "  make run N=1000 P=500 V=v1 CUDA=1     # Versão CUDA"; \
		exit 1; \
	fi
	@echo "Executando versão $(TYPE) com N=$(N), P=$(P), V=$(V)"
	@mkdir -p $(OUTDIR)
	@output_file="$(OUTDIR)/particulas_$(N)_passos_$(P)_$(V).out"; \
	$(PROGRAM) $(N) $(P) > $output_file; \
	@echo "✓ Arquivo gerado: $(OUTDIR)/particulas_$(N)_passos_$(P)_$(V).out"

# Target para compilar apenas (sem executar)
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

# Target para limpar arquivos compilados e de saída
.PHONY: clean
clean:
	@echo "Removendo arquivos..."
	@rm -f nbody nbody_cuda
	@rm -rf out_cpu out_cuda_* $(OUTDIR)
	@echo "✓ Limpeza concluída"

# Target para limpar apenas arquivos de saída
.PHONY: clean-output
clean-output:
	@echo "Removendo arquivos de saída..."
	@rm -rf out_cpu out_cuda_* $(OUTDIR)
	@echo "✓ Arquivos de saída removidos"

# Target para listar todas as pastas de saída existentes
.PHONY: list-outputs
list-outputs:
	@echo "Pastas de saída existentes:"
	@ls -la | grep "^d.*out_" || echo "Nenhuma pasta de saída encontrada"

# Target para limpar pasta específica
.PHONY: clean-specific
clean-specific:
	@if [ -z "$(DIR)" ]; then \
		echo "Erro: Use 'make clean-specific DIR=nome_da_pasta'"; \
		echo "Exemplo: make clean-specific DIR=out_cuda_T512"; \
		exit 1; \
	fi
	@echo "Removendo pasta $(DIR)..."
	@rm -rf $(DIR)
	@echo "✓ Pasta $(DIR) removida"

# Target para limpar apenas executáveis
.PHONY: clean-bin
clean-bin:
	@echo "Removendo executáveis..."
	@rm -f nbody nbody_cuda
	@echo "✓ Executáveis removidos"

# Target para listar configurações
.PHONY: show-config
show-config:
	@echo "Configurações atuais:"
	@echo "Tipo: $(TYPE)"
	@echo "Programa: $(PROGRAM)"
	@echo "Código fonte: $(SOURCE)"
	@echo "Compilador: $(COMPILER) $(COMPILE_FLAGS)"
	@echo "Diretório de saída: $(OUTDIR)"
	@echo ""
	@echo "Parâmetros definidos ($(words $(PARAMS)) execuções):"
	@for param in $(PARAMS); do \
		N=$$(echo $$param | cut -d':' -f1); \
		P=$$(echo $$param | cut -d':' -f2); \
		V=$$(echo $$param | cut -d':' -f3); \
		echo "  N=$$N, P=$$P, V=$$V -> $(OUTDIR)/particulas_$${N}_passos_$${P}_$${V}.out"; \
	done

# Target para debug - mostra arquivos que serão gerados
.PHONY: debug-targets
debug-targets:
	@echo "Arquivos que serão gerados ($(words $(OUTPUTS)) arquivos):"
	@for output in $(OUTPUTS); do \
		echo "  $$output"; \
	done
	@echo ""
	@echo "Teste de parsing dos primeiros 3 parâmetros:"
	@for param in $(wordlist 1,3,$(PARAMS)); do \
		N=$$(echo $$param | cut -d':' -f1); \
		P=$$(echo $$param | cut -d':' -f2); \
		V=$$(echo $$param | cut -d':' -f3); \
		echo "  $$param -> N=$$N, P=$$P, V=$$V"; \
	done

# Target para verificar se os compiladores estão disponíveis
.PHONY: check-deps
check-deps:
	@echo "Verificando dependências..."
	@which gcc > /dev/null 2>&1 && echo "✓ gcc encontrado" || echo "✗ gcc não encontrado"
	@which nvcc > /dev/null 2>&1 && echo "✓ nvcc encontrado" || echo "✗ nvcc não encontrado"
	@[ -f nbody.c ] && echo "✓ nbody.c encontrado" || echo "✗ nbody.c não encontrado"
	@[ -f nbody_cuda.cu ] && echo "✓ nbody_cuda.cu encontrado" || echo "✗ nbody_cuda.cu não encontrado"

# Target de ajuda
.PHONY: help
help:
	@echo "Makefile para execução sequencial do nbody (CPU ou CUDA)"
	@echo ""
	@echo "Targets principais:"
	@echo "  run-all              - Compila e executa todas as configurações (CPU)"
	@echo "  run-all CUDA=1       - Compila e executa todas as configurações (CUDA)"
	@echo "  run-all OUTDIR=pasta - Usa pasta de saída customizada"
	@echo "  run                  - Executa uma configuração específica"
	@echo "  run-file             - Executa para gerar um arquivo específico"
	@echo ""
	@echo "Targets de compilação:"
	@echo "  build-cpu       - Compila apenas a versão CPU"
	@echo "  build-cuda      - Compila apenas a versão CUDA"
	@echo "  build-both      - Compila ambas as versões"
	@echo ""
	@echo "Targets de limpeza:"
	@echo "  clean           - Remove executáveis e todas as pastas de saída"
	@echo "  clean-output    - Remove apenas todas as pastas de saída"
	@echo "  clean-bin       - Remove apenas executáveis"
	@echo "  clean-specific  - Remove pasta específica (use DIR=nome)"
	@echo "  list-outputs    - Lista todas as pastas de saída existentes"
	@echo ""
	@echo "Targets utilitários:"
	@echo "  show-config     - Mostra as configurações atuais"
	@echo "  check-deps      - Verifica se compiladores e arquivos existem"
	@echo "  debug-targets   - Mostra arquivos que serão gerados (debug)"
	@echo "  help            - Mostra esta ajuda"
	@echo ""
	@echo "Exemplos de uso:"
	@echo "  make run-all                                         # CPU -> out_cpu/"
	@echo "  make run-all CUDA=1                                  # CUDA -> out_cuda/"
	@echo "  make run-all OUTDIR=meus_resultados                  # CPU -> meus_resultados/"
	@echo "  make run-all CUDA=1 OUTDIR=teste                     # CUDA -> teste/"
	@echo "  make run N=1000 P=500 V=v1 CUDA=1                    # Execução única"
	@echo "  make run-file FILE=particulas_1000_passos_500_v1.out # Arquivo específico"
	@echo "  make list-outputs                                    # Ver pastas existentes"
	@echo "  make clean-specific DIR=out_cuda_T256                # Limpar pasta específica"
	@echo "  make show-config                                     # Ver configurações"

# Target padrão
.DEFAULT_GOAL := help

# Target para compilar o programa
.PHONY: compile
compile: $(PROGRAM)

$(PROGRAM): $(SOURCE)
	@echo "Compilando versão $(TYPE)..."
	@echo "Comando: $(COMPILER) $(SOURCE) -o $(PROGRAM) $(COMPILE_FLAGS)"
	@$(COMPILER) $(SOURCE) -o $(PROGRAM) $(COMPILE_FLAGS)
	@echo "✓ Compilação $(TYPE) concluída"
	@echo "---"

# Cria o diretório de saída
$(OUTDIR):
	@mkdir -p $(OUTDIR)

# Regra genérica para execução individual (para o target 'run')
$(OUTDIR)/%.out: $(PROGRAM) | $(OUTDIR)
	@echo "Executando $(TYPE): $(notdir $@)"
	@FILENAME="$(notdir $@)"; \
	N=$(echo $FILENAME | sed 's/particulas_\([^_]*\)_passos_\([^_]*\)_\([^.]*\)\.out/\1/'); \
	P=$(echo $FILENAME | sed 's/particulas_\([^_]*\)_passos_\([^_]*\)_\([^.]*\)\.out/\2/'); \
	V=$(echo $FILENAME | sed 's/particulas_\([^_]*\)_passos_\([^_]*\)_\([^.]*\)\.out/\3/'); \
	echo "Parâmetros: N=$N, P=$P, V=$V"; \
	$(if $(CUDA),$(PROGRAM) $N $P > $@,$(PROGRAM) $N $P > $@)
	@echo "✓ Concluído: $@"

# Target para executar arquivo específico por nome
.PHONY: run-file
run-file: compile
	@if [ -z "$(FILE)" ]; then \
		echo "Erro: Use 'make run-file FILE=nome_do_arquivo.out [CUDA=1]'"; \
		echo "Exemplo: make run-file FILE=particulas_1000_passos_500_v1.out"; \
		exit 1; \
	fi
	@echo "Gerando arquivo específico: $(OUTDIR)/$(FILE)"
	@mkdir -p $(OUTDIR)
	@$(MAKE) $(OUTDIR)/$(FILE)
.PHONY: run
# Target para executar com parâmetros específicos
.PHONY: run
run: compile
	@if [ -z "$(N)" ] || [ -z "$(P)" ] || [ -z "$(V)" ]; then \
		echo "Erro: Use 'make run N=valor P=valor V=valor [CUDA=1]'"; \
		echo "Exemplos:"; \
		echo "  make run N=1000 P=500 V=v1            # Versão CPU"; \
		echo "  make run N=1000 P=500 V=v1 CUDA=1     # Versão CUDA"; \
		exit 1; \
	fi
	@echo "Executando versão $(TYPE) com N=$(N), P=$(P), V=$(V)"
	@mkdir -p $(OUTDIR)
	@output_file="$(OUTDIR)/particulas_$(N)_passos_$(P)_$(V).out"; \
	$(PROGRAM) $(N) $(P) > $output_file; \
	@echo "✓ Arquivo gerado: $(OUTDIR)/particulas_$(N)_passos_$(P)_$(V).out"

# Target para compilar apenas (sem executar)
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

# Target para limpar arquivos compilados e de saída
.PHONY: clean
clean:
	@echo "Removendo arquivos..."
	@rm -f nbody nbody_cuda
	@rm -rf out_cpu out_cuda_* $(OUTDIR)
	@echo "✓ Limpeza concluída"

# Target para limpar apenas arquivos de saída
.PHONY: clean-output
clean-output:
	@echo "Removendo arquivos de saída..."
	@rm -rf out_cpu out_cuda_* $(OUTDIR)
	@echo "✓ Arquivos de saída removidos"

# Target para listar todas as pastas de saída existentes
.PHONY: list-outputs
list-outputs:
	@echo "Pastas de saída existentes:"
	@ls -la | grep "^d.*out_" || echo "Nenhuma pasta de saída encontrada"

# Target para limpar pasta específica
.PHONY: clean-specific
clean-specific:
	@if [ -z "$(DIR)" ]; then \
		echo "Erro: Use 'make clean-specific DIR=nome_da_pasta'"; \
		echo "Exemplo: make clean-specific DIR=out_cuda_T512"; \
		exit 1; \
	fi
	@echo "Removendo pasta $(DIR)..."
	@rm -rf $(DIR)
	@echo "✓ Pasta $(DIR) removida"

# Target para limpar apenas executáveis
.PHONY: clean-bin
clean-bin:
	@echo "Removendo executáveis..."
	@rm -f nbody nbody_cuda
	@echo "✓ Executáveis removidos"

# Target para listar configurações
.PHONY: show-config
show-config:
	@echo "Configurações atuais:"
	@echo "Tipo: $(TYPE)"
	@echo "Programa: $(PROGRAM)"
	@echo "Código fonte: $(SOURCE)"
	@echo "Compilador: $(COMPILER) $(COMPILE_FLAGS)"
	@echo "Diretório de saída: $(OUTDIR)"
	@echo ""
	@echo "Parâmetros definidos ($(words $(PARAMS)) execuções):"
	@for param in $(PARAMS); do \
		N=$$(echo $$param | cut -d':' -f1); \
		P=$$(echo $$param | cut -d':' -f2); \
		V=$$(echo $$param | cut -d':' -f3); \
		echo "  N=$$N, P=$$P, V=$$V -> $(OUTDIR)/particulas_$${N}_passos_$${P}_$${V}.out"; \
	done

# Target para debug - mostra arquivos que serão gerados
.PHONY: debug-targets
debug-targets:
	@echo "Arquivos que serão gerados ($(words $(OUTPUTS)) arquivos):"
	@for output in $(OUTPUTS); do \
		echo "  $$output"; \
	done
	@echo ""
	@echo "Teste de parsing dos primeiros 3 parâmetros:"
	@for param in $(wordlist 1,3,$(PARAMS)); do \
		N=$$(echo $$param | cut -d':' -f1); \
		P=$$(echo $$param | cut -d':' -f2); \
		V=$$(echo $$param | cut -d':' -f3); \
		echo "  $$param -> N=$$N, P=$$P, V=$$V"; \
	done

# Target para verificar se os compiladores estão disponíveis
.PHONY: check-deps
check-deps:
	@echo "Verificando dependências..."
	@which gcc > /dev/null 2>&1 && echo "✓ gcc encontrado" || echo "✗ gcc não encontrado"
	@which nvcc > /dev/null 2>&1 && echo "✓ nvcc encontrado" || echo "✗ nvcc não encontrado"
	@[ -f nbody.c ] && echo "✓ nbody.c encontrado" || echo "✗ nbody.c não encontrado"
	@[ -f nbody_cuda.cu ] && echo "✓ nbody_cuda.cu encontrado" || echo "✗ nbody_cuda.cu não encontrado"

# Target de ajuda
.PHONY: help
help:
	@echo "Makefile para execução sequencial do nbody (CPU ou CUDA)"
	@echo ""
	@echo "Targets principais:"
	@echo "  run-all              - Compila e executa todas as configurações (CPU)"
	@echo "  run-all CUDA=1       - Compila e executa todas as configurações (CUDA)"
	@echo "  run-all OUTDIR=pasta - Usa pasta de saída customizada"
	@echo "  run                  - Executa uma configuração específica"
	@echo "  run-file             - Executa para gerar um arquivo específico"
	@echo ""
	@echo "Targets de compilação:"
	@echo "  build-cpu       - Compila apenas a versão CPU"
	@echo "  build-cuda      - Compila apenas a versão CUDA"
	@echo "  build-both      - Compila ambas as versões"
	@echo ""
	@echo "Targets de limpeza:"
	@echo "  clean           - Remove executáveis e todas as pastas de saída"
	@echo "  clean-output    - Remove apenas todas as pastas de saída"
	@echo "  clean-bin       - Remove apenas executáveis"
	@echo "  clean-specific  - Remove pasta específica (use DIR=nome)"
	@echo "  list-outputs    - Lista todas as pastas de saída existentes"
	@echo ""
	@echo "Targets utilitários:"
	@echo "  show-config     - Mostra as configurações atuais"
	@echo "  check-deps      - Verifica se compiladores e arquivos existem"
	@echo "  debug-targets   - Mostra arquivos que serão gerados (debug)"
	@echo "  help            - Mostra esta ajuda"
	@echo ""
	@echo "Exemplos de uso:"
	@echo "  make run-all                                          # CPU -> out_cpu/"
	@echo "  make run-all CUDA=1                                   # CUDA -> out_cuda/"
	@echo "  make run-all OUTDIR=meus_resultados                   # CPU -> meus_resultados/"
	@echo "  make run-all CUDA=1 OUTDIR=teste                      # CUDA -> teste/"
	@echo "  make run N=1000 P=500 V=v1 CUDA=1                     # Execução única"
	@echo "  make run-file FILE=particulas_1000_passos_500_v1.out  # Arquivo específico"
	@echo "  make list-outputs                                     # Ver pastas existentes"
	@echo "  make clean-specific DIR=out_cuda                      # Limpar pasta específica"
	@echo "  make show-config                                      # Ver configurações"

# Target padrão
.DEFAULT_GOAL := help