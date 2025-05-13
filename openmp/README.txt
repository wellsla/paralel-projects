Compilar: 
    gcc nbody_omp.c -o nbody_omp -fopenmp -lm
Executar:
    ./nbody_omp <Número de Threads> <Número de Partículas> <Número de Repetições>
Inputs:
    - Modifiquei o formato de inserção de valores, para poder adicionar o parâmetro de Nº de Threads.
Outputs:
    - O output é feito dentro do arquivo output.txt, contém os tempos de execução sequencial/paralela e ao final as coordenadas resultantes das partículas.
    - Resultados coletados para 4 partículas em 10 repetições na minha máquina pessoal:
        Análise Sequencial: 0.000004 segundos
        Paralelismo com 2 threads: 0.000107 segundos > 0.0310 speedup
        Paralelismo com 4 threads: 0.000179 segundos > 0.0172 speedup
        Paralelismo com 8 threads: 0.000502 segundos > 0.0065 speedup
        Paralelismo com 16 threads: 0.005304 segundos > 0.0006 speedup
    - Resultados coletados para 16384 partículas em 100 repetições na minha máquina pessoal:
        Análise Sequencial: 179.884373 segundos
        Paralelismo com 2 threads: 100.990276 segundos > 1.7812 speedup
        Paralelismo com 4 threads: 56.619489 segundos > 3.1770 speedup
        Paralelismo com 8 threads: 33.877403 segundos > 5.3098 speedup
        Paralelismo com 16 threads: 28.857768 segundos > 6.2334 speedup
