CONTINUOUS DATA RIVER — ANÁLISE CRÍTICA E EXTENSÕES
===============================================================
Documento de trabalho. Complemento ao conceito original.


ÍNDICE

PARTE I    — ANÁLISE CRÍTICA DO DOCUMENTO ORIGINAL
PARTE II   — TEORIA DA RESISTÊNCIA
PARTE III  — DENTRO DO PROCESSADOR: L1/L2/L3 E AS BARREIRAS REAIS
PARTE IV   — QUEBRAR FRONTEIRAS: SEIS INVERSÕES
PARTE V    — SÍNTESE E PRÓXIMOS PASSOS


================================================================
PARTE I — ANÁLISE CRÍTICA DO DOCUMENTO ORIGINAL
================================================================

I.1 O QUE ESTÁ CORRECTO

O reframe central é válido:

   A inferência de LLM em CPU é um problema de movimento de dados
   disfarçado de problema de computação.

A secção 34 (bytes movidos / MAC útil) é a melhor linha do documento.
É a métrica correcta e quase ninguém optimiza directamente para ela.

Secções 3, 4 e 5 (ownership, credits, buffers de elasticidade) são
engenharia de sistemas sólida, herdada do design de NICs e redes.
Funcionam.

Secções 6 e 29 (memória preditiva) exploram algo real e subaproveitado:
o dataflow de um transformer é totalmente conhecido à partida. Isso é
uma vantagem enorme sobre prefetchers genéricos que adivinham a partir
de padrões de acesso.


I.2 A FALHA ESTRUTURAL

O documento confunde LATÊNCIA com BANDWIDTH, e a tese inteira assenta
nessa confusão.

   Pipelining esconde latência.
   Pipelining NÃO faz nada por bandwidth.

A descodificação autoregressiva é bandwidth-bound da forma mais brutal
possível: para produzir UM token é preciso ler TODOS os pesos.

Exemplo concreto:

   modelo 40 GB INT4
   NVMe a 7 GB/s
   rio perfeito, zero stalls, ocupação 100%
   T_total = max(T_storage, T_compute) exactamente como desenhado

   resultado: 0,17 tokens/segundo

O pipeline está impecável e o sistema é inutilizável.

A própria fórmula do documento prevê isto. max() continua dominado
pelo estágio mais lento, e o storage é 30-100x mais lento que a RAM.

   O rio leva-te à parede mais depressa.
   Não move a parede.

Consequência: a secção 20 (streaming weights, "o modelo pode ser muito
maior que a RAM") está escrita como funcionalidade principal quando é
a afirmação mais fraca do documento.


I.3 PARA ONDE ISSO APONTA — O DESBLOQUEIO REAL

A saída não é mover bytes mais depressa.
É NÃO MOVER bytes que não são precisos.

Ou seja: esparsidade.

   - MoE: carregar apenas os experts activos (~10-20% por token)
   - esparsidade de neurónios FFN: activações pós-ReLU/SwiGLU são
     altamente esparsas E previsíveis uma layer antes

E aqui está o ponto importante para a tese:

   Assim que o carregamento passa a ser esparso, torna-se
   imprevisível — e é exactamente aí que a arquitectura do rio
   deixa de ser opcional.

Prefetch estático não ajuda. É preciso um preditor, loads
especulativos, backpressure e descarte por mispredição.
As secções 28 e 29 passam de decorativas a estruturais.

Prior art a estudar: "LLM in a Flash" (Apple), PowerInfer, FlexGen.
Resolveram peças; nenhum construiu a camada de controlo unificada.


I.4 SECÇÕES A CORTAR OU DESAFIAR

§14 CROSS-CORE RIVER — armadilha provável.
Passar tiles entre cores gera cache-line ping-pong e tráfego de
coerência. Num único socket, dividir o mesmo kernel em data-parallel
bate pipeline-parallel entre cores quase sempre.
(NOTA: isto refere-se a mover TILES. Mover ACTIVAÇÕES é outra coisa
completamente — ver Parte IV, inversão 1.)

§15 HARDWARE AGNOSTIC — em tensão com o resto.
Não se pode ser agnóstico ao ISA e simultaneamente saturar portas de
execução. Resolver por camadas:
   agnóstico ao nível do PLANO
   microkernels afinados por ISA ao nível do KERNEL
É o que BLIS e llama.cpp fazem na prática.

§23 PROCESSING NEAR STORAGE — área de investigação real, hardware
consumidor praticamente inexistente. Arquivar.

§9 / §10 / §11 / §12 — correctas mas terreno já muito pisado.
Microkernels estilo BLIS e o ggml do llama.cpp já fazem isto.
Pouca diferenciação.

§19 + §36 O COMPILADOR — território TVM/IREE/tinygrad. Projecto de
20 pessoas-ano. A secção mais perigosa, porque é onde o documento
silenciosamente se torna não-entregável.


I.5 O QUE É REALMENTE NOVO E ENTREGÁVEL

Removendo o que já existe noutro lado, sobra a camada de MEDIÇÃO e
CONTROLO:

   §33  Pipeline Occupancy
   §39  profiler de ponto de resistência
   D    digital twin do hardware
   B    auto-tuner
   C    feedback loop

Ninguém construiu isto. Não exige substituir o llama.cpp — pode
assentar POR CIMA dele. E é o que diria quais das outras 38 secções
valem a pena.

Também genuinamente subexplorado:
   §F  compute-aware storage layout
Reordenar pesos no disco pela ordem de consumo é barato, mensurável,
e ninguém o faz a sério.


I.6 A PERGUNTA QUE DEFINE O PROJECTO

   Alvo = modelos que CABEM na RAM?
      → optimização de latência/ocupação
      → mais tratável, ganho menor

   Alvo = modelos que EXCEDEM a RAM?
      → guerra de bandwidth, só se ganha com esparsidade

São dois projectos diferentes. O documento tenta ser os dois.


================================================================
PARTE II — TEORIA DA RESISTÊNCIA
================================================================

II.1 FORMALIZAR A METÁFORA

"Resistência" não é só imagem. Formaliza-se:

   R = tempo que o consumidor espera / bytes úteis entregues

Daí sai uma álgebra real:

   resistências em série  → somam-se   (T_storage + T_memory + T_compute)
   resistências em paralelo → reduzem-se

O rio, quando funciona, converte série em paralelo. É literalmente o
que max() significa. A fórmula do documento já é a lei de Ohm do
sistema.

Produto: uma MATRIZ DE RESISTÊNCIA da máquina, medida e não assumida.
Não "quanta bandwidth tem o SSD", mas "quanto é que este par
produtor→consumidor desperdiça".


II.2 DOIS TIPOS DE RESISTÊNCIA

RESISTÊNCIA FÍSICA
   bandwidth da DRAM, latência do NVMe, velocidade dos sinais.
   Não se quebra. Só se contorna movendo menos bytes.

RESISTÊNCIA ARTIFICIAL
   - conversões de formato (INT4→FP16, transposições, row-major→tile)
   - cópias redundantes entre camadas de software
   - barreiras e locks onde bastaria ownership
   - page faults, syscalls, o allocator
   - o scheduler do SO a mover threads entre cores
   - descobrir em runtime o que já se sabia em compile time

Num sistema real, a resistência artificial é facilmente 2-5x a física.

   É aí que "quebrar a resistência" tem significado literal,
   sem precisar de vencer nenhuma lei da física.


II.3 A RESISTÊNCIA INVISÍVEL — IMPEDÂNCIA DE FORMATO

Sempre que os dados MUDAM DE FORMA sem produzir informação nova,
isso é fricção pura:

   desempacotar INT4
   transpor uma matriz
   reordenar para tiles
   alinhar
   converter layouts entre bibliotecas

Em electrónica isto chama-se desadaptação de impedância. E a solução
também: adaptar as impedâncias em vez de dissipar energia na
transição.

Aplicado ao rio: desenhar a cadeia inteira DE TRÁS PARA A FRENTE,
a partir da largura do registo.

   tile que o compute quer
      define o tile que a cache quer
         define o bloco que a RAM quer
            define o layout no disco

Se todos os estágios tiverem granularidade compatível, a água nunca
muda de forma. Zero transformações entre origem e unidade de execução.

Versão radical da §21:
   o ficheiro do modelo deixa de ser dados a interpretar e passa a ser
   um binário compilado para aquela máquina.
   Carregar = memcpy para o sítio certo. Nada a descodificar.


II.4 INVERTER A DIRECÇÃO — HORÁRIO EM VEZ DE REACÇÃO

O rio do documento é PUSH: o produtor antecipa.

Alternativa: PULL DETERMINÍSTICO. O compute publica um HORÁRIO —
"no instante T preciso do tile X" — e tudo a montante corre a horário
fixo, como uma rede TDMA ou um sistema de tempo real.

Diferença filosófica:

   prefetch reactivo → precisa de queues, credits, backpressure
                       porque não se sabe quando as coisas chegam

   horário fixo      → as queues só absorvem JITTER,
                       não incerteza

E numa LLM o horário é conhecido todo à partida. É provavelmente a
workload mais determinística que existe.

   Está a usar-se um mecanismo probabilístico
   para um problema que não é probabilístico.


II.5 A ESPECULAÇÃO COMO TRANSFORMADOR DE IMPEDÂNCIA

O problema de fundo do batch=1 é intensidade aritmética ~= 1:
um MAC por byte lido. Com batch=32, ~= 32. Os mesmos bytes alimentam
32x mais trabalho útil.

Mas um utilizador único não tem batch.

   A decodificação especulativa FABRICA batch a partir de um
   fluxo sequencial.

Não é um truque de velocidade. É literalmente um transformador que
converte uma workload de baixa intensidade numa de alta intensidade,
sem mudar o hardware.

Na linguagem do rio: é o único componente que REDUZ A RESISTÊNCIA
FÍSICA, não só a artificial. Todo o resto esconde latência; este
reduz bytes-por-operação-útil.

   Se houvesse que promover uma secção a princípio central em vez
   de técnica auxiliar, seria a 28.


II.6 UMA TENSÃO A ASSUMIR

A §32 diz "se todos os estágios estiverem ocupados, a latência fica
escondida". Fisicamente há um travão:

   saturar load units, vector units e DMA ao mesmo tempo faz o CPU
   descer de clock por limite térmico e de corrente.
   AVX-512 a 100% corre a frequência mais baixa que AVX2 a 60%.

Ou seja: a ocupação óptima do rio provavelmente NÃO é 100%.
Há um ponto de máxima transferência de potência, tal como em
circuitos. Ideia mais rica que "encher tudo", e mantém-se dentro
da própria metáfora.


================================================================
PARTE III — DENTRO DO PROCESSADOR
================================================================

III.1 A LEI QUE GOVERNA TUDO — LITTLE'S LAW

   Bytes em voo = Bandwidth x Latência

Para sustentar 20 GB/s com latência DRAM de 80 ns são precisos
~1600 bytes permanentemente em trânsito — 25 cache lines em voo,
sempre.

O problema: cada core tem apenas ~10-16 LINE FILL BUFFERS. São os
únicos slots para misses L1 pendentes.

   12 LFBs x 64 B = 768 bytes em voo
   com 80 ns de latência → ~9,6 GB/s POR CORE, tecto absoluto

Independentemente do código, do AVX-512, de tudo.

Esta é a resistência mais dura dentro do CPU e quase ninguém fala
dela. Explica porque um único core nunca satura a DRAM, e porque são
precisos 4-6 cores só para gerar PARALELISMO DE MEMÓRIA suficiente —
não por causa do compute.

   Consequência: a métrica de ocupação (§33) deve incluir
   occupancy dos fill buffers. É o verdadeiro indicador de
   saturação do rio dentro do core.


III.2 A VERDADE INCÓMODA SOBRE AS CACHES NA DESCODIFICAÇÃO

Em batch=1, cada peso é lido exactamente UMA VEZ e nunca reutilizado.

A hierarquia de caches existe para explorar reutilização.
Aqui não há nenhuma.

   Para os pesos, as caches não são um benefício. São um imposto.

Cada byte de peso atravessa L3→L2→L1, ocupa espaço, despeja coisas
úteis, gasta tags e coerência — para um único uso.

Isto inverte a estratégia inteira. Não se quer "melhor uso das caches"
no sentido clássico. Quer-se SEPARAR O QUE ATRAVESSA DO QUE RESIDE:

   DEVE RESIDIR                        DEVE ATRAVESSAR
   ------------------------------      ------------------
   vector de activação                 pesos
     (4096 x FP32 = 16 KB → L1)
   acumuladores (registos)
   KV working set (L2/L3)

Objectivo real: os pesos passarem pela cache com pegada mínima,
sem despejar o que tem reutilização.


III.3 AS BARREIRAS CONCRETAS, POR ORDEM DE RETORNO

1. TLB — A BARREIRA MAIS SUBESTIMADA

   Modelo de 40 GB com páginas de 4 KB = 10 milhões de páginas.
   TLB tem ~1500-2500 entradas. Page walk permanente.

   Huge pages de 2 MB reduzem isto 512x.

   Bónus que quase ninguém sabe:
   o prefetcher L2 NÃO atravessa fronteiras de página de 4 KB.
   Pára, e tem de ser re-armado.
   Com huge pages corre livremente durante 2 MB inteiros.

   Uma flag de configuração desbloqueia o rio a nível de hardware.

2. RFO — 50% DA BANDWIDTH DE ESCRITA DESPERDIÇADA

   Ao escrever uma cache line, o CPU LÊ-A PRIMEIRO
   (Read-For-Ownership), mesmo que vá ser sobrescrita inteira.
   Escrever 1 GB custa 2 GB de tráfego.

   Non-temporal stores (MOVNTPS / _mm512_stream_ps) eliminam isto.
   Para KV cache e activações de saída é ganho directo e imediato.

3. POLUIÇÃO DE L3

   PREFETCHNTA traz para L1 marcando como LRU, limitando o estrago
   em L2/L3. Instrumento certo para pesos: entram, usam-se, saem
   sem despejar o KV.

   ARMADILHA CLÁSSICA: MOVNTDQA (load não-temporal) só funciona em
   memória WC em x86. Em memória normal comporta-se como load vulgar.

4. DISTÂNCIA DE PREFETCH

   Não é constante mágica:

      distância = latência_do_nível / tempo_por_iteração

   Se cada iteração demora 5 ns e a DRAM está a 80 ns → 16 iterações
   à frente.

   Auto-tunável. Deve ser dos primeiros parâmetros do auto-tuner (§B).

5. PORTAS DE EXECUÇÃO

   Intel faz 2 loads + 1 store por ciclo.
   Um GEMM ingénuo precisa de 2 loads por FMA → fica LOAD-BOUND
   com as unidades FMA paradas.

   Register blocking (tile de C nos registos, reutilizado por várias
   linhas) é o que quebra isto. É exactamente a §9, mas a razão é
   esta e não "paralelismo genérico".

6. TOPOLOGIA L3

   Zen: cada CCX tem a sua fatia de L3. Atravessar CCX passa pelo
   Infinity Fabric — muito mais caro que L3 local.
   Intel: ring/mesh com latências não uniformes por slice.

   Thread pinning não é micro-optimização aqui. É estrutural.


III.4 A IDEIA MAIS RADICAL DESTA PARTE — CACHE COMO SCRATCHPAD

A Intel tem CAT (Cache Allocation Technology): permite PARTICIONAR
O L3 POR WAYS e atribuir partições a grupos de threads.

Existe, está documentado, e praticamente ninguém a usa para LLMs.

Com isso pode fazer-se explicitamente:

   - reservar N ways de L3 EXCLUSIVAMENTE para o KV cache
   - dar aos pesos uma janela de streaming pequena e fixa
   - garantir que o KV NUNCA é despejado por pesos de uso único

Deixa de ser uma cache automática que se sofre.
Passa a ser memória gerida.

É o mais perto que se chega do scratchpad de uma GPU num CPU normal,
e encaixa exactamente na filosofia: as caches não desaparecem,
tornam-se estágios do rio controlados.


III.5 CORRECÇÃO À §8

O documento diz: "não remover L1/L2/L3 — torná-las transparentes".

Proposta contrária, e mais forte:

   TORNÁ-LAS OPACAS E DELIBERADAS.

   Decidir explicitamente o que reside e o que atravessa, e impor
   essa decisão com huge pages, NT stores, PREFETCHNTA e CAT.

Transparência é o que temos hoje.
E é precisamente a origem da resistência.


================================================================
PARTE IV — QUEBRAR FRONTEIRAS: SEIS INVERSÕES
================================================================

Regra: cada ideia tem de inverter uma suposição que hoje ninguém
questiona.


IV.1 PARAR DE MOVER OS PESOS. MOVER OS TOKENS.

SUPOSIÇÃO QUEBRADA: que os pesos vão ao compute.

Hoje movem-se gigabytes de pesos para processar um vector de
activação de 16 KB. Move-se a coisa enorme para junto da coisa
minúscula. Visto de fora, é absurdo.

INVERSÃO:
   cada core adopta permanentemente uma fatia do modelo, presa no
   seu L2, e nunca mais a larga. Os TOKENS é que viajam.

   Core 0 [layers 0-3 residentes em L2]
      ↓ activação (16 KB)
   Core 1 [layers 4-7 residentes em L2]
      ↓ activação (16 KB)
   Core 2 [layers 8-11 residentes em L2]

O tráfego DRAM durante a descodificação passa a ser ZERO.
Não reduzido. Zero.

NÚMEROS:
   EPYC Genoa: até 384 MB de L3 agregado
   modelo ternário 1-2 B parâmetros: 200-400 MB

   Estamos exactamente na fronteira da viabilidade.
   Não é ficção científica. É aritmética.

Isto muda a pergunta de design:

   de: "como faço streaming eficiente de 40 GB"
   para: "qual é o maior modelo que cabe permanentemente dentro
          do silício"

Uma é uma guerra que se perde. A outra é um problema de compressão.

NOTA: isto contradiz a §14, rejeitada na Parte I. E contradiz com
razão — rejeitou-se mover TILES entre cores. Mover ACTIVAÇÕES é
outra coisa: 16 KB contra gigabytes.


IV.2 GERAR OS PESOS EM VEZ DE OS TRANSPORTAR

SUPOSIÇÃO QUEBRADA: que um peso tem de ser lido de algum lado.

O sistema está bandwidth-bound com as unidades FMA PARADAS.
Há compute grátis. Logo: qualquer troca que gaste compute para
poupar bytes é lucro puro, mesmo que gaste 10x mais FLOPs.

Se um bloco de pesos puder ser reconstruído por uma função
determinística barata — seed + estrutura + resíduo pequeno — então
o que atravessa o rio deixa de ser o peso. É a RECEITA do peso.

   tradicional:  4096 x 4096 INT4  = 8 MB transportados
   gerado:       seed + resíduo    = ~200 KB transportados
                 + reconstrução em registos (FMA que estavam paradas)

Descompressão aritmética, não por tabela. O bloco nasce dentro dos
vector registers e morre lá. Nunca toca em memória nenhuma.

Isto inverte o roofline: em vez de deslizar pela linha de bandwidth,
sobe-se para a região limitada por compute — que é onde há folga.


IV.3 O RIO NUNCA ESPERA POR UMA PEDRA

SUPOSIÇÃO QUEBRADA: que o transporte de dados tem de ser exacto e
completo.

A mais herética, e a que melhor serve a metáfora.

Todo o stack de computação assume entrega garantida: retries,
ordenação, ECC, coerência. Tudo isso existe para proteger correcção
binária.

Mas uma LLM NÃO É BINARIAMENTE CORRECTA. É estatística.
Um tile em falta é uma perturbação minúscula numa soma de milhares
de termos.

INVERSÃO: proibir o stall por construção.

   tile chegou?  →  usa
   não chegou?   →  usa a aproximação (tile anterior, ou zero)
                    e continua
                    nunca bloqueia, nunca sincroniza

CONSEQUÊNCIAS RADICAIS:
   - desaparece o backpressure — não é preciso
   - desaparecem os credits — não é preciso
   - a ocupação do pipeline passa a ser 100% POR DEFINIÇÃO
   - a qualidade degrada-se suavemente sob pressão,
     em vez de a velocidade colapsar

Ganha-se uma propriedade que nenhum sistema de inferência tem hoje:

   THROUGHPUT CONSTANTE E GARANTIDO, COM QUALIDADE VARIÁVEL.

O modelo corre a 30 tok/s sempre. Em hardware bom com qualidade
total, em hardware fraco com qualidade ligeiramente inferior.
Nunca a 0,2 tok/s.

É literalmente como funciona o streaming de vídeo adaptativo.
Ninguém o aplicou a pesos de LLM.


IV.4 A POSIÇÃO FÍSICA NO DISCO É O PROGRAM COUNTER

SUPOSIÇÃO QUEBRADA: que o armazenamento guarda dados e o scheduler
decide a ordem.

Eliminar o scheduler.

   Escrever o modelo no disco pela ordem exacta, byte a byte, em
   que vai ser consumido — derivada da matriz de latência medida
   daquela máquina específica.

Ler o ficheiro sequencialmente à velocidade natural do NVMe produz
AUTOMATICAMENTE o dado certo no instante certo.

   sem seeks
   sem decisões
   sem queues de I/O

O disco vira um leitor de fita, e a fita é o programa.
A ordem física É o horário.

COROLÁRIO: o mesmo modelo lógico produz ficheiros fisicamente
diferentes em máquinas diferentes. O ficheiro deixa de ser um
modelo — é um modelo já compilado para uma máquina concreta.


IV.5 ESPECULAR EM LARGURA PORQUE A LARGURA É GRÁTIS

SUPOSIÇÃO QUEBRADA: que a especulação serve para poupar tempo.

Não serve. Serve para FABRICAR INTENSIDADE ARITMÉTICA.

Uma árvore de 64 candidatos lê exactamente os mesmos pesos que
1 candidato. Os bytes são idênticos.

   batch=1:       1 MAC / byte
   árvore de 64:  64 MAC / byte    ← mesmos bytes lidos

Logo a estratégia correcta não é "especular com precisão".
É especular tão largo quanto as unidades FMA aguentem, mesmo com
ramos de probabilidade ridícula. O custo marginal de mais um ramo é
praticamente zero, porque o custo real já foi pago ao ler os pesos.

Inverte-se o critério de qualidade do draft model:
deixa de importar que acerte muito.
Importa que gere DIVERSIDADE suficiente para encher as unidades
de execução.


IV.6 UM ÚNICO KERNEL ETERNO

SUPOSIÇÃO QUEBRADA: que se chamam funções.

Nada de kernel launches, nada de chamadas, nada de dispatch.

Um único loop, pinado, que arranca e nunca retorna até a geração
acabar. Corpo estável, cabe na micro-op cache, o frontend nunca mais
volta a descobrir a estrutura da mesma computação.

   O modelo deixa de ser dados que um programa percorre.
   O modelo É o programa.


================================================================
PARTE V — SÍNTESE
================================================================

V.1 O PRINCÍPIO POR TRÁS DE TUDO

Todas as inversões da Parte IV têm o mesmo denominador:

   BYTES SÃO CAROS.
   FLOPS SÃO GRÁTIS.
   SINCRONIZAÇÃO É O INIMIGO.

Cada uma gasta abundância (compute ocioso, largura de especulação,
cache não usada) para comprar escassez (bandwidth, previsibilidade,
ausência de stalls).

E convergem num ponto único, que é a formulação mais forte da ideia
original:

   O RIO DEIXA DE TRANSPORTAR PESOS.
   TRANSPORTA APENAS ACTIVAÇÕES E RECEITAS.
   E NUNCA, EM CIRCUNSTÂNCIA NENHUMA, PÁRA.


V.2 REGRAS DE OPTIMIZAÇÃO — REVISÃO DA §35

Mantendo as 10 originais e acrescentando:

   11. Não transportar o que pode ser gerado.
   12. Não mover o grande até ao pequeno; mover o pequeno até
       ao grande.
   13. Não garantir entrega exacta a um sistema que é estatístico.
   14. Não decidir em runtime o que a ordem física pode decidir.
   15. Não especular para acertar; especular para encher.
   16. Não deixar a cache decidir o que reside — decidir e impor.
   17. Não medir ocupação sem medir fill buffers.
   18. Não confundir esconder latência com reduzir bandwidth.


V.3 PRÓXIMA COMBINAÇÃO A EXPLORAR

   IV.1 + IV.3

   modelo residente em cache + transporte não-bloqueante

É a que tem menos peças a inventar e a que muda mais radicalmente
o comportamento do sistema.


V.4 PRIMEIRO PASSO CONCRETO (SE E QUANDO SE PASSAR À PRÁTICA)

Antes de escrever qualquer runtime, construir o roofline de UMA
máquina específica. Medir:

   - NVMe sequencial e 4K aleatório
   - bandwidth de RAM por nó NUMA
   - throughput de GEMM alcançável em INT4/INT8
   - breakdown de stalls do llama.cpp actual
   - occupancy dos line fill buffers

Depois escolher UMA afirmação falsificável e tentar matá-la.
Sugestão:

   "reordenação de pesos compute-aware + prefetch com credits
    melhora tokens/s num modelo memory-constrained em >20%"

Se sobrevive, há projecto.
Se não sobrevive, aprendeu-se numa semana em vez de num ano.


================================================================
FIM
================================================================
