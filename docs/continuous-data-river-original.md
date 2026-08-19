CONCEITO — CONTINUOUS DATA RIVER PARA INFERÊNCIA DE LLMs
===============================================================

VISÃO

A proposta não é eliminar RAM, caches, SSD, CPU, GPU ou NPU.
Também não é construir uma solução dependente de NVIDIA, Intel, AMD ou
de uma CPU específica.

A ideia é criar uma camada de software que transforme todos esses
componentes em partes de um fluxo contínuo de dados e computação.

O problema fundamental não é apenas a latência de cada componente.
É a soma das esperas entre componentes.

Paradigma tradicional:

SSD → RAM → L3 → L2 → L1 → registers → execution units

A proposta:

SSD → RAM → cache → registers → compute
       ╰──────── fluxo continuamente antecipado ────────╯

Não se elimina a hierarquia. Eliminam-se, tanto quanto possível, as
resistências entre os seus níveis.

PRINCÍPIO CENTRAL

Não tentar tornar todos os componentes infinitamente rápidos.

Tentar garantir que, quando um estágio termina, o próximo já tem
trabalho disponível.

Idealmente:

T_total tradicional ≈ T_storage + T_memory + T_compute

Pipeline:

T_total ≈ max(T_storage, T_memory, T_compute)

depois de o pipeline estar cheio.

A latência não desaparece fisicamente; fica escondida por sobreposição.

================================================================
1. O DATA RIVER
================================================================

A arquitectura deve ser pensada como um "rio" de dados.

Cada componente é um endpoint:

- Storage
- RAM
- LLC/L3
- L2
- L1
- registers
- vector units
- matrix units
- execution units
- outros cores
- GPU/NPU quando existirem

As caches e a RAM não desaparecem.
Tornam-se estágios transparentes de um fluxo.

O modelo de execução deixa de perguntar:

"Onde está este dado?"

e passa a perguntar:

"Para onde está este dado a ir?"

================================================================
2. SOFTWARE-DEFINED LINKS
================================================================

Criar uma abstracção de links entre produtores e consumidores.

Conceptualmente:

Link<Storage, RAM>
Link<RAM, Cache>
Link<Cache, Compute>

Cada link possui:

- producer
- consumer
- ring buffer
- capacidade
- occupancy
- bandwidth estimada
- latência estimada
- requests em voo
- credits
- prefetch distance
- backpressure
- ownership

Os links são canais permanentes e assíncronos, não chamadas bloqueantes.

Em vez de:

read()
wait()
process()
read()
wait()

usar:

request A
request B
request C
request D

enquanto o compute processa os anteriores.

================================================================
3. EVERYTHING IS A QUEUE
================================================================

A arquitectura deve favorecer queues em todos os níveis:

Storage Queue
Memory Queue
Cache/Tile Queue
Decode Queue
Compute Queue
Reduce Queue
Output Queue

Sempre que possível usar single-producer/single-consumer rings para
evitar locks.

A propriedade de um bloco de dados muda de estágio:

DMA owns tile
→ Compute owns tile
→ Reduce owns tile
→ Output owns tile

Em vez de vários componentes modificarem simultaneamente o mesmo
estado.

================================================================
4. CREDIT-BASED FLOW CONTROL
================================================================

Cada consumidor anuncia a capacidade disponível.

Exemplo:

Compute:
capacity = 32 tiles

Storage recebe:

32 credits

Cada tile consumido devolve um credit.

Isto cria flow control sem necessidade de esperar por perguntas
constantes do produtor.

É semelhante à ideia de controlo de fluxo de redes, mas especializado
para comunicação local de baixa latência.

================================================================
5. ELASTICITY / JITTER HIDING
================================================================

Cada ponte deve possuir apenas o buffer necessário para absorver
variações de latência.

Não procurar buffers gigantes.

Procurar buffers de elasticidade.

Exemplo:

SSD
 ↓
Ring buffer
 ↓
RAM
 ↓
Ring buffer
 ↓
Compute

Se o SSD sofrer um pequeno atraso, o compute continua a trabalhar.

================================================================
6. PREDICTIVE MEMORY
================================================================

A LLM é extremamente previsível na estrutura do cálculo.

Antes de executar Layer N já sabemos que virão:

Layer N+1
Layer N+2
Layer N+3

Logo:

Compute Layer N
RAM prepara Layer N+1
Storage prepara Layer N+2
Scheduler prepara pedidos para Layer N+3

Não esperar que o consumidor peça os dados.

Antecipar.

================================================================
7. MICRO-TILES
================================================================

Não trabalhar apenas ao nível de layers.

Dividir pesos e activações em tiles pequenos.

Exemplo:

Layer
→ T0
→ T1
→ T2
→ T3
→ ...

Enquanto T0 está a ser calculado:

T1 está a chegar
T2 está a ser preparado
T3 está em fila

Isto aumenta o paralelismo e reduz a necessidade de esperar por uma
layer completa.

================================================================
8. NÃO REMOVER L1/L2/L3 — TORNÁ-LAS TRANSPARENTES
================================================================

O software não deve depender de saber exactamente onde uma linha está.

Em vez disso, deve gerar padrões de acesso que permitam ao hardware
manter o fluxo:

- software prefetch
- hardware prefetch
- cache-aware tiling
- alignment
- streaming accesses
- non-temporal operations quando apropriado
- NUMA-aware placement
- huge pages quando apropriado

A arquitectura optimiza o fluxo; a CPU continua a gerir as caches.

================================================================
9. RIO DENTRO DO PRÓPRIO PROCESSADOR
================================================================

O conceito deve continuar dentro do core.

Pipeline conceptual:

LOAD
→ DECODE
→ VECTOR
→ MATMUL/FMA
→ REDUCE
→ NEXT OPERATION

Não esperar que toda a operação anterior termine.

Usar software pipelining, loop unrolling, instruction-level parallelism
e múltiplas operações em voo.

Enquanto alguns registadores estão a ser usados:

outros podem preparar o próximo tile.

Exemplo conceptual:

ZMM0-7   → input tile
ZMM8-15  → weights
ZMM16-23 → accumulators
ZMM24-31 → next tile

Não significa que o compilador seja obrigado a usar exactamente esta
disposição; é um modelo conceptual de pipeline de registos.

================================================================
10. EXECUTION-PORT AWARENESS
================================================================

O scheduler deve considerar que uma CPU possui diferentes recursos:

- load units
- store units
- ALUs
- vector units
- FMA
- shuffle/permutation
- matrix engines quando existirem

Se um kernel saturar apenas uma unidade e deixar outras paradas,
existe uma resistência interna.

Objectivo:

LOAD       COMPUTE       STORE
████████   ██████████    ████

sobrepostos sempre que possível.

================================================================
11. FRONTEND DO CPU
================================================================

Optimizar também:

- instruction cache locality
- micro-op cache
- branch predictability
- loop unrolling
- JIT especializado
- kernels estáveis
- redução de instruction overhead

Para workloads previsíveis, evitar que o frontend esteja constantemente
a descobrir novamente a estrutura da mesma computação.

================================================================
12. BRANCHLESS / DETERMINISTIC EXECUTION
================================================================

Branchless é apenas uma parte.

Para o caminho crítico da inferência, procurar uma sequência de
instruções previsível:

LOAD
LOAD
DOT
DOT
REDUCE
LOAD
DOT
...

Quanto menos decisões dinâmicas, maior a capacidade de preparar o
pipeline antecipadamente.

================================================================
13. REDUZIR DEPENDÊNCIAS ARTIFICIAIS
================================================================

Evitar cadeias desnecessárias:

A → F → G → H

quando o trabalho puder ser organizado como:

F(A0)
F(A1)
F(A2)
F(A3)

para aumentar instruction-level parallelism.

A unidade fundamental passa a ser o trabalho independente, não a
instrução isolada.

================================================================
14. CROSS-CORE DATA RIVER
================================================================

O rio pode continuar entre cores:

Core 0 → decode
Core 1 → quantization
Core 2 → matrix compute
Core 3 → reduction
Core 4 → attention

Mas só quando:

trabalho poupado > custo de comunicação/coerência

A topologia deve ser descoberta em runtime.

NUMA também faz parte do rio.

Preferir:

CPU0 → RAM0

em vez de atravessar para RAM1 quando isso for vantajoso.

================================================================
15. HARDWARE AGNOSTIC
================================================================

Nunca depender de:

- NVIDIA
- Intel
- AMD
- Apple
- AVX-512
- AMX
- CUDA
- uma determinada GPU
- uma determinada geração de SSD

O runtime descobre capacidades.

Exemplo:

VECTOR_WIDTH
INT8
FP16
AVX2
AVX512
NEON
AMX
GPU
NPU
DMA
P2P
NVMe
io_uring
SPDK
NUMA
RAM bandwidth
Storage bandwidth
Cache sizes

Depois cria o execution plan.

Se uma capacidade não existir, usa outra.

Exemplo:

AMX
→ AVX-512
→ AVX2
→ NEON
→ scalar

Storage:

P2P/DMA
→ SPDK
→ io_uring
→ async POSIX
→ standard I/O

O programa continua a funcionar em hardware diferente.

================================================================
16. HARDWARE CAPABILITY GRAPH
================================================================

Representar a máquina como um grafo:

nodes:
CPU cores
RAM regions
storage
accelerators
NUMA nodes

edges:
bandwidth
latency
capacity
supported operations
DMA capability
coherency properties

O scheduler procura o caminho mais eficiente para cada fluxo.

A topologia física deixa de ser uma suposição do programador.

================================================================
17. LLM COMO DATAFLOW GRAPH
================================================================

Não tratar o Transformer como uma sequência rígida:

Token
→ Layer 1
→ Layer 2
→ ...
→ Layer 80

Tratar como grafo:

Attention
MLP
Residual
Normalization
KV
Projection

Cada componente pode produzir streams consumidos pelos seguintes.

================================================================
18. FUSION
================================================================

Evitar materializar resultados intermédios.

Tradicional:

GEMM
→ Tensor B
→ RAM/cache
→ RMSNorm
→ Tensor C
→ RAM/cache
→ Activation

Proposto:

GEMM
→ RMSNorm
→ Activation
→ resultado

O resultado intermédio pode existir apenas no estágio local do
pipeline.

A regra:

"Não materializar o que pode ser transmitido."

================================================================
19. LLM COMPILER
================================================================

Criar um compilador específico:

LLM
→ Graph IR
→ Dataflow IR
→ Hardware capability graph
→ Execution plan
→ Continuous River

O compilador conhece simultaneamente:

- modelo
- dependências
- quantização
- tamanho dos tiles
- hardware
- bandwidth
- latência
- storage
- memória
- cache
- compute resources

O modelo é compilado para a máquina no arranque.

================================================================
20. STREAMING WEIGHTS
================================================================

Os pesos não precisam de ser vistos como um objecto gigante que tem
de ser "carregado".

Podem ser um stream:

NVMe
→ tile 0
→ tile 1
→ tile 2
→ ...

O modelo pode ser muito maior que a RAM disponível.

O objectivo é manter o pipeline cheio.

================================================================
21. EXECUTABLE / COMPUTE-AWARE MODEL FORMAT
================================================================

Investigar um formato de modelo que armazene:

- pesos
- quantização
- tile layout
- operação
- tamanho
- destino
- dependências
- próximo bloco

Exemplo conceptual:

operation = GEMM
format = INT4
tile = 32x32
source = stream 42
destination = compute stage 3
next = offset X

O storage deixa de guardar apenas "dados".
Guarda dados optimizados para o fluxo de computação.

================================================================
22. QUANTIZAÇÃO COMO PARTE DO RIO
================================================================

Evitar:

INT4
→ RAM
→ dequantize
→ FP16
→ compute

Sempre que o hardware permitir:

INT4
→ tile
→ compute

Guardar e transportar os dados no formato mais compacto possível,
mas sem introduzir mais computação de descodificação do que a bandwidth
poupada justifica.

================================================================
23. PROCESSING NEAR STORAGE
================================================================

Investigar armazenamento com capacidade de:

- decompress
- unpack
- reorder
- tile
- filter

antes de enviar os dados.

Não necessariamente como requisito inicial, mas como evolução da
arquitectura.

Objectivo:

"não transferir dados que ainda precisam de ser transformados".

================================================================
24. DIRECT DEVICE LINKS
================================================================

Sempre que o hardware permitir, procurar:

storage → accelerator
storage → memory
device → device

em vez de forçar todos os caminhos através do CPU.

Mas a arquitectura não depende disto.

É uma optimização opcional do link.

================================================================
25. KV CACHE COMO STREAM
================================================================

O KV cache pode ser tratado como um fluxo e não simplesmente como
um bloco de memória crescente.

Classificar informação como:

HOT
WARM
COLD

e adaptar:

- localização
- precisão
- frequência de acesso
- prefetch
- retenção

O objectivo é reduzir tráfego sem destruir contexto relevante.

================================================================
26. ADAPTIVE PRECISION
================================================================

Não assumir que todos os pesos precisam da mesma precisão.

Explorar:

INT2
INT3
INT4
INT8
FP16
BF16

dependendo de:

- sensibilidade da layer
- canal
- tensor
- contexto
- hardware

Mais precisão onde realmente produz valor.
Menos precisão onde bandwidth é mais importante.

================================================================
27. ADAPTIVE COMPUTE
================================================================

Nem todo o input precisa necessariamente do mesmo custo computacional.

Investigar:

- token importance
- layer sensitivity
- activation importance
- dynamic layer skipping
- approximate computation

Mas qualquer mecanismo de skipping tem de ser validado contra
qualidade do modelo.

================================================================
28. SPECULATIVE DATAFLOW
================================================================

Speculative decoding não deve ser tratado apenas como uma técnica de
LLM.

Pode tornar-se parte do rio.

Modelo pequeno:
→ produz hipóteses

Modelo grande:
→ verifica

Enquanto o grande verifica:
→ pequeno continua a especular

Isto cria dois rios probabilísticos em paralelo.

================================================================
29. PREDICTION DENTRO DO PIPELINE
================================================================

Como a estrutura da LLM é conhecida, o runtime pode preparar:

- próximos pesos
- próximos tiles
- próximos buffers
- próximos pedidos de I/O
- próximos kernels

antes de precisar deles.

É uma forma de aplicar a ideia de branch prediction ao dataflow da LLM.

Se a previsão estiver correcta, a latência fica escondida.
Se estiver errada, descarta-se o trabalho especulativo.

================================================================
30. MODELO COMO PROGRAMA DE DATAFLOW
================================================================

A representação final do modelo pode aproximar-se de:

LOAD_TILE
DECODE_TILE
MATMUL
REDUCE
NORMALIZE
ATTENTION
STORE_STATE
PREFETCH
...

Não necessariamente como instruções físicas, mas como uma IR universal
para o runtime.

Isso cria uma espécie de "ISA virtual de inferência".

================================================================
31. TRÊS RIOS SIMULTÂNEOS
================================================================

A arquitectura completa possui três fluxos.

RIO FÍSICO:

SSD → RAM → cache → registers → compute

RIO COMPUTACIONAL:

GEMM → norm → activation → attention → residual

RIO PROBABILÍSTICO:

context → prediction → speculation → verification → next token

O objectivo é coordenar os três.

================================================================
32. LATENCY ILLUMINATION
================================================================

Não tentar necessariamente reduzir cada latência.

Tentar sobrepor as latências.

Exemplo:

Storage:
████████████████████

Memory:
████████████████████

Decode:
████████████████████

Compute:
████████████████████

Se todos estiverem ocupados simultaneamente, a latência individual
fica escondida.

================================================================
33. PIPELINE OCCUPANCY
================================================================

Criar uma métrica central:

Pipeline Occupancy

Medir durante a geração:

SSD utilization
RAM utilization
cache activity
load units
store units
vector units
matrix units
core utilization
queue occupancy
stall cycles

Objectivo:

nenhum estágio permanece vazio durante períodos significativos.

Quando um estágio está vazio, investigar a ponte anterior.

================================================================
34. BYTES MOVED PER USEFUL OPERATION
================================================================

Não medir apenas FLOPS.

Criar:

Bytes moved / useful MAC

O objectivo é reduzir a quantidade de informação movimentada para
realizar cada operação útil.

Esta métrica pode ser mais relevante para inferência CPU do que
simplesmente FLOPS.

================================================================
35. NOVA REGRA DE OPTIMIZAÇÃO
================================================================

1. Não esperar por algo que podemos antecipar.
2. Não materializar algo que podemos transmitir.
3. Não mover algo que pode permanecer onde está.
4. Não calcular algo que podemos evitar.
5. Não usar precisão superior à necessária.
6. Não criar sincronização quando ownership pode resolver o problema.
7. Não assumir hardware específico.
8. Não optimizar apenas compute; optimizar o caminho inteiro.
9. Não deixar uma unidade parada quando existe trabalho independente.
10. Medir sempre o pipeline inteiro.

================================================================
36. POSSÍVEL ARQUITECTURA FINAL
================================================================

                    LLM GRAPH
                        │
                        ▼
                GRAPH OPTIMIZER
                        │
                        ▼
                DATAFLOW COMPILER
                        │
                        ▼
              HARDWARE CAPABILITY GRAPH
                        │
                        ▼
                 EXECUTION PLAN
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
     STORAGE          MEMORY           COMPUTE
        │               │                │
        └───────────────┼────────────────┘
                        ▼
                 CONTINUOUS RIVER
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
     Attention          MLP              KV
        │               │                │
        └───────────────┼────────────────┘
                        ▼
                    NEXT STATE
                        │
                 SPECULATION
                        │
                        ▼
                    NEXT TOKEN

================================================================
37. A IDEIA MAIS RADICAL
================================================================

Não construir "uma CPU melhor".
Não construir "uma GPU melhor".
Não construir "um SSD melhor".

Construir uma camada que faça CPU, RAM, SSD, GPU, NPU e outros
componentes comportarem-se como uma única máquina de fluxo.

O hardware continua a ser heterogéneo.
O software cria as pontes.

A abstração fundamental deixa de ser:

CPU + RAM + DISK

e passa a ser:

SOURCE → TRANSFORM → COMPUTE → STATE → TRANSFORM → COMPUTE

================================================================
38. POSSÍVEIS EXTENSÕES REVOLUCIONÁRIAS
================================================================

A. LATENCY BUDGETING

Atribuir um orçamento de latência a cada estágio.

Se um link tem 10 µs de latência e o consumidor consegue trabalhar
durante 50 µs, o scheduler tenta garantir pelo menos 5 requests
em voo.

Não optimizar apenas bandwidth; optimizar o produto:

latency × outstanding work

B. AUTO-TUNING DO RIO

Durante o arranque e durante a execução, testar:

- tile sizes
- prefetch distance
- ring size
- número de requests em voo
- número de threads
- core placement
- precision
- fusion

e escolher automaticamente a configuração com maior throughput e
menor stall.

C. FEEDBACK LOOP

O runtime mede continuamente:

queue occupancy
cache misses
stall cycles
bandwidth
compute utilization

e altera o pipeline.

Ou seja, o scheduler não é estático.

É um sistema de controlo.

D. DIGITAL TWIN DO HARDWARE

Criar um modelo interno da máquina:

latency matrix:
component A → component B

bandwidth matrix:
component A → component B

capability matrix:
component A supports operation X

O planner utiliza esse modelo para encontrar caminhos.

E. ZERO-COPY SEM SER OBCECADO POR ZERO-COPY

"Zero-copy" não deve ser um objectivo absoluto.

O objectivo é:

mínimo de bytes transferidos + máxima sobreposição.

Uma cópia extremamente barata e perfeitamente sobreposta pode ser
melhor do que uma operação complexa de zero-copy que cria stalls.

F. COMPUTE-AWARE STORAGE LAYOUT

Organizar fisicamente o modelo no storage segundo a ordem em que será
consumido.

O layout no disco passa a ser parte da optimização do runtime.

G. SELF-COMPILING MODEL

O modelo, hardware e runtime podem gerar um execution plan específico
no primeiro arranque.

O mesmo ficheiro de modelo pode produzir pipelines completamente
diferentes em máquinas diferentes.

================================================================
39. OBJECTIVO FINAL
================================================================

A pergunta tradicional:

"Quanto tempo demora a carregar este tensor?"

deve desaparecer.

A pergunta passa a ser:

"Conseguimos manter o consumidor alimentado?"

Se sim, o sistema está a fluir.

Se não, o profiler identifica a resistência:

storage
→ link
→ memory
→ cache
→ decode
→ register
→ execution
→ cross-core
→ synchronization
→ model computation

E a próxima optimização é aplicada precisamente nesse ponto.

A visão final é uma máquina de inferência em que os dados nunca estão
simplesmente "parados".

Eles estão sempre:

a chegar,
a transformar-se,
a ser consumidos,
a ser reduzidos,
ou a preparar o próximo estágio.

O objectivo não é eliminar a arquitectura existente.

É transformar a arquitectura existente num RIO.
