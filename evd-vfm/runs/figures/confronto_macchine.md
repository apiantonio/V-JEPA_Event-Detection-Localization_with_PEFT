### Le due macchine — `e5b2_efficienza_gpuq` contro `e5b2_efficienza_aiq`

⚠️ **Le latenze di NVIDIA A100-SXM4-80GB e NVIDIA H100 80GB HBM3 non si confrontano fra loro**: qui compaiono solo come **rapporto**, e l'escursione di quel rapporto fra varianti è la riga che dice se «k volte più veloce» sia una frase lecita.

| variante | batch | NVIDIA A100-SXM4-80GB | NVIDIA H100 80GB HBM3 | rapporto |
|---|:-:|---:|---:|---:|
| dora-r32 | 1 | 241,4 ms | 146,4 ms | 1,649× |
| dora-r32 | 2 | 459,0 ms | 271,0 ms | 1,694× |
| dora-r32 | 4 | 879,5 ms | 513,3 ms | 1,713× |
| dora-r32 | 8 | 1.730,0 ms | 1.000,4 ms | 1,729× |
| dora-r32-fusa | 1 | 131,6 ms | 78,0 ms | 1,688× |
| dora-r32-fusa | 2 | 250,3 ms | 144,5 ms | 1,733× |
| dora-r32-fusa | 4 | 477,5 ms | 273,5 ms | 1,746× |
| dora-r32-fusa | 8 | 936,6 ms | 534,3 ms | 1,753× |
| frozen | 1 | 131,7 ms | 77,9 ms | 1,689× |
| frozen | 2 | 250,3 ms | 144,5 ms | 1,732× |
| frozen | 4 | 477,6 ms | 272,7 ms | 1,752× |
| frozen | 8 | 937,2 ms | 533,2 ms | 1,758× |
| lora-r32 | 1 | 187,7 ms | 112,6 ms | 1,666× |
| lora-r32 | 2 | 362,1 ms | 211,4 ms | 1,713× |
| lora-r32 | 4 | 697,3 ms | 401,9 ms | 1,735× |
| lora-r32 | 8 | 1.375,3 ms | 785,0 ms | 1,752× |
| lora-r32-fusa | 1 | 131,7 ms | 77,7 ms | 1,695× |
| lora-r32-fusa | 2 | 250,5 ms | 144,4 ms | 1,734× |
| lora-r32-fusa | 4 | 477,7 ms | 273,5 ms | 1,747× |
| lora-r32-fusa | 8 | 936,5 ms | 534,0 ms | 1,754× |
| vptdeep-n576 | 1 | 133,6 ms | 78,1 ms | 1,712× |
| vptdeep-n576 | 2 | 251,8 ms | 143,5 ms | 1,755× |
| vptdeep-n576 | 4 | 486,4 ms | 272,1 ms | 1,788× |
| vptdeep-n576 | 8 | 952,6 ms | 534,1 ms | 1,784× |

Rapporto mediano **1,733×**, escursione 1,649–1,788×.

**L'overhead, che invece è confrontabile** — è un rapporto interno a ciascuna esecuzione, ed è il numero di C7:

| variante | batch | NVIDIA A100-SXM4-80GB | NVIDIA H100 80GB HBM3 | scarto |
|---|:-:|---:|---:|---:|
| dora-r32 | 1 | 83,40% | 87,85% | -4,45 punti |
| dora-r32 | 2 | 83,34% | 87,51% | -4,17 punti |
| dora-r32 | 4 | 84,13% | 88,24% | -4,11 punti |
| dora-r32 | 8 | 84,58% | 87,64% | -3,05 punti |
| dora-r32-fusa | 1 | -0,02% | 0,05% | -0,07 punti |
| dora-r32-fusa | 2 | -0,02% | -0,04% | 0,02 punti |
| dora-r32-fusa | 4 | -0,02% | 0,31% | -0,33 punti |
| dora-r32-fusa | 8 | -0,06% | 0,21% | -0,27 punti |
| lora-r32 | 1 | 42,58% | 44,56% | -1,98 punti |
| lora-r32 | 2 | 44,65% | 46,27% | -1,62 punti |
| lora-r32 | 4 | 45,99% | 47,37% | -1,39 punti |
| lora-r32 | 8 | 46,74% | 47,23% | -0,49 punti |
| lora-r32-fusa | 1 | 0,04% | -0,28% | 0,32 punti |
| lora-r32-fusa | 2 | 0,06% | -0,05% | 0,11 punti |
| lora-r32-fusa | 4 | 0,02% | 0,29% | -0,28 punti |
| lora-r32-fusa | 8 | -0,08% | 0,15% | -0,23 punti |
| vptdeep-n576 | 1 | 1,51% | 0,20% | 1,31 punti |
| vptdeep-n576 | 2 | 0,60% | -0,71% | 1,31 punti |
| vptdeep-n576 | 4 | 1,84% | -0,22% | 2,05 punti |
| vptdeep-n576 | 8 | 1,64% | 0,17% | 1,48 punti |

**L'ordinamento dei metodi per overhead si conserva?** **NO** — il verdetto di C7 dipende dalla macchina, e va dichiarato

* ⚠️ l'ordinamento dei metodi per overhead NON è lo stesso sulle due macchine nemmeno a meno di 1 punti: [[['dora-r32-fusa', 'lora-r32-fusa'], ['vptdeep-n576'], ['lora-r32'], ['dora-r32']], [['dora-r32-fusa', 'lora-r32-fusa', 'vptdeep-n576'], ['lora-r32'], ['dora-r32']]] — il verdetto di C7 dipende dall'hardware e va dichiarato tale
