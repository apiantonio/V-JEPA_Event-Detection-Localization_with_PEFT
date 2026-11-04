---
titolo: "Mappa della codebase: cosa esiste, cosa fa, perché"
versione: "1.86"
data: 2026-10-10

---

# Mappa della codebase

> **Questo documento è descrittivo, non normativo.** Dice *cosa esiste oggi e perché*.
> A chi serve: a un agente o a una persona che apre il repository e deve capire in dieci minuti
> dove mettere le mani, senza leggere 10 documenti di progetto o 5000 righe di codice.

---

## 1. Dove sta cosa

Il workspace `TESI/codice` ha sei sottoalberi principali con permessi diversi:

| Sottoalbero | Cos'è | Ci si scrive? |
|---|---|---|
| `vjepa2/` | junction al clone di `facebookresearch/vjepa2` — il **riferimento numerico**; il commit locale accettato è `b921227` | mai |
| `archivio/` | copie congelate e provenienza di materiali sostituiti dalle versioni vive | solo per archiviazione esplicita |
| `evd-vfm/` | **il repository della tesi**: tutto il codice che produce numeri | sì |
| `lab/` | esplorazione di V-JEPA 2.1: script usa-e-getta e note di lettura | sì |
| `hpc/` | accesso al cluster: variabili, template SLURM, trasferimenti | sì |
| `data/` | video, annotazioni, checkpoint, feature (fuori da git) | solo gli script |

---

## 2. L'idea che tiene insieme `evd-vfm`

Tutto il repository è costruito attorno a **una catena sola**, la *fetta verticale*:

```
video → finestre causali → backbone → testa → score per chunk → metriche primarie
      → proiezione causale sui frame → metriche derivate → figura → registry
```

Ogni modulo di `src/evd/` è **un anello** della catena, e ogni anello ha un'interfaccia
(`typing.Protocol`) così che il pezzo dentro si possa sostituire senza toccare gli altri. Lo
smoke test percorre la catena intera in ~35 s su CPU: quando è verde, la spina dorsale è integra.

Tre regole spiegano il 90% delle scelte che si incontrano leggendo:

1. **Le definizioni sono fissate; i valori di protocollo no.** `k` conta segmenti di evento
   annotati per classe, `m` limita quante finestre di training si derivano da ciascun segmento,
   `target_fps` è una frequenza richiesta e lo span di N campioni è `(N-1)/target_fps`. I valori
   ancora da approvare restano `???` nei config (`docs/04`, ADR-024).
2. **Ogni conversione ambigua avviene in un punto solo.** timestamp target→indici nativi,
   secondi→intervalli e finestre→frame. È la difesa contro R11.
3. **Un numero senza commit non esiste.** Il tracker rifiuta di registrare un run ufficiale da
   working tree sporco (invariante 2).

---

## 3. `src/evd/` — la libreria, modulo per modulo

### 3.1 Le fondamenta

#### [types.py](../src/evd/types.py) — la lingua franca

**Cosa fa:** definisce gli oggetti immutabili che attraversano tutto il sistema.

| Tipo | Cos'è |
|---|---|
| `PreprocessSpec` | come si trasformano i pixel: fps, frame, size, geometria, backend, versione della libreria, normalizzazione, dtype. **Nessun campo ha default** |
| `WindowSpec` | una finestra **causale** con gli indici nativi effettivi; supporta passi non uniformi e termina in `end_frame`, mai centrata |
| `VideoMeta` | ciò che si sa di un file prima di decodificarlo |
| `BackboneOutput` | uscita uniforme di ogni backbone: `tokens [B,N,D]` + `grid`. Da ADR-029 porta anche `levels: tuple[int, ...]` — vuota per un layer solo, altrimenti i blocchi impilati **sull'asse token** in quell'ordine, e `layer` deve esserne l'ultimo |
| `TokenLayout` | dove stanno i token non-patch (serve a VPT) |
| `EventClass`, `WindowScores`, `FrameScores` | classi, score di anomalia e probabilità per classe a livello chunk, più la loro proiezione sui frame |

**Perché sta qui e non accanto ai suoi moduli:** `docs/02` §5 vieta che `data` importi `models`
e viceversa. Ma `PreprocessSpec` la *dichiara* un backbone e la *consuma* un dataset: metterla
in uno dei due creerebbe la dipendenza vietata.

**Il dettaglio che conta:** `PreprocessSpec.hash()` è la chiave della cache delle feature, e
copre anche la *versione della libreria di resize* — S2 ha misurato che cambiare backend sposta
l'uscita dell'encoder del 3–7%.

**`concat_levels(outs) -> BackboneOutput`** *(ADR-029, 26 agosto)* — funzione libera nello stesso
modulo, **unica implementazione** di cosa significa "multilevel": `torch.cat(..., dim=1)` sui
token di più `BackboneOutput`, nell'ordine dato, con `grid` di un livello e `levels` valorizzato.
Sta qui e non in `models/` perché la usano sia il percorso pixel
(`models/backbones/base.py::encode_multilevel`) sia quello che legge il feature store
(`data/dataset.py`), e `data` non può importare `models` (`docs/02` §5). Con un solo elemento
restituisce l'ingresso invariato — è la ragione per cui il caso a layer singolo non ha un
percorso a sé.

#### `utils/`

| File | Cosa fa | Perché |
|---|---|---|
| [env.py](../src/evd/utils/env.py) | legge le variabili d'ambiente del progetto | I default stanno **solo** in `hpc/env.sh`; questo modulo li legge e non li reinventa. `device()` **verifica** `torch.cuda.is_available()` invece di fidarsi: il job 531853 su `fatq` è morto perché SLURM mette le variabili GPU a `-1` invece di lasciarle vuote |
| [hashing.py](../src/evd/utils/hashing.py) | `stable_hash()`, `file_sha256()` | Hash stabili fra portatile e cluster, fra oggi e settembre. Mai `hash()` builtin, che su str è randomizzato per processo |
| [io.py](../src/evd/utils/io.py) | scritture **atomiche**, JSON, JSONL append-only | Un job ucciso al Wall Time a metà di una `json.dump` lascia un file che *esiste* — e il passo dopo lo legge |
| [logging.py](../src/evd/utils/logging.py) | formato uniforme dei log | Un solo posto dove si decide |
| [seeding.py](../src/evd/utils/seeding.py) | `seed_all()` | Promette che la sequenza di scelte casuali si ripeta, **non** bitwise: su CPU il forward non è bitwise riproducibile nemmeno dentro lo stesso processo (S1–S3) |

#### [vendor/](../src/evd/vendor/) — le copie verbatim del sorgente Meta
**Cosa c'è:** `vjepa21_rope.py` (3D-RoPE), e da ADR-028 `vjepa21_randaugment.py`,
`vjepa21_randerase.py`, `vjepa21_video_transforms.py` — le trasformazioni della catena di
training del probe.
**Regola:** copie **non modificate**, ciascuna con file, commit del clone e **sha256**
dell'originale in testa, e la riga «cosa è cambiato» (invariante 11, `CLAUDE.md` §3.2). Le
funzioni sono estratte con `ast`, non ricopiate a mano.
**`ruff` esclude questa cartella** (`pyproject.toml`): un file riformattato — anche solo
`flip((-1))` → `flip(-1)` — non è più una prova di fedeltà, è una nostra versione che le somiglia.

### 3.2 `data/` — dai file ai tensori

#### [decoder.py](../src/evd/data/decoder.py)
**Cosa fa:** legge frame da un video **per indice**. `PyAVDecoder` cerca il keyframe che precede
il primo indice richiesto e ricostruisce gli indici assoluti dai PTS; se il container non
permette una mappatura sicura, **ripiega sulla lettura sequenziale**.
**Produce:** `[T,H,W,3]` uint8 RGB, nell'ordine chiesto.
**Perché un'interfaccia (ADR-008):** `decord` — quello che usa Meta — non si installa su Windows
oltre Python 3.10; `torchcodec` non è su nessuna delle due macchine. Il decoder è un campo di
config. Il fallback sequenziale è deliberato: **meglio lento che disallineato**.

#### [preprocess.py](../src/evd/data/preprocess.py)
**Cosa fa:** pixel uint8 → clip normalizzata, secondo una `PreprocessSpec`.
**Produce:** `[C,T,H,W]`, oppure `[V,C,T,H,W]` con le tre viste spaziali.
**Perché è delicato:** S2 ha scoperto che nel repository di Meta convivono **tre** catene di
preprocessing, e quale si esegue dipende perfino dal *tipo* dell'argomento passato a
`resize_clip`. La differenza vale L2 relativa ~0,6 sull'uscita — quanto cambiare vista. Qui le
tre alternative sono nomi espliciti (`eval_three_views`, `eval_center`, `demo_center`) e la
scelta fra loro resta al protocollo (Q15).

#### [augment.py](../src/evd/data/augment.py) — l'augmentation di training *(ADR-028)*
**Cosa fa:** pixel uint8 nativi → clip **aumentata** e normalizzata `[C,T,H,W]`, replicando la
catena di training del probe ufficiale V-JEPA 2.1.
**Riferimento:** `vjepa2/evals/video_classification_frozen/eval.py:414-447` (che i parametri li
**cabla**: nessuna chiave YAML li tocca, quindi la catena è identica per k400/ssv2/coin/diving48/
jester) applicata da `utils.py:104-131`.
**L'ordine non è commutativo:** RandAugment sui frame **interi** → `ToTensor` → normalizzazione →
`random_resized_crop` → flip (che l'ufficiale **non** usa) → random erasing.
**Non riscrive le trasformazioni:** sono copie verbatim in `vendor/` (sotto). Qui c'è solo
l'orchestrazione, il seeding e la conversione ai tipi del repo.
**Dove NON si applica:** query e validazione episodica, mai. Il confine è nell'API —
`WindowDataset` solleva se riceve store e augmentation insieme.
**Perché semina e ripristina i generatori globali:** upstream pesca da `random`/`numpy`/torch
globali; qui ogni finestra ha un seme da (seme del run, `window_id`, pescaggio, worker) e lo stato
globale viene rimesso com'era. Senza, l'augmentation sposterebbe anche l'inizializzazione della
testa e lo shuffle, e due run con la stessa config darebbero pesi diversi (invariante 2).

#### [windows.py](../src/evd/data/windows.py) — il cuore temporale
**Cosa fa:** `CausalWindowSampler` genera le finestre di un video.
**Produce:** liste di `WindowSpec`.
**Le regole:** ogni finestra **termina in t**, mai centrata. I timestamp target sono ancorati a
una griglia temporale assoluta e convertiti negli indici nativi con arrotondamento half-up; non
si accumula un errore di hop e il passo fra indici può essere non uniforme.

**Da ADR-047 (31/8) il modulo porta DUE griglie, e il campo `grid` dice quale.** Restano in un
modulo solo perché l'invariante 4 vuole un windowing solo:

| `grid` | come si definisce una finestra | la coda del video | video più corto di una finestra |
|---|---|---|---|
| `endpoint` (storica, ADR-024) | dal suo **endpoint**, campioni a `i/target_fps` a ritroso | l'ultima finestra termina sull'ultimo frame | **nessuna** finestra |
| `fixed` (in vigore, `edp_v2`) | dal suo **inizio**, `k · hop`, durata `window_s`, estremi semiaperti | la prima finestra più corta di `window_s` **si valuta e chiude il video** | **una** finestra, troncata |

Sulla `fixed` i `num_frames` campioni si distribuiscono **uniformemente sulla finestra** (la
`linspace` di `loadvideo_decord`), quindi il contenuto osservato coincide con l'intervallo
etichettato — al prezzo di **3,89 fps effettivi** invece di 4,0 (`effective_fps` lo riporta).
L'API pubblica è per **span**: `spans_for_video`, `window_for_span`, `span_at`, più
`window_frames` e `hop_frames`; `windows_for_video` e `window_at` restano come involucri.
**Da F11 (28/8) due metodi per il campionamento dinamico, che `edp_v2` non usa
(`train_sampling: static`) e che restano per `edp_v1`:**
`window_at(meta, end_frame)` costruisce la finestra a un endpoint che **non** sta sulla griglia
a passo hop, e `endpoint_cell(meta, end_frame)` dà gli endpoint ammessi attorno a un punto della
griglia — la cella è **centrata** e larga un hop, tagliata sui bordi del video. ⚠️ Il centraggio
e la larghezza sono una **scelta nostra dichiarata**: ADR-031 punto 6 fissa il meccanismo
(«endpoint estratto uniformemente dentro la sua cella di hop»), non la convenzione.
`windows_for_segment(..., max_windows=m)` resta, ma è del regime few-shot: nessun percorso in
vigore lo chiama.

#### [dataset.py](../src/evd/data/dataset.py)
**Cosa fa:** `WindowDataset` — da `WindowSpec` a tensori pronti per il backbone.
Su un hit del `FeatureReader` restituisce token/grid/layer; su un miss decodifica i pixel. Il
collate rifiuta batch misti. Il confine importante è nell'API: `build_frozen_loader` accetta la
cache, `build_adapted_loader` non ha quel parametro. `cache_in_memory` è un'altra cosa: tiene i
**pixel** del piccolo support in RAM anche durante PEFT.
**Da ADR-029, `WindowItem.cache_key: str | None` è `cache_keys: tuple[str, ...]`** (rinominato,
non affiancato — un secondo campo avrebbe aperto un secondo percorso): una chiave per livello.
La lettura è **tutto-o-niente** — una sola chiave mancante è un miss, mai un hit parziale — e
`collate_windows` rifiuta un batch che mescola `levels` diversi, accanto a `grid` e `layer`.
`feature_key`/`FeatureStore.key` non cambiano (ADR-007 intatto): si producono N chiavi con la
funzione esistente.
**Da F11 (28/8) il dato porta il proprio peso e sa cambiare finestra.** `WindowItem.weight` è il
peso di campionamento non normalizzato: vive nel dato e non in una config del loader perché due
famiglie PEFT devono condividere **lo stesso** campionamento senza riconfigurarlo (invariante 5).
Lo consumano i due `build_*_loader` con `weighted=True`, che costruiscono un
`WeightedRandomSampler` con reimmissione e rifiutano un peso nullo — sarebbe una finestra nel
pool che non viene mai pescata, cioè scartata senza dirlo. `resample=` è il contratto
`Resampler` `(item, pescaggio, worker) -> item` del campionamento dinamico: il dataset **non sa**
cosa cambia — la regola di positività è protocollo e vive in `evd.protocol.chunks` — e si limita
a servire la finestra del pescaggio, saltando la cache RAM che conterrebbe la clip di un altro
endpoint. `resample` e `cache` si escludono con un errore esplicito: una finestra estratta a caso
non ha una chiave stabile.

#### [feature_store.py](../src/evd/data/feature_store.py) · [feature_extraction.py](../src/evd/data/feature_extraction.py) *(F4)*
**Cosa fanno:** il primo salva feature congelate come un safetensors per (video, layer), con
indice SQLite e metadati verificati; il secondo applica finestre causali, sharding deterministico,
resume e limite di spazio prima di estrarre. Se servono più layer della stessa finestra, usa un
solo forward e separa le uscite soltanto al momento della scrittura.
**Le difese:** chiave SHA256 su backbone/checkpoint/preprocessing/layer/finestra; dtype esplicito
fp16 o fp32, mai bf16; scrittura atomica; un indice per task dell'array e merge che rifiuta
duplicati. Il replay dello smoke deve avere hit rate 100%.
⚠️ **La griglia va passata, e dal 31 agosto si può** (ADR-049). `windows_for_records` costruiva
il sampler senza `grid` né `window_s`, cioè sempre `endpoint`: su `edp_v2`, che usa `fixed`, le
chiavi sarebbero state disgiunte da quelle cercate dal run — **zero hit**, e nessun errore, perché
un miss è un caso legittimo. La griglia **non** entra in `store_meta.json` (`preprocess_hash`
copre fps, frame, size e geometria, non l'hop né l'ancoraggio), quindi due estrazioni con griglie
diverse hanno lo stesso hash e chiavi disgiunte: la si dichiara nel report dell'estrazione.

#### [inventory.py](../src/evd/data/inventory.py) — l'anagrafe dei video
**Cosa fa:** scansiona i video (shardabile), ne misura fps/durata/frame/codec **decodificandoli
davvero**, e fonde gli shard usando gli split ufficiali come canone.
**Produce:** `inventory.csv`, `pathologies.csv`, `excluded.csv`, `summary.json`,
`MANIFEST.sha256`.
**Cosa ha trovato sui dati veri:** 1950 file fisici, di cui **1900 canonici** (i 50 di
`z_Normal_Videos_event` sono copie fuori split); **21 coppie byte-identiche** dentro gli split,
otto delle quali attraversano train/test, e una — `Assault050`/`Robbery138` — è lo stesso file
etichettato con due classi diverse (Q19).

#### [synthetic.py](../src/evd/data/synthetic.py) — il mondo giocattolo
**Cosa fa:** genera "sorveglianza finta" deterministica: video 64×64, 120 frame, un quadrato che
attraversa la scena e in un intervallo noto diventa **veloce e luminoso**.
**Produce:** mp4 veri + `TemporalAnnotation` perfette.
**Perché:** permette di testare l'intera pipeline senza dati veri e senza GPU. La luminosità è
stata aggiunta *misurando*: con la sola velocità la differenza restava sotto il 5% dei pixel,
troppo poco perché il segnale fosse ovvio per un ViT a pesi casuali.

### 3.3 `annotations/` — cosa è successo, e quando *(F3)*

#### [schema.py](../src/evd/annotations/schema.py)
Due tipi gemelli, e la differenza è il punto:
- **`TemporalAnnotation`** — il tipo *valido*: `(video_id, intervalli)` in frame. **Solleva** su
  un intervallo invertito. Un video normale è `intervals=()`: non esiste un "intervallo di
  normalità", esiste l'assenza di eventi.
- **`RawInterval`** — il tipo *tollerante*: una riga di file prima di essere validata, con
  l'unità (`frames`|`seconds`) e l'origine (`file:riga`).

Più `frame_labels()` / `frame_class_labels()` (annotazione → vettore per frame) e `to_frames()`,
**l'unico punto di conversione secondi→frame del sistema**.

#### [ucf_classes.py](../src/evd/annotations/ucf_classes.py)
Le 13 classi di anomalia + `Normal`, e la mappa cartella-di-split → classe. È un fatto del
dataset, non protocollo: il manifest sigilla comunque la propria tabella `classes`.

#### [validate.py](../src/evd/annotations/validate.py) — il report per il laboratorio
**Cosa fa:** incrocia le righe di un file di annotazioni con l'inventario reale e la tabella
delle classi.
**Produce:** un `ValidationReport` con rilievi tipizzati (`interval_inverted`, `unknown_class`,
`interval_beyond_num_frames`, `overlap_different_class`, `class_mismatch`,
`anomalous_video_without_intervals`, …), statistiche per classe in frame **e in secondi**, e
l'analisi 0-based vs 1-based. Esporta `.json` e `.md`.
**La regola:** **non solleva mai**. Il destinatario del report è chi il file lo ha prodotto, e
una traceback non gli serve.

#### [ucf_sultani.py](../src/evd/annotations/ucf_sultani.py)
Il fallback pubblico: `Abuse028_x264.mp4  Abuse  165  240  -1  -1`. Solo test set (290 video).
`index_base` è **un'ipotesi dichiarata**, non un fatto: il file non dice se gli indici siano 0-
o 1-based, e il validatore conta quanti intervalli sforerebbero sotto ciascuna convenzione.

#### [ucf_lab.py](../src/evd/annotations/ucf_lab.py)
L'adapter di FS-UCF-Crime — **il file è arrivato il 19 agosto** (R1 chiuso). Una riga per video,
`path (sigla inizio fine)+`, indici in **frame**, sigle sciolte da `ALL_target_classes.json`; una
riga col solo path è un video *senza eventi*, che non è la stessa cosa di un video assente.
Il sidecar `format.json` non c'è più: era un'astrazione nata prima della seconda implementazione
(`../CLAUDE.md` §5bis-7).

Due cose che l'adapter fa e che nessun tipo dice: **`clamps()`** elenca gli 8 intervalli che
terminano oltre `num_frames - 1` e `load()` ne registra il numero (Q20) — un clamp silenzioso è il
modo esatto in cui un fps sbagliato passa inosservato; e **la classe si legge dalla sigla, mai
dalla cartella**, perché 216 intervalli su 1831 discordano. `class_from_relative_path` resta ma
risponde a un'altra domanda: «in che cartella dello split ufficiale sta questo video».

> Ogni adapter espone **due stadi**: `parse()` tollerante (per il validatore) e `load()` strict
> (è l'interfaccia `AnnotationSource`). Convenzione fra i due: `class_name == ""` significa
> «riga che il parser non ha saputo leggere», conservata invece che scartata.

### 3.4 `protocol/` — chi vede quali etichette *(F3, riscritto il 27/8)*

> ⚠️ **Due regimi convivono in questo package, e non è disordine.** Dal 27 agosto
> (ADR-031) il protocollo in vigore è `edp_v1`, fully-supervised: lo descrivono
> `split.py` e `chunks.py`, entrambi nuovi e coperti da test. `manifest.py` e
> `builder.py` descrivono il regime **few-shot**, che non eseguiamo più: restano
> perché i 35 manifest di `fsp_v1` devono restare leggibili e i run che li hanno
> consumati riproducibili (invariante 2). Non sono il percorso di produzione.

#### [split.py](../src/evd/protocol/split.py) — **in vigore**
**Cosa fa:** `SplitSpec`, le tre partizioni di video (train/val/test) sigillate con
uno sha256 sul contenuto senza il campo stesso, più `annotation_source` che lega
lo split alla versione esatta delle annotazioni.
**Le tre difese:** `save_split` rifiuta di sovrascrivere · `load_split` verifica
schema **e** sigillo · un `protocol_version` che comincia per `fsp` viene rifiutato
esplicitamente, perché è un file di un'altra *forma* e caricarlo darebbe uno split
senza partizioni.
**Una cosa che non c'è, ed è deliberato:** la griglia dei chunk. È una funzione
pura di (split, annotazioni, geometria), e sigillarne 104.767 righe duplicherebbe
un'informazione derivabile — la forma d'errore che ha già invalidato lo store E0d.
`ha_validazione` esiste perché «non c'è una partizione di validazione» è uno stato
**dichiarato** (Q-C), non un motivo per ripiegare in silenzio sul test.

#### [split_builder.py](../src/evd/protocol/split_builder.py) — **in vigore** *(28/8)*
**Cosa fa:** costruisce quello split. `ValidationRules` (nessun campo con default,
come `SamplingRules`) più `build_split`, che applica il dedup Q19 al pool di
training, raggruppa i video in **strati** — l'insieme delle classi che compaiono
nei loro intervalli, letto dall'annotazione e mai dalla cartella — e ne sposta in
validazione `round(fraction · n)` per strato, con un RNG seminato dalla chiave
dello strato perché aggiungere una classe domani non riordini la validazione delle
altre.
**Perché non riusa `builder.py` né `exp/e0f.py`:** il primo campiona segmenti di
support a budget `k`, il secondo costruisce lo split del *pilota* E0f, dove la
quota è in video per classe e finestre per video. Qui il problema è partizionare in
due un insieme già dato, senza scartare niente.
**Due cose che restituisce e che non sono il file:** `SplitReport`, con la tabella
per strato che si guarda **prima** di sigillare, e l'elenco degli strati troppo
piccoli per la quota — le 28 combinazioni rare che restano intere in training
(33 video, 2,1%). Sono dichiarati invece di essere corretti con un pavimento a 1,
che manderebbe metà di uno strato da due video in validazione (ADR-035).

#### [chunks.py](../src/evd/protocol/chunks.py) — **in vigore**
**Cosa fa:** `enumerate_chunks` produce **tutti** i chunk di **tutti** i video di
una partizione, ciascuno con etichetta binaria, classe (per l'analisi, non per il
modello) e peso di campionamento `w_video · w_overlap`.
**Tre decisioni di protocollo vivono qui**, e ciascuna con la sua ragione nel
docstring: positività per **intersezione qualunque**; `w_video = 1/n_chunk(video)`,
senza cui `Normal_Videos308` (9 h) vale da solo il 7,8% del training set; e il
**pavimento a 0,1** del peso di sovrapposizione, che non è arrotondamento —
azzerare quei chunk reintrodurrebbe di nascosto la soglia di overlap che ADR-031
scarta.
**`riepilogo` misura, non dichiara:** il tasso di positivi effettivo è una
proprietà emergente dei pesi (10,93% uniforme → 22,59% pesato sul training reale),
quindi va scritto in `meta.json` dopo averlo calcolato — e da F11 il runner lo fa,
per tutte e tre le partizioni.
**Da F11 la regola di positività sta in una funzione sola:** `chunk_at(meta, ann,
sampler, end_frame, ...)` etichetta **il** chunk che finisce lì, e sia la griglia
statica (`chunks_for_video`) sia il campionamento dinamico la chiamano. Due
funzioni che etichettano lo stesso chunk in due modi sono un bug che si scopre
dentro un numero; un test lo tiene fermo confrontando le due strade su tutta la
griglia.


#### [manifest.py](../src/evd/protocol/manifest.py) — *storico, regime few-shot*
**Cosa fa:** definisce `FewShotManifest` — quali segmenti etichettati un metodo ha visto a un
dato budget `(k, m, seed)`, e su quali video si misura. Gli intervalli del manifest sono secondi
half-open `[start_s, end_s)`; le annotazioni native restano frame inclusivi.
**Produce:** JSON **sigillati** con uno sha256 calcolato sul contenuto senza il campo stesso.
**Le tre difese:** `save_manifest` rifiuta di sovrascrivere · `load_manifest` verifica schema
**e** sigillo · un manifest modificato dopo il commit invaliderebbe in silenzio ogni numero che
ci poggia sopra (invariante 1), quindi non deve poter accadere.

#### [builder.py](../src/evd/protocol/builder.py) — *storico, regime few-shot*
**Cosa fa:** campiona support e query secondo le regole di `docs/04` §3.
**Produce:** manifest sigillati, verificati contro le proprie regole.
**Due cose da capire:**
- **`SamplingRules` non nasconde decisioni.** k, `window_cap_m`, durata del support normale,
  vincolo di diversità e regola sui duplicati (Q19) arrivano dalla config. Il manifest sigilla
  esplicitamente l'unità `annotated_event_segments_per_class` insieme a `k` e `m`.
- **La monotonia in k è strutturale.** Per ogni (classe, seed) si estrae una permutazione **una
  volta sola** e i k crescenti *estendono* la selezione: il support a k=2 contiene quello a k=1
  per costruzione, non per fortuna. `check_monotonic()` lo verifica lo stesso.
- **Il bilanciamento normale non è ancora chiuso dal solo manifest.** `match_event_segments`
  sceglie lo stesso numero di segmenti normali dei segmenti-evento; `m` viene applicato quando si
  materializzano le finestre. Il binding reale deve quindi contare le finestre effettive e
  sottocampionare deterministicamente le normali fino alla quota approvata. Finché quel controllo
  non esiste, il builder non può rivendicare parità esatta fra finestre normali ed evento.

### 3.5 `models/` — il backbone e la testa

| File | Cosa fa |
|---|---|
| [registry.py](../src/evd/models/registry.py) | `@BACKBONES.register("tiny")`: un componente nuovo è un file più un decoratore, non un `if` nel runner |
| [backbones/base.py](../src/evd/models/backbones/base.py) | il contratto `VideoBackbone`: `encode()`, `encode_layers()`, `freeze()`, `eval()`, `preprocess_spec()`, `module()`, `blocks()`, `target_map()`, `token_layout()`, `predictor()`, `hierarchical_layers()` |
| [backbones/tiny.py](../src/evd/models/backbones/tiny.py) | il ViT giocattolo: 2 blocchi, dim 32, tubelet 2×8×8. **Ha la QKV fusa apposta** e una RoPE finta attivabile — riproduce le trappole del modello vero dove costano millisecondi |
| [backbones/vjepa21_source.py](../src/evd/models/backbones/vjepa21_source.py) | costruisce l'encoder dal clone Meta senza modificarlo, verifica la factory e carica il checkpoint con `strict=True`; `ArchSpec` lega in un solo punto ID, checkpoint, factory, chiave e layer di B/L/g/G |
| [backbones/vjepa21.py](../src/evd/models/backbones/vjepa21.py) | wrapper registrato con ID distinti `vjepa21b`, `vjepa21l`, `vjepa21giant`, `vjepa21gigantic`: stesso contratto e percorso per le quattro scale, con controllo ID↔architettura |
| [heads/base.py](../src/evd/models/heads/base.py) · [heads/linear.py](../src/evd/models/heads/linear.py) | il contratto `Head` e `LinearProbe` (pooling medio + `Linear`). Non è la testa buona: è il **detector del collasso** di E2c. **Rifiuta un ingresso multilevel** (ADR-029): mediare su livelli concatenati darebbe un numero plausibile e sbagliato, e la diagnosi di E2c si legge un layer alla volta |
| [heads/pooling.py](../src/evd/models/heads/pooling.py) · [heads/attentive.py](../src/evd/models/heads/attentive.py) · [heads/detection.py](../src/evd/models/heads/detection.py) | clone PyTorch dell'AttentivePooler Meta, verificato per parità numerica; probe frozen e testa 13+Normal condividono lo stesso pooling. **`AttentiveProbe` non cambia una riga per il multilevel** — vede solo una sequenza più lunga, che è il punto centrale di ADR-029; guadagna solo `use_activation_checkpointing: bool = False` (default in firma, non in YAML, per non spostare `_config_sha` delle celle a layer singolo) |
| [exp/preadapt.py](../src/evd/exp/preadapt.py) | **il preadattamento** (ADR-065, E3-S): una famiglia PEFT arriva **gia addestrata** da un altro run, e il blocco `preadattamento` del config dice cosa farne. Due modi, e non sono lo stesso meccanismo benche siano la stessa idea: **`fondi`** agisce sul **backbone** e deve accadere prima di ogni `apply()` (si costruisce il reparam, gli si riportano i pesi, `merge_()`, `unload_()`); **`congela`** agisce su uno **slot di una strategia gia applicata** e non potrebbe accadere prima, perche iniettare i prompt e poi reparam rompe la risoluzione dei nomi. ⚠️ `fondi` **verifica** che il merge abbia cambiato i pesi: un merge che non scrive produrrebbe una cella identica al VPT da solo, con numeri plausibili e un nome che promette altro. **`prepara_backbone()`** (25 settembre, R24) è la traduzione config → backbone preadattato e ha **due chiamanti**, il runner e `exp/demo.motore_live`: fino al 24 settembre la faceva solo il runner, e ogni cella `fondi` ricostruita fuori da lì — valutazione esterna, dimostratore `live`, explainability — girava sul backbone nudo senza sollevare |
| [exp/esterno.py](../src/evd/exp/esterno.py) | **la valutazione cross-dataset** (ADR-066): un run gia chiuso, misurato su un test set che non ha mai visto (MIVIA-Crime). Non addestra, non sceglie e non stima nulla — ricostruisce il modello con `demo.motore_live`, passa da `InferenceEngine` (invariante 4) e compone le metriche con `eval/valutazione.py`. Scrive una **cartella di run vera** piu la riga di registry, quindi il dimostratore la rigioca in replay. ⚠️ **tau si trasferisce dal run sorgente e si registra `fissa`**, con `trasferita_da`: MIVIA-Crime non ha una validazione. Il blocco `soglia_oracolo` — Youden calcolato sul dataset stesso — **non e protocollo** e serve solo a separare ordinamento da calibrazione |
| [exp/descrittori.py](../src/evd/exp/descrittori.py) | **un descrittore per finestra** (ADR-070): la media dei token del layer 23, cioe' 1024 numeri per chunk, da **due** sorgenti — `descrittori_da_store()` rilegge il feature store senza costruire nessun modello, `descrittori_da_forward()` ricalcola con un loader solo su tutto il corpus (e non con `stream()`, che decodifica in serie: sarebbe ADR-038). E' la stessa grandezza che `explain/concepts.py` chiama gia' «descrittore», non un secondo significato. ⚠️ Un livello solo, o la media mescolerebbe token concatenati; media in **fp32**, o sommarne 9216 in bf16 perderebbe cifre dove il descrittore si decide. `salva_descrittori()`/`carica_descrittori()` sono l'unico formato su disco, e il modulo **non importa torch a livello di modulo**, cosi' l'ambiente `viz` puo' rileggerlo |
| [eval/valutazione.py](../src/evd/eval/valutazione.py) | **la tabella di `docs/04` par.4.3, con un autore solo** (ADR-066). Viveva dentro `ExperimentRunner._valuta`; quando e arrivato il secondo chiamante si e **estratta** invece di copiarla — due composizioni della stessa tabella si mettono una accanto all altra in tesi e divergono in silenzio (ADR-052). ⚠️ **Non stima nessuna soglia e non sceglie la partizione**: tau arriva gia decisa, con la sua provenienza, perche e quello che distingue «stimata sulla validazione di questo run» da «ereditata da un altro dataset» |
| [adapt/composite.py](../src/evd/models/adapt/composite.py) | **due famiglie sullo stesso backbone** (ADR-062, E3-C): le componenti stanno in slot nominati per famiglia (`componenti.reparam`, `componenti.prompt`) e l'ordine lo impone la classe, perche' il verso opposto rompe la risoluzione dei nomi qualificati. ⚠️ Il verso **giusto** fallisce in silenzio — `freeze()` dentro il secondo `apply()` ricongela il primo adattatore — quindi `apply()` **ri-arma** `requires_grad` sui `trainable_parameters()` di ogni componente, conta quanti tensori ha ri-armato, e il confronto `n_trainable` ↔ somma delle formule in `AdaptationReport` rende il difetto impossibile invece che improbabile. `can_merge()` e' falso se una componente non fonde: e' un **risultato** per C7, non un limite. ⚠️ **Falso non vuol dire «niente da fondere» (ADR-064)**: `merge_parts()` vale `("reparam",)`, `merge_()` fonde la componente reparam e `unload_merged_()` ne toglie i wrapper lasciando i prompt — E5b misura ogni combinazione anche **fusa parziale** |
| [adapt/base.py](../src/evd/models/adapt/base.py) | il contratto `AdaptationStrategy` + `AdaptationReport`/`AdaptedBackbone`: è ciò che permette a E3-X di confrontare le due famiglie con **un solo percorso di codice** |
| [adapt/target_maps.py](../src/evd/models/adapt/target_maps.py) | **l'antidoto alla QKV fusa** |
| [adapt/reparam.py](../src/evd/models/adapt/reparam.py) | LoRA, rsLoRA e DoRA in PyTorch puro (ADR-019), con le **quattro fattorizzazioni** del reticolo di S3. `LowRankBlock` riparametrizza **un blocco di righe** e `DoRABlock` decora una direzione qualunque: da lì metodo × fattorizzazione escono da tre classi invece di otto |
| [adapt/prompt.py](../src/evd/models/adapt/prompt.py) | VPT shallow/deep (ADR-004), con la `pos_policy` **verificata realizzabile** e non solo dichiarata; i banchi possono avere larghezze diverse e i register si dichiarano per blocco |
| [adapt/apt.py](../src/evd/models/adapt/apt.py) | APT (`wgcban/apt`): prompt come **chiavi e valori** dentro l'attenzione. La sequenza in uscita non si allunga, quindi `extra_tokens` è 0 e la griglia non si contamina; il costo è lineare nei prompt e non quadratico. `KV_PROMPT_ATTENTION` è il registry per cui ogni attenzione porta il suo forward |
| [adapt/davpt.py](../src/evd/models/adapt/davpt.py) | DA-VPT (`Noahsark/DA-VPT`): gli ultimi banchi sono **proxy di classe** in una Proxy-Anchor. Il termine arriva al `Trainer` per il contratto `AuxiliaryLoss`, non per un ramo |
| [adapt/prompt_family.py](../src/evd/models/adapt/prompt_family.py) | la factory di famiglia: una chiave in `ADAPTERS`, tre meccanismi dietro `PROMPT_VARIANTS`. Sta in un modulo a sé per non chiudere il ciclo `prompt → apt → prompt` |
| [vendor/vjepa21_rope.py](../src/evd/vendor/vjepa21_rope.py) | l'unico codice **copiato** da `../vjepa2` (commit `b921227e`): `RoPEAttention.forward` con l'iniezione K/V di APT. Incapsulare non bastava — l'iniezione è in mezzo al forward. Copiato è il solo forward: la rotazione si risolve a runtime da upstream |

Su `reparam.py` e `prompt.py`, tre cose che è meglio sapere prima di leggerli.

**Il riferimento non è `peft`: è `microsoft/LoRA` e `NVlabs/DoRA`.** `FusedQKVLoRALinear` non è
una nostra estensione — è la semantica di `MergedLinear` (un adattatore indipendente per proiezione
abilitata, che upstream ottiene con `F.conv1d(..., groups=n)`), riscritta con fette esplicite. E le
due correzioni che S3 aveva lasciato aperte sono confermate sul sorgente **ufficiale NVlabs**: la
magnitudine di DoRA è una per **riga di uscita**, e si scrive `V·(m/‖V‖)`, mai `(m·V)/‖V‖` — con la
prima forma l'init è bitwise il modello congelato, con la seconda no.

**I prompt vanno in coda e si dichiarano `n_registers`.** `rotate_queries_or_keys` di Meta *già*
lascia non ruotati i token in coda: `rope_identity` non si implementa, si dichiara — e
`PromptPEFT` solleva se il backbone non offre dove dichiararla, invece di fingere di averlo fatto.
Il corollario buono: senza policy, su V-JEPA 2.1 non esce un numero sbagliato ma un errore di shape.

**Il contratto `VideoBackbone` ha due metodi in più da F5**, `module()` e `blocks()`: senza,
un'`AdaptationStrategy` dovrebbe indovinare l'attributo che contiene la rete e quello che contiene
i blocchi. Vedi il riquadro in `docs/03` §4.1.

**`encode_multilevel(backbone, clips, levels)` (ADR-029) è una funzione libera in
`backbones/base.py`, non un metodo di `VideoBackbone`.** Un solo forward via `encode_layers`
(esiste dal F4) + `concat_levels` di `types.py`: serve al trainer e allo scorer, non a ogni
implementazione del Protocol (ISP, `CLAUDE.md` §3.1 punto 4). Con una tupla di lunghezza 1 dà
esattamente ciò che dà `encode()` oggi — è il test di non-regressione di E2b-ML.

Su `target_maps.py` vale la pena fermarsi. Su V-JEPA 2.1 le tre proiezioni dell'attenzione sono
**un unico** `Linear(768, 2304)` chiamato `qkv` (verificato in S1). Chi arriva da Hugging Face
cerca `q_proj`/`k_proj`/`v_proj` e non li trova — e una lista di pattern che non matcha niente
produce **zero adattatori e un training che gira senza imparare**. Quindi: ogni backbone
*dichiara* la sua `TargetModuleMap`, la mappa dichiara anche l'*ordine* delle fette, e
`resolve()` restituisce riferimenti con la fetta di output (il che rende `qk_only` esprimibile
anche su una QKV fusa — ADR-019).

### 3.6 `train/`, `eval/`, `infer/` — addestrare e misurare

#### [train/budget.py](../src/evd/train/budget.py)
Il budget di training **è un dato**: passi, batch, griglia LR, precisione, TF32, tipo di
attenzione. Rifiuta la coppia `(bf16, eager)`, che produce NaN su V-JEPA 2.1 — non si scopre a
metà training. Il TF32 sull'A100 vale 7,28×, quindi va **dichiarato** o due tempi non sono
confrontabili. **Da ADR-030** `BudgetSpec` guadagna `lr_schedule: "constant" | "cosine"`
(default `"constant"`, nessuna cella esistente ne risente) e la funzione libera `lr_al_passo`,
che replica `WarmupCosineLRSchedule` del riferimento (`eval.py:490-515`, warmup a zero passi) —
unica implementazione dello schedule, usata sia per applicare il LR a ogni passo sia per il
controllo di coerenza alla ripresa di un checkpoint. **Da ADR-047** guadagna
`model_selection: "last" | "val_loss" | "val_auroc"`, che decide **quali pesi si tengono** a fine
training — è ortogonale a `early_stop_patience`, che decide *quando fermarsi*, e il protocollo
del laboratorio chiede il primo senza il secondo.

#### [train/trainer.py](../src/evd/train/trainer.py)
**Cosa fa:** ~190 righe, un solo loop per probe e PEFT. Cosa è addestrabile lo decide la
strategia di adattamento, non il loop.
**Perché non Lightning (ADR-006):** la parità di budget fra famiglie PEFT è un *risultato* della
tesi e va difesa in seduta; deve stare in un file leggibile. Con `adapt=None` il forward gira
sotto `no_grad`: non è un'ottimizzazione, è la garanzia che nessun gradiente tocchi il backbone.
Lo stesso loop accetta i batch `features` del ramo frozen e non richiama l'encoder; se è presente
un adapter li rifiuta, applicando il confine della cache anche durante il training.
**Da F6** accetta un `val_loader` — la validazione *episodica*, mai il query set — e applica davvero
`budget.early_stop_patience`, che conta **valutazioni** consecutive senza miglioramento e
cambia il numero di passi eseguiti. Il campo esisteva da M0 e non entrava in nessun loop: era
il terzo caso della famiglia di `EVD_SEED` e `precision`. `FitResult` porta ora `steps`
**eseguiti** (non più una copia del budget), `best_val`, `best_step` e `stopped_early`.
**Da D5** accetta anche un `checkpoint`: al confine di passo salva, e all'avvio riprende — pesi
della testa, adattamento, ottimizzatore, RNG **e batch già consumati**. Quest'ultimo non è un
di più: `_cycle` riparte da zero a ogni `fit`, quindi senza il salto i due tronconi vedrebbero
due volte gli stessi esempi, e c'è un test che lo dimostra togliendolo. `interrupted` è distinto
da `stopped_early` — il primo è un Wall Time arrivato prima, il secondo una decisione di
protocollo che chiude il training.
**Da F11 (28/8) il loop ha un `objective`**, e non lo deduce: `"binary"` è il regime in
vigore (ADR-031 punto 4) — un logit, bersaglio `batch["label"]`, `BCEWithLogitsLoss` **senza**
`pos_weight` (ADR-033 §3) — e `"multiclass"` resta perché resta il suo consumatore, E0c su
UCF101. Dedurlo da `head.n_classes` renderebbe una testa a un logit addestrata in
cross-entropy — possibile e sbagliata — un caso che nessun errore segnala; in binario il loop
**rifiuta** una testa a più logit invece di prenderne la prima colonna.
**Da ADR-030** il LR si aggiorna a ogni passo con `lr_al_passo` e `history[i]["lr"]` porta il
valore realmente applicato (non più il picco `chosen_lr` a ogni riga, che con lo schedule acceso
sarebbe stato un metadato che mente). Corretto nello stesso passaggio un buco trovato
implementando: il controllo di coerenza alla ripresa di un checkpoint confrontava l'LR
ripristinato con `chosen_lr` invece che con lo schedule valutato al passo salvato — con
`lr_schedule="cosine"` avrebbe rifiutato **ogni** ripresa.

#### [train/checkpoint.py](../src/evd/train/checkpoint.py) *(F6, D5)*
**Cosa fa:** il checkpoint intra-training e la ripresa (ADR-025 punto 4). `CheckpointSpec` dice
dove e ogni quanto, `CheckpointIdentity` a chi appartiene, `TrainingState` cosa si salva,
`Checkpointer` scrive in modo **atomico** e rilegge verificando l'identità.
**Perché esiste:** sul cluster il Wall Time è 7 ore per fetta e `--requeue` **non lo estende**
(job E0f `534484_0`, `TIMEOUT` senza rientrare in coda). Tre proprietà non negoziabili: mai i
pesi del backbone — identificato da `backbone_id`/`ckpt_sha`, `docs/08` §4 punto 4; un
checkpoint di un altro run viene **rifiutato nominando il campo** discorde, perché riprendere
nel posto sbagliato non produce un errore ma un numero; e la scrittura è atomica, o un
`torch.save` ucciso a metà lascerebbe un file che *esiste* ed è troncato.
**Due scelte da conoscere.** `git_sha` è provenienza e **non** identità — è la lezione di
`code_version` nel feature store, dove verificarla è costata 57 GB per un commit di soli
documenti. E `restore_adapter_state` non aggiunge un metodo al `Protocol` (ISP): sfrutta il
fatto che `state_dict_adapter()` restituisce **viste** dei parametri, e **verifica** che sia
vero invece di assumerlo — una famiglia che restituisse copie renderebbe la ripresa un no-op
silenzioso. Il controllo è parametrizzato su tutte e sette le config `adapt/`.
⚠️ `signal.SIGUSR1` **non esiste su Windows**: `install_interrupt_handler` installa ciò che
trova e dichiara cosa ha installato, senza mai sollevare. Il meccanismo su cui il loop poggia —
e che i test esercitano — è la **flag**.

#### [train/selection.py](../src/evd/train/selection.py) *(F6, riscritto in F11)*
La regola di `docs/04` §5.3 resa eseguibile, e **solo** quella. **Dal 28 agosto il regime in
vigore è uno solo e non dipende da `k`:** `regime_fully_supervised(n_lr, n_val)` dà `val_split`
se la griglia ha più di un punto e una validazione esiste, `griglia_a_un_valore` altrimenti — e
**solleva** se la griglia ha più punti e la validazione è vuota, perché lì non c'è un ripiego:
sceglierla sul training misurerebbe la capacità di memorizzare, sceglierla sul test è
l'invariante aggirato di lato. `regime_per_k` ed `episodic_split` restano come **storia** del
regime few-shot: nessun percorso in vigore li chiama, e i run già registrati devono restare
leggibili.
Non sa cosa siano un backbone o un `Trainer` — riceve dal chiamante una funzione «dato un LR,
dammi il punteggio sulla validazione», così la conoscenza di *come si addestra* resta nel runner
(direzione degli import, `docs/02` §5). La ricevuta (`SelectionOutcome`: LR, regime, provenienza,
prove, dimensione della validazione) finisce in `metrics.json`: la LR di un run è protocollo, non
un dettaglio implementativo. Che la validazione non sia il test lo garantisce `SplitSpec`, che
rifiuta due partizioni che si toccano — il `Trainer` non ha modo di sapere da dove arrivi un
loader.

#### [eval/frame_map.py](../src/evd/eval/frame_map.py)
**Cosa fa:** proietta score di anomalia e probabilità per classe dai chunk ai frame. La policy
primaria `causal` mantiene a ogni frame l'ultimo chunk già disponibile; il warm-up resta scoperto.
`max`, `mean` e `last` restano analisi offline secondarie.
Dal 1 settembre 2026 **una funzione per policy** (`_causal`, `_max`, `_mean`, `_last`) risolte da un
dizionario, e il nome della policy si valida in `__post_init__`: prima il `raise` per una policy
sconosciuta stava **dentro il ciclo sulle finestre**, cioè scattava alla prima finestra invece che
alla costruzione, e gli accumulatori di `mean` erano creati sotto un `if`, letti sotto un secondo e
normalizzati sotto un terzo. `to_frames` ha ora un docstring — era l'unico metodo pubblico di
`eval/` a non averne uno, su una classe che `docs/03` nomina come contratto.

#### [eval/metrics.py](../src/evd/eval/metrics.py)
AUC, AP e P/R/F1 **chunk-level primarie**, con positività per intersezione qualunque; le
funzioni `frame_*` restano per la sola timeline del dimostratore. Le metriche per classe sono la
verticalizzazione degli errori binari. Tutte verificate contro `sklearn`, casi degeneri inclusi;
nessun positivo → `nan` con warning, mai 0.5.

**Da ADR-047 (31/8) quattro funzioni in più, e sono il protocollo del laboratorio §3–4:**
`threshold_youden(scores, labels)` — la soglia come `argmax(TPR − FPR)` sulla ROC di
validazione, che rifiuta una verità degenere invece di ripiegare su un quantile;
`restrict_to_anomalous(scores, gt)` — il sottodizionario dei soli video anomali, da cui **AUC_a
e AP_a** riusando `dataset_chunk_auc`/`_ap` invece di due metriche nuove;
`video_decision_scores` e `video_decision_prf` — la decisione per video **asimmetrica** (su un
video anomalo il massimo corre sui soli chunk positivi, su uno normale su tutti), che è anche la
ragione per cui AUROC e AP a livello di video non si calcolano più: con quella regola lo score
decisionale dipende da τ, quindi ordinare i video non è definito indipendentemente da τ.

**Da ADR-048 (31/8) il livello EVENTO, e la definizione è NOSTRA:** `event_prf` restituisce
**`EventPRF`** — non `PRF`, e la differenza è sostanziale: precision e recall hanno
**denominatori diversi** (`n_allarmi` contro `n_eventi`), quindi non passa da `_prf`. Le due
regole: un evento è **rilevato** se una finestra sopra soglia lo interseca **anche di un solo
frame** — lo stesso `window_label` con cui il protocollo etichetta un chunk e con cui §4.2 decide
su un video anomalo, quindi il predicato è **uno solo per tutte e tre le granularità**; le
finestre sopra soglia **consecutive** si fondono in un **allarme** (`_allarmi`), falso se non
interseca alcun evento. `n_eventi_irraggiungibili` conta gli eventi che nessuna finestra tocca:
con `edp_v2` deve essere **zero**, e un valore diverso dice che la griglia non è quella del
protocollo. ⚠️ **Precondizione:** tutti i video e la griglia intera di ognuno — l'adiacenza è
quella della griglia degli score, non una distanza temporale.

#### [eval/strata.py](../src/evd/eval/strata.py) *(ADR-060, 4 settembre 2026)*
**Non *quanto* il sistema e' bravo, ma *dove*.** Le metriche di `metrics.py` danno un numero
aggregato; qui si verticalizza lungo **quattro assi**, che sono le tre difficolta' che il PdD §1
nomina rese misurabili: la **copertura** della finestra (livello chunk — sbaglia sui chunk di
*bordo*?), e **classe** · **durata** · **quota anomala del video** (livello evento — quali
classi, gli eventi brevi si perdono, la rarita' pesa). ⚠️ **E' diagnostica, non protocollo:**
nessuna riga e' una metrica del laboratorio e i numeri di `metrics.json` non cambiano.

Due stratificatori e quattro assi, e la separazione e' quella che rende estendibile il modulo:
`chunk_strata(rows, key, labels, threshold)` consuma le righe di `chunk_records` — **lo stesso
schema di `scores.jsonl`**, quindi la stessa funzione serve il run in corso e la ri-analisi di un
run chiuso; `event_strata(scores, gt, fps, key, ...)` **non riconta niente**, partiziona gli
intervalli e richiama `event_prf` e `detection_latency` su ogni sottoinsieme, che e' la mossa di
`restrict_to_anomalous`. Un asse nuovo e' quindi una funzione chiave, non un `if` (CLAUDE.md
§3.1 punto 2).

Quattro cose da sapere prima di toccarlo:
* **i negativi sono condivisi**, e il calcolo e' `metrics.breakdown_vs_negatives`, estratto da
  `per_class_chunk` a numeri identici. Con negativi per strato due AUC differirebbero anche per
  il denominatore, e la riga «peggiore» potrebbe esserlo solo perche' ha negativi piu' difficili;
* ⚠️ **a livello di evento esce il solo recall**: il denominatore della precision sono gli
  **allarmi**, che a uno strato di eventi non appartengono (ADR-048);
* ⚠️ **la durata non e' un asse di chunk**, ed e' una rinuncia dichiarata: servirebbe una
  **seconda** regola di spareggio accanto a `window_overlap`, la famiglia di ADR-052;
* **le fasce sono ancorate alla geometria** (`hop 4 s`, span 7,75 s) e i **bordi viaggiano nel
  risultato**: senza, non ci si accorge che due run sono stati analizzati con fasce diverse. Le
  etichette sono **ASCII** (`"<=4 s"`, non `"≤4 s"`) perche' diventano chiavi JSON e righe
  stampate a terminale, e la console di Windows e' cp1252.

Il gate: gli assi di **evento** seguono `eval.metrics.event_level`, come `event_prf` e per la
stessa ragione. ⚠️ Il flag `strata` invece **si legge nel runner ma non si dichiara in
`configs/eval/`**: `_config_sha` esclude dal gruppo `eval` i soli `num_workers` e `precision`,
quindi una chiave in piu' farebbe **rifiutare i checkpoint** gia' scritti — il conto di ADR-054.

#### [eval/scorers.py](../src/evd/eval/scorers.py)
`HeadScorer`: backbone + testa → un punteggio per chunk e le probabilità di classe. Per output
multiclasse l'indice di `Normal` è obbligatorio; in UCF-Crime è 13, non uno zero implicito.

#### [eval/_figure.py](../src/evd/eval/_figure.py) *(1 settembre 2026)*
Le due righe che ogni figura ripete: `agg()` (backend `Agg`, che sui nodi di calcolo non è
opzionale) e `salva()` (mkdir, `tight_layout`, `savefig`, `close`). Erano **sei** copie — tre in
`plots.py`, due in `report.py` — e avevano già cominciato a divergere nell'ordine di `mkdir` e
`tight_layout`. Modulo a sé e non dentro `report.py` perché `plots.py` non deve tirarsi dietro
scikit-learn.

#### [eval/plots.py](../src/evd/eval/plots.py)
La timeline. **Le figure si generano, non si disegnano** (ADR-014).
⚠️ **La policy `causal` ammette due finestre con lo stesso endpoint** (ADR-050): su `edp_v2` la
coda del video si valuta, e se l'ultima finestra piena termina gia' sull'ultimo frame la coda viene
troncata li'. Sono due finestre diverse — inizi diversi, contenuto diverso — e sul frame condiviso
vince quella che **comincia dopo**. Fino al 31 agosto era un `ValueError`, e ha ucciso due run
**dopo** che avevano finito di addestrare. Il controllo resta sugli **span**, che e' il difetto che
davvero voleva prendere. Nessuna metrica di tesi passa di qui: il livello frame e' abbandonato. Lo stesso codice serve lo
smoke, l'analisi qualitativa E5c e il dimostratore. Da F6 c'è anche `curve_png`: la curva
prestazione-vs-k con barre d'errore, disegnata **dal CSV tidy** di `SeedAggregator`. I punti
con un solo seed non hanno barra invece di averne una nulla, e i k non numerici (`full`,
`na`) restano fuori — su quell'asse non hanno una posizione.
Da ADR-049 c'è anche `training_png`: le curve di **un run** — loss di training e di validazione
sopra, AUROC di validazione sotto, il passo selezionato marcato su entrambi. Il runner la scrive
accanto a `history.jsonl` **sempre**, non dietro `make_figure`: quel flag governa la timeline di
*un video*, cioè una scelta su cosa mostrare, mentre questa descrive il run. I punti di validazione
stanno solo sui passi in cui la validazione è stata fatta — interpolarli sui 5524 passi
disegnerebbe una misura che nessuno ha fatto — e un guasto della figura non porta via le metriche.

#### `strata_png` in [eval/report.py](../src/evd/eval/report.py) *(ADR-060)*
Un pannello per asse: barre orizzontali col **conteggio nell'etichetta** (una barra su sei chunk
non si legge come una su 277 — la stessa avvertenza di `per_class_png`) e, sugli assi di evento,
la **latenza mediana** accanto alla barra di recall, perche' due strati con lo stesso recall e
latenze diverse sono due comportamenti diversi. ⚠️ **Le fasce vuote restano nel disegno**, a zero
e con `n=0`: toglierle nasconderebbe che una fascia non ha eventi, che spesso *e'* il risultato.
Legenda e latenza vivono nella striscia a destra e non sopra le barre — qui una barra a 1,0 e' il
caso normale, non l'eccezione. Entra in `FIGURE_DA_METRICHE`, quindi `make_figures.py --per-run`
la produce senza modifiche (ADR-014).

#### [eval/aggregate.py](../src/evd/eval/aggregate.py) *(F6)*
`SeedAggregator`: da tanti run a `(k, metodo) → media ± dispersione sui seed`, in CSV tidy
(`docs/03` §6.4). Tre regole applicate invece che suggerite: `n` si riporta sempre; con un
solo seed la deviazione è `None` e **non** 0 (misurato e nullo ≠ non misurato); un gruppo
sotto il calendario dei seed si dichiara invece di essere aggregato in silenzio. Non conosce
il registry — riceve chiavi e metriche già estratte, e il ponte sta in `exp/registry.py`.
Il perché è misurato, non stimato: `docs/10` §E0c dà 1,3 pp di rumore di riesecuzione dove
l'accuratezza è alta e **8,4** dove è bassa, e su UCF-Crime si sarà nel secondo regime.

#### [eval/efficiency.py](../src/evd/eval/efficiency.py) *(F4)*
Le primitive di E0b.1: warmup e iterazioni misurate con sincronizzazione CUDA, latenza,
throughput e picco VRAM (`None` su CPU), conteggio dei parametri e stima FLOP con fonte
dichiarata. E0b.1 misura `final` e `hierarchical` senza duplicare il blocco finale e salva ogni
cella in un JSON atomico; un rilancio sullo stesso commit riprende dalle celle mancanti. E5b
costruirà qui sopra il profiler completo di adapter e prompt.
Le clip sintetiche del profilo sono seminate da `EVD_SEED` con un **seme derivato per cella**
(`exp/e0.py::clip_seed`): seminare una volta all'avvio non basterebbe, perché il profilo è
riprendibile e un requeue salta le celle già fatte, cambiando la sequenza del RNG. `seed_all` è
chiamato con `deterministic=False` — forzare gli algoritmi deterministici cambierebbe la selezione
dei kernel, cioè proprio ciò che il profilo misura.

#### [eval/deployment.py](../src/evd/eval/deployment.py) · [eval/deployment_plots.py](../src/evd/eval/deployment_plots.py) · [exp/e5b_deploy.py](../src/evd/exp/e5b_deploy.py) *(E5b.2, ADR-059)*
Lo studio di **deployment**: dal costo del modello a quello del sistema, **componendo misure gia'
pagate** invece di rimisurarle. Tre moduli e tre ruoli, e la separazione non e' estetica.

`eval/deployment.py` e' **l'aritmetica**, e non importa ne' `torch` ne' `matplotlib`: e' quello che
la rende verificabile su CPU e senza backend grafico. Compone il modello (E5b), il decoder (E0d) e
la `detection_latency` delle celle E2/E3 — che esisteva dal 31 agosto e che nessuno aveva mai messo
accanto al costo di calcolo. ⚠️ **Somma e massimo hanno due nomi**: decoder e modello stanno su
device diversi, quindi in regime stazionario si sovrappongono (`ms_per_finestra_stazionario`, il
massimo) mentre la latenza di **una** risposta e' la somma (`latenza_risposta_ms`) — confonderli e'
il modo classico di sbagliare un conto di pipeline. `verifica_wallclock` adatta
`osservato = a + b · previsto` sulle passate di test vere e **solleva con meno di due punti**,
perche' una retta a due parametri su un punto passa esatta e il residuo nullo non direbbe niente.

`eval/deployment_plots.py` sono le **cinque figure** (latenza scomposta, capacita', Pareto,
scaling, coda dei percentili). Modulo a se' e **non** dentro `report.py`: quel modulo si dichiara,
nella prima riga del suo docstring, «le figure di *una* cella», e queste sono figure fra metodi e
fra stadi — metterle li' renderebbe falsa quella riga.

`exp/e5b_deploy.py` e' il **ponte**, e sta in `exp/` perche' li' si sa cosa sono i `run_id`
(`docs/02` §5 mette `eval` sotto `exp`): stessa posizione di `exp/registry.py` rispetto a
`eval/aggregate.py`. ⚠️ **Due filtri, entrambi pagati**: le celle di **E0i** portano lo stesso
`adapt` nel `run_id` di quelle di E3 ma girano su `edp_v1` e si fermano alla selezione — senza
filtro ogni variante risultava a **nove** semi invece di tre, e le loro durate a **zero minuti**
falsavano il fit. I criteri sono *l'esperimento sta nella matrice* e *la cella ha valutato la
griglia giusta*, quest'ultimo confrontato con `Geometria.n_chunk_valutazione`.

#### [exp/fidelity_job.py](../src/evd/exp/fidelity_job.py) · [scripts/confronta_store.py](../scripts/confronta_store.py) *(pilota E0d)*
Il caso d'uso che apre due store, confronta e registra il run. Il confronto ha un **asse
dichiarato** (`ASSI`): `_assert_comparable` pretende che i due store differiscano *solo* lungo
quello e coincidano su tutto il resto — backbone, checkpoint, `preprocess_hash` e l'altro dtype —
con il riferimento sempre fp32. Senza quel controllo la rel_L2 misurerebbe due effetti insieme e
il numero sembrerebbe comunque sensato.

I due assi rispondono a domande diverse e **solo uno ha un verdetto**. `storage_dtype` («rileggere
da una cache fp16 rende le stesse feature di una fp32?») è arrotondamento puro: la soglia si deriva
dall'epsilon del dtype. `precision` («il calcolo in AMP BF16 rende le stesse feature di fp32?») è
errore *accumulato* attraverso 24 blocchi, non ha un limite a priori, e il job passa
`derive_tolerance=False`: misura e non giudica, perché una soglia inventata darebbe un verdetto che
sembra un risultato. Il nome del report porta asse e geometria — sotto un run_id convivono più
confronti e un `fidelity.json` fisso li sovrascriverebbe in silenzio. Un criterio non superato
**non è un errore**: il report si scrive e si registra lo stesso, il verdetto sta in `passed`, e
solo la CLI esce diversa da zero perché si veda in `sacct`.

#### [eval/fidelity.py](../src/evd/eval/fidelity.py) *(pilota E0d)*
Il criterio con cui si deciderà il **dtype persistente** del feature store. `relative_l2` è un
rapporto adimensionale calcolato in fp64 (mai un `max`: su input diversi varia 2–5×, la rel_L2
1,1–1,2× — S3 §6); `storage_dtype_tolerance` **deriva** la soglia da `finfo(dtype).eps / 2`, il
limite dell'arrotondamento al più vicino, con un margine dichiarato — non è una costante scelta
dopo aver visto il numero (regola §12.3 del piano F4). `compare_stores` legge due `FeatureReader`
— non due store completi: chi verifica non deve poter scrivere — e tiene separate tre cose che
un solo numero confonderebbe: chiavi mancanti (il pilota vuole hit rate 100%), valori non finiti
(fp16 satura a 65504: lì il tensore non è impreciso, è `inf`) e la rel_L2 vera e propria.

`scale_reference` — con la CLI [`scripts/scala_feature.py`](../scripts/scala_feature.py) — dà il
**denominatore** con cui quelle rel_L2 si leggono. Serve perché una perturbazione del 2,3% (quella
misurata fra calcolo BF16 e fp32) non significa niente da sola: significa qualcosa solo accanto a
quanto distano fra loro le finestre che la testa a valle deve separare. Campiona coppie di finestre
*diverse* dentro un solo store, tenendo distinte due popolazioni — stesso video (la separazione che
il frame-level AUC deve sfruttare) e video diversi (la scala grossolana) — e raggruppa **per layer**,
perché mescolarli misurerebbe la distanza fra spazi diversi. La statistica che decide è il 5°
percentile, non la media: è la separazione più stretta che una perturbazione potrebbe ribaltare.

#### [infer/engine.py](../src/evd/infer/engine.py) — l'unico windowing del sistema
**Cosa fa:** un video → score per chunk → proiezione causale sui frame → intervalli rilevati (soglia con
**isteresi**: si apre a `on`, si chiude a `off`).
**Perché è importante (invariante 4):** valutazione, profiling e dimostratore passano tutti da
qui. Se un giorno una figura di tesi e il video della seduta mostrassero curve diverse sullo
stesso video, la causa **non potrebbe essere** il windowing. La garanzia è che non esiste un
secondo percorso.

### 3.7 `exp/` — l'identità e la memoria dei run

#### [exp/run_id.py](../src/evd/exp/run_id.py)
La grammatica `{exp}_{dataset}_{backbone}_{adapt}_k{K}_s{S}`, validata anche per E0e/E0f e
`k32`. Un nome che si *parsa*
permette a settembre di rispondere a «da dove viene questo numero» interrogando il registro
invece di ricordare. Esiste `kna` («questo run non sta sulla curva») e **non** esiste `k0` —
sarebbe *zero-shot*, che è fuori matrice (ADR-016/017). `EXPS` guadagna `e0h` (ADR-030, il
pilota di `weight_decay`/schedule): stessa convenzione di `e0g`, un nome ammesso dalla grammatica
perché l'esperimento esiste, non perché sia in matrice per costruzione.

#### [exp/tracker.py](../src/evd/exp/tracker.py)
**Cosa fa:** apre un run, scrive sempre `meta.json` e `config.yaml`, poi gli artefatti propri
dell'esperimento (`history.jsonl` per il training, `efficiency.json` per il profiling,
`metrics.json` alla chiusura) e accoda una riga a `runs/registry.jsonl`.
**Le due garanzie:** il registry è **append-only** (fondere due copie è un'unione, non una
risoluzione di conflitti); e **un run ufficiale con working tree sporco viene rifiutato**.
**Due correzioni di F11 (28/8).** `manifest_sha` si chiama ora `protocol_sha`: senza manifest,
quel nome sarebbe un metadato che mente — porta il sigillo dello split di `edp_v1`, o quello del
manifest per i run few-shot già registrati. E `extra_meta` era un parametro **inerte**: accettato
dal 21 agosto e **mai scritto**, quindi la ricevuta dell'adattamento che il runner gli passava non
è mai finita in `meta.json`. È la famiglia di difetto che `CLAUDE.md` §3.1 elenca per prima.

#### [exp/lr_trials.py](../src/evd/exp/lr_trials.py) *(29/8, ADR-038)*
`RicevutaProve`: le prove **finite** della griglia LR, in `runs/{run_id}/lr_trials.json`.
Nasce da un fallimento — la campagna E0i del 28 agosto ha perso 21 prove complete su 45
insieme alle celle che le contenevano — e **non** è una deroga al divieto di checkpoint dentro
le prove: una prova *a metà* resta inutilizzabile (riprenderla renderebbe quel punto non
confrontabile con gli altri), una prova **finita** è un fit completo e riusarla non introduce
asimmetrie. L'identità — `run_id`, `config_sha`, `split_sha`, griglia — è la condizione perché
la ripresa sia definita: su disaccordo il file si **ignora** e si ricomincia, dicendolo nel log,
che è la stessa regola di `CheckpointIdentity` e per la stessa ragione.

#### [exp/runner.py](../src/evd/exp/runner.py) *(F6)*
**Il motore**, e la Facade di `docs/03` §8: `run(cfg) -> RunResult`. Prima di F6 la fetta
verticale viveva cablata dentro `exp/smoke.py` e ogni blocco E0 aveva la sua CLI; da qui E2
congelata ed E3 PEFT attraversano **le stesse righe**, e fra loro cambia una config, non un
ramo (invariante 5: due percorsi di codice non reggono un confronto a parità di budget).
Tre confini che fa rispettare, e che sono la ragione per cui è un file solo:
lo **split è l'unica sorgente delle partizioni** (non sa costruirne uno, sa solo leggerne uno
sigillato — e la griglia che ne deriva la **ricalcola**, invece di leggerla da un artefatto che
si disallineerebbe in silenzio); il **feature store lo vede solo il regime congelato** — la
scelta fra `build_frozen_loader` e `build_adapted_loader` si fa **una volta**, guardando
`preserves_backbone_output(strategia)`, e un run PEFT non riceve nemmeno il riferimento alla
cache; **learning rate, early stop e soglia si decidono sulla validazione**, mai sul test.
`_verifica_geometria` confronta la griglia dichiarata dal protocollo con quella del backbone —
ha già intercettato un `target_fps` 7,5 contro 8,0 — e `_n_classes` rifiuta un `head.n_classes`
che contraddica il **protocollo** (con `output: binary` la testa ha un logit, non 14).

**Da F11 (28/8) esegue `edp_v1`, e i nove passi di §5.0 sono tutti qui.** In ordine di
esecuzione: `split_path_for`/`_split` caricano lo split sigillato e **si fermano** se non
dichiara una validazione — un ripiego silenzioso lì sarebbe il test usato per scegliere;
`_chunks` ricalcola la griglia delle tre partizioni con `enumerate_chunks`, contando e
**dichiarando** i video più corti di una finestra; `_items` porta il peso `w_video · w_overlap`
dentro `WindowItem`, e `_loader(..., weighted=True)` lo porta fino al `WeightedRandomSampler`;
`_resampler` costruisce la chiusura del campionamento **dinamico**, che sposta l'endpoint dentro
la cella di hop e **ricalcola l'etichetta** con `chunk_at` — la stessa regola della griglia
statica; `_seleziona_lr` sceglie sulla validazione (`regime_fully_supervised`), e da lì l'early
stop è applicabile anche al fit finale, che una validazione ce l'ha; `_soglia` stima τ sui chunk
**negativi della validazione**; `_valuta` misura sullo **split di test**. `_verifica_regime` si è
**invertita**: rifiutava `edp_*` finché il runner non lo sapeva eseguire, adesso rifiuta `fsp_*`,
che è il regime che non esegue più.

**Cosa `metrics.json` porta di nuovo, e perché conta:** `chunks` — conteggio, positivi e i due
tassi (uniforme e **pesato**) per train/val/test — è la **firma verificabile** della griglia, che
non essendo sigillata va provata a ogni run (`docs/04` §8). Lo stesso blocco finisce in
`meta.json` **prima** del training, così resta leggibile anche se il run non arriva in fondo.
Spariscono `k`, `m` e `n_support`; `manifest_sha` diventa `split_sha`.
**Da ADR-029, `_layer` è `_levels`**: il campo `layer` accetta un intero, una lista, o il
sentinello `"hierarchical"` (risolve a `backbone.hierarchical_layers()`, la parte che rende il
braccio multilevel parametrico sulla scala). Restituisce sempre una tupla, e il caso a layer
singolo è la tupla di lunghezza 1 — non un ramo a parte. `chiave_di` restituisce ora
`tuple[str, ...]` (una chiave per livello) invece di `str`.

#### [exp/sources.py](../src/evd/exp/sources.py) *(F6)*
`DatasetBundle` + registry `DATASOURCES` (`synthetic`, `ucf_crime`): dove stanno i file,
quanti frame hanno, e qual è la verità sulle query — cioè le tre cose che il manifest non
contiene, e non per un limite: un file di protocollo che contenesse i path non sarebbe più
valido su un'altra macchina (ADR-001). È **uno solo** perché gli stessi ingredienti servono
al `ManifestBuilder` che genera e al runner che consuma: tenerli in due posti significherebbe
costruire il protocollo su un inventario ed eseguirlo su un altro. Sta in `exp/` e non in
`data/` perché compone `data` + `annotations` e usa il registry di `models`.

> Da D7 il modulo espone **`run_id_for`** e — da F11 — **`split_path_for`** come funzioni
> libere: `campagna.py` deve sapere quale `run_id` una cella produrrà e quale split consumerà
> **prima** di eseguirla, e ricalcolarlo altrove sarebbe il secondo percorso che diverge in
> silenzio. `run_id_for` prende il `dataset_id` e non il bundle proprio per questo — costruire
> un `DatasetBundle` significa toccare i dati, e un piano non deve. `split_path_for` legge
> `protocol.split_file` e non compone un nome da `(k, seed)`: nel regime nuovo c'è **uno** split
> per protocollo, quindi la convenzione di nome che `manifest_path` doveva tenere allineata fra
> scrittura e lettura non serve più.

#### [exp/registry.py](../src/evd/exp/registry.py) *(F6)*
Il lato **lettura** del registry: `read_registry` (tollerante — una riga rotta non impedisce
di sapere cosa manca), `completed`/`mancanti`/`is_complete` (severi), `latest_per_run` e il
ponte `grouped_for_aggregation`. Due fatti sul formato che sbagliarli produce risposte
plausibili: c'è **una riga per esecuzione**, non per `run_id` — la chiave è
`(run_id, started_at)` — e non tutti i `run_id` storici rispettano la grammatica
(`f5_minifit_…`, `e0f_ucfcrime_…`). Si leggono lo stesso e **non si correggono**: la
grammatica si impone dal runner in avanti. Il commit **non** entra nel criterio di
completezza — è la lezione di `code_version`, che costò 57 GB di riestrazione.

#### [exp/campagna.py](../src/evd/exp/campagna.py) *(F6, D7)*
**Cosa fa:** espande una griglia dichiarata in `configs/sweep/` in celle
(`SweepSpec` → `Cella`), chiede al registry cosa manca, e costruisce i comandi `hpc/invia.sh`
a fette da ≤8 task.
**Le tre decisioni che portano tutto il resto.** *(a)* Il **dedup viene dal registry**, non
dalle cartelle: `runs/{run_id}/` esiste anche se il run è morto a metà, e da D5 esiste appena
parte il primo checkpoint. *(b)* **L'indice di array è la posizione nel piano intero**, quindi
un rilancio parziale è un array *sparso* (`0,3,7`) e lo stesso indice non può puntare a due
celle diverse in due sottomissioni. *(c)* Il `run_id` di una cella lo calcola
`runner.run_id_for`, cioè la funzione che poi lo scrive — e lo split lo risolve
`runner.split_path_for`: due copie di quelle regole sarebbero il posto in cui piano e run
guardano due cose diverse.
**Perché `configs/sweep/` non è un gruppo Hydra:** un gruppo entrerebbe nella config del run e
finirebbe in `runs/{run_id}/config.yaml` dicendo il falso — un run non sa di far parte di una
griglia, e due run della stessa cella lanciati da due sweep diversi devono restare lo stesso
run. Quel file descrive *un'esecuzione*, non *un esperimento*.
**Due controlli che sembrano pignoleria e non lo sono:** `official` dichiarato deve coincidere
con quello dedotto dalla grammatica (uno sweep di prova che si credesse ufficiale finirebbe in
una tabella di tesi), e due celle non possono avere lo stesso `run_id` (un asse che non entra
nel `run_id` — `train.max_steps` — genera punti indistinguibili nel registry, e il checkpoint
dell'uno riprenderebbe l'altro).

#### [exp/smoke.py](../src/evd/exp/smoke.py)
La fetta verticale eseguita: mondo sintetico → **split giocattolo sigillato** → griglia dei
chunk → probe → valutazione → soglia **dalla validazione** → metriche → figura → riga nel
registry marcata `official: false`. F4 aggiunge estrazione Tiny e replay attraverso
`WindowDataset`, con hit rate richiesto del 100%.
**Da F6 non ha più un percorso proprio:** costruisce il mondo, genera lo split giocattolo in
cache e chiama `ExperimentRunner` — uno smoke che passasse da un secondo percorso verificherebbe
il secondo percorso, e i due divergerebbero in silenzio. Gli restano il mondo sintetico e
`_feature_store_replay`, che è un'asserzione sulla cache e non un passo di esperimento.
Attraversa anche l'adapter (LoRA r=2, VPT n=2), che è ciò che rende vera la DoD «dati →
backbone → **adapter** → testa → metrica → figura».
**Da F11 lo split giocattolo sostituisce il manifest giocattolo** (`_split_giocattolo`): test =
il pool di query del mondo, validazione ritagliata dal training **per video e stratificata per
classe** — la stessa forma del ripiego dichiarato in `docs/04` §5.3 per il dataset vero, applicata
a un mondo inventato. Una classe con un solo video di training non ne cede nessuno: toglierlo la
farebbe sparire dall'addestramento.
⚠️ **La guardia va tarata sul regime che misura, e il 28 agosto non lo era.** Il frame AUC 0,947
di `PIANO.md` F1 era già scaduto con ADR-024; la guardia chunk-AUC ≥ 0,70 che l'ha sostituito è
scaduta con ADR-031, e il modo in cui si è visto è che lo smoke usciva **sotto il caso** —
0,225 · 0,328 · 0,367 sui semi 0/1/2. Non era un difetto della pipeline: 20 passi a batch 4 con
lr 1e-3 erano tarati su una testa a **4 vie** addestrata in cross-entropy su ~26 finestre di
support, e su 91 chunk con **un logit** e una BCE non muovono niente. Col budget rimisurato
(`configs/train/probe_toy.yaml`: 20 passi, batch 8, lr 3e-2) gli stessi tre semi danno
**1,000 · 0,984 · 1,000**. La lezione da portarsi dietro: **quando cambia l'obiettivo, la guardia
va rimisurata, non riletta** — altrimenti dice «bug del codice» quando il codice è giusto.

#### `exp/feature_jobs.py` · `exp/e0.py` · `exp/e0c.py` *(F4)*
`feature_jobs` stima spazio e finestre prima di caricare il checkpoint, estrae shard riprendibili
e consolida gli indici. `e0.py` ispeziona architettura/stato/token layout e produce la griglia
VRAM/throughput più la tabella degli hop, con progresso atomico per cella. `e0c.py` prepara la
sanity UCF101 ma delega l'ottimizzazione al `Trainer` comune, esclusivamente su feature congelate,
e confronta la top-1 con un riferimento esplicito. Le probe lunghe possono usare lo stesso
checkpoint intra-training del runner F6, con identità distinta per replica; un'interruzione non
produce metriche parziali. I percorsi ufficiali
passano dal tracker e quindi rifiutano un working tree sporco.

#### [exp/minifit.py](../src/evd/exp/minifit.py) *(F5)*
**Cosa fa:** applica le dodici varianti di `COSTRUTTORI` a una scala B/L/g/G scelta esplicitamente, addestra una dozzina di
passi su video UCF101 veri decodificati al volo, e dà un **verdetto** sulle cinque cose che i test
su Tiny non possono provare — che la mappa risolva su tutti i blocchi veri, che forward *e backward*
passino sotto AMP BF16, che il gradiente arrivi solo agli adattatori, che
l'adattatore abbia davvero imparato, e che `merge_()` regga sul modello vero.

**Il criterio del quarto punto non è «la loss scende».** Su una dozzina di passi la loss oscilla, e
un cancello così renderebbe il test intermittente — che è peggio di assente, perché si impara a
ignorarlo. È deterministico e in **tre** parti: ogni tensore riceve un gradiente non nullo (il grafo lo
raggiunge), ogni tensore che AdamW *potrebbe* muovere si è mosso (l'ottimizzatore li possiede), e
l'uscita dell'encoder è cambiata. Le prime due erano una sola fino al job 533070: separarle serve
perché su APT il passo di AdamW può stare **sotto l'ULP del float32**, e un parametro che non
cambia per aritmetica non è un cablaggio rotto. La soglia si deriva nel run (`passo_adamw` contro
l'ULP), mai scritta a mano. La traiettoria della loss si riporta come dato, senza verdetto.

#### [exp/e0e.py](../src/evd/exp/e0e.py) · [scripts/e0e_geometria.py](../scripts/e0e_geometria.py) *(pilota E0e)*

Il fattoriale 2×2×2 che ha prodotto il ripiego corrente per Q21 e il confronto storico di Q22
(`docs/04` §4.4): risoluzione × ritaglio × span, a probe
congelata. **Non contiene una pipeline**: `run_arm` mette in fila `run_extract_job`,
`merge_feature_shards` e `run_e0c`, che esistono già — se ne avesse una propria, il pilota
misurerebbe la propria estrazione invece di quella che poi userà la matrice. Le parti nuove sono
`arms()` (il piano, in ordine stabile perché l'indice del job array è la cella), `scegli()` (il
criterio dichiarato prima dei risultati) e `aggrega()` (che rifiuta di aggregare un fattoriale
parziale). `tokens_per_window` viene da `data/feature_extraction`, non riscritto qui.

**Quattro modi, e l'ordine non è un consiglio** (`e0e.sbatch`): `stima` dimensiona il job senza
GPU, `calibra` fissa il **budget di dati** — cercando la top-1 della sola cella di riferimento
dentro una banda dichiarata — `cella` gira una cella del piano, `aggrega` applica il criterio.
`calibra` **deve** precedere `cella`, e il modo `cella` lo impone rifiutandosi di partire senza
`E0E_TRAIN_VIDEO_PER_CLASSE`: un budget scelto dopo aver visto il confronto sarebbe una scelta
dentro l'esperimento. `confronti_appaiati()` implementa il test dei segni; un confronto è appaiato
tre volte (stessa replica, stessi video di test, altri due assi fissati), che è l'unica
configurazione in cui la differenza osservata è attribuibile all'asse in esame.

**Idempotenza per seme.** `run_arm` scrive un record in `runs/<run_id>/semi/` appena un seme
chiude, e al rilancio salta quelli già misurati. Non è una comodità: il TIMEOUT delle celle a 384
(job 533781) avrebbe fatto ripagare ~8 GPU-ora di risultati già ottenuti. L'artefatto su disco è
la fonte di verità, non lo stato in coda — i log, uccisi al Wall Time, erano vuoti dopo il
preambolo perché lo stdout era ancora nel buffer.

#### [exp/e0f.py](../src/evd/exp/e0f.py) · [scripts/e0f_split.py](../scripts/e0f_split.py) *(pilota E0f)*

**Cosa fa:** costruisce lo split del pilota sul dominio vero **e la mappa delle etichette per
finestra**. Non contiene un fattoriale: quello è `e0e.py`, già indipendente dal dataset, e
riscriverne una copia per UCF-Crime significherebbe avere due piloti che divergono in silenzio.

**Il vincolo che governa il file:** lo split si costruisce da `Anomaly_Train.txt` e basta.
`costruisci_split` legge `Anomaly_Test.txt` **solo** per verificare che l'intersezione sia vuota,
e solleva se non lo è. Non esiste un modo di puntarlo al test set, e la sua assenza è voluta.

**Tre scelte che definiscono il compito**, tutte e tre dichiarate perché tutte e tre potevano
essere fatte diversamente: (a) una finestra è positiva di classe *c* se **termina** dentro un
intervallo `c` — `window_class_label`, la stessa regola con cui il sistema verrà valutato;
(b) è normale solo se l'intero **span** dista da ogni evento più del margine di guardia, e ciò che
sta in mezzo non riceve etichetta e **non entra**; (c) le finestre normali si prendono dagli
**stessi video di anomalia**, il che costa zero e toglie la scorciatoia della scena.

**Perché la stessa mappa vale per tutte le celle.** `window_id` contiene video, endpoint, numero
di campioni e profilo di campionamento: `s{stride}` quando gli indici nativi sono uniformi,
`i{hash}` quando la quantizzazione dei timestamp produce passi non uniformi. Non contiene né la
risoluzione né la variante di ritaglio. A parità di video, `num_frames`, `target_fps` e hop, tutte
le celle costruiscono gli stessi timestamp e quindi **le stesse finestre con le stesse etichette**:
il confronto resta appaiato nel senso forte senza fingere che esista sempre uno stride intero.

`bilancia_per_classe()` riduce il lato di valutazione a una quota uguale per classe, distribuita
fra i video: senza, `Normal` porterebbe 11 443 finestre contro le 67 di `Explosion` e una testa
che risponde sempre «normale» starebbe sopra il 95%.

#### [exp/e0b2.py](../src/evd/exp/e0b2.py) *(F5-bis)*
**Cosa fa:** misura il costo di **addestrare** un adattatore — forward, backward e aggiornamento —
su una griglia `size × variante × precision × tf32 × batch`. È l'unica evidenza che può
dimensionare `peft_budget_a`, e non lo fissa: la scelta resta di protocollo.

**Cronometra il `Trainer` vero**, con un loader di un solo batch già in memoria, invece di
riscrivere il passo: un profiler che duplica il loop misura il profiler, e la prima volta che il
loop cambia le due strade divergono dentro un numero. I pixel sono sintetici e la pipeline dati è
fuori dalla misura, come in E0b.1. Riprendibile per cella, con le clip seminate dalla cella.

**Due dettagli che sembrano prudenza e non lo sono.** Il backbone si **ricostruisce per ogni
variante**, perché `merge_()` scrive nei pesi e riusarlo farebbe partire la variante dopo da un
encoder che non è più quello pre-addestrato — sintomo: una loss leggermente diversa, cioè niente.
E il modulo gira anche **su CPU con Tiny e video sintetici**: un test di integrazione che si può
provare solo sull'hardware di destinazione si scopre rotto proprio quando serve.

---

### 3.8 `explain/` — dove il modello guarda *(ADR-063, 9 settembre 2026)*

ADR-067 estende questo package con la vista offline, distinta dalla proiezione causale:

| Componente | Responsabilità |
|---|---|
| `explain/attribution.py` | Gradient×Activation pre/post norma, A×Gradient tramite V e gradiente dell'output effettivo, IG adattivo e gradiente positivo per LeGrad |
| `explain/representation.py` | PCA accumulata, EigenCAM non centrato secondo il paper, cosine con query esplicita |
| `explain/perturbation.py` | Deletion/insertion sui pixel preprocessati, riferimenti blur/mean, griglia a pari volume e controlli casuali |
| `explain/comparison.py` | Curve appaiate con i casuali e bootstrap per video, distinto per modello/categoria/riempimento/segno |
| `exp/explanation_campaign.py` | Campagna a checkpoint selezionato, batch uno, mappe grezze e provenienza, selezione deterministica delle finestre |
| `explain/selection.py` | La maschera che si mostra: quella di effetto massimo **fra le sole localizzate**. Senza `torch`, perche' la usa anche `apps/` |
| `explain/perturbed.py` | Le undici viste perturbate dell'ingresso — tre occlusioni, freeze locale, sette ablazioni globali — e i nomi `pert__{vista}__{metodo}` con cui le loro mappe tornano al metodo base |
| `explain/concepts.py`, `explain/vtcd.py`, `explain/volumes.py`, `explain/methods.py` | Registro dei cinque metodi: direzione TCAV-style, prototipi binari, SLIC 3D + Convex NMF + CRIS finale, occlusione e ablazione del movimento |
| `exp/explanation_references.py`, `exp/explanation_store.py`, `exp/explanation_advanced.py` | Manifest dal training senza video visualizzati o duplicati, preparazione per checkpoint, ricevute atomiche per metodo e finestra, verifica della griglia e del logit contro la sorgente; cosine da feature salvate con query e identita verificate, senza nuovi forward |
| `apps/demo/explanation_advanced.py` | Originale/perturbato con score confrontabili per le ablazioni globali; **un filmato per ognuna delle 45 maschere** con i due punteggi in testata e la griglia letta dai riquadri; ranking CRIS aggregato e rendering con scale comuni al modello; ripresa per filmato; **quadro 2x2** di un'ablazione di concetto (`render_quadri_concetto`), con le due mappe sopra i loro ingressi e una sola scala |
| `apps/demo/explanation.py` | Video offline affiancati e heatmap sola, pannelli delle tubelet e curve temporali dai soli artefatti |
| `explain/boxes.py` | Da una mappa ai **rettangoli** che la circondano: maschera del concetto, taglio top-k per conteggio, componenti connesse a 8 e coordinate in token/frazioni/pixel. Senza `torch`, perche' la usano sia `apps/` sia la campagna |
| `exp/ig_topk.py` | La vista `ig_top10_black`: il primo 10% di `ig_ensemble` coperto di nero e **tutte le mappe ricalcolate** su quell'ingresso, in un **gruppo nuovo** — l'identita' di una campagna comprende `explain/methods.py`, che qui non si tocca |
| `apps/demo/explanation_boxes.py` | I filmati con i bounding box, alla risoluzione **nativa** del video, in due varianti che mostrano lo **stesso video intero** e differiscono solo in dove si disegna (`intero` ovunque, `annotati` sui soli frame annotati positivi); `boxes.json` con le coordinate nei tre sistemi e `boxes_per_frame.csv` come forma consegnabile dell'annotazione; `caselle_della_campagna()` e' l'unica via maschera → caselle, usata anche dal taglio `side_by_side_bbox` di `explanation.render_campaign` (video · sovrapposto · box), che disegna con `explanation.disegna_caselle()` (30/9); accetta `fusione` (`evd.explain.boxes.RegolaFusione`, proposta non adottata: `docs/varie/proposta_fusione-bbox_30-09.md`, pilota in `scripts/pilota_fusione_box.py`); `render_bounding_box_composto()` (4/10) fa un filmato e un'annotazione per video coi box del dizionario della classe di ogni evento (schema 2: finestre per `(window_id, classe)`, colonna `classe` nel CSV) |

`project.py` conserva il mapping causale e aggiunge `offline_assignment`/`offline_plane`; dal 29 settembre `offline_assignment` ha due politiche, `fine` (default) e `overlap_positivo` (ADR-077), e la politica di una campagna si legge solo con `apps/demo/explanation.assegnazione_di()`. La politica sceglie la mappa (e testata e cella accesa); la curva degli score di `timeline_di` resta sempre sulla `fine`.
`HeadScorer.prepare_clips` e `autocast` condividono la preparazione delle viste e il regime
numerico fra inferenza ordinaria e spiegazioni. Il protocollo del detector non cambia.

| File | Cosa fa |
|---|---|
| [explain/attention.py](../src/evd/explain/attention.py) | legge i pesi della **cross-attention del pooler** (`depth=4`, una query appresa, 9216 token alla geometria di `edp_v2`) con un `forward_hook` — `pooling.py` e' la replica fedele del sorgente Meta e non si tocca — e li **verifica ricalcolando l'uscita** (1,0e-03 in bf16). Porta `Contrasto` e `contrasto_di()` con quattro modi (`massimo`, `percentile` di default, `scarto`, `finestra`): la mappa grezza e' quasi piatta — entropia normalizzata 0,9985-0,9990 su `dora-r32` — e su una colormap non stirata sembra un fondino |
| [explain/project.py](../src/evd/explain/project.py) | porta le mappe **dalle finestre ai frame nativi** con la stessa regola con cui ci arrivano gli score: non riscrive l'assegnazione, la **chiede** a `FrameScoreMapper` (invariante 4) codificando l'indice della finestra in un punteggio. Se ADR-050 cambia idea su quale finestra vince un frame condiviso, questo modulo cambia con lei senza essere toccato |
| [exp/explain.py](../src/evd/exp/explain.py) | l'orchestrazione: dal `run_id` e da un video ai tagli video e ad `attention.json`/`attention.npz` |

⚠️ **`attention.npz` salva le mappe NON contrastate**, quindi il contrasto si
rifa' senza rieseguire il forward e **nessun numero di tesi dipende dal disegno**.
⚠️ Uno stiramento senza la sua scala **mente** (`percentile` fa occupare tutta la
colormap a qualunque mappa), quindi due contromisure obbligatorie: la **barra di
scala** con gli estremi in multipli del peso uniforme, e il blocco
`concentrazione` in `attention.json` — entropia, min/max e massa del top-1%
**prima** del contrasto. E' la disciplina di `effetto_adattatore` in E5b, e c'e'
un test che la difende.

---

## 4. Gli script eseguibili

### 4.1 In `evd-vfm/scripts/`

Aggiornamento 1/10: gli script `vtcd_grid_*` leggono modello, gruppo di
output e vista del catalogo dallo scope, con controllo del run autorizzato.
`vtcd_reference_worker.py` esegue il CNMF originale con prodotti a blocchi
e checkpoint ogni 10 iterazioni; il supervisore può alternare calcolo della
prima variante ed estrazione della seconda, e ammette gli shard di estrazione
prerequisiti del fit iniziale. Le configurazioni
`localizzazione_it2000_n{260,3298}_r32_k8_vtcd` compaiono nella sezione VTCD
delle pagine di localizzazione test del modello VPT-shallow su LoRA.
Vedi [campagna corretta](varie/vtcd_vpt_riferimenti_20261001.md).
Estensione del 1/10 ai 61 video, poi 62 con Stealing079: preflight e renderer leggono conteggi e
numero di shard dallo scope, mantenendo i default storici. I riferimenti
sigillati restano invariati; i nuovi video sono registrati nello scope e
controllati contro i riferimenti. Il rendering riprende solo i lotti mancanti.
Regressioni in `test_vtcd_grid_extension.py`: riuso dei 57 video e controllo
del leak per i quattro aggiunti.

Estensione del 2/10: quote per evento (17 per classe, 442 riferimenti),
conteggi e copertura per split indipendenti, esclusione delle copie tramite
SHA256. Un alias verificato condivide il fit tra validation e test; shard e
risorse CPU sono configurabili. I riferimenti possono conservare solo gli
input del CNMF, senza duplicare i token grezzi. Catalogo diretto nelle due
sezioni `localizzazione_eventi17` e `localizzazione_eventi17_val`, con tutti
i sei tagli. Regressioni mirate in `test_vtcd_event_campaign.py`; insieme
alle precedenti: sei test passati. Piano e avvio in
[campagna per eventi](varie/vtcd_eventi17_20261002.md).
`render_methods_per_batch` configura da uno a tre metodi per lotto,
mantenendo invariati metodi, tagli e copertura; la campagna completa usa
un metodo per rispettare le quattro ore di fatq sui video più lunghi.
`render_cuts` seleziona i tagli senza cambiarne i default storici.
`vtcd_reassign.py` riassegna su CPU descrittori e regioni salvati al nuovo
dizionario, usando `assign_concepts` e le mappe canoniche; verifica modello,
preprocessing, SLIC, manifest e checksum, salva nuove identità e provenienza.
Il DAG usa `preextracted_references`, `calculation_mode: reassign` e
`calculation_partition: fatq` per evitare inferenza ed estrazione ripetute.
La versione corrente del piano per eventi riguarda solo i 290 video test,
CNMF fino a 10000, affiancato/sovrapposto con e senza interfaccia.

- `vtcd_classi.py` + `slurm/vtcd_classi.sbatch` (2/10): VTCD **per classe**, per classe **senza normali** e **globale senza normali** sopra la campagna eventi17. `fit` adatta i 27 dizionari col solver di `vtcd_reference_worker` (senza `fit_binary_references`, indefinita senza normali); `process` riassegna le finestre di val/test dai descrittori SLIC che eventi17 ha salvato (`w*_vtcd_concepts`), verificando prima di riprodurne l'assegnazione, e rende i sei tagli con `vtcd_grid_render._work`; `catalog` riscrive l'indice; `supervise` orchestra con budget di spazio e, a fine campagna, rilancia il catalogo se un catalogo precedente toglie le sezioni `vtcdclassi_*`. `boxes` (3/10) rende i bounding box del concetto scelto per classe (`box-concetti.json`): soli box nativi con `boxes.json`/CSV, coppia e affiancato `*_annotati` nel lotto `render_box`, solo sugli eventi della classe della pagina; dal 4/10 **un rendering per video** (`<video>__box/`): su ogni evento i box del dizionario della sua classe (`explanation.render_campaign(composizione=...)`, `explanation_boxes.render_bounding_box_composto`, `boxes.json` schema 2), solo coi due tagli con l'interfaccia, e i box per classe del 3/10 tolti dalle pagine. Con `dictionaries_ready` nello scope salta estrazione e fit (dizionari già adattati, come la «prima selezione» del 3/10); lo sbatch legge la cartella delle operazioni da `VTCD_OPS`, che il supervisore esporta. Dal 5/10 lo scope può dichiarare `cuts` (un sottoinsieme dei quattro tagli delle maschere): MIVIA-Crime rende le sole viste con l'interfaccia, nel gruppo `vtcdclassi_classennprima_mivia_vtcd`, che il catalogo riconosce (`PARTIZIONI_VTCD_CLASSI`). Piano e stato: [campagna del 2/10](varie/vtcd_classi_20261002.md).
- `vtcd_reference_worker.py` (30/9): riuso verificato di feature/SLIC dei riferimenti, estrazione GPU a shard con SLIC CPU concorrente e fit finale CPU. I riferimenti da 260 e 3.298 finestre usano lo stesso CNMF della matrice; `vtcd_grid_preflight.py` accetta manifest con più finestre per video, `vtcd_grid_supervisor.py` orchestra estrazione/fit/calcolo/rendering e pubblica ciascuna configurazione appena pronta. `vtcd_grid_render.py` produce sei tagli, con e senza interfaccia, e verifica il catalogo e lo streaming HTTP prima della chiusura. Piano e stato: [campagna del 30/9](varie/vtcd_riferimenti_20260930.md).
- `explain_video.py`: CLI comune per run, video/manifest, metodi, ID finestra e rendering
- `estendi_spiegazioni.py`: preflight, pilot locale e ripresa dei metodi su sviluppo e due video completi; default Anomaly Direction e VTCD, `--cosine-only --render` riusa le feature salvate. `explain_video.py --extend-campaign` espone lo stesso runner per una sorgente singola
  senza inferenza; `--select-development` fissa le categorie dagli score di validazione.
- `confronta_spiegazioni.py`: JSON e tabella Markdown dalle curve salvate, bootstrap per video.
- `slurm/explain.sbatch`: launcher predisposto per modello e manifest; nessun submit implicito.
- `render_bounding_box.py`: `--elenca` dice **quale gruppo porta quale mappa** prima di renderizzare niente; poi rende le quattro combinazioni (sorgente x variante) a fette, un processo per video. Solo CPU.
- `localizzazione_vtcd.py` + `slurm/localizzazione.sbatch` *(ADR-077)*: VTCD di base sul modello migliore (OVAD, guardia 30 s) sui 57 video della selezione — `piano` (in locale, dalla selezione congelata della griglia; task 0 = fit dei riferimenti, un pilota in coda), `rendi` (anteprime, raw attention e concetti, a fette su `fatq`; con `--box vtcdK` anche i bounding box, con `--force` sopra i filmati esistenti) e `verifica` (`--box` pretende i manifest dei box); `--pilota` e' ripetibile. `localizzazione_supervisor.py` porta la campagna in fondo da sola sul login node (calcolo su `gpuq`, render in sei fette su `fatq`, verifica, catalogo; tetto di 9 job dell'utente, arresto con `ALLERTA.txt` al primo fallimento o a una ricevuta mancante; `--box` e `--force` per un rendering nuovo, e all'avvio riconosce chiusi dalle ricevute i soli calcoli; `--affiancato-box vtcdK` dal 30/9: l'affiancato dei concetti coi box, e si rende di nuovo solo cio' che ha tagli diversi da quelli attesi). Il calcolo resta `campagna_explain.py`, i box `render_bounding_box.py --sorgente vtcdK`.
- `campagna_ig_topk.py`: l'unico passo con GPU — la vista `ig_top10_black` sui video della selezione, ripresa per video.
- `esporta_score_chunk.py`: il pacchetto degli score sui chunk (CSV + JSONL + video + intervalli + `meta.json` + README) con AUC e AP **ricalcolate e confrontate** con `metrics.json`.
- `slurm/bounding_box.sbatch` (`fatq`, nessuna GPU) e `slurm/ig_topk.sbatch` (`aiq`, una scheda): nessun submit implicito.
- `vtcd_grid_plan.py`, `vtcd_grid_preflight.py`, `vtcd_grid_references.py`, `vtcd_grid_supervisor.py`, `vtcd_grid_render.py` e `slurm/vtcd_grid.sbatch` *(ADR-074)*: la matrice VTCD della granularità spaziale e del dizionario — piano congelato delle celle pulite (le storiche restano archiviate e fuori dal confronto), controllo di input e isolamento prima di sottomettere, riferimenti annidati da 52 a 104/208 finestre bilanciate per classe, supervisore senza job array (rispetta `MaxSubmit`) e rendering riprendibile dei soli video coi concetti. Gli script si caricano nei test per percorso (`importlib`), non come package. ⚠️ Il testo di ADR-074 descrive quattro coppie `(regioni, concetti)`, il piano ne congela 27: vale il codice.
- `confronto_sota_xlsx.py` e `metriche_per_classe_xlsx.py` *(E4)*: il primo genera `docs/incontri/confronto_sota_modello-migliore_20260918.xlsx` dai `metrics.json` di DoRA r=32 s42 e dalle tabelle del laboratorio; il secondo appende, in modo idempotente, le tabelle per classe a `docs/incontri/metriche_confronto-sota.xlsx`, che è mantenuto a mano. Dipendono da `openpyxl`.

| Script | Comando | Cosa produce | Perché esiste |
|---|---|---|---|
| [smoke.py](../scripts/smoke.py) | `.\scripts\smoke.ps1` | `runs/smoke_…/` + figura; exit 0 = verde | Il controllo «la spina dorsale è integra». Esiste **senza Hydra** apposta: se passa questo e non `run_experiment.py`, il problema è la composizione delle config, non il codice |
| [run_experiment.py](../scripts/run_experiment.py) | `python scripts/run_experiment.py +experiment=<id>` | il run vero, via Hydra | La facciata Hydra, **entrypoint unico dal 21 agosto** (ADR-025): ogni `exp_id` diverso da `smoke` va dritto a `ExperimentRunner.run()`; lo smoke genera prima mondo e manifest giocattolo, poi passa dallo stesso runner. I piloti E0 conservano le CLI dedicate che hanno prodotto numeri registrati |
| [inventario.py](../scripts/inventario.py) | `… scan --shard-index N`, poi `… merge` | i 5 artefatti dell'inventario | Decodificare 1900 video è lavoro CPU: si spezza in shard e gira su `fatq` |
| [e0d_throughput.py](../scripts/e0d_throughput.py) | `--protocol … --n-windows 512 --workers 0,1,2,4,8,16,32` | finestre/s per cella e il **ginocchio**, cioè quante CPU chiedere | Chiude la gamba di throughput di E0d, aperta dal 14 agosto: tutti gli altri profili girano su clip sintetiche già sulla GPU, quindi sul decoder non c'era un numero. Estrae le finestre **coi pesi del sampler** e le converte con `runner.window_items`, cioè lo stesso percorso dei run |
| [e0d_cache_benefit.py](../scripts/e0d_cache_benefit.py) *(1 settembre)* | `--protocol … --n-draws 3000 --cache-mb 0,166,500,1000,2000,4000 --augment photometric` | throughput e RSS per budget di cache, **sotto il campionamento vero** | Risponde alla domanda che `e0d_throughput.py --cache-mb` **non può** rispondere: quello passa una volta sola su un elenco fisso, quindi ogni indice è un miss **per costruzione** e misura il *costo* della cache, non il beneficio. Qui si pesca **con reimmissione** dai pesi reali del training (`w_video = 1/n_chunk`), com'è in un run vero, e si attraversa la **stessa sequenza** a ogni budget — così l'unica differenza fra le righe è il budget. **Ha risposto no** (job 541451): 3,743 fin/s a 0 MB, poi monotonicamente in giù fino a 0,840× a 4000. ⚠️ I candidati sono **per worker**: per tradurli in `CACHE_RAM_MB` si moltiplica per `num_workers` (12 su `edp_v2`). Registra `n_indici_unici` accanto ai tempi, ed è quel campo a rendere leggibile il risultato: **2772 su 3000**, cioè la cache non aveva quasi occasione di colpire |
| [replay_store.py](../scripts/replay_store.py) | `--protocol … --store-dir …` | hit rate dello store contro la **griglia corrente** del protocollo | È il controllo che allo store E0d è mancato per dieci giorni: **un artefatto derivato da una funzione pura si verifica contro la regola corrente, non si assume**. Ricostruisce le chiavi dalla griglia e le confronta con l'indice in dieci secondi. ⚠️ Senza `--expected-count` dice solo che nessuna voce d'indice è orfana — compatibile con uno store a cui mancano 1072 feature, che è esattamente quello che è successo |
| [explain_attention.py](../scripts/explain_attention.py) *(9 settembre)* | `--run … --video … --contrasto percentile` (o `--selezione`, per tutti i video di una scelta) | `attention.mp4`, `attention_affiancato.mp4`, `attention_mappa.mp4`, `attention.json`, `attention.npz` | La richiesta §Analisi.2 dell'incontro del 9 settembre. Il terzo taglio — la sola heatmap, senza il frame sotto — e' il **controllo**: un video in cui l'originale non c'e' non puo' avere il problema che Antonio ha segnalato («si nota troppo il video originale»), quindi dice se una mappa poco visibile e' debole o solo coperta |
| [scelta_video_explain.py](../scripts/scelta_video_explain.py) *(9 settembre)* | `--run … --out …` | i 54 video da guardare | Migliore e peggiore per ciascuna delle 13 classi anomale, due casuali per classe, quattro normali: la selezione e' una **regola dichiarata**, non una scelta a occhio fatta dopo aver visto i risultati |
| [scheda_analisi_qualitativa.py](../scripts/scheda_analisi_qualitativa.py) *(9 settembre)* | `--selezione … --out …` | `docs/varie/analisi-qualitativa-video_0909.md` | Prepara la scheda con le **colonne misurate piene** e quelle da guardare vuote: le compila Antonio, e cio' che il codice sa non si ridigita a mano |
| [e0j_autocast.py](../scripts/e0j_autocast.py) | `--protocol …` | quanto costa in *decisioni* valutare in `bf16` invece che in `fp32` | Il numero che regge ADR-044 (la valutazione gira in bf16 come il training) e la guardia di ADR-053 (τ non si riusa se `eval.precision` ≠ `train.precision`): **6 finestre su 1024 cambiano lato della soglia**. Un confronto di regime numerico si misura sulle decisioni, non sulla rel_L2 |
| [conta_chunk.py](../scripts/conta_chunk.py) | `--protocol configs/protocol/edp_v2.yaml --inventario … --split-dir … --annotazioni …` | conteggi per **le tre partizioni**, tassi di positivi, finestre troncate e con frame ripetuti, intervalli raggiungibili, e **i passi di un'epoca** | I numeri di `docs/04` §3.6 venivano da uno script ad hoc, e la prima volta che qualcuno li ha rifatti col codice vero uno era sbagliato (1831 → **1825** intervalli raggiungibili). Chiama `CausalWindowSampler` e `chunks_for_video`, cioè lo stesso percorso dei run: una seconda implementazione del windowing misurerebbe se stessa. **Da ADR-047 la geometria si legge dal protocollo e le partizioni dallo split sigillato**: prima aveva default cablati e leggeva `Anomaly_Train`/`Anomaly_Test`, cioè due cose sbagliate insieme. `--num-frames`/`--target-fps`/`--hop-s` restano come override espliciti, per confrontare due geometrie candidate — è così che è nata la tabella di ADR-034. È anche la sorgente di `max_steps`: `ceil(88.371 / 16) = 5524` |
| [sync_inventario.py](../scripts/sync_inventario.py) | `python scripts/sync_inventario.py` | scarica dal cluster e **valida** prima di promuovere | Un artefatto scaricato a metà non deve sostituire quello buono |
| [scarica_annotazioni.py](../scripts/scarica_annotazioni.py) | `--source sultani` | il file + `PROVENIENZA.md` (URL, data, sha256) | Registra lo sha alla prima esecuzione e lo **verifica** dalla seconda: «le annotazioni sono cambiate» è una notizia, non un aggiornamento |
| [valida_annotazioni.py](../scripts/valida_annotazioni.py) | `--source sultani --path … --inventory …` | `report.json` + `report.md` | Il `.md` è l'allegato per il laboratorio. Exit ≠ 0 se ci sono errori |
| [build_manifests.py](../scripts/build_manifests.py) | `--protocol … [--data configs/data/ucf_crime.yaml]` oppure `--synthetic --dry-run` | i manifest sigillati dopo l'approvazione | Valida anche `k` e `m`; sul protocollo reale si ferma finché `approved: false`, e da D9 il binding **dietro** quel rifiuto è scritto: bundle da `DATASOURCES`, regole dal YAML firmato, rifiuto di riscrivere un manifest esistente (i manifest sono immutabili). **Entrambi i modi costruiscono il mondo con `DATASOURCES`**, la stessa sorgente del runner: prima ricostruiva mondo e split a mano con una regola di partizione *diversa*, e i manifest così generati parlavano di uno split che il runner non avrebbe mai prodotto |
| [build_split.py](../scripts/build_split.py) | `--protocol configs/protocol/edp_v1.yaml [--dry-run\|--verifica]` | `manifests/edp_v1/ucf_crime/split.json` sigillato, più la tabella per strato | Legge quota, stratificazione e seme dal **YAML firmato** e non dagli argomenti, come `build_manifests.py` (D9): una quota passata a riga di comando vivrebbe nella cronologia di una shell invece che nel file che si rilegge partendo da un `run_id`. `--verifica` rigenera lo split e lo confronta col file su disco — è il controllo che allo store E0d è mancato, e qui costa un secondo |
| [importa_split.py](../scripts/importa_split.py) | `--protocol configs/protocol/edp_v2.yaml --csv … [--dry-run\|--verifica]` | `manifests/edp_v2/ucf_crime/split.json` sigillato, più il rapporto delle verifiche | **Non genera niente**: la partizione arriva dal laboratorio, e questo script la trascrive nel formato sigillato *dopo averla messa alla prova*. È la differenza con `build_split.py`, che applica una regola e può quindi rigenerare ciò che ha scritto; qui la regola non c'è, e al suo posto ci sono cinque verifiche — copertura, geometria (frame e fps contro l'inventario F2), etichette, **griglia video per video** contro i conteggi consegnati, duplicati che attraversano le partizioni. La quarta è quella che vale di più, ed è quella che ha trovato la finestra vuota sui video che finiscono esattamente su un confine di passo |
| [campagna.py](../scripts/campagna.py) | `piano\|stato\|valida\|esegui\|invia <sweep>` | la griglia, cosa manca, i comandi `hpc/invia.sh` | `piano` gira **prima del freeze** (comporre non risolve i `???`), `valida` no — ed è lì che si fermano. `invia` **stampa e non sottomette** senza `--conferma`. Il dedup viene dal registry, non dalle cartelle |
| [e0_profilo.py](../scripts/e0_profilo.py) | `e0a …` / `e0b1 …` | ispezione wrapper e tabelle di costo E0b.1 | Separa il controllo meta-device, eseguibile senza checkpoint, dal run ufficiale su A100 |
| [lista_video_split.py](../scripts/lista_video_split.py) | `--split val test --out-dir …` | un file per partizione, una riga per video | Proietta la partizione **sigillata**: l'elenco che `--videos-from` consuma non si scrive a mano. L'intestazione porta lo sha dello split, perché queste liste decidono cosa finisce in uno store da centinaia di GB |
| [estrai_feature.py](../scripts/estrai_feature.py) | `--videos-from … --dry-run`, shard, `--store-shard`, `--grid`/`--window-s`, merge `--expected-count` | store safetensors + indici SQLite | Non esiste `--all`: l'elenco dei video e tutti i campi di protocollo devono essere espliciti. Per un **append** a uno store popolato: fetta `0..n-1` (o non la si copre tutta) e `--store-shard` spostato oltre gli indici esistenti |
| [e0c_sanity.py](../scripts/e0c_sanity.py) | config E0c risolta | top-1 UCF101 e caveat di comparabilità | Legge solo feature: un miss è un errore, non un fallback GPU |
| [profilo_peft.py](../scripts/profilo_peft.py) | architettura + assi della griglia; il checkpoint canonico deriva dalla prima | `efficiency_peft.json`: s/passo e picco VRAM per variante | E0b.2. **È un esperimento**, non un test: run ufficiale, quindi rifiuta un working tree sporco. Riprendibile per cella dopo un requeue |
| [minifit.py](../scripts/minifit.py) | architettura + split UCF101; il checkpoint canonico deriva dalla prima | verdetto sulle dodici varianti sul modello vero | È il livello *Integration GPU* di `docs/08` §1: esce **1** se una variante fallisce. Il criterio non è «la loss scende», ma gradienti, aggiornamento ed effetto sull'uscita |
| [confronta_store.py](../scripts/confronta_store.py) | `--axis storage_dtype\|precision` | `fidelity-{asse}-{geom}.json` | Varia **un asse dichiarato**; esce ≠ 0 se il criterio non passa, così il verdetto si vede in `sacct` |
| [scala_feature.py](../scripts/scala_feature.py) | `--reference-store-dir … --candidate-store-dir …` | tabella per layer + JSON | Il **denominatore** di una rel_L2: quanto distano fra loro finestre diverse. Senza, «2,3%» non decide niente |
| [aggrega_augment.py](../scripts/aggrega_augment.py) | il `registry.jsonl` con le celle di **E0g** | il verdetto del criterio dell'augmentation: per catena e per k la mediana appaiata, i semi concordi, la catena da adottare e **la ragione** | Il criterio sta in `docs/10` ed è stato scritto *prima* dei run; questo lo esegue. Le differenze sono **appaiate per seme**, mai medie sottratte, e la regola è **asimmetrica**: `official` è la ricetta del riferimento e non deve vincere, deve solo non essere dimostrabilmente dannosa (mediana ≤ −0,01 e 4 semi su 5, a entrambi i k). Scritto e testato prima che la campagna rientri, per la stessa ragione per cui esiste `aggrega_geometria.py` |
| [aggrega_lr.py](../scripts/aggrega_lr.py) | il `registry.jsonl` con le celle di **E0i** (`--esperimento e0i2` per la seconda campagna) | per ciascuna delle quattro famiglie: le LR scelte seme per seme, la **mediana** che si adotta, la concordanza, e il file di config in cui il numero va scritto | Il criterio sta in `docs/10` ed è stato scritto *prima* dei run; questo lo esegue. **Mediana** e non media — la media di due punti della griglia non sta sulla griglia, cioè è una LR che nessuna prova ha eseguito — e non minimo, che inseguirebbe il seme più fortunato. Ha una **guardia di bordo**: una mediana su un **estremo della griglia della cella** non si adotta, perché dice che l'ottimo sta fuori dalla banda misurata — gli estremi si leggono dalle prove, non da due costanti, ed è ciò che permette alla seconda banda di concludere su `1e-4`, che nella prima era un bordo. Esce **1** finché il verdetto non copre tutte e quattro le famiglie |
| [aggrega_geometria.py](../scripts/aggrega_geometria.py) | la cartella `ricalcolo-938/` con le quattro celle | le letture del criterio, l'esito di geometria e — con `--verifica` — il confronto con l'artefatto in archivio | Il criterio che congela la geometria di tutta la matrice era in uno script mai committato. Non sceglie: se nessun asse conclude applica il ripiego dichiarato. `--con-controllo-robustezza` aggiunge la media entro seme, che **non** e' il criterio del pilota |
| [confronta_predizioni.py](../scripts/confronta_predizioni.py) | due o più `predizioni-*.json`; opzionalmente una mappa `window_id -> classe` | quante unità cambiano decisione e top-1 ricalcolate sul sottoinsieme verificato | Confronto **appaiato**: a metrica satura due accuracy non distinguono «equivalenti» da «test senza potere». Il filtro conserva gli artefatti originali e verifica etichette e cardinalità prima del ricalcolo |
| [ricalcola_latenza.py](../scripts/ricalcola_latenza.py) *(ADR-061)* | `--run runs/<run_id>` oppure `--tutti` (+ `--set`, `--tavola`, `--dry-run`) | `latenza.json` dentro il run, piu' la tavola a confronto | Ricalcola la latenza con **entrambe** le regole dagli score gia' su disco, in secondi e senza GPU. **La guardia che lo rende credibile**: ricalcola anche la regola storica e **rifiuta di scrivere** se non riproduce `metrics.json` — la geometria viene dal **sampler** e non dal file (ADR-055), quindi una griglia sbagliata darebbe un numero nuovo *plausibile e falso*. Un secondo controllo confronta gli esclusi con i mancati di `event_prf`. ⚠️ **Non scrive dentro `metrics.json`**: e' la ricevuta di quella esecuzione |
| [analisi_strati.py](../scripts/analisi_strati.py) *(ADR-060)* | `--run runs/<run_id>` (+ `--set` per i percorsi, `--bordi-*` per le fasce) | `strata.json` dentro il run, piu' le quattro tabelle a terminale | Rilegge `scores.jsonl` e produce l'analisi stratificata **in secondi e senza GPU**: le quindici celle di `edp_v2` chiuse fra il 1 e il 3 settembre non si rifanno per una tabella diagnostica. La geometria la ricostruisce il **sampler**, non il file (ADR-055), quindi `--split` decide la griglia e non solo le righe. **Non riscrive `metrics.json`**: il risultato porta `provenienza.origine: ricalcolo_post_hoc`, perche' un numero prodotto durante il run e uno ricalcolato dopo sono due cose diverse. ⚠️ Servono le **annotazioni**: `scores.jsonl` porta etichetta e classe, non gli **intervalli** |
| [valuta_esterno.py](../scripts/valuta_esterno.py) *(ADR-066)* | `--run runs/<run_id> --dati mivia_crime` (+ `--set` per i percorsi, `--limite` per una prova, `--backbone-checkpoint`) | `runs/<run_id>__<dataset>/` con `config.yaml`, `meta.json`, `metrics.json`, `scores.jsonl`, piu la riga di registry | Valuta un run **gia chiuso** su un test set esterno. ⚠️ **Non e una rilettura da disco**: servono il checkpoint del run, i pesi del backbone e una GPU. Sui 270 video di MIVIA-Crime sono 3485 finestre. ⚠️ **La RAM e il vincolo, non la GPU**: un item del loader e la finestra a risoluzione **nativa**, e 13 video sono in 4K (~800 MB per finestra) |
| [estrai_descrittori.py](../scripts/estrai_descrittori.py) *(ADR-070)* | `--cella dora_ucf --run runs/<run_id>` (+ `--sorgente store --store <dir>` per il congelato su UCF-Crime, `--dati mivia_crime --scores <run>__mivia/scores.jsonl` per le celle esterne) | `<out>/descrittori/<cella>.npz` (valori, `window_id`, `video_id`, etichetta, classe, score, provenienza) piu' il `.json` accanto | Un descrittore per finestra di test: la media dei token del layer 23. ⚠️ **Le etichette vengono da `scores.jsonl`, mai da un ricalcolo**, e il sampler ri-enumera le finestre: se le due sequenze non coincidono elemento per elemento lo script si ferma (ADR-055). ⚠️ Con `--sorgente store` non costruisce nessun modello e non tocca la GPU |
| [riduci_descrittori.py](../scripts/riduci_descrittori.py) *(ADR-070)* | `--out runs/analisi_proiezioni_<data> --spazio frozen=congelato_ucf,congelato_mivia --spazio dora=dora_ucf,dora_mivia` | `proiezioni.npz`, `figure/*.png` (quattro pannelli: etichetta, classe, dataset, **video**), `diagnostiche.json` | PCA, UMAP e t-SNE sui descrittori. ⚠️ Gira nell'ambiente **`viz`** e **non importa torch**, come `apps/` (ADR-015): `umap-learn` tira dentro numba, e l'env dei run resta intoccato. ⚠️ Le coordinate si salvano **prima** di disegnare, e `--solo-figure` ridisegna senza rifare venti minuti di t-SNE (ADR-014) |
| [e5b_deployment.py](../scripts/e5b_deployment.py) *(E5b.2)* | gli artefatti gia' su disco: `efficiency_inference.json`, `throughput.json`, le celle di E2/E3, `registry.jsonl` | `deployment.json`, `deployment.md` e quattro PNG in `runs/<run E5b>/` | **Non e' un esperimento e non chiede una GPU**: compone, non misura, quindi **non passa dal tracker** — non ci sono un seme ne' un regime numerico da certificare, e la provenienza di ogni numero sta nel payload (`sorgenti`). Gira sul portatile, ed e' la parte dello studio che non costa un secondo di coda |
| [make_figures.py](../scripts/make_figures.py) *(F6)* | `--exp <id>` sul registry locale, **ripetibile** (`--exp e2d --exp e3p --exp e3r` **è** E3-X.1) | CSV tidy, tabella `.md` e figura `.png` in `$EVD_RUNS_DIR/figures/`, più le figure di ogni cella con `--per-run` | ADR-014: **le figure si generano dal registro, mai numeri copiati a mano in tesi**. Sottile: compone `read_registry` → `grouped_for_aggregation` → `SeedAggregator` → `curve_png`. Ufficiali e non ufficiali **non si mescolano** (`--non-ufficiali` li prende *al posto* degli altri); dove la curva non esiste (tutti `kna`) lo dice e produce comunque la tabella |
| [ispeziona_norms_block.py](../scripts/ispeziona_norms_block.py) | `<checkpoint.pt>` | quali `norms_block` sono addestrate | Controllo binario (distanza dall'init, non una soglia): per B/L solo l'ultima lo è |

E in `scripts/slurm/`: [inventario.sbatch](../scripts/slurm/inventario.sbatch),
[smoke.sbatch](../scripts/slurm/smoke.sbatch), [e0.sbatch](../scripts/slurm/e0.sbatch),
[estrai_feature.sbatch](../scripts/slurm/estrai_feature.sbatch) (modi `extract`/`merge`/`fidelity`/`scala`), [e0c_inventory.sbatch](../scripts/slurm/e0c_inventory.sbatch) ed
[e0b2.sbatch](../scripts/slurm/e0b2.sbatch),
[e0b3.sbatch](../scripts/slurm/e0b3.sbatch) (modi `frozen`/`peft`, che leggono geometria
e config PEFT dai file di protocollo invece di ricopiarne i valori) ed
[e0c_scarica.sbatch](../scripts/slurm/e0c_scarica.sbatch), che legge i digest da
`configs/data/ucf101.yaml` invece di farseli passare a mano; più
[e0e.sbatch](../scripts/slurm/e0e.sbatch) ed [e0f.sbatch](../scripts/slurm/e0f.sbatch)
(sottocomando + job array, con `SLURM_ARRAY_TASK_ID` come indice di cella) e
[minifit.sbatch](../scripts/slurm/minifit.sbatch),
[campagna.sbatch](../scripts/slurm/campagna.sbatch) (il lanciatore della matrice: job array,
`--signal=B:USR1@300`, `EVD_CAMPAGNA_EXTRA` per gli argomenti),
[e0d_throughput.sbatch](../scripts/slurm/e0d_throughput.sbatch) (asse `workers`, `augment`,
`cache_mb`) e [e0d_cache_benefit.sbatch](../scripts/slurm/e0d_cache_benefit.sbatch). Sono
**quattordici**. Sono sotto git perché un run dev'essere ricostruibile dal repository
(invariante 2); i nuovi job richiedono gli input di protocollo via ambiente e non li scelgono.
Il controllo che precede ogni submit è **`bash -n`** su ognuno: un apostrofo dentro un
`${VAR:?messaggio}` rompe lo script e bash accusa una riga venti più giù (16 agosto, job
532963).

### 4.2 In `hpc/`

| File | Cosa fa |
|---|---|
| [env.sh](../../hpc/env.sh) | **la fonte** di tutti i default (path, precisione, seed). Lo sourcia ogni job |
| [invia.sh](../../hpc/invia.sh) | sottomette un job controllando prima ciò che SLURM segnala male o tardi: l'account `-A`, le risorse, il requeue `-R n` |
| [sync.ps1](../../hpc/sync.ps1) | trasferimenti portatile↔cluster. Il codice viaggia come **git bundle**, non come copia di file. `-Azione pull-run` riporta a casa registry e artefatti — l'unico modo per cui un risultato esiste fuori dal cluster |
| [tools/job_env.py](../../hpc/tools/job_env.py) | diagnostica **solo-stdlib**: gira sul login node, dove torch potrebbe non esserci |
| [tools/ricognizione.sh](../../hpc/tools/ricognizione.sh) | raccoglie i fatti sul cluster che non si indovinano da fuori |
| [tools/scarica_ucf_crime.sh](../../hpc/tools/scarica_ucf_crime.sh) | porta i 111 GB su `/shared`, con verifica sha |
| [tools/scarica_ucf101.sh](../../hpc/tools/scarica_ucf101.sh) | scarica archivio e split ufficiali, verifica hash forniti esplicitamente ed estrae solo il sottoinsieme E0c scelto. **Rieseguibile**: salta il download quando il file c'è già *e* il digest torna — senza la guardia, `-C -` su un file completo prende un 416 e `set -e` uccide lo script prima dell'estrazione |
| [tools/estrai_classi_rar.py](../../hpc/tools/estrai_classi_rar.py) | estrae le classi dichiarate dal `.rar` in **una** passata, via `libarchive.so.13`: sul cluster non c'è né `unrar` né `7z` e non si possono installare |
| [tools/verifica_ucf101.py](../../hpc/tools/verifica_ucf101.py) | digest + integrità: legge **tutti i blocchi** di tutte le entry, che è l'unico modo di far verificare i CRC a libarchive. Un `.rar` corrotto in mezzo si elenca benissimo fino al punto in cui si rompe |
| [tools/smoke_torch.py](../../hpc/tools/smoke_torch.py) | torch vede la GPU, calcola, e a che velocità |
| [setup/crea_ambiente.sh](../../hpc/setup/crea_ambiente.sh) | crea `.venv-hpc` (una volta sola, sul login node) |
| [setup/requisiti-hpc.txt](../../hpc/setup/requisiti-hpc.txt) | il `pip freeze` esatto dell'ambiente che produce i numeri, **in git** |
| `jobs/*.sbatch` | i template SLURM (cpu, gpu, array, ricognizione) |

### 4.3 In `lab/`

Gli script delle tre sessioni di studio del modello. Non producono numeri di tesi: producono
**capire**.

- [s1_forward_minimo.py](../../lab/scripts/s1_forward_minimo.py) → la QKV è fusa,
  `Linear(768→2304)`; l'architettura giusta è `app/vjepa_2_1/`, non `src/models/`
- [s2_preprocessing.py](../../lab/scripts/s2_preprocessing.py) → esistono **tre** preprocessing,
  e la geometria vale quanto cambiare vista
- [s3_target_modules.py](../../lab/scripts/s3_target_modules.py) → ~800 righe di
  LoRA/DoRA/rsLoRA funzionanti, 16 controlli su 16. **Questo codice si porta in `evd/` in F5**:
  non è «implementare LoRA», è «portare codice che gira già, con due correzioni note»

Le conclusioni stanno in `lab/notes/`, e separano sempre *osservato* da *dedotto*.

---

## 4bis. `apps/demo/` — il dimostratore *(ADR-055, 2 settembre 2026)*

**Aggiornamento 9 settembre (ADR-063):** `apps/demo/attention.py` disegna i tre
tagli video della mappa di attenzione — overlay, affiancato, sola heatmap — con
la barra di scala in multipli del peso uniforme. ⚠️ **Di `torch` non sa niente**:
a calcolare e' `evd/explain/`, ed e' la divisione di ADR-015 che il test
`test_apps_demo_non_costruisce_modelli` gia' difende. `COLORMAP_COOLWARM` **non
esiste in OpenCV 5.0.0** — 22 tavole, nessuna divergente — e su una mappa di
scarto una colormap sequenziale renderebbe «molto sotto la media» e «poco sopra»
la stessa cosa scura: la tavola divergente e' costruita a mano, tre nodi BGR e
centro chiaro. Nel dimostratore e' il flag `--attention`.

**Aggiornamento 7 settembre:** `src/evd/eval/demo_metrics.py` raccoglie
metriche del singolo video e tempi di calcolo; `src/evd/exp/demo_stream.py`
gestisce l'inferenza temporizzata e il diario incrementale della sessione.
Il lettore usa `assets/metrics.js` e `assets/realtime.js`.
`InferenceEngine.stream(before_window=...)` consente di attendere la finestra
prima del forward e richiede batch 1 quando il callback ? presente.
Gli override di geometria del dimostratore restano locali alla sessione.

**Aggiornamento 6 settembre:** `serve.py` offre la console locale su loopback,
senza nuove dipendenze web. `assets/console.html` contiene libreria, scelta del
run, ricerca, upload e stato delle analisi; la CLI esistente gira in un processo
annullabile. Il servizio espone solo file registrati e supporta Range HTTP per
il seek. `Avvia dimostratore.cmd` avvia il browser; `apps/demo/README.md` è la
guida operativa. Le aperture sono isolate in `runs/demo-local/sessions/`.
`run_demo.py --batch-size` permette batch 1 in locale e la didascalia riporta
`val_hop_s` quando il replay usa effettivamente la griglia di validazione.

Il deliverable del PdD §6: **dato un video, la timeline degli score e gli
intervalli rilevati sovrapposti**. Era l'ultima riga di DoD di M3 rimasta
scoperta, ed era rimandata da tre mesi (R7).

La regola che ne governa la forma è una sola, e vale la pena tenerla in testa
leggendo la tabella: **qui non si costruisce nessun modello e non si fa
windowing**. La composizione — aprire un run, decidere da dove vengono gli
score, costruire il motore — sta in `evd/exp/demo.py`; `apps/` compone e disegna.
A dirlo non è più solo questa frase:
`tests/unit/test_demo.py::test_apps_demo_non_costruisce_modelli` legge gli import
di ogni file sotto `apps/` e fallisce se compare `torch`, `evd.models`,
`evd.train`, `evd.data.dataset`, `hydra` o `omegaconf`.

| File | Cosa fa |
|---|---|
| `run_demo.py` | La CLI. `--run` e `--video` bastano; `--modo {auto,live,replay}` sceglie la sorgente degli score, `--annotazioni` accende la riga di confronto, `--elenca` dice quali video il run può rigiocare. Non contiene decisioni: mette in fila le chiamate e sceglie i nomi dei file |
| `payload.py` | **`DemoPayload`**, il dato che alimenta tutti e tre i renderer — ed è la ragione per cui non possono divergere. `Segmento` è un intervallo con il suo picco; `as_dict()` è la forma JSON (score a 4 cifre, copertura in **run-length**), `scrivi_json()` è `demo.json` |
| `timeline.py` | `timeline.png`. **Chiama** `eval.plots.timeline_png`, che quel modulo prometteva dal giorno in cui è stato scritto: non ridisegna niente |
| `overlay.py` | `annotated.mp4` con OpenCV: video a sinistra, pannello di lettura a destra (score, stato, allarmi), striscia della timeline in fondo col cursore. Lo sfondo della striscia si disegna **una volta** e si copia per frame |
| `viewer.py` + `assets/viewer.html` | `index.html`, la console interattiva. Un file solo, JSON incorporato, nessun server e nessuna build: si apre con un doppio clic anche offline |

**I quattro artefatti**, in `runs/{run_id}/demo/{video_id}/`: `index.html`,
`annotated.mp4`, `timeline.png`, `demo.json`. Il video viene **copiato accanto**
per default, così la cartella si zippa e si apre su un'altra macchina — che è il
caso vero, perché la seduta non si tiene davanti a questo disco
(`--no-copia-video` la lega invece a questo).

**Le due sorgenti di score** stanno in `evd/exp/demo.py` (`analizza`):

- **live** — backbone dal registry, testa dal registry, adattamento dalla
  factory, **pesi selezionati** dal checkpoint (non quelli dell'ultimo passo: su
  E2d sono 4600 contro 5524), poi `InferenceEngine`. È il caso del PdD: un video
  qualunque, anche mai visto. **Gira** — `test_il_modo_live_gira_davvero`
  percorre la sequenza intera su `TinyBackbone`. **Dal 6 settembre è verificato
  anche su ViT-L/E2d e RTX 4060 locale**, su una clip di una finestra caricata
  tramite la console; la prova CPU reale è stata interrotta per lentezza.
  Il checkpoint E2d selezionato è ora anche sul PC, con hash verificato.
  Su una macchina diversa da
  quella del run serve `--backbone-checkpoint`, perché la config registra il
  percorso dei pesi del backbone dov'erano allora.
- **replay** — gli score si rileggono da `scores.jsonl` (ADR-051) con
  `evd/infer/replay.py`, e **nessun modello viene costruito**. Vale sui video che
  il run ha valutato, e serve a mostrare i numeri *della tesi* senza una GPU.

⚠️ **La geometria è la stessa nei due modi**: le finestre le produce sempre
`sampler_for_protocol`, la proiezione sui frame sempre `risultato_video`
(la funzione libera in cui è stato fattorizzato `InferenceEngine._risultato`,
perché ha un chiamante in più). Fra live e replay cambia chi calcola il
punteggio, non dove cade una finestra né come si apre un allarme.

⚠️ **La partizione viaggia con gli score, e la prima prova l'ha dimostrato.**
`ScoreVideo` porta il campo `split` perché in `edp_v2` la validazione ha una
griglia contigua (ADR-047): rigiocare un video di validazione col sampler del
test non trova **nessuna** delle sue finestre. È successo su `Arrest009_x264`, e
la guardia l'ha detto invece di lasciar disegnare una timeline plausibile e
sbagliata.

⚠️ **Lo stato mostrato è binario, e la classe scritta è quella ANNOTATA.** La
testa emette un solo logit (ADR-031 punto 4): «evento rilevato di classe X» non è
una frase che il sistema può dire. Il PdD §6 la promette — deviazione, `docs/01`
§3bis.

**Verificato su dati veri** il 2 settembre: `e2d_ucf_vjepa21l_frozen-attentive_s42`
in replay su `Arson016_x264` (60 s, 14 finestre) → rilevati 24–28 s e 32–60 s
contro i tre annotati 2,9–10,4 · 17,4–24 · 27–59,8 s. **69 dei 290 video di test
sono in locale** e rigiocabili senza GPU.

## 5. Cosa **non** c'è ancora

I file lo dicono dove serve (`SEGNAPOSTO`, `F4`/`F5`/`F6`).

### 5.-1 La migrazione a `edp_v2` — **fatta il 31 agosto (ADR-047)**

Il protocollo del laboratorio è **eseguibile**: le sei config della matrice
puntano a `protocol=edp_v2`, `train=edp_v2`/`peft_budget_v2`, `eval=chunk_lab`,
`augment=photometric`, e lo split sigillato esiste
(`manifests/edp_v2/ucf_crime/split.json`, 1303 · 306 · 290).

| # | Cosa | Dove è finito |
|---|---|---|
| 1 | La **griglia** ancorata all'inizio, con la coda del video valutata | `data/windows.py::grid="fixed"`, `spans_for_video`, `window_for_span` |
| 2 | I 32 campioni che **coprono** la finestra | `windows.window_for_span` (`linspace`), `effective_fps` per riportare i 3,89 fps |
| 3 | La **validazione** con la sua griglia contigua | `protocol.temporal.val_hop_s` → `runner._sampler(validazione=True)` |
| 4 | Lo split **consegnato**, verificato invece che rigenerato | `scripts/importa_split.py` (cinque verifiche) |
| 5 | La soglia di **Youden** | `metrics.threshold_youden` · `runner._soglia` (che ora applica `threshold.rule`) |
| 6 | **AUC_a/AP_a** e la decisione per video asimmetrica | `metrics.restrict_to_anomalous`, `video_decision_prf` |
| 6bis ⚠️ | Il **livello evento**, con definizione **nostra** (ADR-048) | `metrics.event_prf`/`EventPRF`/`_allarmi` · `chunk_lab.yaml::event_level` · `runner._valuta` |
| 7 | La **selezione del modello** sull'AUROC di validazione | `budget.model_selection` · `trainer._val_metriche` + snapshot · `TrainingState.selected_*` |
| 8 | L'augmentation **solo fotometrica** | `types.AugmentSpec.rand_augment_exclude` · `augment.catena_randaugment` · `configs/augment/photometric.yaml` |

**Cosa resta fuori, e va detto.** *(a)* Il livello **evento** gira dal 31 agosto
(ADR-048) ma con una definizione **nostra**: il documento consegnato non ne
contiene una, quindi quella riga della tabella **non è confrontabile coi loro
numeri** finché Q-B non ha risposta — e va dichiarato, non solo saputo
(`docs/01` §3bis **D10**). *(b)* Il blocco `eval.metrics` è **quasi tutto
descrittivo** — il runner legge `video_auc`, `video_ap` ed `event_level`, i primi
due perché il protocollo dice di non calcolarli e il terzo perché la sua
definizione è nostra; gli altri campi descrivono ciò che `_valuta` produce
comunque. È un'asimmetria nota, e si sistema togliendo i flag inerti o facendoli
leggere, non aggiungendone altri che non tocchino il calcolo. *(c)* **Nessun run
ha ancora eseguito `edp_v2`.**

### 5.0 La migrazione del runner al regime fully-supervised — **fatta il 28 agosto (F11)**

Dal 27 agosto il protocollo in vigore è `edp_v1` (ADR-031). Fino al 28 il codice
che lo *descriveva* esisteva ed era testato (`protocol/split.py`,
`protocol/chunks.py`, metriche, testa binaria, augmentation senza ritaglio), ma il
codice che lo **eseguiva** no: `exp/runner.py` consumava ancora manifest, support
e query, e una guardia esplicita **rifiutava** un protocollo `edp_*` invece di
eseguirlo col disegno sbagliato.

I nove passi sono stati fatti tutti nella stessa sessione — che era il punto:
migrarne una metà avrebbe lasciato due percorsi che divergono in silenzio.

| # | Cosa | Dove è finito |
|---|---|---|
| 1 | Lo **split** al posto del manifest | `exp/runner.py::split_path_for`, `_split` — e `_split` **si ferma** se lo split non dichiara una validazione |
| 2 | Training set da `enumerate_chunks`, col **peso** fino al sampler | `runner._chunks`/`_items` · `WindowItem.weight` · `build_*_loader(weighted=True)` → `WeightedRandomSampler` |
| 3 | Modalità di campionamento **dinamica** | `windows.endpoint_cell`/`window_at` · `chunks.chunk_at` · `runner._resampler` · `WindowDataset(resample=)`, incompatibile con la cache **per costruzione** |
| 4 | LR ed early stop sulla **validazione** | `selection.regime_fully_supervised` (regime `val_split`) · `runner._seleziona_lr` — e il fit finale, che ora una validazione ce l'ha, tiene la pazienza dichiarata |
| 5 | Soglia dalla **validazione** (`threshold.source: val`) | `runner._soglia`, sui chunk **negativi** della validazione |
| 6 | Valutazione sullo **split di test** | `runner._valuta` |
| 7 | Loss **binaria** (BCE, nessun `pos_weight`) e `n_classes: 1` | `trainer.Objective`/`_perdita`/`_bersaglio`/`_logit` · `runner._objective`/`_n_classes` |
| 8 | **Tasso di positivi misurato** e conteggio dei chunk | `runner` → `chunks` in `metrics.json` **e** in `meta.json` (dove è servito riparare `tracker.start(extra_meta=…)`, che era inerte) |
| 9 | Protocollo giocattolo nella forma nuova | `configs/protocol/synthetic_toy.yaml` + `exp/smoke.py::_split_giocattolo` |

**Tre cose scoperte facendola, che valgono più dell'elenco.**

*(a)* **`extra_meta` del tracker era un parametro accettato e mai scritto.** Il
runner gli passava la ricevuta dell'adattamento dal 21 agosto, e in `meta.json`
non c'è mai finita. Nessun test lo prendeva perché tutti guardavano
`metrics.json`. È la famiglia di `EVD_SEED` e `budget.precision`, alla terza
occorrenza.

*(b)* **`manifest_sha` è diventato `protocol_sha`** in `tracker`, `checkpoint` e
`RunResult`. Tenere il vecchio nome sarebbe costato zero righe e avrebbe prodotto
un campo che nomina un artefatto che non esiste più — cioè esattamente ciò che
questo repository chiama un metadato che mente.

*(c)* **La guardia dello smoke era tarata sul regime precedente**, e questo l'ha
resa illeggibile per mezz'ora: col budget vecchio il chunk AUC usciva **sotto il
caso** (0,225 · 0,328 · 0,367), che somiglia moltissimo a un segno invertito nel
codice. Non lo era: 20 passi a batch 4 con lr 1e-3 erano il budget di una testa a
4 vie in cross-entropy su ~26 finestre, e su 91 chunk con un logit e una BCE non
muovono niente. Rimisurato (`configs/train/probe_toy.yaml`), gli stessi semi danno
1,000 · 0,984 · 1,000. **Quando cambia l'obiettivo, una guardia si rimisura — non
si rilegge.**

**~~Cosa resta fuori~~ — chiuso il 28 agosto (ADR-035).** Questo paragrafo diceva
che lo split vero di UCF-Crime non esisteva, che mancava la **quota** della
validazione, e che lo script per costruirlo restava fuori proprio per non
scegliere quella quota nel codice. Antonio l'ha decisa — **20% dei video di
training, per video, stratificato per classe** — e sta dove doveva stare: nel
blocco `validation` di `configs/protocol/edp_v1.yaml`, non in una CLI. Lo split
esiste (`manifests/edp_v1/ucf_crime/split.json`: **1278 train · 312 val · 290
test**, 20 video esclusi dal dedup Q19) e lo genera `scripts/build_split.py`.
La guardia del runner **resta**: uno split senza validazione ferma il run
nominando Q-C, perché la domanda al laboratorio è ancora aperta e la loro
partizione sarebbe `source: laboratory`, cioè un protocollo nuovo.

### 5.1 Il resto

> **⚠️ Questa tabella è cresciuta per accumulo e molte righe sono superate.** Lo
> stato vero al **1 settembre 2026**, in tre righe:
>
> - **Manca solo `apps/demo/`.** È l'unico modulo del layout di `docs/02` §5 che
>   non esiste. Il checkpoint di E2d c'è, il codice è riuso di `InferenceEngine`
>   + `FrameScoreMapper` (invariante 4), ed è l'ultima riga di DoD di M3 mai
>   fatta — **R7** in `docs/09`, salita a media × alto.
> - **Le righe su E0b.2, E0c, E0d, E0h, E2b-ML e i manifest sono storia.** **E0
>   è chiuso per intero**: l'estrazione `edp_v2` è fatta e verificata *dentro* E2
>   (`cache_hit_rate 1,0`), E2b-ML è **ritirata** (ADR-029), E0h è stato
>   **sostituito** da E0h-3k, e l'ultima gamba — **E0d-cache** — ha chiuso il 1
>   settembre con esito negativo.
> - **I wrapper IVN-L e DINOv3 restano non scritti, ed è deliberato**: E2a è
>   opzionale dal 25 agosto (`docs/01` §3bis D7), quindi C1 e C6 si dichiarano
>   non verificati.
>
> Le righe sotto si leggono come *cronaca di cosa mancava quando*, non come
> elenco di cosa manca oggi.

| Manca | Fase | Cosa serve |
|---|---|---|
| ~~**`apps/demo/`**~~ | ~~F7~~ | ✅ **scritto il 2 settembre 2026** (ADR-055). Vedi §4bis |
| ~~`CACHE_RAM_MB` da abbassare~~ | ~~prima di E3~~ | **fatto il 1 settembre** (ADR-054): vale **0**, la cache RAM del `WindowDataset` è spenta. `e0d_cache_benefit` (job 541451) aveva chiuso con esito **negativo** — non paga, e sopra ~500 MB per worker costa fino al **16%** sulle celle PEFT, che decodificano tutto. Zero e non 166 perché fra i due il throughput non sceglie (0,991×), quindi sceglie la RAM. A budget nullo `ExperimentRunner._loader` spegne anche `cache_in_memory`, o si loggherebbe «budget esaurito a 0/N finestre» a ogni run. ⚠️ Il beneficio resta un **limite inferiore** (3000 pescaggi contro gli 88.384 di un'epoca): chi la riaccende rifà la misura nel regime vero |
| wrapper IVN-L e DINOv3-frame | **F4-bis** | prima ADR-023 sulla dipendenza runtime `transformers`; senza, E2a non è eseguibile |
| ~~APT~~ | ~~F5-bis~~ | **fatto il 16 agosto** (`adapt/apt.py` + `vendor/vjepa21_rope.py`). Resta un'osservazione aperta: su tutti e 24 i blocchi il gradiente del **primo** banco è esattamente nullo, per cancellazione a 4608 chiavi (job 533071 in bf16, 533073 in fp32). Sui blocchi profondi impara; quale intervallo usare in E3-P.3 è una scelta di matrice |
| ~~DA-VPT~~ | ~~stretch~~ | **fatto il 16 agosto** (`adapt/davpt.py`). Il termine di metric learning arriva al `Trainer` per il contratto `AuxiliaryLoss`, quindi non ha richiesto F6. ⚠️ Resta stretch come **esperimento**: pesi della loss e definizione di classe in regime few-shot sono decisioni del 24 agosto, e `configs/adapt/davpt_n8.yaml` ha `n_classes: ???` apposta |
| run E0b.2 | **F4+F5** | il codice c'è (`exp/e0b2.py`, job 533078 lanciato il 16 agosto); resta da registrare la griglia in `docs/10` e da portarla al 24 agosto insieme a `peft_budget_a` |
| run E0c/E0d | **F4, sperimentale** | E0c richiede il subset UCF101 esplicito; E0d parte dal pilota end-to-end e aspetta geometria/hop per l'estrazione piena |
| ~~`exp/runner.py`~~ | ~~F6~~ | **fatto il 21 agosto**: `ExperimentRunner` esiste, `exp/smoke.py` lo chiama, `scripts/run_experiment.py` non solleva più `NotImplementedError`. Con lui `exp/sources.py`, `exp/registry.py`, `eval/aggregate.py`, `models/adapt/factory.py`, `train/selection.py`, `train/checkpoint.py` e `scripts/make_figures.py` |
| ~~checkpoint e ripresa intra-training~~ | ~~F6~~ | **fatto il 21 agosto** (`train/checkpoint.py`, D5): `runs/{run_id}/checkpoint/` con testa + `state_dict_adapter()` + ottimizzatore + RNG + batch consumati (**mai** il backbone), identità verificata alla ripresa, e la flag installata dietro guardia perché `signal.SIGUSR1` **non esiste su Windows**. Serviva oltre il Wall Time di 7 h, dove `-R` non aiuta: il job E0f `534484_0` è terminato `TIMEOUT` senza rientrare in coda |
| ~~`scripts/campagna.py` + `scripts/slurm/campagna.sbatch`~~ | ~~F6~~ | **fatto il 21 agosto** (D7): `piano/stato/valida/esegui/invia`, dedup **dal registry** e non dalla cartella, fette da ≤8 task perché ogni task di un array conta sul tetto di 10 job dell'account, `--signal=B:USR1@300` che è la metà che rende utile il checkpoint di D5 |
| ~~`apps/demo/`~~ | ~~F7~~ | **promosso in cima alla tabella**: è l'unica cosa che manca davvero |
| E2a — wrapper IVN-L e DINOv3 | **opzionale dal 25/8** | `docs/01` §3bis D7: V-JEPA 2.1-L e' il backbone di tutti gli esperimenti, E2a scende a opzionale. `configs/experiment/e2a_headtohead.yaml` **non e' scritta**, ed e' deliberato: una config che non si puo' eseguire e' un segnaposto che sembra pronto |
| ~~copertura dello store E0d al 100%~~ | ~~prima di E2~~ | **fatto il 25 agosto**: la coda di 1072 feature (l'ultima finestra di 268 video su 290) è stata estratta; hit rate 1,000000 sulla griglia di `fsp_v1` |
| ~~budget a epoche fisse invece che a passi~~ | ~~prima di E2d~~ | **fatto il 25 agosto** (ADR-027): `BudgetSpec` ha `epochs`/`max_steps_cap`, `e2d_curve` (primaria) ed `e2d_curve_passifissi` (sensitivity) esistono entrambe |
| E2b-ML rientra sul cluster | **prima di potare lo store** | codice e config pronti (ADR-029, 26 agosto): 5 celle, `layer: hierarchical`. Nessun job sottomesso — si sottomette una cella sola (`seed=1`) per misurare wall time e hit rate prima delle cinque (`CLAUDE.md` §2 punto 8). Finché non ha un esito, la potatura post-E2b (R6, `docs/09`) resta sospesa |
| E0h rientra sul cluster | **prima di scrivere `weight_decay`/schedule in `probe_e2*.yaml`** | codice e config pronti (ADR-030, 26 agosto): 80 celle (`weight_decay` × `lr_schedule` × k × seme). Nessun job sottomesso — si sottomette una cella sola prima delle 80 (`CLAUDE.md` §2 punto 8). Finché non ha un esito, `probe_e2*.yaml` resta `weight_decay: 0.0`/`lr_schedule: constant` |
| ~~binding reale di `build_manifests.py --protocol` e `manifests/fsp_v1/`~~ | ~~dopo il freeze~~ — **fatto il 25 agosto**: 35 manifest sigillati (7 budget x 5 semi). Il binding ha retto al primo comando con una correzione, `content_sha` che non arrivava al builder — invisibile finche' `dedup` era `none`. Testo storico: | integrare inventario e annotazioni FS-UCF-Crime; i manifest non devono esistere prima dell'approvazione (invariante 1). **Il binding è scritto** (21 agosto, D9): `--protocol` costruisce il bundle da `DATASOURCES`, legge le regole dal YAML firmato — non dalla CLI, o file approvato e manifest generati potrebbero divergere — e rifiuta di riscrivere un manifest esistente. Resta **irraggiungibile** finché `approved: false`, ed è il punto: il giorno dell'approvazione è un comando, non una fase |

---

## 6. Stato e comandi

**10 ottobre 2026 — VTCD selezionata (ADR-080):** nuovi moduli
`exp/vtcd_sampling.py` (selezione riproducibile dei chunk distinti e matching
degli eventi) e `eval/spatial_bbox.py` (sette impostazioni bbox).
`scripts/vtcd_spatial_selected_{plan,worker,supervisor,pull}.py` e
`scripts/slurm/vtcd_spatial_selected.sbatch` concretizzano ed eseguono il
DAG con ripresa, fold annidati, selezione bbox e trasferimento MIVIA.
`tests/unit/test_vtcd_spatial_selected.py` verifica matching, bbox e assenza
di uso della classe esclusa in calibrazione. Conteggi statici rimisurati:
**131** file Python in `src/evd`, **22** in `apps`, **86** in `scripts`,
**113** in `tests`. Diciotto test mirati passati e lint dei file nuovi
pulito; suite completa non rieseguita. Replay reale A100 dodici finestre
passato con errore zero. [Avvio](varie/avvio_vtcd_selected_20261010.md).

Ripresa pomeridiana: tre test di regressione per consumo concorrente dei
marker SLIC, JSON corrotto e task pronti rinviati dal limite job. Otto test
nel file dedicato passati; conteggi statici invariati. Il supervisore
recupera i task WAITING anche fra tick e non considera errore un marker
già consumato. Dettagli e ricevuta live nel documento di avvio.

Autonomia rafforzata nello stesso pomeriggio: retry con backoff delle sole
query SLURM indisponibili, persistenza immediata dei job e contatore di
convergenza basato sulle ricevute. Altri tre test di regressione passati,
**11 nel file dedicato**; gli errori scientifici mantengono il blocco.

**9 ottobre 2026 — campagna spaziale VTCD (ADR-079):** `eval/spatial.py`
calcola conteggi native/token, IoU e matching delle istanze; riusa i box
canonici e conserva i pesi di piani, video e classi. Quattro nuovi CLI
`scripts/vtcd_spatial_{prepare,worker,supervisor,pull}.py` preparano snapshot
Forma e fold, eseguono il DAG, mantengono la supervisione cron e verificano
il rientro. SLIC in streaming su cinque nodi `fatq`, A100 per i soli forward
mancanti. Dettagli e comandi in [avvio](varie/avvio_vtcd_spatial_20261009.md).
Conteggi statici rimisurati con `Path.rglob('*.py')`: **129** file in `src/evd`,
**22** in `apps`, **81** in `scripts`, **112** in `tests`. Dodici test mirati
passati, prima dieci anche sul cluster; pilota reale di dodici finestre passato.
La suite completa non è stata rieseguita e questi conteggi non la certificano.

**7 ottobre 2026 — «Applica a più piani» in Forma:** `assets/workflow.js` porta i box del
piano corrente su un intervallo di piani (sostituisci/aggiungi, bozza, protezione dei confermati,
un solo Ctrl+Z); pulsante nel pannello «Lavoro nel tempo» e Shift+P. Solo frontend; dettagli in
[docs/20](20_annotatore_spaziale.md).

**6 ottobre 2026 — statistiche in Forma:** `apps/annotator/stats.py` riassume
l'avanzamento per video (frame d'evento, piani di base, eventi; per classe), `Project.stats()` lo
espone su `GET /api/stats` e `assets/stats.js` lo mostra nella finestra **Statistiche**. Contratto
e verifiche in [docs/20](20_annotatore_spaziale.md).

**6 ottobre 2026 — piani VTCD di base in Forma:** senza sorgente, ogni evento ha i suoi
piani dal protocollo `edp_v2`, con la stessa regola delle sorgenti VTCD (`overlap_positivo`,
16 piani per finestra) e gli stessi ID di unità: `planes.protocol_units`, `Project.protocol_view`
e `GET /api/video/<id>/units`. La traduzione protocollo → finestre è una sola
(`planes.protocol_windows`) e il protocollo si legge con `geometry.load_protocol()`, senza
omegaconf. I piani si salvano nella tabella `protocol_units` con la loro impronta e il
catalogo ne riporta il numero, su cui la raccolta si può ordinare insieme a nome, classe
annotata ed eventi. Contratto e verifiche in [docs/20](20_annotatore_spaziale.md).

**6 ottobre 2026 — fonte esterna Liu & Ma in Forma:** `apps/annotator/liu_ma.py`
converte `Test_annotation.pkl` di Liu e Ma (2019) in coppie importabili, solo sul
test UCF a 320×240 e solo dentro gli eventi del laboratorio; `scripts/prepara_liu_ma.py`
scrive raccolta e `audit.json`. Le coppie sono di tipo `external_keyframes`: un'unità
di revisione per fotogramma campionato, nessuna estensione ai vuoti. Il server non
cambia (l'import passa da `imports.import_pair`); gli asset guadagnano la modalità
«Campioni fonte» e «Usa box della fonte». 79 test dell'annotatore passati, import
completo provato su copia del database; contratto e limiti in [docs/20](20_annotatore_spaziale.md).

**5 ottobre 2026 — estensione ergonomica di Forma:** ora sette moduli Python
(incluso `__init__.py`) e otto asset web. `preload.py` aggiunge cache JPEG
limitata e cancellazione al cambio video; `geometry.token_box` discretizza e
registra le coordinate token senza cambiare i pixel. `workflow.js` contiene
le interazioni e le scelte di conferma/copia; `playback.js` il precaricamento
nel browser e la riproduzione basata sugli FPS e sulle durate dei piani.
63 test mirati in tre file, collaudo browser e inventario completo dei test
(290 UCF + 270 MIVIA, tutti disponibili). Contratto e ricevute in [docs/20](20_annotatore_spaziale.md).

**5 ottobre 2026 — griglia, raccolte e Guida in Forma:** `apps/annotator/` conta
sei moduli Python (incluso `__init__.py`) e sei asset web. `geometry.py` riproduce
la proiezione dei bordi dei token del renderer; `imports.py` verifica e importa
coppie su più video attraverso il contratto esistente. Le preferenze iniziali
sono separate dalle annotazioni umane. `box-tools.js` gestisce griglia e menu
contestuale, `collection.js` la verifica e importazione per lotto, `guide.js`
la tabella comune delle scorciatoie e la Guida. Snap, ID e colore sono salvabili
e annullabili; eliminato il tipo di oggetto. **54 test Python passati**, prove
browser su copia isolata e layout a 1440×1000 e 1000×800. La Guida nell'app e
[docs/20](20_annotatore_spaziale.md) vanno aggiornati a ogni modifica funzionale.

**3 ottobre 2026 — correzione diretta in Forma:** i box della sorgente sono
subito modificabili, con originale visibile durante la correzione e soli numeri
come etichette. Ctrl+S salva anche i campi ancora focalizzati; Ctrl+Z/Y conserva
le azioni dopo salvataggio e cambio video, senza il limite di 25, memorizzando
solo i frame cambiati. Prove browser isolate e 35 test Python dell’annotatore
passati; dettagli e ricevute in [docs/20](20_annotatore_spaziale.md).

**2 ottobre 2026 — annotatore spaziale Forma:** `apps/annotator/` contiene
originariamente quattro moduli Python (incluso `__init__.py`) e tre asset web. `project.py` importa
e verifica le coppie JSON/CSV del renderer e gestisce sorgenti, revisioni,
recupero e collegamenti ai video; `server.py` serve frame nativi per indice
ordinale ed esportazioni coerenti. `planes.py` ricostruisce e verifica le unità
temporali usando il mapping del renderer; `explanation_boxes.py` esporta anche
gli indici delle finestre in `temporal_layout`. Nessun algoritmo numerico modificato.
CLI: `python scripts/annota_spaziale.py`; avvio Windows: `Avvia annotatore.cmd`.
Il progetto iniziale ha 560 video, 554 file disponibili e 29 sorgenti verificate
con proposte su 30.272 frame. Guida: [docs/20](20_annotatore_spaziale.md).
Verifica mirata rimisurata: **35 test dell'annotatore + 38 dei box = 73 passati**; revisione per
piano, piani vuoti e protezione delle correzioni puntuali verificati nel browser; Ruff pulito sui file nuovi, JavaScript sintatticamente valido.
Prova browser con disegno, propagazione, salvataggio/riapertura, export e MIVIA;
nessun errore JavaScript, layout controllato a 1440×1000 e 1000×800.
Questa verifica non è una nuova esecuzione della suite completa.

**27 settembre 2026 — OVAD:** `scripts/esporta_score_ovad.py` converte gli score
EDP-v2 con estremi espliciti; `scripts/valuta_ovad.py` calcola le metriche col
codice ricevuto in `ovad_protocol/`, salva le tabelle intermedie e genera
classifiche per dataset, con una versione selezionata e con tutte le
versioni. Dal 28 settembre sceglie la versione dalla media dei mean UCF/MIVIA
s42, esplicitamente sul test, e pubblica tutte le metriche per seme;
`--solo-classifiche` rigenera le tabelle senza rivalutare gli score.
`--da-export <raccolta>` riusa piano e input convertiti dopo controllo degli
hash; `--guard 30` rivaluta validazione e test con la guardia comune richiesta
per il confronto SOTA. La raccolta del 28 settembre è in
`runs/classifiche/ovad_guard30/`, distinta da quella iniziale con guardia 8 s.
Il comando riproducibile e la ricevuta completa sono in
`runs/classifiche/ovad/README.md` e `manifest.json`; sintesi e limiti in `docs/19`.
Conteggi statici rimisurati: **128** file Python in `src/evd`, **13** in
`apps`, **68** in `scripts`, **101** in `tests`; i **7** file Python della
consegna `ovad_protocol/` sono esterni a questi quattro alberi.

**22 settembre 2026 (ADR-072):** nascono `src/evd/explain/boxes.py`,
`src/evd/exp/ig_topk.py`, `apps/demo/explanation_boxes.py`,
`scripts/{render_bounding_box,campagna_ig_topk,esporta_score_chunk}.py`, due
`.sbatch` e tre file di test (29 + 5 + 4 funzioni, verdi su CPU; +2 in
`test_explain_viste.py`). Conteggi statici rimisurati: **125** file Python in
`src/evd`, **13** in `apps`, **60** in `scripts`, **96** in `tests` — gli scarti
oltre i file di questa voce vengono da lavoro F14 non committato nello stesso
working tree. ⚠️ La suite intera **non** e' stata rieseguita, e
`test_explanation_advanced.py` ha **2 fallimenti preesistenti** dovuti al lavoro
`class_direction` non committato in `explain/methods.py`, non a questa voce.

**21 settembre 2026 (ADR-070):** nascono `src/evd/exp/descrittori.py`,
`scripts/estrai_descrittori.py`, `scripts/riduci_descrittori.py`,
`scripts/slurm/descrittori.sbatch` e `tests/unit/test_descrittori.py` (19 test,
tutti su CPU, verdi). Conteggi statici rimisurati: **123** file Python in
`src/evd`, **12** in `apps`, **57** in `scripts`, **93** in `tests` — ⚠️ gli
scarti oltre i quattro file di questa voce vengono da lavoro F14 non ancora
committato, che condivide il working tree. La suite intera **non** e' stata
rieseguita in questa sessione: verdi sono i 19 test nuovi e i 230 di regressione
sui moduli toccati (motore di inferenza, cache di valutazione, dimostratore,
campagna di spiegazione, `test_runner.py`).


**14 settembre 2026:** `scripts/occlusioni.py` esegue uno shard di campagne
gia selezionate riusando `explanation_advanced.extend_campaign`, con ripresa
per metodo. `explain.volumes` costruisce maschere in token e le espande
esattamente sui pixel; `explain.methods` condivide le maschere tra occlusione
e freeze locale. Conteggi statici rimisurati: **119** file Python in `src/evd`,
**12** in `apps`, **46** in `scripts`, **89** in `tests`.

**14 settembre 2026, sera — eseguita e renderizzata.** Otto shard sul cluster
(job 562113-562127) hanno prodotto le 150 finestre della raccolta
`runs/analisi_explain/volumi_20260914`: 48 campagne, **187 forward completi per
finestra** (135 occlusione + 52 movimento), **62,5 minuti di GPU** e 18 minuti di
wall-clock. La geometria eseguita e **45 volumi sovrapposti** — 25 da 6x6 token
con passo 4, 9 da 8x8 con passo 6, 11 unioni speciali — ciascuno esteso a
**tutti e 16** i piani temporali: quindi l'occlusione localizza nello spazio, non
nel tempo, ed e l'ablazione del movimento a porre la domanda temporale.
⚠️ Il numero che il piano portava (144 volumi disgiunti su 4 piani, 288 forward)
non e mai stato eseguito, e la figura della presentazione che lo disegnava e
stata corretta.

Il rendering e girato **sul cluster** (job 562175, partizione `fatq`, 12 CPU,
**5 minuti 02 s**) grazie a `--jobs N`, nuovo in `scripts/render_spiegazioni.py`:
le scale comuni al `run_id` si calcolano **prima** e una volta sola, quindi il
parallelismo non puo cambiare un colore, e ogni campagna scrive solo dentro la
propria cartella. Lo stesso lavoro in serie sulla GPU locale costava 38,6 minuti.
Dal cluster sono tornati 416 MP4 e 1 200 PNG (139 MB); i `*_features.npz`
(5,2 GB) **restano la**, perche non servono ne al catalogo ne alle figure.

**Cinque metodi, 12 settembre 2026:** conteggio statico rimisurato con
`Path.rglob('*.py')`: **119** file in `src/evd`, **12** in `apps`, **45** in
`scripts`, **89** in `tests`. La campagna richiesta e di 75 finestre per modello;
il completamento si legge da `rapporti/advanced_rendering.json`. Il
[dossier dei metodi](varie/interpretabilita_cinque_metodi.md) documenta formule,
scelte della campagna e comandi. La verifica CPU include l'esecuzione dei cinque
metodi sul backbone sintetico, la ripresa senza nuovi forward e la decodifica
dei filmati; i risultati GPU non si deducono da questi test.

**Explainability, 10 settembre 2026 (ADR-067):** conteggio statico rimisurato:
112 file Python in `src/evd`, 11 in `apps`, 44 in `scripts`, 88 in `tests`.
`apps/demo/explanation_catalog.py` genera le due liste Sviluppo e Metodi completi e riunisce le campagne nella pagina di ciascun video/modello, con famiglie, descrizioni e titoli IG uniformi;
`scripts/render_spiegazioni.py` rigenera una raccolta, riprende i rendering mancanti
e costruisce il catalogo. I video parziali esportano solo i segmenti coperti.
La verifica mirata e gli artefatti reali sono descritti nel
[dossier](varie/explainability_vjepa21_1009.md). Le prove di costo non chiudono la
campagna: il confronto perturbativo completo resta da eseguire; i tre video
inizialmente mancanti sono stati recuperati dal cluster con SHA256 verificato.

**Chiusura del 9 settembre 2026:** ⚠️ **nessun test, smoke, lint o raccolta
pytest eseguito**, su richiesta esplicita di Antonio — gli esiti riportati sotto
sono **fotografie storiche** e non certificano lo stato committato oggi. I soli
numeri rimisurati sono statici (`Path.rglob("*.py")`, `__pycache__` escluso):
**104** file Python in `src/evd` (erano 99), **9** in `apps` (8), **40** in
`scripts` (36), **86** in `tests` (81). I cinque file di test nuovi
(`test_composite.py`, `test_explain.py`, `test_classifiche_modelli.py`,
`test_scelta_video_explain.py`, `test_scheda_analisi_qualitativa.py`) sono
**scritti ma non eseguiti in questa passata**.

**Analisi E3 del 7 settembre:** `scripts/analisi_ablation_e3.py` verifica
score/etichette/span, AUC/AP, Youden, evento e latenza storica; produce
`runs/analisi_e3_20260907/analisi.json` e diagnostica per cella.
Usa `exp/rilettura.py`, `eval/metrics.py` e `eval/strata.py`.
Dossier: `docs/15_e3_ablation_tre_semi.md` e `15_e3_ablation_strati.md`.
39 celle rilette con esito coerente; nessun test di suite o smoke eseguito.
Conteggi statici: 99 file Python in src/evd, 8 in apps, 36 in scripts,
81 in tests. Il conteggio dei test resta storico.


**Chiusura del 7 settembre 2026:** nessun test, smoke, lint o raccolta pytest
eseguito, su richiesta di Antonio. Il conteggio statico dei file Python
(`Path.rglob("*.py")`) ? 99 in `src/evd`, 8 in `apps`, 35 in
`scripts` e 81 in `tests`. Il numero di test e gli esiti sotto sono
fotografie storiche, non una certificazione dello stato committato oggi.

**Verifica del 6 settembre 2026, dimostratore:** 34 test mirati passati
(`test_demo.py` e `test_demo_console.py`), Ruff check/format puliti sugli 8 file
Python controllati, sintassi dei due script JavaScript verificata con Node.
Raccolta globale: **1416 test**, senza eseguire la suite completa.
Conteggio dei file Python rimisurato con `Path.rglob('*.py')`: **97** in
`src/evd`, **8** in `apps`, **35** in `scripts`, **79** in `tests`.
Replay reale E2d su Arson016 e inferenza reale ViT-L su RTX 4060 con un upload
di una finestra; artefatti e limiti in `apps/demo/README.md`. Verifica visiva
nel browser non eseguita: nessun browser disponibile all'automazione.

**Verifica del 5 settembre 2026, limitata alle aggiunte:** due test di
`test_complementarita_far.py` passati; Ruff check/format sui due file nuovi.
Le due nuove sweep sono risolvibili sul cluster. Rilettura di 15/15 celle
con verifica esatta di score, etichette, span e conteggi evento. Nessuna nuova
esecuzione della suite completa: il conteggio storico sotto resta datato.

`scripts/analisi_complementarita_far.py` produce
`runs/analisi_20260905/complementarita_far.json`: 45 punti operativi, 30 coppie
chunk e confronti evento per classe. Usa `exp/rilettura.py` e le metriche
esistenti; non modifica i risultati ufficiali. Le config
`e3p_vpt_n64` e `e3r_lora_r8` aggiungono un gruppo adapt, experiment e sweep
ciascuna, con il seme 42 e il budget originale.

```powershell
conda activate pytorch
cd evd-vfm

.\scripts\smoke.ps1        # la spina dorsale è integra? (~35 s, CPU)
python -m pytest           # la suite completa
python -m ruff check .     # lint
```

**Ultima verifica locale: 24 settembre 2026, sera, dopo la fusione dei branch (v1.80).**
**1780 test raccolti**, 1778 passati e 2 saltati, in un'unica invocazione
(`python -m pytest tests`, smoke compresi, output su file, 38 minuti): esce **0**.
Prima della fusione **due** test fallivano dal 17 settembre
(`test_explanation_advanced.py`, riferimenti per classe mancanti al test) e sono
corretti. `ruff check` resta con **un** errore preesistente (E731 in
`scripts/analisi_ablation_e3.py`, già su `0404490`) e `ruff format --check` con
**13** file da formattare, nessuno toccato dalla fusione: sono lavoro del 17–23
settembre (i `vtcd_grid_*` coi loro test, `test_vtcd_ablation.py`,
`apps/demo/explanation{,_catalog}.py`, `confronto_sota_xlsx.py`) più due già non
formattati prima (`analisi_ablation_e3.py`, `test_figura_training.py`). Conteggi statici:
**128** file Python in `src/evd`, **13** in `apps`, **66** in `scripts`, **100** in
`tests`.

**Verifica precedente: 4 settembre 2026, dopo ADR-060 (v1.62).** **1393 test
raccolti**, tutti verdi, e per la prima volta **in un'unica invocazione**:
`python -m pytest tests -q` con l'output rediretto su file esce **0** e arriva al 100%
(smoke compreso — `addopts` è solo `-q --strict-markers`, quindi `slow` e smoke sono
inclusi). ⚠️ La nota «il giro intero supera il limite della shell» delle verifiche
precedenti riguardava il **terminale**, non pytest: redirigendo su file il problema non
si presenta, e le partizioni restano utili solo per leggere l'esito a pezzi. `ruff check`
e `ruff format --check` puliti su **207 file** (erano 203: i quattro in più sono
`src/evd/eval/strata.py`, `scripts/analisi_strati.py`, `tests/unit/test_strata.py` e
`tests/unit/test_analisi_strati.py`).

I **36 in più** rispetto ai 1357 di E5b stanno in `test_strata.py` (25),
`test_analisi_strati.py` (8) e `test_report_figure.py` (3), e due meritano di essere
nominati perché sono la ragione per cui l'analisi stratificata si può credere.
**`test_gli_strati_di_evento_non_possono_contraddire_il_totale`** somma gli eventi e i
rilevati degli strati e li confronta con `event_prf` sull'insieme intero: è il test che
giustifica la scelta di *filtrare `gt` e richiamare le funzioni esistenti* invece di
ricontare, e senza di esso una riga di strato potrebbe dire una cosa diversa dal numero
aggregato che le sta accanto nella stessa tabella. **`test_appiattisci_lascia_fuori_i_booleani`**
copre un difetto silenzioso: `True` è un `int` in Python, quindi senza il controllo
esplicito il flag `livello_evento` sarebbe entrato nella tavola come una metrica costante
a 1,0 — un numero perfettamente plausibile che nessuno ha misurato.

**Verifica sul branch `worktree-e5b-deployment`, 4 settembre 2026, sera (v1.61.2; il branch e' stato fuso il 24 settembre, e il conteggio dopo la fusione sta nella verifica qui sopra).** **1382 test raccolti**;
eseguiti **in partizioni** come sempre — `tests/unit` senza `test_runner.py` (esce **0**),
`test_runner.py` da solo (esce **0**), `tests/smoke/` (esce **0**). `ruff check` e
`ruff format --check` puliti su **208 file** (erano 203: i cinque in più sono
`eval/deployment.py`, `eval/deployment_plots.py`, `exp/e5b_deploy.py`,
`scripts/e5b_deployment.py` e `tests/unit/test_deployment.py`).

I **25 in più** stanno tutti in `test_deployment.py`, e quattro meritano di essere nominati.
**`test_il_tempo_stazionario_e_il_massimo_e_la_risposta_e_la_somma`** e
**`test_quando_il_decoder_e_piu_lento_e_lui_a_comandare`** difendono il modello di pipeline
dai due lati: somma e massimo non si confondono, e `stadio_dominante` non è una costante
travestita da misura — se comandasse sempre il modello, la conclusione «il collo di bottiglia
è il modello» sarebbe vera per costruzione invece che per i numeri.
**`test_la_verifica_rifiuta_un_solo_punto_perche_sarebbe_vacua`** è il controgruppo del fit sul
wall-clock: con un punto una retta a due parametri passa esatta, quindi quel caso deve
**sollevare** — è la stessa forma della verifica del merge su `init: kaiming_b_zero`.

⚠️ **Corretto nella stessa sessione un test che passava solo grazie all'ordine degli altri**:
`test_e5b._pezzi` costruiva `TinyBackbone` dal RNG **globale**, quindi
`test_una_cella_fusa_fuori_tolleranza_si_marca_invece_di_passare` passava dentro la suite e
**falliva da solo** — su certe inizializzazioni il merge esce *bitwise esatto* e lo scarto è
0,0, cioè il test asseriva senza dirlo che l'arrotondamento fp32 debba per forza essere diverso
da zero. La fixture è ora seminata.

**Verifica precedente: 3 settembre 2026, dopo E5b (v1.61).** **1357 test raccolti**;
eseguiti **in partizioni** come sempre — `tests/unit` senza `test_runner.py` (esce **0**),
`test_runner.py` da solo (esce **0**), `tests/smoke/` (esce **0**). `ruff check` e
`ruff format --check` puliti su **203 file** (erano 200: i tre in più sono `exp/e5b.py`,
`scripts/e5b_efficienza.py` e `tests/unit/test_e5b.py`).

I **28 in più** rispetto ai 1329 di ADR-055 stanno in `test_e5b.py` (21) e in
`test_registry.py` (7), e due meritano di essere nominati perché sono la ragione per cui E5b
si può credere. **`test_dopo_una_cella_fusa_il_backbone_torna_quello_di_partenza`** confronta
il digest dei pesi prima e dopo una variante fusa: `merge_()` scrive **in place**, e senza
ripristino ogni cella successiva misurerebbe pesi corrotti con tempi perfettamente plausibili.
**`test_senza_perturbazione_la_riga_si_dichiara_non_messa_alla_prova`** è il controgruppo del
controllo di fedeltà: a perturbazione nulla ΔW resta zero — `init: kaiming_b_zero` — quindi il
merge riesce e la riga deve uscire comunque **`sospetta`**, perché un controllo che non può
fallire non è un controllo. Se quel test passasse con `ok`, E5b starebbe certificando il merge
sulla base di niente.

**Verifica precedente: 2 settembre 2026, dopo ADR-055 (v1.58).** **1329 test raccolti,
tutti verdi** (`addopts` è solo `-q --strict-markers`, quindi `slow` e smoke sono inclusi).
⚠️ **Eseguiti in partizioni**, per la stessa ragione di sempre: il giro intero supera il limite
della shell. Le invocazioni sono state `tests/unit` senza `test_runner.py` (esce **0**),
`test_runner.py` da solo (esce **0**) e `tests/smoke/` (esce **0**). `ruff check` e
`ruff format --check` puliti su **200 file** (erano 190: i dieci in più sono `apps/`).

I **21 in più** rispetto ai 1308 di ADR-054 sono tutti in `test_demo.py`, e tre meritano di
essere nominati: **`test_apps_demo_non_costruisce_modelli`** legge gli import di ogni file sotto
`apps/` e fallisce su `torch`/`evd.models`/`evd.train`/`hydra` — è la prima volta che la regola
di ADR-015 può fallire invece di essere solo scritta; **`test_il_modo_live_gira_davvero`** monta
un run finto con `TinyBackbone` e percorre la sequenza intera del modo `live` (registry, factory,
pesi selezionati dal checkpoint, forward, proiezione), che senza di lui sarebbe codice mai
eseguito; **`test_la_palette_del_video_e_quella_della_pagina`** confronta `overlay.COLORI` col
blocco `:root` del template — il docstring afferma che il video e la pagina sono un tema solo, e
la prima stesura lo affermava con i colori di *un'altra* versione del template.

**Verifica precedente: 1 settembre 2026, sera, dopo ADR-054 (v1.57).** **1308 test
raccolti, tutti verdi** (`addopts` è solo `-q --strict-markers`, quindi `slow` e smoke sono
inclusi). ⚠️ **Eseguito in partizioni, non in un processo solo**: il giro intero dura ~55
minuti e supera il limite della shell, quindi i 70 file di `tests/unit/` sono stati percorsi in
tre gruppi (più i più lenti — `test_runner.py` da solo passa i 10 minuti — uno per volta) e
`tests/smoke/` a parte. Ogni invocazione esce **0**. `ruff check` e `ruff format --check` puliti
su 190 file.

I **quattro in più** rispetto ai 1304 di ADR-052/053 sono di ADR-054: due guardie nuove in
`test_loader_workers.py` (`test_la_costante_di_progetto_tiene_la_cache_ram_spenta` e
`test_a_budget_zero_la_cache_non_si_accende_nemmeno`, quest'ultimo parametrizzato su tre valori
di `num_workers`).

**Verifica precedente: 1 settembre 2026, dopo ADR-052/053 (v1.56).** **1304 test raccolti,
giro completo verde** in un processo solo. I **120 in più**
rispetto ai 1184 del 31 agosto sono della passata di riuso e del riuso degli score:
`test_selezione_modello.py` (fra cui **quello che conta**: gli score riusati per τ devono
essere quelli del **passo selezionato**, con la sua contro-prova), `test_contratto_docs03.py`
(21), `test_e0d_throughput.py` ampliato con gli assi `cache_mb` e `cache_benefit`, più le
guardie nuove in `test_configs.py` (`verifica_campi_noti` su **ogni** file di `configs/train/`
e `configs/augment/`), `test_chunks.py` e `test_feature_jobs.py`.

`ruff check` **e** `ruff format --check` sono **puliti** su 190 file. ✅ Il debito di forma dei
7 file di ADR-052/053 — `scripts/e0d_throughput.py`, `scripts/replay_store.py`,
`src/evd/annotations/schema.py`, `src/evd/eval/metrics.py`, `src/evd/eval/report.py`,
`src/evd/exp/sources.py`, `tests/unit/test_chunks.py` — è stato chiuso con
`python -m ruff format .` prima del commit, com'è richiesto dalla definizione di fatto
(`CLAUDE.md` §5). Erano differenze di sola forma: righe vuote fra `def` di primo livello e una
chiamata che stava su una riga.

**Verifica precedente: 31 agosto 2026, ADR-047 (v1.50).** **1184 test raccolti, 1184
verdi**, giro completo (`slow` e smoke inclusi, `-o addopts=`); `ruff check` e
`ruff format --check` puliti su 181 file; smoke end-to-end su CPU **verde** (chunk AUC 1,000).
I settantatré in più rispetto ai 1111 della verifica precedente sono tutti del protocollo
del laboratorio: **10** in `test_griglia_lab.py` (la geometria a 30 fps letta dal documento,
la coda troncata che chiude il video, il video più corto di una finestra che ora ne produce
una, i campioni che **coprono** la finestra, la frequenza effettiva che **non** è quella
nominale, e i frame ripetuti sulla coda di validazione), **8** in `test_metriche_lab.py`
(Youden contro il quantile sullo **stesso** campione, la restrizione ai video anomali, e le
due righe **asimmetriche** della decisione per video — quella su un video anomalo ha la sua
contro-prova dentro, perché la regola simmetrica lo dichiarerebbe rilevato), **6** in
`test_selezione_modello.py` — fra cui **quello che conta**: due fit identici che differiscono
per il solo `model_selection` devono restituire **pesi diversi**, altrimenti lo snapshot non
viene ripristinato — e i restanti distribuiti fra `test_configs.py` (la matrice gira su
`edp_v2`, il budget è un'epoca, la selezione non è early stop), `test_augment.py` (la catena
`photometric` aggiunta alla parametrizzazione che c'era già, che è la lezione del 29 agosto
applicata prima di pagarla) e `test_runner.py`, dove il test sull'**orchestratore** ha trovato
che `_budget` non leggeva `model_selection`.

**Verifica precedente: 29 agosto 2026, ADR-038 (v1.44).** **1111 test raccolti, 1111
verdi**, giro completo (`slow` e smoke inclusi, 1162,45 s, `-o addopts=`); `ruff check` e
`ruff format --check` puliti su 176 file. I ventisei in più rispetto ai 1085 della verifica
precedente sono tutti della correzione del loader: **10** in `test_loader_workers.py`
(`budget.num_workers` arriva al `DataLoader` invece dello zero cablato; la cache RAM che si
**divide** fra i worker invece di moltiplicarsi; il campo che entra nel `config_sha`, perché
cambia i pixel), **11** in `test_lr_trials.py` (la ricevuta delle prove finite, il rifiuto di
una ricevuta di un'altra cella su ognuno dei quattro campi d'identità, il file troncato che non
ferma il run, e **il test che conta**: con la griglia intera in ricevuta `_seleziona_lr` non
tocca nemmeno il `Trainer`, perché `costruisci` solleva apposta), **4** in
`test_e0d_throughput.py` per l'asse `augment` — fra cui quello che verifica che la catena
**arrivi al loader**, non che finisca nei metadati — e **1** in `test_augment.py`: la riga
`official_nocrop` che mancava alla parametrizzazione, cioè l'unica catena che i run montano.
Quest'ultimo è stato scritto con la sua **contro-prova**: senza `crop=bool(gruppo.get("crop"))`
in `_augment` fallisce, con essa passa.

**Verifica precedente: 28 agosto 2026 sera, E0i pronta (v1.43).** **1085 test raccolti,
1085 verdi**, giro completo (`slow` e smoke inclusi, 1238,09 s, `-o addopts=`); `ruff check` e
`ruff format --check` puliti su tutto il repository (173 file). I test in più rispetto alla
verifica precedente sono: **18** in `test_aggrega_lr.py` (il criterio di E0i, scritto e testato
prima che la campagna parta — mediana e non media, la guardia di bordo, la cardinalità della
griglia che deve coincidere fra famiglie), **4** in `test_runner.py` per `select_only` — fra cui
quello che **conta i fit eseguiti**, perché è lì che sta il comportamento e non nei metadati — e
**9** in `test_configs.py` per le due campagne di E0i, fra cui `eval_every == max_steps` (la riga
che separa 45 ore da 1.100) e la sentinella che verifica che nessun esperimento della matrice
abbia `select_only` acceso.

**Verifica precedente: 28 agosto 2026, la migrazione del runner (v1.41).** **1007 test
raccolti, 1007 verdi**, giro completo (`slow` e smoke inclusi, 710,13 s, `-o addopts=`);
`ruff check` e `ruff format --check` puliti su tutto il repository; smoke end-to-end su CPU
**verde**. I trentacinque test in più rispetto ai 972 della sessione precedente sono: dieci in
`tests/unit/test_campionamento.py` (il peso che **sposta davvero** la frequenza di
campionamento, con la contro-prova senza `weighted`; il rifiuto di un peso nullo; la cella di
hop centrata e tagliata sui bordi; il dinamico che **ri-etichetta**; `chunk_at` identico alla
griglia statica su tutta la griglia; il dinamico che esclude la cache), tre in `test_trainer.py`
per l'obiettivo binario — fra cui quello che smaschera la confusione fra `label` e `class_label`
mettendoli **diversi** nello stesso batch — quattro in `test_selection.py` per
`regime_fully_supervised`, più gli adeguamenti di `test_runner.py` (split al posto del manifest,
`_n_classes` col protocollo binario, i tre campi di protocollo che toccano il calcolo) e di
`tests/smoke/` (la firma della griglia in `metrics.json` **e** in `meta.json`).

**Verifica precedente: 27 agosto 2026, il cambio di protocollo (v1.39).** **972 test
raccolti, 972 verdi**, giro completo (`slow` e smoke inclusi); `ruff check` pulito su tutto il
repository; smoke end-to-end su CPU **verde**. I trentatré test in più rispetto ai 939 della
sessione precedente sono: `test_split.py` e `test_chunks.py` (il protocollo nuovo — sigillo,
disgiunzione, positività per intersezione, i due pesi col pavimento che non può essere zero, il
tasso di positivi **misurato**), quattro in `test_augment.py` per il campo `crop` (**fra cui la
contro-prova che ha bocciato la prescrizione di ADR-032 §3**), quattro in `test_metrics.py`
(verticalizzazione per classe, latenza di rilevazione, e il test che distingue l'intersezione
qualunque dalla regola dell'ultimo frame), più gli adeguamenti di `test_run_id.py` alla
grammatica a quattro campi. ⚠️ Il conteggio va riletto senza un secondo `-q` (vedi la nota in
tabella): `-o addopts=` è il modo affidabile.

**Verifica precedente: 27 agosto 2026, protocollo fully-supervised (v1.39).** 939 test al momento della riscrittura; il giro verde da 972 sopra è la misura successiva, dopo ADR-034.

**Verifica precedente: 26 agosto 2026, E0h (v1.38).** **939 test raccolti, 939 verdi**, giro completo (`slow` e smoke inclusi, exit 0, nessun marcatore `F`/`E`); `ruff check` pulito sui file toccati. Smoke end-to-end su CPU **verde** (chunk AUC 0,850, hit rate store 1,0). `scripts/campagna.py valida` risolve le 80 celle di `e0h_wd_schedule` e, invariate, le 5 di `e2b_multilevel` e le 20 di `e2b_layers`. I sei test nuovi sono in `test_utils.py` (validazione di `BudgetSpec.lr_schedule`, `lr_al_passo` costante/coseno/schedule ignoto), `test_trainer.py` (il log per-passo riflette l'LR applicato, non il picco) e `test_checkpoint.py` (`test_fit_ripreso_con_schedule_cosine_riproduce_il_fit_intero`, il test "spezzato = intero" della correzione alla ripresa).

**Verifica precedente: 26 agosto 2026, E2b-ML (v1.37).** 933 test raccolti, 933 verdi, giro completo; `ruff check` pulito; `mypy` pulito su `types.py` ed `eval/scorers.py`. I ventisette test nuovi allora erano in `tests/unit/test_multilevel.py` più gli adeguamenti dei test esistenti a `cache_keys`/`levels`; quello che conta di più è la non-regressione: `levels=(-1,)` dà lo stesso risultato, bit a bit, del vecchio `layer=-1`.

**Verifica precedente: 25 agosto 2026, augmentation (v1.34).** 906 test raccolti, 906 verdi, giro completo; `ruff check` e `ruff format --check` puliti. I trenta test nuovi allora erano di `tests/unit/test_augment.py`, e fra loro c'è quello che conta di più: la nostra catena confrontata con la classe `VideoTransform` **importata dal clone di Meta**, identiche a `rtol = atol = 0`.

| Cosa | Valore |
|---|---|
| Fasi software chiuse | M0 · F2 · F3; F4 software e F5 chiuse. **F6 completa** (D1–D9). **F11 completa** (28 agosto): il runner esegue `edp_v1` — split sigillato, tutti i chunk, campionamento pesato, uscita binaria, validazione per LR/early stop/soglia, valutazione sul test. **Lo `split.json` di UCF-Crime esiste dal 28 agosto** (ADR-035, quota di validazione al 20%), quindi il blocco software è tolto: quello che manca a E2/E3 sono i **costi** (E0b.3) e le **feature**, non il codice. Le sei campagne validate il 26 agosto erano del regime few-shot e vanno rifatte |
| Milestone | M2 **non chiusa**: E0a, E0b.1, E0b.2, E0c, fedeltà E0d ed E0e sono completi; **E0f v2 chiuso sul piano delle run** (25 agosto: quattro celle a 7/7 semi, aggregazione in sensitivity analysis in `runs/…/ricalcolo-938/aggrega-geometria-938.json`, **geometria congelata al ripiego `256/eval_center/fps 7,5`**, ratifica della lettura primaria ai supervisori — Q26 di `docs/12` v1.3); **store E0d query-side completo e verificato contro il protocollo** (290/290 video per layer, **147.548 feature, copertura 1,000000** della griglia di `fsp_v1`; ~650 GiB allocati). Il protocollo è congelato (ADR-026) e i 35 manifest esistono: M2 resta aperta solo sulla gamba di *throughput* di E0d |
| Test | **1111 raccolti, 1111 verdi** (giro completo, 1162,45 s, 29 agosto, dopo ADR-038: 26 in più per il loader coi worker, la ricevuta delle prove e il campo `crop` che `_augment` perdeva). Prima: **1007 raccolti, 1007 verdi** (giro completo, `slow` e smoke inclusi; 710,13 s, misurati il 28 agosto con `-o addopts=` dopo F11). Trentacinque in più rispetto ai 972 della verifica precedente, tutti della migrazione del runner: campionamento pesato e dinamico, obiettivo binario, regime di selezione sulla validazione. Prima di questi, trentatré in più rispetto ai 939, tutti della riscrittura del protocollo che li ha preceduti — **ADR-034 non ne ha aggiunto nessuno**: ha cambiato due valori e adeguato i due test di geometria di `test_configs.py`. Sei in più rispetto ai 933 della sessione precedente, per E0h (ADR-030): validazione di `BudgetSpec.lr_schedule` e di `lr_al_passo` (`test_utils.py`), il log per-passo che riflette l'LR applicato invece del picco (`test_trainer.py`), e il test "spezzato = intero" della ripresa con schedule attivo (`test_checkpoint.py`). Prima di questi, ventisette in più rispetto ai 906 della sessione ancora precedente, per E2b-ML (ADR-029): `tests/unit/test_multilevel.py` — `concat_levels`, `encode_multilevel` uguale al `torch.cat` manuale e identico a `encode()` con un solo livello, il dataset che impila i livelli nell'ordine dichiarato con un miss tutto-o-niente, il collate che rifiuta `levels` misti, il sentinello `hierarchical`, e **il test che conta**: `levels=(-1,)` bit-a-bit identico al vecchio `layer=-1` — più gli adeguamenti a `cache_keys`/`levels` in `test_heads.py`, `test_cache_valutazione.py`, `test_dataset_feature_cache.py`, `test_augment.py`, `test_minifit.py`. ⚠️ `pyproject.toml` ha già `addopts = "-q --strict-markers"`: un secondo `-q` da riga di comando dà `-qq` e **sopprime la riga di riepilogo**, così un giro senza fallimenti sembra identico a uno che non ha raccolto niente — costato due diagnosi sbagliate il 25 agosto, una il 26 (E2b-ML) e una terza lo stesso giorno (E0h): il conteggio va riletto con `--collect-only -q -o addopts=` o verificando che non compaia nessun marcatore `F`/`E` nell'output, non la riga di riepilogo |
| `ruff` | pulito (`check` e `format --check`: il 22 agosto 26 file sono stati riportati alla formattazione di ruff 0.16) |
| `mypy` | pulito su `evd/protocol`, `evd/eval`, `data/windows.py` e `types.py`; fuori da queste aree gli errori noti sono invariati rispetto a `HEAD` |
| Smoke | guardia chunk-AUC **≥ 0,70 superata su CPU**: **1,000 · 0,984 · 1,000** sui semi 0/1/2 del mondo, rimisurata il 28 agosto col budget giocattolo nuovo (`configs/train/probe_toy.yaml`: 20 passi, batch 8, lr 3e-2). ⚠️ **Col budget precedente lo smoke usciva sotto il caso** — 0,225 · 0,328 · 0,367 — e non per un difetto della pipeline: era una guardia tarata su una testa a 4 vie in cross-entropy, applicata a un logit con BCE su 91 chunk (§5.0, punto *c*). **Il seme del mondo resta parte della guardia.** Frame-AUC fuori protocollo dal 27 agosto; replay feature Tiny con hit rate **100%**; la fetta attraversa anche l'adapter (LoRA r=2, VPT n=2) |
| E0a locale meta-device | ViT-L: 302 tensori, 304.680.960 parametri, output `[1,4,1024]`, grid `(1,2,2)` |
| Strict load locale reale | ViT-L: 302 tensori, zero mismatch, materializzazione CUDA in 10,59 s e 1,135 GiB allocati (diagnostica, non numero di tesi) |
| E0a cluster | **completato**, job 532132 su A100 in 9 s, commit pulito `f35934a`: strict load di 302 tensori, output `[1,4608,1024]`, grid `[8,24,24]`, valori finiti; artefatti e registry rientrati localmente. I timeout diagnostici 532120/532131 hanno portato ai fix meta-load e PyAV `SLICE` |
| E0b.1 cluster | **completato**, job 532148 su A100 in 1:18:10, commit pulito `e11c0c0`: 96/96 celle valide, nessun OOM fino a batch 32. Massimi `final`: 79,95 finestre/s a 224 e 23,47 a 384 con AMP BF16; quattro layer aggiungono lo 0,31% mediano di latenza. Artefatti e log rientrati localmente |
| F5/E0b.2-scale | **completati e rientrati**, commit pulito `6c5d4cd`: mini-fit B/g/G job 533123–533125; E0b.2 B/g/G job 533126/533127/533130 con 56/55/48 celle valide su 56. A 384px G è OOM a batch 8 per tutte le varianti e DoRA già a batch 4. Diagnostica prompt g in FP32/TF32: job 533132 |

Le due caselle che questo paragrafo indicava — la partizione di validazione (Q-C) e la rimisura
dei costi (E0b.3) — **sono chiuse** dal 28 agosto: lo split `edp_v1` è sigillato (1278/312/290,
ADR-035) e il budget di E2/E3 è fissato (3000 passi a batch effettivo 16, ADR-036).

**E0i ha girato**, 12 celle su 12 il 29 agosto, dopo che la prima campagna (28 agosto, job
`539643`) era morta per intero per il difetto del loader corretto in **ADR-038**. Ma **non
produce i quattro numeri**: tutte e quattro le famiglie cadono su `1e-4`, il bordo basso della
griglia, e la guardia scritta prima dei numeri dice di non adottarlo (`docs/10`, voce E0i).

Quindi `train.lr_grid` resta `???` in `configs/train/probe_edp.yaml` e in
`configs/train/peft_budget_a.yaml`, e Hydra continua a fermare ogni cella di E2 e di E3
nominandolo: è ancora l'unico `???` fra qui e la matrice.

**La seconda campagna è partita il 29 agosto: `e0i2`**, banda `3e-5 · 5e-5 · 1e-4 · 2e-4`
decisa da Antonio, con `1e-4` **interno** — sul bordo la guardia sarebbe scattata di nuovo
sul valore che la prima campagna aveva già scelto quattro volte su quattro. Dodici celle in
due ondate (9 PEFT + 3 congelate: il tetto dell'account è 10 job in coda), ~29 h di H100 sui
tempi misurati. Finché non rientra, il `???` resta dov'è.

> **✅ Chiuso — e i due paragrafi qui sopra sono cronaca.** `e0i2` è rientrata il
> 29 agosto e **ADR-039** ha fissato `1e-4` **per tutte e quattro** le famiglie,
> come deroga dichiarata: VPT lo sceglie all'**interno** della banda, le altre tre
> cadono su `2e-4` **al bordo**, e la ragione per non adottare quel bordo è
> scritta nel criterio, non dedotta dopo. Quindi `train.lr_grid` **non è più un
> `???`** e non è più l'ultimo cancello fra qui e la matrice: i budget in vigore
> sono `train/edp_v2.yaml` e `train/peft_budget_v2.yaml` (ADR-047), e la matrice
> ha già eseguito le sue prime due celle (E2, 1 settembre). `probe_edp.yaml` e
> `peft_budget_a.yaml` restano su disco perché E0h ed E0i li hanno **consumati**,
> e un file di budget già consumato non si riscrive.


### Aggiornamento 30 settembre 2026: risultati completi E3-SV

Otto celle E3-SV complete e riportate in locale, 16 test EDP UCF/MIVIA e
16 rivalutazioni OVAD g30 verificati. Classifiche a 38 modelli / 98 celle EDP
e 176 risultati OVAD; documenti aggiornati, E3-SV esclusa da Excel per decisione
di Antonio. Ricevuta finale del cluster verificata: campagna chiusa.
Nessuna modifica alla struttura della codebase per questa passata
e nessuna suite completa nuova; 64 test mirati e 16 originali OVAD passati.

### Aggiornamento 26 settembre 2026: E3-SV

Nessun nuovo modulo Python di produzione: `exp/preadapt.py` legge il campo
`checkpoint_state` (`last` storico, `selected` esplicito senza fallback).
`run_id.py` registra `e3sv`; sedici nuovi YAML ereditano gli otto esperimenti
E3-S e definiscono le otto sweep al solo seed 42. `classifiche_modelli.py`
distingue E3-SV da E3-S. Test sui pesi effettivi in entrambi i modi e sui config.


### Estensione della localizzazione: presentazione completa e MIVIA

`vtcd_spatial_present.py` genera nove categorie di tabelle filtrabili e mantiene tutte le colonne, senza riscrivere dati scientifici; vista iniziale `boxes`, distinta da `mask` e `largest`. Il preparatore accetta `--dataset mivia_crime --external --reviewed-videos-only`: esporta i 39 video interamente confermati senza fold MIVIA. Snapshot/audit in `tmp/vtcd-spatial-extension/gt-mivia/`. Tredici test mirati passati, incluso il controllo sugli ID/duplicati dei box.

Conteggi statici: 129 file Python in src/evd, 22 apps, 82 scripts, 112 tests. La suite completa non e' rieseguita. Le alternative di formazione bbox sono solo proposte: [discussione](varie/estensione_vtcd_box_mivia_discussione.md). MIVIA non ha ancora metriche spaziali della nuova campagna.
