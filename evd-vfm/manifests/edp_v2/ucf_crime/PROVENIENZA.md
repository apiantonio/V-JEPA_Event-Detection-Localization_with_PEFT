# `edp_v2` / UCF-Crime — provenienza dello split

**Lo split non è nostro.** È la partizione consegnata dal laboratorio MIVIA il
**31 agosto 2026** insieme al documento di protocollo
(`docs/varie/protocollo-laboratorio_FS-UCF-Crime.md`). Chiude Q-C dalla fonte
giusta e supera ADR-035, che descriveva un ripiego — il 20% del training
ritagliato da noi, per video e stratificato per classe — e lo dichiarava tale.

| file | cos'è |
|---|---|
| `split-FS-UCF-Crime.csv` | la consegna, verbatim: 1899 righe con `video_id, split, label, total_frames, fps, durata_s, finestre_valutazione` |
| `split.json` | la stessa partizione nel formato sigillato del repository, prodotta da `scripts/importa_split.py` |

## Composizione

| partizione | video | anomali | normali | finestre di valutazione |
|---|---:|---:|---:|---:|
| train | 1303 | 668 | 635 | 88.371 (hop 4 s) |
| val | 306 | 141 | 165 | **8383** (hop 8 s, contigue) |
| test | 290 | 140 | 150 | **9116** (hop 4 s) |

Le due colonne in grassetto sono quelle che il documento di protocollo dichiara
(§1.3 e §3.1) e che `importa_split.py` verifica **video per video**, non solo in
somma. Il conteggio del training è nostro e non è nel CSV: la griglia di training
usa lo stesso hop del test.

`Burglary019_x264` è l'unico dei 1900 video canonici che il laboratorio ha
lasciato fuori. Non compare in nessuna partizione, e nessun numero lo riguarda.

## Come si rigenera e come si verifica

```
python scripts/importa_split.py \
    --protocol configs/protocol/edp_v2.yaml \
    --csv manifests/edp_v2/ucf_crime/split-FS-UCF-Crime.csv \
    --verifica
```

⚠️ Questo file **non si rigenera da una regola**, a differenza di
`manifests/edp_v1/ucf_crime/split.json`: la partizione arriva da fuori, e non
esiste nessun criterio nostro che la riproduca. Al posto della rigenerazione ci
sono cinque verifiche, tutte passate al momento della scrittura:

1. **copertura** — 1899 video su 1900 canonici stanno nell'inventario F2;
2. **geometria** — `total_frames` e `fps` del CSV coincidono con quelli misurati
   dal nostro probe su tutti e 1899 i video, senza una sola eccezione;
3. **etichette** — `label` (anomalous/normal) coincide con le annotazioni
   FS-UCF-Crime su tutti e 1899 i video;
4. **griglia** — la nostra implementazione del §1 del protocollo riproduce
   `finestre_valutazione` **per ogni video** delle due partizioni valutate:
   8383 su validazione e 9116 su test, zero scarti;
5. **duplicati** — contati e dichiarati, vedi sotto.

La verifica 4 è quella che vale di più: dice che leggiamo il loro documento come
lo intendono loro. È anche il controllo che allo store E0d è mancato — un
artefatto derivato che si disallinea in silenzio dalla regola che lo genera è già
costato 648 GB (`CLAUDE.md` §4).

## ⚠️ Duplicati byte-identici che attraversano le partizioni

UCF-Crime contiene 21 coppie di file con lo stesso sha256. **Dieci di queste
coppie attraversano le partizioni di questo split**, e otto toccano il test:

| | |
|---|---|
| `Normal_Videos592_x264` [train] | `Normal_Videos_203_x264` [test] |
| `Normal_Videos267_x264` [train] | `Normal_Videos_603_x264` [test] |
| `Normal_Videos363_x264` [train] | `Normal_Videos_621_x264` [test] |
| `Normal_Videos447_x264` [train] | `Normal_Videos_634_x264` [test] |
| `Normal_Videos654_x264` [train] | `Normal_Videos_894_x264` [test] |
| `Normal_Videos764_x264` [train] | `Normal_Videos_641_x264` [test] |
| `Normal_Videos221_x264` [val] | `Normal_Videos_597_x264` [test] |
| `Normal_Videos594_x264` [val] | `Normal_Videos_217_x264` [test] |
| `Normal_Videos349_x264` [val] | `Normal_Videos616_x264` [train] |
| `Normal_Videos380_x264` [train] | `Normal_Videos625_x264` [val] |

È **leakage**, ed è reale: sei video di training sono copie esatte, byte per
byte, di sei video di test. `edp_v1` li escludeva (`dedup:
sha256_representative`, 20 video fuori); `edp_v2` **non li esclude**, e la
ragione è una sola: questa è la partizione su cui il laboratorio riporta i suoi
numeri, e i conteggi 9116 e 8383 si riproducono esattamente solo prendendola
verbatim. Escludere i due video di validazione, per esempio, porterebbe la
griglia di validazione sotto le 8383 finestre e romperebbe il confronto.

Cosa va scritto in tesi, e non altrove:

- tutti e dieci i gruppi sono video **`Normal`**, quindi il leakage agisce sui
  negativi e non sugli eventi;
- i sei video di training duplicati nel test sono lo **0,46%** del training set
  (6 su 1303) e il **2,1%** dei video normali di test (6 su 150);
- l'effetto atteso è di **gonfiare** le metriche, quindi il limite è nella
  direzione conservativa per ogni claim del tipo «il PEFT non basta»;
- la scelta è dichiarata in **ADR-047** ed è di Antonio, il 31 agosto 2026.

## Storia

`manifests/edp_v1/ucf_crime/split.json` (1278 train · 312 val · 290 test, 20
video esclusi dal dedup) resta su disco immutato: descrive la partizione del
regime precedente. Non si cancella, non si riscrive, non si riusa — invariante 1.
Il suo test set è **identico** a questo, il che è una coincidenza utile ma non
una garanzia: i due file si distinguono per il resto.
