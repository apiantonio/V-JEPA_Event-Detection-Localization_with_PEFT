---
titolo: "Registro degli esperimenti"
versione: "1.62"
data: 2026-09-30
---

# Registro degli esperimenti

> Questo è il **gemello leggibile di `runs/registry.jsonl`**

## Regole

1. **Una voce per blocco sperimentale** (E0b, E2a, E3-R.1, …), non per run: i run stanno nel
   registry, che è generato. Qui ci va ciò che un file JSON non può contenere.
2. **Entro 48 ore** dalla chiusura del blocco (regola di `06_roadmap_milestones.md` §M5),
   mentre si ricorda ancora perché si era fatto così.
3. **Ogni numero citato porta il suo `run_id`.** Un numero senza run_id non entra qui e non
   entra in tesi (invariante 2 e 7).
4. **Le decisioni prese guardando i risultati** (il layer dopo E2b, il gate IVN dopo E3-R.1,
   `peft_budget_a` dopo E0b) si scrivono qui come **mini-ADR**, con un rimando da `07_adr.md`
   se cambiano un contratto.
5. **Anche i risultati negativi e le sorprese.** Un esperimento che non ha funzionato, se si
   capisce perché, è una sezione del capitolo sui limiti — cioè una tesi più forte, non più
   debole. Un numero che non torna e non si spiega si marca ⚠️ e resta marcato.
6. **Mai correggere un numero.** Se un run è sbagliato si butta e si rilancia, e qui si scrive
   che è successo (R12).
7. **Il blocco non è chiuso finché i risultati non sono tornati a casa.** `runs/` è gitignored
   (ADR-009, confermata da ADR-020): finché i file stanno solo sul cluster, il numero vive su
   una macchina il cui accesso scade nell'agosto 2027. Prima di scrivere la voce:
   `.\hpc\sync.ps1 -Azione pull-run`, che porta registry e artefatti di testo e **fonde** il
   registry invece di sovrascriverlo.

## Indice

**27–28 settembre: E-OVAD**, la rivalutazione degli score già su disco col codice
OVAD consegnato dal laboratorio, eseguito senza modificarlo (ADR-076,
[voce](#e-ovad-20260927), sintesi e limiti in [`docs/19`](19_protocollo_ovad.md)).
⚠️ Righe, soglia, livello video ed eventi hanno regole diverse da `edp_v2`: i numeri
stanno in una raccolta separata e **non** sostituiscono quelli della Tavola.

**30 settembre: E3-SV chiusa a 8/8 celle**, riportate in locale con 16 test EDP e 16 valutazioni OVAD g30 verificati; ricevuta finale del cluster `complete`. **26 settembre: E3-SV avviata** (ADR-075, [voce](#e3sv-20260926)): le otto celle di
E3-S al seme 42 col prior ereditato dal checkpoint **selezionato** su validazione
invece che dall'ultimo passo. Nessun risultato ancora.

**24 settembre: E3-S CHIUSA**, 24/24 celle sui tre semi, verificate: in fila le due
famiglie non guadagnano più che addestrate insieme, e nessuna configurazione batte
il proprio reparam da solo. [Numeri, verifiche e limiti](#e3s-20260924).
⚠️ **Le valutazioni MIVIA-Crime delle quattro celle `fondi` non valgono** (la
valutazione esterna non rifonde il reparam nel backbone, `docs/09` R24).
Nella stessa giornata hanno una voce anche il **confronto con lo stato dell'arte**
([E4](#e4-sota-20260924)) e la **campagna di explainability**
([F14](#f14-explain-20260924), con i metodi per esteso in
[`docs/17`](17_explainability_per_la_tesi.md)).

**25 settembre: R24 è corretto** (`docs/09`). Le valutazioni MIVIA-Crime delle celle `fondi` di E3-S sono rifatte col motore che rifonde il reparam, e il «crollo» di −0,12…−0,13 **non esiste**: perdono come tutte le altre. Le letture corrette stanno in [E3-S](#e3s-20260924) e in testa a [E-MIVIA.2](#e-mivia2-20260922).

**24 settembre, sera: i due branch che portavano voci di registro sono fusi**, e le
voci ora stanno qui. [**E-MIVIA.2**](#e-mivia2-20260922) — le 36 celle a seme 42 su
MIVIA-Crime, 22 settembre, anche in `runs/classifiche/classifiche.md` §6 e
`runs/classifiche/cross_dataset_mivia.csv` — si legge **senza** le quattro righe
`fondi` (R24, nota in testa alla voce). [**E5b.2**](#e5b2-20260904) — la
composizione del costo di sistema, con la tabella **D-bis** e **ADR-059** — compone
la latenza con la regola **storica**: la conclusione (il ritardo è quasi tutto griglia)
regge anche con quella di ADR-061, i numeri puntuali no (nota in testa alla voce).
Il punto d'ingresso per chi scrive la tesi è [`docs/18`](18_fonti_per_la_stesura.md).

**9 settembre: ablation DoRA del rango CONCLUSA**, 12/12 celle sui tre semi.
R32 ha le medie migliori (AUC 0,9422 ± 0,0017; AP 0,7947 ± 0,0037);
r64 peggiora in tutti i semi. [Risultati, verifiche e recupero locale](varie/e3-dora-risultati-20260909.md).
La ricevuta di avvio dell'8 settembre sotto resta storica.

**8 settembre, DoRA rango: AVVIATA, non conclusa.** Nove nuove celle
r8/r16/r64 sui semi 42/67/104 in esecuzione; tre r32 riusate dopo
verifica del checkpoint. [Job, risorse e supervisore](varie/e3-dora-rank-20260908.md).

**15 settembre: E3-M ed E3-C CHIUSE**, 30/30 celle sui tre semi. Né le tre
epoche né le combinazioni spostano il verdetto di E3: **DoRA r=32 a una epoca
resta prima** su chunk-AUC e chunk-AP fra tutti e 28 i modelli.
[Numeri, verifiche e artefatti](#e3m-e3c-20260915).

**Ultimo completamento:** [E3-S](#e3s-20260924).
Prima: [E3-M ed E3-C](#e3m-e3c-20260915), [E3-R.2/E3-P.2 su tre semi](#e3-ablation-20260907).

**Aggiornamento del 5 settembre:** [complementarità, FAR e profili schema 2](#analisi-20260905).
Le tabelle complete e le indicazioni per la stesura sono nel
[dossier 14](14_analisi_complementari_stesura.md). Questa voce aggiorna le
limitazioni hardware delle voci storiche E5b, senza sostituirne le misure.

| Blocco | Data | Esito in una riga | Voce |
|---|---|---|---|
| **E3-S** ⭐ | **24 set 2026** (celle 15–22 set) | **chiusa, 24/24 celle sui tre semi: in fila come insieme, e la sequenza non guadagna sul reparam da solo.** Contro il **reparam da solo a una epoca** tutte e otto le configurazioni hanno delta medio di chunk-AUC **negativo** (−0,0009 … −0,0101), nessuno che separi; contro la **combinazione joint** E3-C, a parità di budget e di componenti, quasi mai si separano (2 righe su 32). Quindi la ridondanza dei due prior **non è un effetto dell'addestrarli insieme**. La migliore, VPT-shallow n=8 sopra DoRA r=32 fuso, sta a **0,9397 ± 0,0080** contro 0,9422 ± 0,0017 del DoRA da solo — e il suo seme 42 (**0,9477**) è la cella singola più alta di tutte le 108, cioè un massimo di seme e non una proprietà del modello. ⚠️ Prior ereditato all'**ultimo** passo, non al selezionato. ⚠️ LR della matrice (`1e-4`), non quella «più bassa» indicata dal prof il 7 settembre: da dichiarare. ⚠️ **Le quattro valutazioni MIVIA `fondi` non valgono** (R24) | [E3-S](#e3s-20260924) |
| **E4** ⭐ | **17–24 set 2026** | **il confronto con lo stato dell'arte, sui numeri del laboratorio e con lo stesso protocollo a 8 s.** DoRA r=32 s42: chunk-AUC **94,41** su UCF-Crime contro **85,52** (VadCLIP) e **84,19** (metodo del laboratorio); **90,29** su MIVIA-Crime contro **81,69** (laboratorio) e **79,90** (VadCLIP), e AP_a **83,00/88,13** contro 60,66/73,05. ⚠️ Su MIVIA-Crime, alla soglia **trasferita**, il nostro sistema **sotto-allarma**: recall di chunk 64,79 contro 74,00 e recall di evento 75,61 contro 80,80–82,11. ⚠️ I numeri degli altri metodi sono del laboratorio, non riprodotti, e la loro supervisione non è dichiarata | [E4](#e4-sota-20260924) |
| **F14** | **10–24 set 2026** | **campagne eseguite, lettura quantitativa dei metodi NON fatta.** Due modelli (DoRA r=32 s42, congelato s42), tre popolazioni (24 finestre di sviluppo · **279** video anomali interi · **46** video della selezione), fino a **31** varianti per finestra, ~**159 h** di GPU per la sola campagna del 15 settembre. Controlli di fedeltà **a zero** (hook, ripetizione del detector, ricostruzione delle occlusioni). Il risultato quantitativo più solido è sulle **ablazioni temporali**: congelare il tempo abbatte il logit delle finestre positive di **~4** su entrambi i modelli e ne cambia la decisione nel **44–48%**, invertire il verso quasi non lo sposta. ⚠️ Deletion/insertion **calcolate e non lette**; nessun metodo raccomandato; bounding box **proposti e non validati** | [F14](#f14-explain-20260924) |
| **E-MIVIA.2** | **22 set 2026**, corretta il 25 | **tutti i 36 modelli a seme 42 su MIVIA-Crime: i valori si trasferiscono, l'ordine solo in parte.** Spearman **+0,638** e Pearson **+0,964** fra chunk-AUC in casa e fuori, con una tassa di dominio quasi costante (delta mediano **−0,039**, σ 0,0067, da −0,054 a −0,027) che vale **anche per le quattro `fondi`** una volta corretta la valutazione (R24, 25 settembre: il loro «crollo» di −0,12…−0,13 era un artefatto). Il primo su UCF-Crime è 14° su MIVIA-Crime | [E-MIVIA.2](#e-mivia2-20260922) |
| **E3-M / E3-C** ⭐ | **15 set 2026** | **chiuse, 30/30 celle sui tre semi: né il budget né la composizione spostano il verdetto di E3.** **Tre epoche** (E3-M, +2× di calcolo) non comprano niente tranne sul modello che era sotto-addestrato: la **probe lineare** guadagna **+0,0102** di AUC e **+0,0384 di AP** con **3/3** semi concordi, tutte le altre restano dove erano e **VPT-shallow n=8 peggiora** (−0,0122 / −0,0204, **0/3**). La conferma è indipendente e sta nel **passo selezionato**: su attentive e VPT-deep cade dentro il **primo** ciclo 3 volte su 3, sulla linear nel **secondo** 3 volte su 3. Le **combinazioni** (E3-C) dicono che i due prior sono **ridondanti, non complementari**: una combinazione si comporta come il suo **reparam leggermente peggiorato** (negativa su **15 righe metriche su 16**) e come il suo prompt **migliorato** (fino a +0,0412 di AP_a) — il reparam porta il risultato, il prompt non aggiunge. ⚠️ **Nessuna combinazione batte DoRA r=32 da sola a UNA epoca** (0,9422 ± 0,0017, prima su AUC e AP fra 28 modelli), e per pareggiarla su AUC_a costa **3× il training** e perde la **fondibilità** (ADR-064: +42,8%/+83,7% di latenza). ⚠️ **Il fatto di metodo che nessuno dei due blocchi cercava**: il divario fra AUROC di validazione e chunk-AUC di test cresce **monotonicamente col budget** — mediana **+0,0030** (1 epoca) → **+0,0058** (3 epoche) → **+0,0155** (3 cicli da zero) — e su **3 celle E3-M su 6** i due segnali sono **discordi**. Il criterio non è rotto (Pearson 0,957) ma perde **risoluzione** dove si decide; ⚠️ il numero è misurato, il **meccanismo no**. ≈ 639 h di H100 per un risultato **negativo**, ed è quello che rende difendibile la scelta | [E3-M / E3-C](#e3m-e3c-20260915) |
| **E-MIVIA** ⭐ | **10 set 2026** | **chiusa: l'ordinamento si trasferisce e perde 4 punti, la calibrazione vale 0,021 di F1.** chunk-AUC **0,9029** contro 0,9441 e AUC_a **0,8293** contro 0,8704 — ⚠️ **la caduta e' la stessa, −0,041 su entrambe**. ⚠️ **AP e precision NON si sottraggono**: MIVIA-Crime ha il **30,65%** di chunk positivi contro il 16,36%, quindi l'AP piu' alta (0,8335) e' pavimento, e il **lift** scende da 4,87x a **2,72x**. Il punto di Youden calcolato sul dataset stesso (`soglia_oracolo`, **non protocollo**) porta la chunk-F1 da 0,7167 a 0,7380: **due punti** sono il costo massimo di non ricalibrare — ma tau trasferita e' **troppo alta** e il sistema **sotto-allarma** (recall 0,648 contro 0,798), ed e' li' che il livello **video** perde (F1 0,865 contro 0,941). ⚠️ **`Shoplifting` e' di nuovo l'ultima classe** (AUC 0,751), che e' la conferma **cross-dataset** di E-strati. Il laboratorio ha consegnato il suo test set privato (270 video, 135 anomali sulle **stesse 13 classi** di UCF-Crime) e la domanda e' **cross-dataset**. ⚠️ **Non e' un run**: pesi **selezionati** di `e3r_..._dora-r32-qkvomlp_s42`, una passata di inferenza, stessa geometria. ⚠️ **tau si TRASFERISCE** — MIVIA-Crime non ha una validazione — quindi AUC/AP misurano l'**ordinamento** e P/R/F1 misurano ordinamento **piu'** calibrazione. Tre fatti della consegna che nessuno si aspettava: **47 risoluzioni** invece di una, **fps da 6 a 60** invece di 30, e ⚠️ **l'header sbaglia `nb_frames` su 53 video su 270** (fino a 1,285x) — che non e' un metadato ma la **geometria**, e l'ha preso il **dimostratore**, non un test | [E-MIVIA](#e-mivia--il-modello-su-un-dataset-che-non-ha-mai-visto-chiusa-lordinamento-si-trasferisce-e-perde-4-punti-di-auc-la-calibrazione-vale-0021-di-f1) |
| **E-strati** ⭐ | **4 set 2026** | **dove il sistema va meglio e peggio, a costo zero di GPU** (ri-analisi delle 15 celle da `scores.jsonl`, ADR-060). **Il bordo della finestra costa 3,8× di AP**: su DoRA i chunk coperti ≤25% stanno a **0,205** contro **0,783** dei >75%, e la forma è **identica su tutti e cinque i metodi** — è una proprietà del protocollo («intersezione qualunque»), non del metodo. **Shoplifting è il collo di bottiglia**: recall di evento **0,575** sull'attentive dove ogni altra classe sta sopra 0,86 — e **VPT-deep prende 0,858** sulla stessa classe, **+0,28**, pur avendo la chunk-AP più bassa. **La durata è monotona con il taglio a 32 s**: sopra, recall **1,000** su tutti i metodi e tutti i semi. ⚠️ **La rarità NON è monotona** — peggiore è la fascia intermedia (5–20%, **0,736**) — e **per metà è Shoplifting**: escludendolo sale di ~0,10 **ma resta la peggiore**. Più un fatto sul **metodo**: le F1 di evento di linear e attentive sono **0,6775** e **0,6778**, indistinguibili, dove la chunk-AP differisce di **0,28** — il livello evento perdona l'imprecisione temporale. ⚠️ Diagnostica, non protocollo | [E-strati](#e-strati--dove-il-sistema-va-meglio-e-peggio-chiuso-il-bordo-della-finestra-costa-38-di-ap-e-il-collo-di-bottiglia-ha-un-nome--shoplifting) |
| **E5b.2** ⭐ | **4 set 2026** | **dal costo del modello a quello del sistema, senza un job nuovo.** Componendo E5b + E0d + la `detection_latency` delle celle E2/E3: il ritardo con cui l'allarme arriva e' **2,57–2,68 s** su tutti e sei i metodi, ed e' per il **93% griglia** e per il **7% calcolo** — un'escursione del 4% dove le latenze differiscono del 64%, quindi **accelerare il modello non accorcia la risposta del sistema**. Real-time con margine ovunque (la peggiore usa il 7% del budget dell'hop) e collo di bottiglia sempre il **modello**: e' il conto di **C5**. **La composizione si verifica** su sei passate di test vere (`ainode02`): ordinamento rispettato, **residuo max 1,29%**, overhead fisso 21,4 ms/finestra, rapporto hardware implicito **1,49x**. ⚠️ **Correzione a E5b**: contando la testa (49,3 M, addestrata in ogni cella) il rapporto fuso/non-fuso su un aggiornamento e' **5,6x** e non 21,5x, e al primo deploy **si inverte**. I profili **schema 2** sono girati lo stesso giorno su A100 e H100 (job 557837, 557836). ⚠️ La latenza composta è quella della regola **storica**; con quella di ADR-061 (`runs/classifiche/classifiche.md` §3) il ritardo è 2,91–3,14 s e la quota di calcolo 5,5–9,0%: la conclusione regge, i numeri sono quelli | [E5b.2](#e5b2-20260904) |
| **E5b** ⭐ | **3 set 2026** | **chiusa in 79 secondi, e il trade-off di C7 non è quello che il PdD descrive.** Senza il merge il **reparam è il più caro dei tre**: LoRA **+42,8%** e DoRA **+83,7%** di latenza contro il congelato, mentre **VPT-deep costa +1,68%**, cioè quasi nulla. «Reparam non costa a inferenza» è vero **solo dopo `merge_()`** — fuso torna a **+0,3%**, sul congelato — e quel merge ha un prezzo che nessuno mette in tabella: il delta smette di esistere come oggetto separato e si spediscono **304,7 M parametri invece di 14,2 M (21,5×)**. Il **modello FLOP sovrastima la penale dei prompt di 6,1×** (+10,23% previsto, **+1,68%** misurato) e ⚠️ a batch 2 l'overhead **scende**, che questo esperimento non spiega. Real-time da **16,6× a 30,4×** sul modello, e componendo con E0d (decoder 25,1 fin/s contro 7,61 del modello) il collo di bottiglia resta il modello: è il conto che sostiene **C5**. **Verdetto congiunto: il reparam fuso domina VPT su ogni asse tranne le ore di training e i byte da spedire** | [E5b](#e5b--lefficienza-a-inferenza-la-metà-mancante-di-c7-chiusa-senza-il-merge-il-reparam-è-il-più-caro-dei-tre) |
| **E3** ⭐ | **3 set 2026** | **le tre famiglie a parità di budget, e la lettura congiunta E3-X.1.** **Reparam ≥ probe congelata > prompt**: DoRA **0,9424** ± 0,0024 e LoRA **0,9400** ± 0,0009 di chunk-AUC (due semi) contro **E2d 0,9385** e **VPT-deep 0,8968** (un seme). **VPT sta SOTTO la probe congelata** — −0,042 di AUC e −0,123 di AP a **3,6×** il costo — e sovradatta: `val_loss` in salita, selezione al 3680, peggiora su **12 classi su 13**. Il guadagno del reparam **non sta nel separare i video** ma **dentro il video anomalo**: +0,0015/+0,0039 di AUC contro **+0,038/+0,041 di AP_a**. **LoRA e DoRA non si separano**: il divario di s42 (+0,0048) non si replica a s67 (+0,00004) ed è pari a una σ, mentre DoRA costa **+30%**. ⚠️ VPT è a un seme, quindi E3-X.1 *fra famiglie* resta a un seme e si dichiara | [E3](#e3--le-tre-famiglie-a-parità-di-budget-e-la-lettura-congiunta-e3-x1-chiusa-reparam--probe-congelata--prompt) |
| **E2** ⭐ | **1 set 2026** | **la prima voce su `edp_v2`, e il primo numero di tesi.** Due celle congelate a seme 42: **E2d attentive 0,9385** di chunk-AUC (AP 0,7737, AUC_a 0,8409, video-F1 0,9225) contro **E2c linear 0,8473** (AP 0,4833, AUC_a 0,6535, video-F1 0,8228). **La linear non regge** — −0,091 di AUC e **−0,290 di AP**: la semantica non è linearmente separabile sul layer 23, che è esattamente il lavoro diagnostico di E2c. Cache di valutazione a **hit rate 1,0**; selezione al passo **4600/5524** con l'AUROC di validazione da 0,9315 a 0,9370, cioè **l'80% del budget per mezzo punto**. ⚠️ Un solo seme: nessuna dispersione | [E2](#e2--la-baseline-congelata-su-edp_v2-chiusa-a-un-seme-la-linear-non-regge) |
| **E0d-cache** | 1 set 2026 | **chiuso, esito negativo: la cache RAM non paga.** Sotto il campionamento vero (job **541451**, pescaggi **con reimmissione** dai pesi reali) il throughput scende **monotonicamente**: 0 MB **3,743 fin/s** → 4000 MB **3,143** (**0,840×**). La causa è nell'artefatto stesso: **2772 indici unici su 3000**, cioè **7,6%** di ripetizione — la cache non colpisce e si paga solo la contabilità. ⚠️ 3000 pescaggi **non sono un'epoca** (88.384): il costo è misurato, il beneficio resta **sottostimato**. Si adotta `CACHE_RAM_MB` **basso o zero** | [E0d-cache](#e0d-cache--quanto-costa-e-quanto-rende-cache_ram_mb-chiuso-non-rende) |
| **E0h-3k** | 31 ago 2026 | **chiuso: vince `cosine`**, adottato in ADR-046. Δ mediana **+0,0212** di chunk-AUC contro `constant`, **3/3** semi, con i tre Δ **6-12× sopra** la dispersione fra semi (0,0023). Due limiti dichiarati: la soglia del criterio era +0,02 ed è passata per **+0,00116**, metà di quella dispersione; e **la conferma su `frozen-attentive` del punto 6 non è stata fatta** (decisione di Antonio). Vince il picco **basso** — `1e-4` batte `2e-4` 3/3 — quindi l'argomento della LR media non regge (informa ADR-039). Il punto 1 è **inapplicabile**: la loss di validazione non esiste nel runner | [E0h-3k](#e0h-3k--constant-o-cosine-deciso-a-3000-passi-chiuso-vince-cosine-con-due-limiti-dichiarati) |
| **E0i** | 28–29 ago 2026 | **prima campagna eseguita, 12/12 celle — e il criterio dice di NON adottare**: tutte e quattro le famiglie cadono su `1e-4`, il **bordo basso**, quindi la guardia scatta e i quattro `???` restano. Da `1e-3` in su 35 celle su 36 sono peggiori di un predittore costante (0,3838): fra `3,16e-4` e `1e-3` c'è un gradino. **Non è lo schedule** — il braccio col coseno lo falsifica. **`e0i2` chiusa 12/12**: VPT sceglie `1e-4` **interno** e si adotta, le altre tre `2e-4` **al bordo**. **ADR-039** fissa `1e-4` per tutte e quattro come deroga dichiarata: i quattro `???` sono chiusi | [E0i](#e0i--il-learning-rate-scelto-una-volta-sola-aperto-criterio-scritto-prima-dei-numeri) |
| **E0d-throughput** | 28 · **29** ago 2026 | **chiuso, in due esecuzioni**: un worker decodifica 2,4 fin/s e la scala è quasi ideale; la seconda misura (job 540018, con l'asse **augmentation** che mancava) dà a 8 worker **25,1 fin/s** senza catena e **20,5 con** `official_nocrop` — cioè **l'augmentation costa 1,23×, non 3,2×**, e il loader non è il collo di bottiglia. A **zero** worker sono 4,68 e 3,63: è il difetto che ha ucciso E0i | [E0d-throughput](#e0d-throughput--quante-cpu-servono-davvero-chiuso-24-finestres-per-worker) |
| **E0b.3** | 28 ago 2026 | **chiuso, su A100 e H100**: alla geometria vera il forward congelato fa **9,33 finestre/s** (16,55 su H100 — il rapporto è **1,7×**, non i 2,7× del microbenchmark, e la VRAM è identica) (l'estrapolazione diceva 7,3) e il passo completo va da **2,83** fin/s di VPT-deep a **1,60** di DoRA, che a parità di batch usa **2,6× la VRAM** di LoRA. Il micro-batch massimo lo fissa DoRA, ed è **2** | [E0b.3](#e0b3--i-costi-alla-geometria-edp-v1-chiuso-9216-token-misurati-non-estrapolati) |
| **E0h** | 26 ago 2026 | **aperta, nessun run**: pilota `weight_decay`/schedule di LR (ADR-030), 80 celle, confronto appaiato contro lo status quo. Criterio scritto prima dei numeri | [E0h](#e0h--weight_decay-e-schedule-di-lr-aperto-criterio-scritto-prima-dei-numeri) |
| **E2b-ML** | 26 ago 2026 | **aperta, nessun run**: braccio multilevel di E2b (ADR-029) — quattro layer concatenati sull'asse token. Criterio e previsione scritti prima dei numeri; su ViT-L ci si aspetta che **non concluda** | [E2b-ML](#e2b-ml--il-braccio-multilevel-aperto-criterio-e-previsione-scritti-prima-dei-numeri) |
| **E0g** | 25–26 ago 2026 | **chiuso, 30/30**: si adotta la catena **ufficiale**, perché la prova contraria chiedeva il danno a entrambi i k e a k=8 non c'è. Il risultato vero è l'**interazione con k**: −0,070 di AUC a k=1 (5/5 semi), +0,020 a k=8 | [E0g](#e0g--quale-augmentation-per-misura-chiuso-si-adotta-la-catena-ufficiale) |
| **E0f v2** | 19–25 ago 2026 | **geometria congelata**: quattro celle a 7/7 semi sulla stessa mappa a 938; nessun asse conclude sotto alcuna lettura del criterio → ripiego pre-dichiarato `256/eval_center/fps 7,5` confermato; ratifica della lettura primaria ai supervisori (Q26, `docs/12`) | [E0f v2](#e0f-v2--la-stessa-domanda-con-etichette-vere-geometria-congelata) |
| **E0f** | 19 ago 2026 | **fermato alla calibrazione**: senza annotazioni temporali sui video di training il proxy ha 8,3 pp di escursione utile, e l'effetto atteso vale **meno di un video su 168**. Le otto celle non sono partite | [E0f](#e0f--la-stessa-geometria-misurata-sul-dominio-vero-chiuso-alla-calibrazione) |
| **E0e-bis** | 18–19 ago 2026 | nessun asse conclude → si applicano i ripieghi dichiarati: **`s256-eval_center-fps7p5`**. Q22 è poi fissata da ADR-024; Q21 è stata riaperta per la verifica E0f v2 sul dominio vero | [E0e-bis](#e0e-bis--la-stessa-domanda-con-un-disegno-che-può-rispondere-chiuso) |
| **E0e** | 18 ago 2026 | **esito negativo**: su UCF101 ridotto nessun asse (risoluzione, ritaglio, span) esce dal rumore — il compito è saturo e due semi non bastano. Q21/Q22 restano aperte | [E0e](#e0e--la-geometria-dingresso-per-misura-chiuso-esito-negativo) |
| E0c | 14–15 ago 2026 | pipeline certificata (top-1 ~99%); fp16 non sposta le decisioni più del rumore; i layer vanno 5 < 11 < 17 ≲ 23 e il rumore cresce dove l'accuratezza cala | [E0c](#e0c--sanity-sul-noto-forma-ridotta) |
| E0d *(fedeltà chiusa; store query-side completo al 100%)* | 14 ago · 25 ago 2026 | fp16 sullo store costa 1,8e-4 di rel_L2 (4,7× sotto soglia); AMP BF16 costa 2,3e-2, cioè ~10% della separazione più stretta fra finestre. Il 25 agosto lo store query-side `c6fe0f7cc26c` è stato completato e **copre il 100% della griglia di `fsp_v1`**: 147.548 feature, hit rate 1,000000; la gamba di *throughput* resta non eseguita | [E0d](#e0d--pilota-del-feature-store-fedeltà-chiusa-store-query-side-estratto) |

## 📊 Tavola dei numeri — tutto ciò che si può scrivere in tesi

> **Cos'è.** Il resto di questo documento racconta *come* si è arrivati ai
> numeri, una voce per blocco. Questa sezione li mette invece **tutti insieme e
> in ordine**, perché è da qui che si scrivono i capitoli 6 e 7 senza dover
> ricomporre a mano cinque narrazioni.
>
> ⚠️ **Le tavole sono generate dagli artefatti, non ricopiate.** I valori vengono
> da `runs/*/metrics.json`, `runs/*/history.jsonl` e
> `runs/e5b_efficienza/efficiency_inference.json`; la dispersione usa la stessa
> statistica di `SeedAggregator` (deviazione **campionaria**), quindi coincide
> per costruzione con le figure di `runs/figures/` e con le voci qui sotto. Se un
> numero di questa tavola diverge da uno del testo, **vince l'artefatto** ed è un
> difetto da correggere, non una sfumatura.
>
> **Tutti i numeri sono su `edp_v2`**, il protocollo del laboratorio (ADR-047),
> con lo split consegnato `5e82820a…` (1303/306/290). ⚠️ I numeri delle voci
> datate **31 agosto o prima** sono `fsp_v1` o `edp_v1` e **non appartengono a
> questa tavola**: non sono confrontabili.

### Le cinque avvertenze che devono viaggiare coi numeri

Non sono cautele di forma: ognuna cambia cosa si può affermare, e ometterne una
renderebbe la tabella più forte di quanto sia.

1. ⚠️ **Tre semi, e tre è poco.** Dal 4 settembre tutte e cinque le celle
   girano su `[42, 67, 104]` (ADR-058), quindi ogni riga ha una dispersione e i
   confronti si fanno **appaiati** (tabella A-bis). Ciò che tre semi *non*
   comprano va detto con altrettanta chiarezza: con n=3 un t appaiato ha **2
   gradi di libertà** e la soglia al 5% è |t| > 4,303, quindi solo gli effetti
   **grandi** concludono. Un divario piccolo con la direzione concorde 3/3 —
   DoRA sopra LoRA, per dire — resta **non concluso**, e in tesi si scrive così:
   né «uguali» né «DoRA vince». La riserva per salire a cinque è
   `[42, 67, 104, 88, 1320]`, in quest'ordine.
2. ⚠️ **Il livello evento usa una definizione NOSTRA** (ADR-048): un evento è
   rilevato se una finestra sopra soglia lo interseca anche di un solo frame, e
   gli allarmi consecutivi si fondono. Il documento del laboratorio non definisce
   l'evento. Chunk e video sono confrontabili coi loro numeri, **l'evento no**, e
   in tesi si scrive (`docs/01` §3bis D10).
3. ⚠️ **Lo split contiene leakage, adottato consapevolmente** (ADR-047): dieci
   coppie byte-identiche attraversano le partizioni e sei video di training sono
   copie esatte di video di test. Si dichiara invece di correggerlo, perché è la
   partizione su cui il laboratorio riporta i propri numeri.
4. ⚠️ **Le righe per-classe con `n_pos` a una cifra non sono misure.** `Abuse` ha
   **6** chunk positivi su 9116: la sua AP vale quanto un lancio di moneta, e la
   tabella per-classe va riportata **con `n_pos` accanto**, sempre.
5. **La tabella D conserva il profilo storico A100.** Dal 5 settembre sono
   disponibili anche misure schema 2 su A100 e H100, alla stessa geometria:
   [dossier 14, §3](14_analisi_complementari_stesura.md#3-profili-schema-2-verificati).
   Non sono estrapolazioni. Resta una sola esecuzione del benchmark per GPU;
   la dispersione fra iterazioni non sostituisce repliche su nodi o sessioni.
   Dal 4 settembre esiste anche il costo di **sistema** (tabella **D-bis**), e
   porta due avvertenze proprie: è una **composizione** di misure prese su
   macchine diverse — il modello su A100, il decoder su un nodo H100 — e i suoi
   `stream/GPU` sono un **tetto aritmetico**, senza scheduling né contesa. La
   conclusione che ne dipende («il collo di bottiglia è il modello») sopravvive
   per un fattore ≥ 3, non per un pareggio.

### Il protocollo in una riga

Finestre di **8 s** con passo **4 s**, **32 campioni** per finestra (≈3,89 fps
effettivi) a **384×384 `squash`**; la coda di ogni video si valuta. Training
**88.371** chunk, validazione **8383** su griglia contigua, test **9116** su 290
video di cui **140 anomali**. Budget: **un'epoca**, 5524 passi. Selezione:
**AUROC di validazione** ogni 920 passi. Soglia: **punto di Youden** sulla ROC di
validazione, una sola per chunk e video.

### A. Accuratezza — insieme di test, 9116 finestre su 290 video

Metriche che non dipendono dalla soglia. `AUC_a`/`AP_a` sono le stesse
statistiche ristrette ai **140 video anomali**.

**Tre semi — `[42, 67, 104]` — per tutte e cinque le celle** (ADR-058). La
dispersione è la deviazione **campionaria** su n=3, la stessa di `SeedAggregator`.

| cella | adattamento | semi | chunk-AUC | chunk-AP | AUC_a | AP_a |
|---|---|:-:|---:|---:|---:|---:|
| E2c | `frozen-linear` | 42·67·104 | **0,8473 ± 0,0034** | 0,4788 ± 0,0044 | 0,6495 ± 0,0035 | 0,5205 ± 0,0054 |
| E2d | `frozen-attentive` | 42·67·104 | **0,9310 ± 0,0080** | 0,7632 ± 0,0092 | 0,8355 ± 0,0053 | 0,7873 ± 0,0041 |
| E3-P.1 | `vptdeep-n576` | 42·67·104 | **0,8882 ± 0,0083** | 0,6279 ± 0,0201 | 0,7549 ± 0,0073 | 0,6779 ± 0,0141 |
| E3-R.1 | `lora-r32-qkvomlp` | 42·67·104 | **0,9392 ± 0,0015** | 0,7889 ± 0,0055 | 0,8627 ± 0,0027 | 0,8270 ± 0,0059 |
| E3-R.1 | `dora-r32-qkvomlp` | 42·67·104 | **0,9422 ± 0,0017** | 0,7947 ± 0,0037 | 0,8669 ± 0,0032 | 0,8305 ± 0,0037 |

⚠️ **Il seme 42 era il migliore dei tre per E2d**, e i numeri che questa tavola
portava fino al 3 settembre erano perciò i più ottimisti della baseline: 0,9385
contro una media di 0,9310, quasi una sigma sopra. Non è un difetto del 42 — è
ciò che significa non avere dispersione.

⚠️ **E3-M ed E3-C non entrano in questa tavola, e l'assenza è una scelta.** Le
trenta celle chiuse il 15 settembre ([voce](#e3m-e3c-20260915)) girano a **due e
tre cicli** invece che a una epoca: sono lo stesso protocollo ma **non lo stesso
budget**, e «Budget: un'epoca, 5524 passi» è una riga del protocollo in una riga
qui sopra. Mescolarle darebbe una tabella dove la colonna del metodo e quella del
calcolo speso variano insieme. Il loro esito — **nessuna delle due sposta
l'ordinamento**, e la migliore configurazione resta `dora-r32-qkvomlp` a **una**
epoca — è quello che rende questa tavola ancora quella da citare. ⚠️ In
`runs/classifiche/classifiche.md` dal 25 settembre entra **una** versione per
modello, sotto il nome semplice, e per Linear probe, Attentive probe e LoRA r=32 è
quella a tre epoche ([postilla di E3-M](#versione-riportata-20260925)): lì quelle tre righe **non** sono le righe di
questa tavola. Entrambe le versioni: `classifiche_modelli.py --tutte-le-versioni`.

⚠️ **Per la stessa ragione non entrano E3-S** (tre cicli, famiglia
`sequenziale`, [voce](#e3s-20260924)) **né i confronti esterni**: MIVIA-Crime è un
altro dataset ([E-MIVIA](#e-mivia--il-modello-su-un-dataset-che-non-ha-mai-visto-chiusa-lordinamento-si-trasferisce-e-perde-4-punti-di-auc-la-calibrazione-vale-0021-di-f1))
e il confronto con lo stato dell'arte usa numeri del laboratorio
([E4](#e4-sota-20260924)). Tutti e tre confermano l'ordinamento di questa tavola
invece di sostituirlo.

#### A-bis. Il confronto **appaiato**, che è quello che i tre semi comprano

Tutti i metodi hanno girato sugli **stessi tre semi**, quindi confrontare le
medie con σ indipendenti butta via informazione: la differenza si calcola
**seme per seme** e la sua dispersione è quella delle differenze. `t` è il t
appaiato a 2 gradi di libertà, e con n=3 la soglia bilaterale al 5% è
**|t| > 4,303** — molto esigente, ed è giusto che lo sia.

| confronto | Δ chunk-AUC | Δ AP_a | per seme (AUC) | concordi | verdetto |
|---|---:|---:|---|:-:|---|
| VPT − E2d | **−0,0428 ± 0,0084** | −0,1093 ± 0,0117 | −0,042 · −0,035 · −0,052 | 3/3 sotto | **separa** (t = −8,8) |
| LoRA − E2d | +0,0082 ± 0,0089 | **+0,0397 ± 0,0059** | +0,001 · +0,018 · +0,006 | 3/3 sopra | AUC no (t = 1,6) · **AP_a sì** (t = 11,7) |
| DoRA − E2d | +0,0112 ± 0,0064 | **+0,0432 ± 0,0038** | +0,006 · +0,018 · +0,010 | 3/3 sopra | AUC no (t = 3,1) · **AP_a sì** (t = 19,9) |
| DoRA − LoRA | +0,0030 ± 0,0026 | +0,0035 ± 0,0024 | +0,005 · +0,000 · +0,004 | 3/3 sopra | **non separa** (t = 2,0) |

**Questa tabella è il risultato principale della campagna a tre semi**, e dice
tre cose che a uno o due semi non si potevano dire.

1. **VPT sotto la probe congelata è ora un fatto, non un'osservazione**: −0,043
   di AUC con t = −8,8, sotto su tutti e tre i semi e su tutte le metriche.
2. **Il guadagno del reparam non sta dove lo si cercherebbe.** Su chunk-AUC
   **non separa** (t = 1,6 e 3,1), su **AP_a separa nettissimo** (t = 11,7 e
   19,9). Cioè l'adattamento non aiuta a dire se un video contiene un'anomalia —
   la testa attentiva su feature congelate lo fa già — aiuta a dire **dove**,
   dentro il video anomalo. La lettura di E3 a due semi reggeva, e a tre regge
   con un test appaiato dietro.
3. **LoRA e DoRA non si separano nemmeno a tre semi.** Il divario è +0,0030,
   concorde 3/3, ma a s67 vale **+0,00004**: la direzione è stabile, la
   magnitudine no. E DoRA costa **+30%** di training.

⚠️ **Con n=3 «non separa» non vuol dire «uguali»**: vuol dire che tre punti non
bastano a escludere il caso. La direzione concorde 3/3 è essa stessa debole — un
test dei segni a tre prove dà al meglio p = 0,125 — quindi ciò che regge sono gli
effetti **grandi** rispetto alla dispersione, non i piccoli col segno giusto.

### B. Metriche a soglia — τ = punto di Youden sulla ROC di validazione

Una sola soglia per chunk e video, stimata sulle 8383 finestre di validazione.
La regola di decisione a livello di video è **asimmetrica** (`docs/04`).

| cella | adattamento | τ | chunk P | chunk R | chunk F1 | video P | video R | video F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E2c | `frozen-linear` | 0,1145 ± 0,0158 | 0,2801 | 0,9446 | **0,4318 ± 0,0137** | 0,7231 | 0,9357 | **0,8155 ± 0,0104** |
| E2d | `frozen-attentive` | 0,0977 ± 0,0469 | 0,5678 | 0,8055 | **0,6657 ± 0,0205** | 0,8621 | 0,9167 | **0,8884 ± 0,0296** |
| E3-P.1 | `vptdeep-n576` | 0,1140 ± **0,1349** | 0,4085 | 0,8520 | **0,5480 ± 0,0347** | 0,7759 | 0,9429 | **0,8501 ± 0,0376** |
| E3-R.1 | `lora-r32-qkvomlp` | 0,0690 ± 0,0404 | 0,5700 | 0,8225 | **0,6721 ± 0,0089** | 0,8926 | 0,9429 | **0,9165 ± 0,0132** |
| E3-R.1 | `dora-r32-qkvomlp` | 0,0707 ± 0,0137 | 0,5761 | 0,8326 | **0,6810 ± 0,0056** | 0,9055 | 0,9571 | **0,9306 ± 0,0121** |

⚠️ **La τ di VPT ha una deviazione più grande della sua media** (0,1349 su
0,1140): il punto di Youden di quella cella **non è riproducibile fra semi**, ed
è un fatto che le medie di F1 nascondono. Le altre quattro stanno fra il 14% e il
59% della propria media — alte, ma non degeneri. È una ragione in più per non
appendere una conclusione alle metriche a soglia: l'ordine dei metodi lo decidono
AUC e AP, che una soglia non ce l'hanno.

**Livello evento** — definizione **nostra** (ADR-048), non confrontabile coi
numeri del laboratorio; latenza mediana di rilevazione in secondi:

| cella | adattamento | evento F1 | latenza mediana |
|---|---|---:|---:|
| E2c | `frozen-linear` | 0,6775 ± 0,0229 | 2,3 s |
| E2d | `frozen-attentive` | **0,6778 ± 0,0365** | 2,5 s |
| E3-P.1 | `vptdeep-n576` | 0,6124 ± 0,0377 | 2,4 s |
| E3-R.1 | `lora-r32-qkvomlp` | 0,6437 ± 0,0183 | 2,4 s |
| E3-R.1 | `dora-r32-qkvomlp` | 0,6609 ± 0,0017 | 2,4 s |

⚠️ **A livello evento la probe congelata non è battuta da nessuno**, ed è il solo
asse dove succede: il reparam sta sotto di 0,014-0,034 e la linear la pareggia.
Con σ fra 0,018 e 0,037 nessuna di queste differenze separa, quindi la
lettura onesta è **«a livello evento i metodi non si distinguono»**, non «il
congelato vince». La latenza è piatta a 2,3-2,5 s per tutti: la decide la
**griglia** (hop 4 s), non il metodo.

### C. Budget, parità e selezione

Un'epoca = **5524 passi**, batch effettivo 16 (2 × 8 accumuli), `lr 1e-4`,
schedule coseno, selezione sull'AUROC di validazione ogni 920 passi.

I tre valori di ogni cella sono i semi **42 · 67 · 104**, in quest'ordine.

| cella | adattamento | par. adattatore | par. testa | passo scelto | AUROC val | GPU |
|---|---|---:|---:|:-:|---|:-:|
| E2c | `frozen-linear` | 0 | 3,073 | 4600 · 5520 · 5520 | 0,7787 · 0,7662 · 0,7683 | A · A · A |
| E2d | `frozen-attentive` | 0 | 49,339,393 | 4600 · 3680 · 2760 | 0,9370 · 0,9150 · 0,9288 | A · H · H |
| E3-P.1 | `vptdeep-n576` | 14,155,776 | 49,339,393 | 3680 · **1840** · 2760 | 0,9084 · 0,8910 · 0,9002 | A · H · H |
| E3-R.1 | `lora-r32-qkvomlp` | 14,155,776 | 49,339,393 | 5520 · 4600 · 3680 | 0,9466 · 0,9484 · 0,9499 | A · H · H |
| E3-R.1 | `dora-r32-qkvomlp` | 14,376,960 | 49,339,393 | 5520 · 4600 · 3680 | 0,9467 · 0,9463 · 0,9481 | H · H · H |

**Il passo scelto è esso stesso un risultato, e va letto per colonne.** Il
reparam sceglie **tardi** — 5520, 4600, 3680, cioè fra il 67% e il 100% del
budget — mentre **VPT sceglie prestissimo**: 3680, **1840**, 2760, e 1840 è un
terzo dell'epoca. La probe congelata sta in mezzo. Un metodo che raggiunge il suo
massimo di validazione a un terzo del budget e poi peggiora sta
**sovradattando**, ed è la stessa diagnosi che E3-P.1 aveva fatto su un seme
solo: qui si vede su tre.

⚠️ **Le celle non hanno girato tutte sulla stessa macchina** (A = A100 `gpuq`,
H = H100 `aiq`), e alcune hanno attraversato entrambe fra un anello di catena e
il successivo. Non e un confondimento — il non-determinismo fra le due vale
**1,8e-7** di AUC, misurato il 1 settembre, sette ordini di grandezza sotto il
pavimento di rumore — ma si dichiara comunque, perché è provenienza.

### D. Costo — training misurato su A100, inferenza su NVIDIA A100-SXM4-80GB

Il costo di training è la **mediana degli intervalli fra checkpoint** ogni 100
passi, con le validazioni escluse; quello di inferenza viene da E5b, batch 1.

| adattamento | s/passo | training | 6 validazioni | cella intera | latenza | overhead | RTF | da spedire |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `frozen-linear` | 1,40 | 2,15 h | 0,16 h | **2,67 h** | — | — | — | — |
| `frozen-attentive` | 1,74 | 2,67 h | 0,23 h | **3,41 h** | 131,4 ms | — | 30,4× | 0 |
| `vptdeep-n576` | 6,59 | 10,11 h | 1,89 h | **12,43 h** | 133,6 ms | +1,68% | 29,9× | 14,155,776 |
| `lora-r32-qkvomlp` | 8,85 | 13,58 h | 2,74 h | **16,90 h** | 187,6 ms | +42,79% | 21,3× | 14,155,776 |
| `dora-r32-qkvomlp` | 11,48 | 17,62 h | 3,42 h | **21,48 h** | 241,4 ms | +83,71% | 16,6× | 14,376,960 |

**Il reparam dopo `merge_()`** — stesso modello, stessa accuratezza, altra forma:

| adattamento | latenza | overhead | RTF | da spedire | scarto del merge (fp32) |
|---|---:|---:|---:|---:|---:|
| LoRA r=32 **fusa** | 131,8 ms | +0,34% | 30,3× | 304,680,960 | 8,94e-08 |
| DoRA r=32 **fusa** | 131,8 ms | +0,33% | 30,3× | 304,680,960 | 4,77e-07 |

⚠️ **La colonna «da spedire» qui sopra è il solo delta lato backbone, e non è ciò
che si spedisce**: vedi **D-bis**. Il numero completo cambia il rapporto
fuso/non-fuso da 21,5× a **5,6×**.

#### D-bis. Il **sistema**, non solo il modello *(E5b.2, ADR-059)*

Le righe di D dicono cosa costa il **modello**. Componendole col decoder (E0d) e
con la `detection_latency` delle stesse celle si ottiene cosa costa il
**sistema** — generato in `runs/e5b_efficienza/deployment.md`, voce
[§E5b.2](#e5b2--dal-costo-del-modello-al-costo-del-sistema-la-composizione-è-chiusa-il-ritardo-dellallarme-è-per-il-93-griglia-non-calcolo).

| adattamento | risposta per finestra | **ritardo dell'allarme** | quota di calcolo | stream/GPU |
|---|---:|---:|---:|---:|
| `frozen-attentive` | 172,2 ms | **2,64 s** | 6,5% | 30 |
| `vptdeep-n576` | 174,4 ms | **2,57 s** | 6,8% | 30 |
| `lora-r32-qkvomlp` | 228,4 ms | **2,66 s** | 8,6% | 21 |
| LoRA **fusa** | 172,6 ms | **2,61 s** | 6,6% | 30 |
| `dora-r32-qkvomlp` | 282,2 ms | **2,68 s** | 10,5% | 17 |
| DoRA **fusa** | 172,6 ms | **2,57 s** | 6,7% | 30 |

**La frase che questa tabella autorizza, e che D da sola non autorizzava**: il
ritardo con cui il sistema segnala un evento è **2,57–2,68 s** su tutti e sei i
metodi — un'escursione del **4%** dove le latenze di calcolo differiscono del
**64%** — perché è per il **93% geometria della griglia** e per il 7% calcolo.
Accelerare il modello non rende il sistema più reattivo; l'hop sì.

⚠️ *Nota del 24 settembre:* il ritardo qui sopra è composto con la regola
**storica** di `detection_latency` (endpoint dentro l'evento). ADR-061 porta in
tesi quella per **intersezione**, e con quella la stessa composizione sta in
`runs/classifiche/classifiche.md` §3: sulle sei varianti **2,91–3,14 s**, quota di
calcolo **5,5–9,0%** ([E5b.2](#e5b2-20260904)). La frase regge; i numeri da citare
sono quelli.

**E i byte che si spediscono davvero**, testa compresa (49.339.393 parametri,
addestrata in ogni cella e assente dal checkpoint base):

| adattamento | **aggiornamento** (delta + testa) | MB fp32 | **primo deploy** | MB fp32 |
|---|---:|---:|---:|---:|
| `frozen-attentive` | 49.339.393 | **188,2** | 354.020.353 | 1.350,5 |
| `vptdeep-n576` · `lora-r32` | 63.495.169 | **242,2** | 368.176.129 | 1.404,5 |
| `dora-r32` | 63.716.353 | **243,1** | 368.397.313 | 1.405,3 |
| LoRA / DoRA **fusa** | 354.020.353 | **1.350,5** | 354.020.353 | 1.350,5 |

⚠️ Su un **aggiornamento** il fuso costa **5,6×** il non fuso, non 21,5×; su un
**primo deploy** il segno **si inverte**, perché il fuso non porta un delta a
parte. Sono due scenari, e con una colonna sola si sceglierebbe male in uno dei
due.

⚠️ **Le colonne di accuratezza non si ripetono qui**: stanno in **A**, e il
confronto fra metodi è quello **appaiato** di **A-bis**, non la sottrazione
delle medie. Questa tabella aggiunge un asse — il costo di sistema — e non una
seconda lettura dello stesso.

### E. Le sei frasi che questi numeri autorizzano

Ognuna con il numero che la sostiene e il limite che le impedisce di dire di più.
⚠️ Erano **cinque** fino al 3 settembre: la sesta è arrivata con E5b.2, ed è
l'unica che parla del **sistema** invece che del modello.

1. **La semantica di questo dominio non è linearmente separabile sul layer 23.**
   La probe lineare perde **8,4 punti** di AUC e **28,4 di AP** contro quella
   attentiva. È il divario più grande della tavola, ben oltre qualunque
   pavimento di rumore, ed è il lavoro diagnostico per cui E2c esiste. ⚠️ E2c è
   anche la cella **più stabile** fra i semi (σ = 0,0034): il suo verdetto non
   dipendeva dal 42.
2. **Il prompt tuning non regge il confronto a parità di budget su questo
   backbone**, e a tre semi è un fatto misurato. VPT-deep con **gli stessi
   14.155.776 parametri** di LoRA sta **sotto la probe congelata** di
   **0,0428 ± 0,0084** di AUC (t appaiato **−8,8**, sotto su 3/3 semi) e di
   0,109 di AP_a, pagando 3,6× il training. Sovradatta, e i tre semi lo mostrano
   dove prima si vedeva su uno: la selezione cade ai passi **3680 · 1840 ·
   2760** — a s67 a **un terzo** del budget — e la sua soglia di Youden ha una
   **σ più grande della media** (0,135 su 0,114), cioè non è riproducibile.
3. **L'adattamento del backbone paga poco sull'ordinamento globale e molto
   dentro il video anomalo**, ed è la frase che i tre semi hanno reso
   verificabile. Contro E2d, appaiato: chunk-AUC **+0,0082** (t = 1,6) e
   **+0,0112** (t = 3,1), che **non separano**; AP_a **+0,0397** (t = **11,7**)
   e **+0,0432** (t = **19,9**), che separano nettissimo. Cioè l'adattamento non
   aiuta a dire *se* un video contiene un'anomalia — la testa attentiva su
   feature congelate lo fa già — aiuta a dire **dove**.
4. **LoRA e DoRA non si separano nemmeno a tre semi.** Il divario appaiato è
   **+0,0030 ± 0,0026** di AUC (t = 2,0, sotto la soglia 4,303), concorde 3/3 ma
   con un seme — il 67 — in cui vale **+0,00004**. DoRA non è mai dietro, su
   nessuna metrica e su nessuno dei tre semi, e costa **+30%** di training.
   **La direzione è consistente, la magnitudine no**, ed è la stessa conclusione
   di due semi con un seme in più a sostenerla: comprare il terzo non ha
   cambiato il verdetto, il che è a sua volta un'informazione.

   ⚠️ **Una quinta lettura, che nessuno aveva chiesto e che i tre semi
   regalano:** il reparam non alza solo l'accuratezza, la **stabilizza**.
   σ(chunk-AUC) vale **0,0080** per la probe congelata e **0,0015 · 0,0017** per
   LoRA e DoRA — cinque volte meno. VPT, che sovradatta, sta con la congelata
   (0,0083). Con n=3 le σ sono stime grossolane e questa osservazione **non è
   un test**, ma la direzione è coerente su tutte e quattro le metriche.

   ⚠️ **E un asse dove il quadro si rovescia:** a livello **evento** nessun
   metodo batte la probe congelata (F1 0,678 ± 0,037 contro 0,644 e 0,661 del
   reparam). Nessuna di queste differenze separa, quindi la lettura è «a livello
   evento i metodi non si distinguono» — ma è il solo posto della tavola dove
   l'adattamento non è avanti, e va riportato invece che omesso.
5. **Il trade-off di C7 non è quello che il PdD descrive.** Senza `merge_()` il
   reparam è il **più caro dei tre** a inferenza (+42,8% e +83,7%) mentre VPT
   costa **+1,68%**. Con il merge il reparam torna sul congelato (+0,3%) e
   **domina VPT su ogni asse tranne due**: ore di training e byte da spedire
   (**1.350 MB contro 242 MB, 5,6×** — testa compresa, D-bis; il **21,5×** che
   questa riga portava fino al 3 settembre confrontava i soli delta e non ciò
   che si spedisce). Il trade-off «opposto» si riduce a un compromesso fra *ore
   di GPU una volta sola* e *megabyte a ogni distribuzione* — una domanda di
   deployment, non di metodo. ⚠️ E su un **primo deploy** il segno si inverte: il
   fuso è il più **piccolo**, perché non porta un delta a parte.

6. **Il sistema risponde in 2,6 secondi, e non è il modello a deciderlo**
   *(E5b.2)*. Componendo latenza del modello, decoder e `detection_latency`, il
   ritardo fra l'inizio di un evento e l'allarme è **2,57–2,68 s** su tutti e sei
   i metodi: un'escursione del **4%** dove le latenze di calcolo differiscono del
   **64%**, perché il ritardo è per il **93% geometria della griglia**. Ne segue
   che sull'asse della **reattività** la scelta del metodo è **libera** — DoRA
   non fusa arriva 0,04 s dopo il congelato — e che chi vuole un sistema più
   pronto deve toccare l'**hop**, non l'adattatore. ⚠️ Il limite: è una
   composizione di misure prese su macchine diverse, e non include acquisizione
   RTSP né dispatch dell'allarme. Regge perché il modello domina il decoder di
   un fattore ≥ 3, non per un pareggio.

### F. Dove stanno gli artefatti

| cosa | dove |
|---|---|
| metriche di una cella | `runs/{run_id}/metrics.json` |
| score per finestra (test **e** validazione) | `runs/{run_id}/scores.jsonl` |
| curve di training | `runs/{run_id}/history.jsonl`, `training.png` |
| ROC, PR, confusione, per-classe, soglia, evento | `runs/{run_id}/*.png` (`make_figures.py --per-run`) |
| confronto fra metodi | `runs/figures/e2c-e2d-e3c-e3m-e3p-e3r_*.png` e `_tabella.md` |
| classifiche di tutti i modelli | `runs/classifiche/` (`classifiche.{json,md}`, `per_video*.csv`, `per_class.csv`, `ranking_chunks_riepilogo.csv`) |
| strati e latenza per intersezione | `runs/{run_id}/{strata.json,latenza.json}` e `strata_*.png`; per le celle di settembre sono **anche** dentro `metrics.json` |
| F12/F13 in particolare | `runs/analisi_f1213_20260915/` (`celle.csv`, `confronto_appaiato.csv`, `analisi.json`, `verifica_score.json`) |
| efficienza a inferenza (**modello**) | `runs/e5b_efficienza/efficiency_inference.json` |
| efficienza di **sistema** (E5b.2) | `runs/e5b_efficienza/deployment.{json,md}` + quattro PNG; i profili schema 2 e i loro studi in `runs/e5b2_efficienza_{gpuq,aiq}/`, il confronto fra le due macchine in `runs/figures/confronto_macchine.*` |
| provenienza di ogni esecuzione | `runs/registry.jsonl` (append-only, una riga per **esecuzione**) |
| ricevute di una campagna | `../tmp/{campagna}/completion-*.json` e `state-finale-*.json` |

Le figure si rigenerano con **un comando solo**, e ADR-014 impone che sia così —
mai numeri ricopiati a mano in tesi:

```powershell
python scripts/make_figures.py --exp e2c --exp e2d --exp e3p --exp e3r `
    --exp e3m --exp e3c `
    --metric chunk_auc --metric chunk_ap `
    --metric chunk_auc_anomalous --metric chunk_ap_anomalous --per-run
```

E le venti tabelle di tutti i modelli — **28 modelli, 84 celle** dal 15 settembre
— con lo stesso comando di sempre; gli `--set` dicono *dove* stanno i file in
locale, mai *cosa* si legge:

```powershell
python scripts/classifiche_modelli.py `
    --set root=../data/ucf_crime_mirror `
    --set inventory=../data/ucf_crime_inventory/inventory.csv `
    --set splits_dir=../data/ucf_crime_mirror/Anomaly_Detection_splits `
    --set annotations.path=../data/annotations/fs_ucf_crime
python scripts/classifiche_modelli.py --solo-per-classe   # + gli stessi --set
python scripts/classifiche_modelli.py --solo-ranking-chunk # + gli stessi --set
```

Lo **studio di deployment** (tabella D-bis) si rigenera con un comando altrettanto
solo, e ⚠️ **non chiede una GPU**: compone artefatti che esistono già, quindi gira
sul portatile e non passa dal tracker.

```powershell
python scripts/e5b_deployment.py
```

---

<a id="e-ovad-20260927"></a>

## E-OVAD — 27 settembre 2026, rivalutazione completa degli score disponibili

Richiesta di Antonio e ADR-076. Eseguito senza modifiche il codice consegnato
in `ovad_protocol/`, sugli score dei checkpoint già selezionati: **108 celle
UCF-Crime e 52 MIVIA-Crime**, 36 versioni di modello. Nessuna inferenza o nuovo
training. Le classifiche principali riportano le 30 versioni storiche, UCF a
tre semi e MIVIA al solo 42; la raccolta completa conserva tutte le versioni e
tutti i semi disponibili. Gli otto E3-SV risultavano ancora in addestramento
al controllo sul cluster, senza score finali: stato archiviato nella raccolta.

Chunk e guard 8 s, passo test 4 s, righe fino a 32 s. Il passo della validazione
salvata è 8 s, accettato con avviso dal codice. Soglia Youden ricalcolata sulle
righe UCF di validazione per ciascun modello/seme e trasferita invariata a MIVIA.

| Partizione | Righe giudicabili | Escluse | Eventi | Eventi non osservati |
|---|---:|---:|---:|---:|
| UCF validazione | 2428 | 128 | 258 | 0 |
| UCF test | 1536 | 144 | 262 | 0 |
| MIVIA test | 724 | 140 | 200 | 0 |

Identità delle etichette e delle maschere di giudicabilità verificata fra tutti
i modelli. La macro usa **129/140** video anomali UCF e **102/135** MIVIA: gli
altri non hanno entrambe le classi giudicabili. Questi denominatori accompagnano
i numeri, perché non sono le metriche EDP-v2 sui chunk.

La prima riga per protocol mean è **DoRA r=8 su UCF: 81,97 ± 0,31%** (tre semi);
su MIVIA è **VPT-shallow n=8 su LoRA r=32 fuso: 83,63%** (seme 42). Sono ordini
descrittivi, non test di superiorità statistica.

**Aggiornamento del 28 settembre:** le tabelle principali ora iniziano da
UCF e MIVIA al solo seed 42, con tutte le 15 metriche e il mean, seguite dalla
media dei due mean e dagli altri semi separati. A s42 UCF è primo DoRA r=16
(mean **83,15%**); MIVIA e media fra dataset sono prime per VPT-shallow n=8
su LoRA r=32 fuso (**83,63%**, media **83,32%**). Fra le sei coppie una/tre
epoche si seleziona la migliore media dei mean sui due test s42: solo Linear
probe rimane a tre epoche. La scelta usa i valori completi, è dichiarata sul
test e non cambia fra semi. L'Excel riporta queste tre nuove tabelle prima
del vecchio protocollo; le 5 tabelle preesistenti sono conservate.

**Verifiche:** 16 test originali e 72 test mirati/regressioni passati; lint e
formattazione puliti. La CLI originale coincide in ogni campo con il pilota.
Audit indipendente sulle 160 celle: tutte le metriche, Youden, conteggi,
aggregazione fra semi, CSV/JSON e hash verificati. Scarto massimo rispetto ad
AUC/AP originali nella conversione: **2,22e-16**. Il codice ricevuto resta invariato.

Artefatti: [`runs/classifiche/ovad/README.md`](../runs/classifiche/ovad/README.md),
`manifest.json`, `verifica_indipendente.json`, `celle/`, `classifiche/` e
`classifiche_tutte_versioni/`. Sintesi metodologica e limiti in [docs/19](19_protocollo_ovad.md).

---

<a id="e3so-20260930"></a>

## E3-SO — selezione OVAD-UCF e tre epoche totali (30 settembre 2026)

Richiesta di Antonio: selezionare le architetture soltanto sulla Protocol Mean
UCF, guardia 30 s, seme 42, e contenere ogni sequenza in **tre epoche totali**.
Le quattro celle nuove accoppiano **DoRA r=16** con **VPT-shallow n=8** e
**VPT-deep n=8**, in entrambe le direzioni (`fondi` e `congela`).
Si riusano i prior delle sorgenti a **una epoca** (5524 passi di budget),
selezionati per AUROC di validazione; il nuovo stadio esegue **11048 passi**,
due cicli di coseno da 5524, LR 1e-4, senza early stopping. Il checkpoint
selezionato può precedere la fine del primo ciclo; il budget contabilizzato
resta quello di un'epoca. Testa e ottimizzatore del nuovo stadio ripartono.

VPT-deep a tre epoche non è un prior ammesso: esaurirebbe già il budget della
sequenza. La quinta variante proposta con quel prior è esclusa. LoRA r=32
non genera coppie architetturali nuove e non è rilanciata in questa campagna.
Le storiche E3-S/E3-SV usano **1+3=4 epoche complessive**; restano intatte e
non costituiscono un confronto a pari budget con E3-SO. La scelta delle
architetture è esplorativa perché deriva dal test UCF; MIVIA non entra nella
selezione. La valutazione finale conserva anche MIVIA come test esterno.

Identità distinta **e3so**, quattro experiment/sweep, campagna isolata
`~/evd/experiments/20260930-e3so`. Preflight su identità, hash e stato selected
dei tre checkpoint sorgente; controllo di rango, alpha/r=2, parametri
addestrabili e congelati. Supervisore con ripresa per checkpoint, priorità
aiq, massimo quattro job propri e limite di dieci per associazione utente-account; artefatti e
classifiche separati in `runs/analisi_e3so_20260930/`.
**Chiusa il 2 ottobre 2026:** quattro training a 11048/11048 passi e quattro
post-processing completi; checkpoint finali validati, artefatti rientrati in
locale e verificati con SHA256 (1003 file, 105 figure nell'archivio). Snapshot
`86397e9`, seme 42. Tutte le metriche EDP-v2 e OVAD g30 sono calcolate su UCF e
MIVIA; Youden OVAD e' stimata sulla validazione UCF con la stessa guardia e
trasferita a MIVIA. Coorti e hash delle annotazioni coincidono con le tabelle
precedenti. Audit indipendente su AUC/AP, macro/micro, righe, video, eventi,
Youden e Protocol Mean; le classifiche EDP non hanno incoerenze.

| Sequenza (1+2 epoche) | AUC UCF EDP | AP UCF EDP | AUC MIVIA EDP | AP MIVIA EDP | Mean UCF OVAD g30 | Mean MIVIA OVAD g30 |
|---|---:|---:|---:|---:|---:|---:|
| VPT-deep n=8 -> DoRA r=16 | 93.47 | 77.65 | 89.20 | 82.24 | 85.54 | 82.94 |
| DoRA r=16 -> VPT-shallow n=8 | 94.26 | 79.28 | 89.95 | 82.71 | 84.90 | 83.24 |
| VPT-shallow n=8 -> DoRA r=16 | 94.16 | 79.11 | 89.99 | 83.06 | 84.31 | 84.09 |
| DoRA r=16 -> VPT-deep n=8 | 94.04 | 78.75 | 91.31 | 84.75 | 84.21 | 85.98 |

**Nessun nuovo migliore secondo la Protocol Mean UCF.** DoRA r=16 su
VPT-deep n=8 congelato e' la migliore nuova sequenza (**85,5432%**), sotto
DoRA r=16 da sola a una epoca (**85,7487%**, -0,2055 punti) e sotto il migliore
preesistente, VPT-deep n=8 su DoRA r=32 fuso con prior val-best, E3-SV
(**85,9440%**, -0,4007 punti). Il confronto con E3-SV resta a budget diverso
(1+2 contro 1+3); anche la baseline a una epoca resta davanti. E' un esito al
solo seme 42, non una prova di significativita' fra semi. MIVIA non entra nel
criterio di scelta. L'Excel di confronto resta byte-identico: questo blocco
entra nel registro e nelle classifiche, senza aggiungere righe al foglio.

Il 1 ottobre le due celle `congela` si erano fermate per ENOSPC durante un
salvataggio; i checkpoint validi sono stati copiati e verificati su /home,
con symlink ai percorsi canonici su /shared, quindi ripresi senza aumentare
il budget. Il 2 ottobre e' stato corretto il conteggio del supervisore:
`MaxSubmitJobs=10` appartiene all'associazione utente-account e non comprende
i job di altri utenti. Restano massimo quattro job della campagna, riserva
sei e tetto prudenziale otto job del nostro utente. Nessun job altrui toccato.

Artefatti: [riepilogo e tabella](../runs/analisi_e3so_20260930/README.md),
[tutte le metriche OVAD delle quattro celle](../runs/classifiche/ovad_e3so_guard30_20261002/risultati.csv),
[classifica UCF completa](../runs/classifiche/ovad_guard30/classifiche_tutte_versioni/ucf_s42.md).
Le raccolte globali comprendono **42 modelli / 102 celle EDP** e **184
valutazioni OVAD g30 (120 UCF, 64 MIVIA)**, con 42 modelli principali e 48
versioni. Ricevuta locale: `runs/analisi_e3so_20260930/final-verification.json`.
I checkpoint restano conservati sul cluster; il rientro leggero include tutte
le figure, le metriche, gli score e i CSV per classe/video/chunk e OVAD.


---

<a id="e3sv-20260926"></a>

## E3-SV — 26–30 settembre 2026, 8/8 celle verificate

**Stato al 30 settembre:** otto training completi, tutti a **16.572 passi** e
al **seme 42**. Verificati e riportati in locale gli otto test UCF e gli otto
test MIVIA con EDP-v2, comprese figure, strati, latenza per intersezione,
CSV per classe e video, ranking dei chunk. Restano sul cluster i checkpoint;
config, metadati, score e risultati sono in `runs/{run_id}/`, con MIVIA annidata
in `runs/{run_id}/{run_id}__mivia/`. Il registry locale include le 16 esecuzioni.

Le celle ereditano il prior **selezionato per AUROC di validazione**:
passo 5520 per LoRA, DoRA e VPT-shallow, 3680 per VPT-deep. Il secondo stadio
seleziona a sua volta il checkpoint su validazione. La ricevuta della campagna
verifica budget, parametri addestrabili, prior e stato `selected`; MIVIA usa
questi pesi e la soglia EDP UCF trasferita invariata. Su ogni cella: 290 video
e 9.116 chunk UCF; 270 video e 3.485 chunk MIVIA. Nessun nuovo training e
nessuna inferenza duplicata in questa passata: le otto valutazioni MIVIA
sono state completate dal supervisore.

### EDP-v2

Valori percentuali; AUC e AP sono a livello di chunk. AUC media = media a
pari peso dei due test. I delta sono in punti percentuali contro la stessa
cella **E3-S s42**, che eredita l'ultimo checkpoint del prior. Sono confronti
a un seme, su otto celle; non sono una media a tre semi.

| Secondo stadio e prior | AUC UCF | AP UCF | AUC MIVIA | AP MIVIA | AUC media | Δ AUC media vs E3-S |
|---|---:|---:|---:|---:|---:|---:|
| VPT-shallow su LoRA fuso | 94.03 | 79.26 | 91.58 | 85.40 | 92.81 | +0.46 |
| DoRA su VPT-shallow congelato | 94.20 | 80.36 | 90.29 | 83.34 | 92.25 | +1.08 |
| VPT-deep su LoRA fuso | 93.86 | 78.71 | 90.42 | 83.37 | 92.14 | -0.04 |
| LoRA su VPT-shallow congelato | 93.79 | 78.53 | 90.48 | 83.51 | 92.14 | -0.15 |
| VPT-deep su DoRA fuso | 94.24 | 79.99 | 89.87 | 82.99 | 92.05 | +0.11 |
| VPT-shallow su DoRA fuso | 94.58 | 79.61 | 89.35 | 81.48 | 91.96 | -0.47 |
| LoRA su VPT-deep congelato | 93.02 | 75.90 | 90.47 | 83.81 | 91.74 | -0.35 |
| DoRA su VPT-deep congelato | 93.49 | 77.08 | 89.88 | 82.63 | 91.69 | +0.15 |

VPT-shallow su LoRA fuso ha AUC media **92,81%**, contro **92,35%** della sua
controparte E3-S (+0,46 punti), e guida il confronto EDP fra i modelli disponibili.
L'effetto del prior selezionato varia per cella: questo confronto riguarda un solo seme.

### OVAD, guardia 30 s

Gli stessi score sono convertiti senza alterazioni e rivalutati col codice
originale del laboratorio. Youden è ricalcolata sulle righe di validazione
UCF con **g=30 s** e trasferita a MIVIA. Stesse etichette e coorti della
raccolta preesistente: 1.425 righe giudicabili e 255 escluse su UCF, 622 e
242 su MIVIA; macro su 95 e 48 video anomali, 262 e 200 eventi giudicabili.
La media dei mean segue la definizione OVAD; non è una media di AUC.

| Secondo stadio e prior | Mean UCF | Mean MIVIA | Media dei mean | Δ media vs E3-S |
|---|---:|---:|---:|---:|
| VPT-deep su LoRA fuso | 85.13 | 85.96 | 85.54 | -0.03 |
| VPT-shallow su DoRA fuso | 85.94 | 84.69 | 85.32 | +0.43 |
| VPT-deep su DoRA fuso | 85.94 | 83.76 | 84.85 | +0.56 |
| LoRA su VPT-shallow congelato | 84.07 | 84.73 | 84.40 | -1.23 |
| VPT-shallow su LoRA fuso | 83.97 | 84.50 | 84.24 | -1.95 |
| DoRA su VPT-deep congelato | 84.48 | 83.72 | 84.10 | -0.66 |
| LoRA su VPT-deep congelato | 83.81 | 83.57 | 83.69 | -0.52 |
| DoRA su VPT-shallow congelato | 84.43 | 82.93 | 83.68 | -0.98 |



Fra i nuovi E3-SV la maggiore media dei mean è **85,54%**, VPT-deep su LoRA
fuso. La classifica OVAD complessiva resta guidata dall'E3-S storico VPT-shallow
su LoRA fuso (**86,19%**). EDP e OVAD restano raccolte distinte.

### Artefatti e verifiche

- [Classifiche EDP](../runs/classifiche/classifiche.md): **38 modelli, 98 celle**;
  sezione MIVIA al solo seme 42 e [CSV cross-dataset](../runs/classifiche/cross_dataset_mivia.csv).
- [Classifiche OVAD guardia 30 s](../runs/classifiche/ovad_guard30/classifiche/classifiche.md):
  **176 valutazioni**, 116 UCF e 60 MIVIA; 38 modelli nella raccolta principale,
  con versioni alternative conservate separatamente. Guardia 8 s conservata.
- Raccolta della sola aggiunta: `runs/classifiche/ovad_e3sv_guard30_20260930/`.
  Figure di confronto E3-SV: `runs/analisi_e3sv_20260930/`.
- [Excel](incontri/metriche_confronto-sota.xlsx): E3-SV esclusa per decisione di Antonio;
  ripristinata integralmente la copia precedente alle aggiunte del 30 settembre.
- Audit indipendente su **176** risultati OVAD, incluse tutte le soglie;
  guardia ricostruita su **292** tabelle e **543.216** righe. Passano i **16/16**
  test originali OVAD. La conversione delle nuove celle riproduce AUC/AP
  EDP con scarto **0,0**; tutte le classifiche EDP hanno **0 incoerenze**.
  Excel verificato identico al file precedente alle aggiunte E3-SV (SHA256).

Snapshot e ricevute: `../tmp/e3sv-20260930/`; chiusura della passata in
`runs/analisi_e3sv_20260930/final-verification.json`, con stato `complete` dopo la verifica della ricevuta finale del cluster. Il rientro iniziale verifica **721 file**; gli originali sono
conservati nell'archivio scaricato e nelle copie prima dell'aggiornamento.

### Chiusura dell'ottava cella e della campagna

Durante questa sessione è terminato anche il training
`e3sv_ucf_vjepa21l_dora-r32-vptdeep-n8_s42` (DoRA su VPT-deep congelato).
Il supervisore ha completato MIVIA e gli artefatti nel post-job `684212_0`;
la cella è stata riportata in locale e rivalutata su entrambi i test in OVAD g30.
Il checkpoint del secondo stadio selezionato è al passo **10.120**, il prior
VPT-deep ereditato al passo 3.680. La ricevuta finale del supervisore è
`status=complete`, **8 training / 16 test EDP**, e la verifica locale comprende
tutte le otto celle e le 16 nuove valutazioni OVAD.

Il supervisore attuale è cron su lnode02; il checkout isolato resta
`~/evd/experiments/20260926-e3sv`, snapshot
`64cd46883a54d37593933c8b7783fe55b8bd742a`. Nessun training o inferenza duplicati.
Il job CPU finale `684353_0` ha chiuso la campagna dopo la correzione del limite
fatq da 7 a 4 ore, verificata con preflight e controllo di regressione.
La suite completa resta quella da
verificare: qui passano i **64** test mirati di OVAD/classifiche e i 16 originali.

---

<a id="e3s-20260924"></a>

## E3-S — Le combinazioni sequenziali *(chiusa: in fila come insieme, e la sequenza non guadagna sul reparam da solo)*

**Data:** celle 15–22 settembre 2026, voce scritta il 24 · **ADR-065** (con le
postille della sera del 9 settembre), **F13-bis** di `PIANO.md` · **Stato:**
**chiusa**, 24/24 celle su `edp_v2`, tre semi `[42, 67, 104]`, riportate in locale
**senza checkpoint**. Campagna sul cluster in `~/evd/experiments/20260915-f13bis`
(supervisore autonomo, preferenza `aiq`), lista di lavoro e registro operativo in
`../tmp/f13bis-20260915/TODO.md`. Analisi in `runs/analisi_f13bis_20260924/`.

### La domanda

E3-C ha detto che una combinazione reparam + prompt si comporta come il suo
**reparam leggermente peggiorato**: i due prior sono **ridondanti**. Ma E3-C
addestra le due famiglie **insieme**, quindi non separa due spiegazioni: che il
guadagno manchi perché addestrarle insieme le fa **interferire**, o perché i due
prior semplicemente **non si sommano**. E3-S mette le due famiglie **in fila**, in
entrambe le direzioni:

| Direzione | Cosa succede | Celle |
|---|---|---|
| `fondi` (reparam → prompt) | LoRA/DoRA r=32 di E3-R, **finito**, si riassorbe nei pesi **prima** del run; si addestra il solo VPT n=8 | `e3s_ucf_vjepa21l_vpt{deep,shallow}-n8-su-{lora,dora}-r32_s*` |
| `congela` (prompt → reparam) | i prompt VPT n=8 di E3-P si caricano e si **fermano**; si addestra il solo LoRA/DoRA r=32, da zero | `e3s_ucf_vjepa21l_{lora,dora}-r32-vpt{deep,shallow}-n8_s*` |

Budget: **16.572 passi** (tre cicli del coseno di periodo 5524), lo stesso di
E3-C, perché il confronto che E3-S deve reggere è **E3-S contro E3-C**. Le
sorgenti sono le celle a una epoca di E3-R (`lora/dora-r32-qkvomlp`) ed E3-P
(`vptdeep/vptshallow-n8`) **allo stesso seme**.

### Le verifiche, fatte prima di leggere i numeri

Nessuna è formale; sono in `runs/analisi_f13bis_20260924/verifica.json`.

- **24/24 complete**: `steps == max_steps == 16572`, `stopped_early` falso ovunque.
- **`trainable_params_adapter` è quello della sola famiglia addestrata**:
  14.155.776 (LoRA) e 14.376.960 (DoRA) sulle `congela`, 196.608 (VPT-deep n=8) e
  8.192 (VPT-shallow n=8) sulle `fondi`. È il controllo che dice che il
  congelamento ha morso (ADR-065) e che il reparam fuso non è stato contato come
  addestrabile.
- **Ogni cella eredita dal run giusto e al seme giusto**: la ricevuta
  `preadattamento` di `meta.json` nomina la sorgente e il passo, e sulla `fondi`
  con DoRA a seme 42 riporta **432 tensori fusi e 96 wrapper tolti**.
- **AUC, AP e τ ricalcolate da `scores.jsonl` coincidono al bit** con
  `metrics.json` su tutte e 24 (scarto 0,0 su 9116 finestre di test e 8383 di
  validazione).

⚠️ **Il prior ereditato è quello dell'ULTIMO passo del run sorgente, non del
passo selezionato.** Tutte le celle ereditano lo stato al passo 5524, mentre la
selezione di quel run aveva scelto un passo precedente: lo scarto va da **4**
passi (niente: 1e-10 sui pesi) a **3.684** (`vptdeep-n8_s67`, selezionato al
1840, fino a 1,32e-2 sui pesi). Trovato il 17 settembre a campagna al 42%, e
**decisione di Antonio: continuare e dichiarare**. Il difetto ha **un verso
solo** — una cella che guadagna è solida a maggior ragione, una che non guadagna
non separa «la sequenza non aiuta» da «il prior ereditato era sciupato» — e morde
soprattutto le `congela` su **VPT-deep**, la famiglia che E3-P ha documentato come
sovradattata.

### Cosa è uscito — test di UCF-Crime, tre semi

Media ± σ campionaria sui tre semi; le righe di riferimento sono le celle a
**una** epoca della matrice e le combinazioni joint di E3-C.

| Coppia | Configurazione | chunk-AUC | chunk-AP | AUC_a | AP_a |
|---|---|---:|---:|---:|---:|
| LoRA r=32 + VPT-deep n=8 | E3-C joint | 0,9387 ± 0,0068 | 0,7814 ± 0,0109 | 0,8563 ± 0,0052 | 0,8141 ± 0,0034 |
| | `congela` (LoRA su VPT congelato) | 0,9338 ± 0,0064 | 0,7680 ± 0,0301 | 0,8549 ± 0,0087 | 0,8089 ± 0,0215 |
| | `fondi` (VPT su LoRA fuso) | 0,9383 ± 0,0018 | 0,7839 ± 0,0095 | 0,8683 ± 0,0055 | 0,8283 ± 0,0014 |
| LoRA r=32 + VPT-shallow n=8 | E3-C joint | 0,9350 ± 0,0052 | 0,7740 ± 0,0190 | 0,8569 ± 0,0098 | 0,8189 ± 0,0154 |
| | `congela` | 0,9364 ± 0,0061 | 0,7804 ± 0,0171 | 0,8594 ± 0,0078 | 0,8225 ± 0,0045 |
| | `fondi` | 0,9373 ± 0,0049 | 0,7887 ± 0,0155 | 0,8634 ± 0,0099 | 0,8292 ± 0,0107 |
| DoRA r=32 + VPT-deep n=8 | E3-C joint | 0,9354 ± 0,0011 | 0,7785 ± 0,0085 | 0,8520 ± 0,0131 | 0,8171 ± 0,0188 |
| | `congela` | 0,9321 ± 0,0048 | 0,7697 ± 0,0284 | 0,8580 ± 0,0063 | 0,8203 ± 0,0085 |
| | `fondi` | 0,9384 ± 0,0022 | 0,7857 ± 0,0106 | 0,8653 ± 0,0108 | 0,8272 ± 0,0158 |
| DoRA r=32 + VPT-shallow n=8 | E3-C joint | 0,9405 ± 0,0016 | 0,7914 ± 0,0037 | 0,8674 ± 0,0031 | 0,8313 ± 0,0026 |
| | `congela` | 0,9354 ± 0,0031 | 0,7864 ± 0,0043 | 0,8606 ± 0,0082 | 0,8298 ± 0,0050 |
| | `fondi` | 0,9397 ± 0,0080 | 0,7915 ± 0,0186 | 0,8653 ± 0,0136 | 0,8290 ± 0,0127 |
| *riferimenti, 1 epoca* | LoRA r=32 | 0,9392 ± 0,0015 | 0,7889 ± 0,0055 | 0,8627 ± 0,0027 | 0,8270 ± 0,0059 |
| | DoRA r=32 | **0,9422 ± 0,0017** | **0,7947 ± 0,0037** | 0,8669 ± 0,0032 | 0,8305 ± 0,0037 |
| | VPT-deep n=8 | 0,9317 ± 0,0024 | 0,7622 ± 0,0035 | 0,8382 ± 0,0019 | 0,7882 ± 0,0014 |
| | VPT-shallow n=8 | 0,9360 ± 0,0030 | 0,7690 ± 0,0081 | 0,8378 ± 0,0068 | 0,7901 ± 0,0060 |

Le metriche a soglia (chunk, video, evento) e la latenza di griglia delle otto
righe stanno in `runs/classifiche/classifiche.md` §1, sotto i nomi *«VPT-… on
fused LoRA/DoRA r=32»* e *«LoRA/DoRA r=32 on frozen VPT-…»*, famiglia
`sequenziale`.

**Il confronto appaiato** — delta **per seme**, t appaiato a 2 gradi di libertà,
soglia bilaterale al 5% |t| > 4,303 (`confronto_appaiato.csv`, 168 righe):

| Cella E3-S | Δ chunk-AUC contro E3-C | Δ chunk-AUC contro il reparam da solo | Δ AP_a contro il prompt da solo |
|---|---:|---:|---:|
| LoRA su VPT-deep congelato | −0,0050 (t −0,7) | −0,0054 (t −1,8; 0/3 sopra) | +0,0207 (t 1,7) |
| VPT-deep su LoRA fuso | −0,0004 (t −0,1) | −0,0009 (t −0,6) | **+0,0401 (t 75,7)** |
| LoRA su VPT-shallow congelato | +0,0014 (t 0,4) | −0,0028 (t −0,7) | **+0,0323 (t 7,0)** |
| VPT-shallow su LoRA fuso | **+0,0023 (t 12,0)** | −0,0019 (t −0,9) | **+0,0390 (t 14,0)** |
| DoRA su VPT-deep congelato | −0,0032 (t −1,0) | −0,0101 (t −4,2; 0/3 sopra) | **+0,0321 (t 7,6)** |
| VPT-deep su DoRA fuso | +0,0030 (t 1,7; 3/3 sopra) | −0,0038 (t −1,7; 0/3 sopra) | +0,0390 (t 4,1) |
| DoRA su VPT-shallow congelato | −0,0051 (t −2,1; 0/3 sopra) | −0,0068 (t −2,6; 0/3 sopra) | **+0,0397 (t 8,8)** |
| VPT-shallow su DoRA fuso | −0,0008 (t −0,2) | −0,0025 (t −0,6) | **+0,0389 (t 10,0)** |

In grassetto i confronti che separano. Tre letture, e nessuna ha bisogno di
un'ipotesi sul meccanismo.

1. **Nessuna sequenza batte il proprio reparam da solo.** Contro LoRA o DoRA a
   **una** epoca, il delta medio di chunk-AUC è negativo **8 volte su 8**, e così
   quello di chunk-AP; nessuno dei sedici separa, e gli unici confronti contro il
   reparam da solo che separano, su tutte e sei le metriche del CSV, sono
   **negativi** (AUC_a di DoRA su VPT-deep congelato, −0,0089, t −4,8, più due F1
   di video e una di evento). Aggiungere prompt a un reparam, prima o
   dopo, **non compra niente** — e per la direzione `fondi` vale anche a costo
   di tre cicli in più sopra un reparam già finito.
2. **In fila o insieme fa lo stesso.** Contro la combinazione joint E3-C della
   stessa coppia, a parità di budget e di componenti, separano **2 righe su 32**
   sulle quattro metriche senza soglia (+0,0023 di chunk-AUC e +0,0142 di AP_a,
   entrambe su una `fondi` con LoRA). Quindi la risposta alla domanda di E3-S è che
   **la ridondanza non è un effetto dell'addestramento congiunto**: i due prior non
   si sommano nemmeno quando non possono interferire.
3. **Il guadagno sopra il prompt è il reparam.** Contro il VPT da solo AUC_a e AP_a
   salgono su **8/8** celle, di +0,017…+0,030 e +0,021…+0,040, e separano 5 e 6
   volte su 8: è dello stesso ordine di ciò che il reparam da solo aggiunge sopra
   la probe congelata (+0,040 e +0,043 di AP_a, Tavola A-bis), cioè di nuovo il
   guadagno **dentro il video anomalo**.

**Le due direzioni fra loro**: `fondi` sta sopra `congela` su chunk-AUC, chunk-AP
e AUC_a **4 volte su 4** (chunk-AUC da +0,0009 a +0,0062), e separa **una volta
sola** (AUC_a di VPT-deep su LoRA, +0,0134, t 7,2). ⚠️ La direzione è coerente con
due spiegazioni che questo disegno **non separa**: in `fondi` il reparam è quello
della matrice, selezionato tardi e quindi quasi integro, mentre in `congela` si
riaddestra da zero sopra prompt ereditati all'ultimo passo — che su VPT-deep è il
prior «sciupato» dell'avvertenza sopra.

⚠️ **La cella singola più alta di tutte le 108 è una E3-S, e non va letta come il
modello migliore.** `vptshallow-n8-su-dora-r32_s42` fa **0,9477** di chunk-AUC —
più di qualunque altra cella della matrice — ma la sua media sui tre semi è
0,9397 ± 0,0080, cioè **sotto** DoRA r=32 da solo, con la σ più grande fra i
modelli in testa. È un massimo di seme. Lo stesso vale per qualunque confronto
fatto **al solo seme 42**, dove quella cella risulterebbe prima.

### MIVIA-Crime: tre semi, e le `fondi` non crollano *(corretta il 25 settembre)*

⚠️ **Fino al 25 settembre questa sottosezione diceva che le `fondi` non valevano**,
ed era vero: la valutazione esterna passa da `exp/demo.motore_live`, che non
rileggeva `preadattamento`, quindi il reparam della sorgente non veniva fuso e i
prompt giravano sul backbone nudo (`docs/09` R24). Il «crollo» di −0,120…−0,131
che E-MIVIA.2 leggeva come risultato controllato era **interamente** quello.

**La correzione** (commit `407d13d`): la traduzione config → backbone preadattato
diventa `preadapt.prepara_backbone()`, con due chiamanti — il runner e
`motore_live` — e la ricevuta della fusione finisce nel `meta.json` della
valutazione esterna (su DoRA: 432 tensori, 96 wrapper tolti). Le celle rifatte
sono **12**: oltre alle quattro a s42, la campagna `20260923-mivia-e3s` aveva
valutato con lo stesso difetto anche i semi 67 e 104, mai riportati in locale.
Job 569258–569260 e 569334 su H100, dal worktree `~/evd/experiments/20260925-mivia-fondi/`.

**La verifica che rende credibile la correzione, fatta prima di misurare**: per
ogni cella il modello ricostruito rifà il forward su tre video di test di
UCF-Crime (15 finestre) e si confronta con `scores.jsonl`, che il runner ha
scritto col modello addestrato. Risultato: **identico al bit** sulle 9 celle
valutate in origine su H100; entro **1,1e-2** di massimo e **1,3e-3** di media
sulle 3 valutate su A100 (`gnode06`) — è la differenza bf16 fra due GPU, e cade
esattamente su quelle tre. La ricostruzione vecchia, sulle stesse finestre,
sbaglia di **0,38–0,41 in media** e fino a 0,996. La stessa verifica sulle 8
`congela` s67/s104 dà **0,0** ovunque: i prompt congelati stanno nello stato
dell'adattatore e si ricaricano, quindi le loro valutazioni del 23 settembre
valgono. Ricevute in `operations/verifica_ucf*.jsonl` della campagna.

⚠️ **La prima soglia della verifica era troppo stretta**, e va detto: chiedeva
uno scarto massimo sotto 1e-2 e ha scartato `vptdeep-n8-su-lora-r32_s104`
(1,07e-2 su una finestra, media 7,4e-4), che era giusta. Il criterio è passato a
media < 5e-3, massimo < 5e-2 e ricostruzione vecchia almeno 50 volte più
lontana, e la cella è stata rifatta (job 569334).

**Cosa è uscito — chunk-AUC su MIVIA-Crime, tre semi** (media ± σ; UCF-Crime agli
stessi semi; delta appaiato sul seme):

| Cella E3-S | MIVIA-Crime | UCF-Crime | Δ | AUC_a MIVIA | solo s42 |
|---|---:|---:|---:|---:|---:|
| VPT-shallow n=8 su DoRA fuso | 0,9039 ± 0,0045 | 0,9397 | −0,0358 ± 0,0095 | 0,8381 | 0,9009 |
| VPT-deep n=8 su DoRA fuso | 0,9034 ± 0,0006 | 0,9384 | −0,0350 ± 0,0018 | 0,8334 | 0,9028 |
| VPT-deep n=8 su LoRA fuso | 0,9028 ± 0,0019 | 0,9383 | −0,0355 ± 0,0026 | 0,8333 | 0,9032 |
| VPT-shallow n=8 su LoRA fuso | 0,8997 ± 0,0147 | 0,9373 | −0,0376 ± 0,0105 | 0,8362 | 0,9056 |
| LoRA su VPT-deep congelato | 0,9022 ± 0,0083 | 0,9338 | −0,0316 ± 0,0035 | 0,8327 | 0,9040 |
| DoRA su VPT-shallow congelato | 0,9017 ± 0,0090 | 0,9354 | −0,0337 ± 0,0061 | 0,8289 | 0,8913 |
| DoRA su VPT-deep congelato | 0,8936 ± 0,0065 | 0,9321 | −0,0385 ± 0,0049 | 0,8198 | 0,8942 |
| LoRA su VPT-shallow congelato | 0,8925 ± 0,0107 | 0,9364 | −0,0438 ± 0,0082 | 0,8207 | 0,9025 |

Tre letture, e nessuna è quella che la versione sbagliata suggeriva.

1. **Le due direzioni perdono allo stesso modo.** Delta da −0,032 a −0,044,
   dentro la tassa di dominio del resto della matrice (mediana −0,039 su 36
   celle a s42, E-MIVIA.2): la direzione di addestramento **non** decide come il
   modello si trasferisce.
2. **`fondi` resta sopra `congela`, come su UCF-Crime**, 4 coppie su 4
   (+0,0006 … +0,0098 di chunk-AUC su MIVIA) — ma a tre semi e con σ fino a
   0,015 **non separa**, e la confusione col prior ereditato all'ultimo passo
   (sopra) vale anche qui.
3. **Nessuna sequenza fa meglio del suo reparam da solo anche fuori casa**: DoRA
   r=32 e LoRA r=32 a s42 fanno 0,9029 e 0,9074; le `fondi` a s42 0,9009–0,9056.
   È la lettura 1 di UCF-Crime, confermata su un dominio mai visto — con il
   limite che per il reparam da solo MIVIA esiste a un seme.

⚠️ **E3-S è l'unica famiglia con MIVIA a tre semi**, e i tre semi stanno **solo
qui**: `runs/classifiche/classifiche.md` §6 riporta il **seme 42 per tutti** i
modelli, come richiesto (`classifiche_modelli.py --semi-mivia`, default 42), perché
una `Mean AUC` a tre semi non si confronta con una a un seme. Nella §6b, la
classifica per AUC media con cui si sceglie,
`vptshallow-n8-su-dora-r32` è **prima** (0,9243: 0,9477 in casa, 0,9009 fuori),
davanti a DoRA r=32 (0,9235) e VPT-shallow su LoRA fuso (0,9234): **tre modelli
in 0,001**, a un seme solo. *(Fino alla sera del 25 settembre era quarto anche
LoRA r=32 a una epoca, 0,9234; da quando le classifiche riportano LoRA r=32 a tre
epoche — [postilla di E3-M](#versione-riportata-20260925) — quella riga fa 0,9226 ed è quinta, dietro
LoRA r=32 su VPT-shallow congelato, 0,9229.)* ⚠️ Vale l'avvertenza
qui sopra sul massimo di seme: a tre semi la stessa cella fa 0,9218 di media fra
i due dataset.

### Cosa concludono, e cosa no

**Concludono**: che su questo backbone comporre prompt e reparam non guadagna **né
insieme né in fila**, in nessuna delle due direzioni, con tre semi e con i
confronti appaiati — il reparam porta il risultato, il prompt non aggiunge; e che
quindi il verdetto di E3 regge anche alla composizione.

**Non concludono** — e si scrive: (a) nulla sul **meccanismo** della ridondanza;
(b) nulla che separi `fondi` da `congela` in modo netto, perché l'unica
differenza sistematica è confusa col difetto del prior ereditato; (c) nulla sul
**costo di inferenza** delle otto celle, che E5b non ha profilato — strutturalmente
una `fondi` costa quanto il suo VPT sopra un backbone fuso e una `congela` quanto
la combinazione a fusione parziale (ADR-064), ma nessuna delle due righe è
misurata; (d) nulla sul **calcolo totale**, che è asimmetrico: dietro una cella
E3-S c'è anche l'epoca del run sorgente; (e) nulla sulla scelta del modello «migliore» fra E3-S e il reparam da solo: su MIVIA-Crime si separano di millesimi, a un seme per il reparam.
⚠️ Come E3-C, **non è a parità di parametri**: la sequenza conta i parametri della
sola famiglia addestrata, ma il modello in esercizio porta anche l'altra.

⚠️ **Un'indicazione del prof non seguita, e non dichiarata finora.** Nell'incontro
del 7 settembre (`docs/incontri/incontro_pre-domanda30gg_07-09.md`, «Altri
esperimenti», punto 2) la richiesta delle combinazioni sequenziali viene con
«usare un lr più basso (a detta del prof è una linea promettente)». Le 24 celle
girano invece con la **LR della matrice**, `1e-4` (`metrics.json`: `lr` 0,0001,
«griglia a un valore»), e né ADR-065 né `docs/05` ne parlano. La scelta ha una
ragione difendibile — la stessa LR di E3-C è ciò che rende E3-S confrontabile con
E3-C — ma è una **decisione da chiedere ad Antonio** e, se resta, da dichiarare in
tesi come deviazione da un'indicazione dei supervisori; il risultato negativo di
E3-S **non** copre il caso a LR più bassa.

### Alimenta

C7 (perché comporre le due famiglie non paga, e perché non è un problema di
ottimizzazione), il capitolo 4 dei risultati accanto a E3-C, e la lista dei limiti
(prior ereditato all'ultimo passo; MIVIA-Crime a tre semi per E3-S e a uno per il reparam da solo).

**Artefatti**: `runs/e3s_ucf_vjepa21l_*_s{42,67,104}/` (metriche, score, figure,
`strata.json`, `latenza.json`, `per_class.csv`, `ranking_chunks*.csv`), le
valutazioni esterne in `runs/{run_id}/{run_id}__mivia/`, e
`runs/analisi_f13bis_20260924/` (`verifica.json`, `analisi_e3s.json`,
`confronto_appaiato.csv`, `LEGGIMI.md`, e lo script che li produce).

---

<a id="e4-sota-20260924"></a>

## E4 — Il confronto con lo stato dell'arte, sulle tabelle del laboratorio *(eseguito sul modello migliore: +8,9 punti di chunk-AUC su UCF-Crime e +8,6 su MIVIA-Crime sul migliore dei metodi forniti)*

**Data:** tabelle consegnate il **17 settembre 2026**, confronto preparato il 18
e il 23, voce scritta il 24 · **Stato:** **eseguito** sul modello DoRA r=32 a seme
42; ⚠️ la soglia F1-max per il nostro modello **non è calcolata**. Incontri:
`docs/incontri/incontro_baseline-sota_15-09.md` (il prof annuncia la tabella: «appena
li ricevo devo inviare al prof il confronto») e
`docs/incontri/incontro_explainability-selezione_17-09.md` (la consegna: «le
metriche da inserire sono solo quelle che abbiamo calcolato»).

### La domanda, e perché non è E4a

`docs/05` §6 prevedeva **E4a**: numeri presi dai paper (RTFM, VadCLIP) al loro
regime weakly-supervised, con caveat di protocollo. Il laboratorio ha fatto di
meglio: ha fornito i risultati di **sei metodi valutati con il nostro stesso
protocollo** — finestre da 8 s, passo 4 s, soglie calibrate sulla validazione di
FS-UCF-Crime (v2) e tenute fisse su MIVIA-Crime — quindi il confronto è a
**geometria e partizione identiche**, cioè quello che E4a non poteva essere.

**Sorgenti.** Le tabelle del laboratorio, trascritte così come consegnate:
`docs/incontri/ucf_proposed_vs_vadclip_8s.md` (UCF-Crime, due metodi) e
`docs/incontri/sota_mivia_8s.md` (MIVIA-Crime, sei metodi), ciascuna a soglia di
**Youden** e a soglia **F1-max**. I nostri numeri vengono da
`runs/e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42/metrics.json` e dalla sua valutazione
esterna `…/e3r_…_s42__mivia/metrics.json`. Il foglio che li affianca è
`docs/incontri/confronto_sota_modello-migliore_20260918.xlsx`, **generato** da
`scripts/confronto_sota_xlsx.py`.

### Cosa è uscito — soglia di Youden, valori in percentuale

In grassetto il migliore di ogni colonna; la riga Δ è il nostro valore meno il
migliore fra gli altri metodi.

**FS-UCF-Crime, test (290 video, 9116 chunk)**

| Metodo | AUC | AP | AUC_a | AP_a | P | R | F1 | R_evt | video P | video R | video F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VadCLIP | 85,52 | 49,34 | 60,71 | 51,36 | 41,54 | 75,25 | 53,53 | 84,33 | 84,25 | 87,86 | 86,01 |
| Proposed (laboratorio) | 84,19 | 56,04 | 66,03 | 60,66 | 34,06 | **85,45** | 48,70 | 87,00 | 76,07 | 88,57 | 81,85 |
| **Nostro — V-JEPA 2.1-L + DoRA r=32, s42** | **94,41** | **79,71** | **87,04** | **83,00** | **58,66** | 82,90 | **68,70** | **92,75** | **91,84** | **96,43** | **94,08** |
| Δ sul migliore della colonna | +8,89 | +23,67 | +21,01 | +22,34 | +17,12 | −2,55 | +15,17 | +5,75 | +7,59 | +7,86 | +8,07 |

**MIVIA-Crime, test (270 video, 3485 chunk) — soglia trasferita da FS-UCF-Crime**

| Metodo | AUC | AP | AUC_a | AP_a | P | R | F1 | R_evt | video P | video R | video F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposed (laboratorio) | 81,69 | 64,06 | 67,45 | 72,46 | 55,92 | **74,00** | 63,70 | 80,80 | 71,79 | 82,96 | 76,98 |
| PEL4VAD | 65,84 | 43,57 | 56,41 | 62,19 | 44,05 | 52,10 | 47,74 | 73,19 | 68,71 | 74,81 | 71,63 |
| GlanceVAD | 72,13 | 54,41 | 63,26 | 69,07 | 52,66 | 48,84 | 50,68 | 73,71 | 72,73 | 77,04 | 74,82 |
| UR-DMU | 68,10 | 47,58 | 56,77 | 62,80 | 44,55 | 58,25 | 50,48 | **82,11** | 66,28 | 84,44 | 74,27 |
| Holmes-VAU | 57,39 | 34,82 | 56,53 | 60,96 | 35,34 | 53,49 | 42,57 | 73,90 | 56,35 | 75,56 | 64,56 |
| VadCLIP | 79,90 | 63,60 | 66,08 | 73,05 | 59,05 | 55,64 | 57,29 | 67,67 | 72,73 | 71,11 | 71,91 |
| **Nostro — DoRA r=32, s42** | **90,29** | **83,35** | **82,93** | **88,13** | **80,19** | 64,79 | **71,67** | 75,61 | **85,51** | **87,41** | **86,45** |
| Δ sul migliore della colonna | +8,60 | +19,29 | +15,48 | +15,08 | +21,14 | −9,21 | +7,97 | −6,50 | +12,78 | +2,97 | +9,47 |

### Cosa dicono

1. **Sulle metriche indipendenti dalla soglia il vantaggio è netto su entrambi i
   dataset**: +8,9 e +8,6 punti di chunk-AUC sul migliore dei metodi forniti, e
   ancora di più sulle metriche **dentro il video anomalo** (AUC_a +21,0 e +15,5,
   AP_a +22,3 e +15,1). È il margine che conta, perché non dipende da come ciascun
   metodo sceglie la propria soglia. Rispetto alla dispersione fra semi del nostro
   modello (σ di chunk-AUC **0,0017** su UCF-Crime) è due ordini di grandezza più
   grande: non dipende dal seme 42, e la media sui tre semi (94,22) direbbe la
   stessa cosa.
2. **Il confronto più pertinente è col metodo del laboratorio**, perché è l'unico
   che condivide anche il **regime di supervisione** — fully supervised sugli
   intervalli di FS-UCF-Crime — oltre al protocollo: +10,2 punti di chunk-AUC su
   UCF-Crime e +8,6 su MIVIA-Crime.
3. ⚠️ **Su MIVIA-Crime, alla soglia trasferita, il nostro sistema sotto-allarma**:
   è il più preciso (P 80,19) e **non** il più sensibile — recall di chunk 64,79
   contro 74,00 del metodo del laboratorio, recall di evento 75,61 contro 80,80 e
   82,11 (UR-DMU). È la stessa lettura di E-MIVIA: l'**ordinamento** si trasferisce,
   la **calibrazione** meno, e τ trasferita è troppo alta per quel dominio. Su
   UCF-Crime l'unica colonna persa è il recall di chunk (−2,55), a favore di una
   precision più alta di 24,6 punti.

### Le sei avvertenze che viaggiano con questa tabella

1. ⚠️ **I numeri degli altri metodi sono del laboratorio e non sono stati
   riprodotti qui.** Il documento dichiara solo che VadCLIP usa i **pesi
   ufficiali**; per PEL4VAD, GlanceVAD, UR-DMU e Holmes-VAU **il regime di
   addestramento e i pesi non sono dichiarati** — ⚠️ DA VERIFICARE col laboratorio
   prima di scrivere una frase sul perché del divario. Nei loro lavori originali
   sono per lo più metodi *weakly supervised*, e il nostro è fully supervised a
   livello temporale: parte del vantaggio può venire dalla **supervisione**, non
   dal backbone, e va detto (è il caveat che `docs/05` E4a già dichiarava).
2. ⚠️ **«Proposed» non ha una citazione nelle tabelle.** È il metodo del
   laboratorio — plausibilmente quello del lavoro che accompagna le annotazioni
   FS-UCF-Crime (Carletti, Greco, Marseglia, Vento, *Online Fully Supervised Video
   Anomaly Detection*, sottomesso) — ⚠️ DA VERIFICARE col prof prima di citarlo per
   nome.
3. ⚠️ **Il livello evento non è confrontabile**: il nostro recall di evento usa la
   definizione **nostra** (ADR-048, `docs/01` §3bis D10), quella del laboratorio
   non è documentata nelle tabelle. La colonna `R_evt` si riporta, ma non si
   commenta come un confronto.
4. ⚠️ **Le soglie non sono confrontabili fra metodi**: ogni metodo ha la propria
   scala di score (la nostra τ è 0,0863, quella del laboratorio 0,0054, quella di
   VadCLIP 0,7069). Per questo il confronto che regge è quello senza soglia.
5. ⚠️ **Mancano delle colonne, per scelta**: BA, P_a, F1_a e video BA del
   laboratorio non le calcoliamo e sono omesse; la tabella a soglia **F1-max** per
   il nostro modello non è calcolata («non ancora calcolato» nel foglio).
6. ⚠️ **Il modello confrontato è il migliore su UCF-Crime a tre semi**, scelto
   **prima** di vedere MIVIA-Crime; il criterio proposto il 17 settembre — scegliere
   il modello che va meglio **congiuntamente** sui due test set — dal 25 settembre
   si può applicare (R24 corretto, §E3-S): a seme 42 per tutti
   (`runs/classifiche/classifiche.md` §6b) il primo è VPT-shallow n=8 su DoRA
   fuso con **0,9243** di AUC media, DoRA r=32 è secondo con **0,9235**, e i primi
   tre stanno in 0,001 — cioè il criterio **non separa** il modello
   confrontato qui dai suoi vicini, e a un seme non potrebbe. Il leakage dello split
   (`docs/01` §3bis D9) riguarda la partizione di UCF-Crime, che il laboratorio usa
   identica; **MIVIA-Crime non ha video byte-identici a UCF-Crime** (verificato il
   24 settembre sugli sha256 dei due inventari: 0 su 270).

⚠️ **Il foglio `docs/incontri/metriche_confronto-sota.xlsx`** (23 settembre, rifatto
il 25) mette accanto alle tabelle del laboratorio **30 modelli** a seme 42 sui due
dataset — una versione per ciascuno dei sei addestrati a una e a tre epoche, la
stessa delle classifiche ([postilla di E3-M](#versione-riportata-20260925)) — più due tabelle per classe
del solo DoRA r=32 (appese da `scripts/metriche_per_classe_xlsx.py`). È **mantenuto a
mano**, quindi per la tesi fa fede la sua sorgente (`runs/classifiche/`); i valori
sono a precisione piena e il foglio ne mostra due decimali. Le quattro righe MIVIA
`fondi` sono quelle corrette del 25 settembre (R24).

### Alimenta

Il capitolo 4 (posizionamento rispetto alla letteratura — la parte della PdD «confronto
tra configurazioni e con metodi esistenti») e la risposta alla prima domanda della
tesi: un backbone video-nativo self-supervised congelato e adattato con PEFT batte,
**sullo stesso protocollo**, i metodi che il laboratorio usa come riferimento, su
entrambi i dataset. ⚠️ La frase va scritta con l'avvertenza 1 accanto.

---

<a id="f14-explain-20260924"></a>

## F14 — L'explainability del detector *(campagne eseguite; la lettura quantitativa dei metodi NON è fatta)*

**Data:** 10–24 settembre 2026 · **ADR-063, ADR-067 … ADR-074** · **F14** di
`PIANO.md` · **Stato:** calcolo e rendering **eseguiti** sulle popolazioni sotto;
⚠️ **nessun metodo è stato raccomandato**, le curve deletion/insertion **non sono
state lette**, e l'ablazione VTCD a griglia è **in corso**. Questa voce registra
cosa è girato e cosa i numeri già autorizzano; **metodi, formule, protocollo e
limiti per esteso stanno in [`docs/17`](17_explainability_per_la_tesi.md)**.

### La domanda, e chi l'ha posta

L'incontro del 9 settembre (`docs/incontri/incontro-presentazione_09-09.md`) chiede
una rappresentazione visiva di **dove il modello guarda** per decidere, sul
migliore e peggiore video di ogni classe, due casuali per classe e quattro
normali, «per capire cosa comporta gli errori e da questo cercare di migliorare il
modello». Il 10 settembre (`incontro_explainability_10-09.md`) l'assistente del
prof giudica le mappe di attenzione del pooler **non informative** e chiede di
sperimentare altri metodi; il 17 (`incontro_explainability-selezione_17-09.md`),
guardando i video, indica **Integrated Gradients** e la direzione **TCAV-style**
come i più promettenti e **VTCD per classe** come promettente per la
localizzazione spaziale; il 22 (`incotro_explainability-boundingbox_22-09.md`)
chiede i **bounding box** attorno al concetto VTCD 0 sui soli frame annotati.

### Cosa è stato eseguito

Due modelli, entrambi a seme 42 e coi pesi **selezionati**: **DoRA r=32**
(`e3r_…_dora-r32-qkvomlp_s42`, passo 5520) e la **probe attentiva congelata**
(`e2d_…_frozen-attentive_s42`).

| Data | Blocco | Cosa | Dove, quanto |
|---|---|---|---|
| 10 set | sviluppo | i dieci metodi che **leggono** il modello (raw attention, Gradient × Activation dopo e prima della LayerNorm, Attention × Gradient, PCA-RGB, EigenCAM, IG con due riferimenti, LeGrad a 1 e 4 livelli, coseno) su **24 finestre** di validazione per modello — 6 TP, 6 TN, 6 FP, 6 FN su 22 video | RTX 4060 locale, ~19 s per finestra con DoRA |
| 12 set | concetti | **Anomaly Direction** (TCAV-style) e **VTCD** sul layer finale, 75 finestre per modello (sviluppo + Arson016 + Fighting047) | locale |
| 14 set | perturbazioni | **occlusione** a 45 maschere × 3 riempimenti, **freeze locale** a 45 maschere, **7 ablazioni temporali globali**; 187 forward completi per finestra, 150 finestre | cluster, 62,5 min di GPU |
| 15 set | corpus e selezione | **corpus**: i **279 video anomali** di validazione e test, interi, con 13 mappe per finestra (raw attention, Gradient × Activation, IG a 5 baseline + ensemble, Anomaly Direction ×2, **Anomaly Prototypes** ×3) — 5.494 finestre per modello; **selezione**: i **46 video** della scheda qualitativa, interi, 1.351 finestre per modello, con in più la famiglia perturbativa, le **viste perturbate** (11 ingressi × 13 mappe = 143 mappe per finestra) e le curve **deletion/insertion** | cluster, **13.690 finestre**, ~**159 h** di GPU su venti shard |
| 17–19 set | concetti per classe | `vtcd_concepts` (VTCD senza CRIS) su corpus e selezione, `class_direction` (**13 CAV**, uno per classe), `vtcd_class` (13 dizionari) sulla selezione DoRA | cluster |
| 20 set | rigenerazione | riferimento TN di IG ricalcolato (43 video col TN interno, 3 col globale), IG-TN ed ensemble rifatti anche sulle viste, con le loro due curve | cluster |
| 20–21 set | ablazione di un concetto | concetto VTCD **0** coperto di nero e IG ricalcolata sull'ingresso oscurato; **quadri 2×2** (originale · IG · concetto a nero · IG) | cluster, 46 video DoRA |
| 22 set | bounding box + IG top-10% | **ADR-072**: box attorno al concetto 0 e al primo 10% di `ig_ensemble`, sul video intero e sui soli frame annotati; vista `ig_top10_black` | `fatq` 6 min + 3 A100, 1.351 finestre |
| 22 set | ensemble a cinque | **ADR-073**: `blur` entra nell'ensemble IG; 44.811 mappe e 19.432 artefatti rifatti | `fatq`, 1 h, nessuna GPU |
| 23–24 set | ablazione VTCD | **ADR-074**, poi estesa a griglia: riferimenti 52/104/208 × regioni 16/32/64 × concetti 4/8/16 (27 celle) su **57 video**, 1.475 finestre per cella | ⚠️ **in corso** |

Tutto è consultabile dal catalogo `runs/analisi_explain/campagna_20260910/index.html`
(755 campagne, 307 pagine al 22 settembre). ⚠️ **In locale la raccolta è
metadati e ricevute**: i filmati del corpus e gli array delle mappe e delle curve
stanno sul cluster (`tmp/explain-20260915/apri-remoto.ps1`).

### Controlli di fedeltà — tutti a zero, ed è la condizione per leggere il resto

- Gli **hook** che estraggono attenzione e gradienti lasciano il logit
  **invariato** su tutte le 48 finestre di sviluppo (delta massimo zero).
- **Due forward identici** sulla clip integra danno lo stesso logit su tutte le
  150 finestre perturbative (`detector_repeat_delta` 0,000000); fra GPU locale e
  cluster lo scarto ha mediana 0,0156 e massimo 0,1094 di logit, e **nessuna
  finestra cambia lato della soglia**.
- L'ingresso perturbato **ricostruito** dalla maschera salvata riproduce **esattamente**
  il logit che l'occlusione aveva misurato (`logit_vs_occlusione` 0,0).
- **IG** soddisfa la *completeness* in 92 combinazioni finestra/riferimento su 96
  sullo sviluppo; sulle 682 finestre della selezione contate da ADR-073 **non
  converge** 34 volte con `blur` (5,0%), 74 con `noise` (10,9%) e 47 con
  `tn_same_video` (6,9%), e i casi si marcano invece di essere scartati.
- La mappa di attenzione è quella che il pooler **ha calcolato**: ricalcolando
  l'uscita dai pesi estratti lo scarto è ~1e-03 in bf16 e ~6e-07 in fp32 (ADR-063).

### Cosa i numeri già dicono — diagnostica, non protocollo

1. **L'attenzione del pooler è quasi uniforme, ed è un fatto del modello**
   (ADR-063, su `Arson016`, DoRA): pesi fra 0,60× e 1,70× il valore uniforme,
   entropia normalizzata **0,9985–0,9990**, il primo 1% dei token porta l'1,26%
   della massa contro l'1,00% di una mappa piatta. Ciò che dipende dal contenuto è
   lo **scarto** dalla mappa media del video (massimo 3,6 σ). È la ragione per cui
   la raw attention è stata giudicata non informativa, e va scritta come misura,
   non come impressione.
2. **Il detector guarda il moto, e non il verso.** Le ablazioni temporali
   globali, lette dalle ricevute (`rapporti/ablazioni_globali_20260924/`):

   | Popolazione | Ablazione | Δlogit mediano, finestre positive (DoRA · congelato) | decisioni cambiate sulle positive |
   |---|---|---|---|
   | 205 finestre comuni ai due modelli (64 positive) | freeze del campione centrale | **+4,08 · +3,79** | 43,8% · 48,4% |
   | | inversione del verso | +0,18 · +0,09 | 7,8% · 15,6% |
   | | cinque permutazioni | −0,41 · **−2,71** | 15,3% · 32,2% |
   | 1.295 finestre DoRA (477 positive) | freeze | +4,46 | 46,3% |
   | | inversione | +0,06 | 5,9% |
   | | permutazioni | +0,47 | 22,7% |

   Congelare il tempo abbatte il logit delle finestre anomale di ~4 su **entrambi**
   i modelli e ne spegne l'allarme quasi una volta su due; invertire il verso quasi
   non lo sposta. ⚠️ «Invariante al verso» non significa «non usa il tempo» — il
   freeze dimostra il contrario — significa che le feature usate sono simmetriche
   rispetto al verso. ⚠️ **Le permutazioni non reggono una conclusione**: il 14
   settembre, su 75 finestre per modello di un'altra popolazione, DoRA risultava
   *più* sensibile del congelato (+1,97 contro −0,69 di mediana); qui il congelato
   reagisce di più, e nel verso opposto (permutare **alza** il suo logit). La
   lettura dipende dalla popolazione e non si scrive.
3. **Nelle occlusioni il riempimento decide l'ampiezza, non la forma**: massimo
   della mappa, mediana su 75 finestre (congelato · DoRA), blur **0,21 · 0,66**,
   colore medio 1,62 · 1,90, nero 1,74 · 2,33. Un numero di occlusione senza il suo
   riempimento non si legge, e i confronti fra metodi si fanno a riempimento fisso.
4. **IG dipende dal riferimento, e l'ensemble lo media.** La baseline `blur` porta
   una differenza di logit piccola e di **segno opposto** (−1,2 contro +3,4 di
   `noise` e `tn_same_video`), perché il detector giudica una clip sfocata quasi
   come l'originale. Entrata nell'ensemble il 22 settembre, ne ha lasciato quasi
   invariata la **forma** (correlazione mediana 0,9928) e ridotto la **scala**
   (|nuovo|/|vecchio| 0,9105) — ma su **18 campagne su 702** la correlazione scende
   sotto 0,90 e su **4** sotto 0,50: video **normali** e **Shoplifting** su DoRA,
   dove le altre componenti quasi si annullano. Sulle finestre davvero normali IG
   attribuisce comunque una massa non trascurabile (differenza di logit fra 2 e 5
   contro 7–14 sulle anomale): la mappa va letta **con la sua scala**, non solo
   col suo disegno.
5. **La LayerNorm rende degenere l'attribuzione prima della norma**: rapporto
   mediano fra le masse assolute pre/post di **2,3e-6** e 2,5e-6 sui due modelli.
   Gradient × Activation si calcola quindi **dopo** la LayerNorm d'ingresso del
   pooler.
6. **I bounding box esistono come coordinate, non come misura.** 84 filmati interi
   (42 per sorgente, VTCD 0 e IG top-10%) con i box sui soli frame annotati
   positivi: **89.788** fotogrammi disegnati, **602.612** box, **0** finestre
   scartate dal filtro sul chunk, 4 video normali dichiarati `skipped`.

### Cosa NON è stato fatto, e va scritto così

- ⚠️ **Nessun metodo è stato promosso con una misura.** Le curve deletion/insertion
  sono **calcolate** su tutte e 13 le mappe della selezione (144 regioni di una
  partizione 4×6×6 della griglia, frazioni
  0–100% in sette passi, riempimenti `blur` e `mean`, cinque ordinamenti casuali come
  riferimento, 784 forward completi per finestra) e **non sono state lette**: `confronta_spiegazioni.py` esiste, il confronto con il bootstrap per
  video no. La scelta di IG, Anomaly Direction e VTCD per la presentazione del 23
  settembre è **qualitativa** (incontro del 17 settembre).
- ⚠️ **Le ricevute `deletion_insertion` e `map_statistics` descrivono l'ensemble a
  QUATTRO componenti**: ADR-073 ha rifatto mappe e filmati, non quelle ricevute.
  Prima di leggere le curve vanno rifatte (una passata CPU).
- ⚠️ **La scheda qualitativa è compilata a metà**: note su una parte dei video,
  sezione «Ricorrenze» vuota (`docs/varie/analisi-qualitativa-video_0909.md`,
  `note_visualizzazione-predizioni.md`). Le ipotesi che contiene sono ipotesi.
- ⚠️ **I bounding box non sono una localizzazione validata**: su UCF-Crime non c'è
  un ground truth spaziale, quindi sono una *proposta* di annotazione automatica,
  senza alcun numero di qualità spaziale.
- ⚠️ **La selezione è descrittiva**: migliore e peggiore si scelgono dopo aver visto
  il test; e tutto è a **un seme** e su due soli modelli.

### Alimenta

Il capitolo 4 — l'analisi dei risultati per tipologia di evento che la PdD chiede
(fase 7), cioè *dove* e *su cosa* il detector sbaglia — e il capitolo 5 (limiti e
sviluppi futuri: la localizzazione spaziale proposta). ⚠️ Dove la regola dei capitoli
lo consente, le mappe entrano come **ipotesi e diagnostica**, mai come risultato di
accuratezza (`docs/16` §6 punto 3).

---

<a id="e3m-e3c-20260915"></a>

## E3-M ed E3-C — Il budget e la composizione *(chiuse: né le tre epoche né le combinazioni spostano il verdetto di E3, e la validazione diventa ottimista)*

**Data:** 9–15 settembre 2026 · **ADR-062** (rivista), **ADR-064**, **F12/F13** di
`PIANO.md` · **Stato:** **chiuse**, 30/30 celle su `edp_v2`, tre semi
`[42, 67, 104]`, campagna supervisionata nel worktree
`~/evd/experiments/20260909-f12-f13` sul cluster.

### Le due domande, e perché sono state pagate

E3 aveva chiuso con **reparam ≥ probe congelata > prompt** a **una epoca**. Due
obiezioni restavano aperte, e nessuna delle due si poteva chiudere con i numeri
che c'erano.

1. **Il budget.** Una epoca è poca: se i metodi convergessero a velocità diverse,
   l'ordinamento a 5524 passi sarebbe un artefatto del budget e non una proprietà
   dei metodi. **E3-M** (F12) rifà i sei modelli a **tre epoche** — coseno
   **ciclico** di periodo 5524, due cicli in **continuazione** dalle celle già
   chiuse con `init_from`, 11.048 passi.
2. **La composizione.** C7 confronta reparam e prompt come alternative, ma non
   dice se i due prior induttivi siano **complementari** o **ridondanti**.
   **E3-C** (F13) addestra LoRA r=32 e DoRA r=32 **insieme** a VPT-shallow n=8 e
   VPT-deep n=8, tre cicli **da zero** (16.572 passi), perché non hanno un
   antenato a una epoca da cui continuare.

⚠️ **Tre semi e non uno.** La stesura del 9 settembre ne dava uno (ADR-062); la
decisione della sera l'ha rovesciata, perché a un seme il confronto con la
matrice sarebbe **indicativo e non un test**, e questi blocchi esistono per
rispondere in sede. Il conto è salito a ≈ 639 h di H100 sui due blocchi.

### Cosa è stato eseguito

| Blocco | Celle | Passi | Avvio | Ore H100 per cella (misurate) |
|---|---|---|---|---|
| **E3-M** | 6 modelli × 3 semi = **18** | 11.048 (2 cicli) | **a caldo** da E2c/E2d/E3-P/E3-R a pari seme | 3,6 (linear) · 4,7 (attentive) · 10,5 (VPT-sh) · 10,7 (VPT-deep) · 15,9 (LoRA) · 20,8 (DoRA) |
| **E3-C** | 4 combinazioni × 3 semi = **12** | 16.572 (3 cicli) | **da zero** | 23,7 · 23,8 · 31,1 · 31,4 |

Le ore sono la mediana degli intervalli fra checkpoint nei log SLURM
(`classifiche_modelli.py --log-slurm`), validazioni escluse: sono il solo
training, non il tempo a parete di un job. ⚠️ **Per E3-M sono le ore della
continuazione**, non il totale: dietro ogni cella c'è anche l'epoca del run
sorgente, e si dichiara.

### Le verifiche, fatte prima di leggere i numeri

* **30 cartelle su 30**, otto artefatti ciascuna, tutte `complete` nello stato del
  supervisore, tutte con la riga nel registry e il checkpoint sul disco;
  `steps == max_steps` e `stopped_early` falso ovunque.
* **`trainable_params_adapter` è la somma delle due componenti** su tutte e dodici
  le celle E3-C — 14.155.776 + 8.192 e 14.155.776 + 196.608 per LoRA, 14.376.960 +
  le stesse due per DoRA. È il controllo che ADR-062 punto 2 chiedeva di fare al
  primo checkpoint, e che qui è verificato a posteriori su ogni cella.
* **`avvio_a_caldo` nomina la sorgente giusta** su tutte e diciotto le celle E3-M,
  col passo 5524 e i momenti dell'ottimizzatore dichiarati reinizializzati.
* **AUC, AP e la soglia di Youden ricalcolate da `scores.jsonl`** riproducono
  `metrics.json` su tutte e trenta: AUC e τ **identiche al bit**, AP entro
  2,2e-16. La guardia di fedeltà di `ricalcola_latenza.py` — che rifiuta di
  scrivere se la regola storica non torna — è passata 30/30, e gli strati
  ricalcolati coincidono con quelli scritti dal run a meno di **un ULP** su una
  sola cifra di AP.
* `verifica_cella` di `classifiche_modelli.py` ricompone P/R/F1 di chunk e video
  dai conteggi per video: **0 incoerenze su 84 celle**.

Gli artefatti stanno in `runs/analisi_f1213_20260915/` (`celle.csv`,
`confronto_appaiato.csv`, `analisi.json`, `verifica_score.json`, `LEGGIMI.md`) e
la ricevuta di completamento in `../tmp/f12-f13-20260909/completion-20260915.json`.

### Cosa è uscito — F12, le tre epoche

Il confronto è **appaiato per seme** contro la stessa cella a una epoca: stessa
partizione di test, stessa inizializzazione.

| Modello a 3 epoche | Δ chunk-AUC | Δ chunk-AP | semi a favore | dove cade la selezione |
|---|---|---|---|---|
| **Linear probe** | **+0,0102** | **+0,0384** | **3/3** su tutte e quattro le metriche | **0/3** nel primo ciclo |
| Attentive probe | −0,0021 | +0,0022 | 1/3 | 3/3 nel primo ciclo |
| VPT-shallow n=8 | **−0,0122** | **−0,0204** | **0/3** | 2/3 nel primo ciclo |
| VPT-deep n=8 | −0,0025 | +0,0042 | 0/3 su AUC, 3/3 su AP | 3/3 nel primo ciclo |
| LoRA r=32 | −0,0006 | −0,0086 | 2/3 su AUC, 1/3 su AP | 1/3 nel primo ciclo |
| DoRA r=32 | −0,0037 | −0,0029 | 0/3 su AUC, 1/3 su AP | 1/3 nel primo ciclo |

**Raddoppiare il budget non compra niente, tranne sul modello che era sotto-addestrato.**
L'unica cella che guadagna è la **probe lineare**, e guadagna su tutte e quattro
le metriche con **3/3** di concordanza fra i semi: +0,0102 di AUC e **+0,0384 di
AP**, che è il salto più grande di tutta la campagna. Tutte le altre restano dove
erano o peggiorano, e **VPT-shallow n=8 peggiora nettamente** — −0,0122 di AUC e
−0,0204 di AP, **0/3**, cioè il segno si ripete su tutti i semi.

⚠️ **LoRA e DoRA non «peggiorano»: non si separano.** Su LoRA i tre delta sono
+0,0057, −0,0084 e +0,0008 — l'escursione fra i semi è **più grande del delta
medio**, che è la stessa forma che ADR-057 aveva già trovato fra LoRA e DoRA. La
frase corretta è *tre epoche non cambiano il risultato*, non *tre epoche fanno
danno*.

**La conferma sta nel passo selezionato, ed è indipendente.** La selezione è
l'AUROC di validazione, e non sa niente di questa tabella: su **VPT-deep** cade
dentro il **primo** ciclo in 3 casi su 3, su **attentive** idem — cioè il secondo
ciclo non ha mai prodotto una validazione migliore e il run ha buttato il budget
che gli era stato dato. Sulla **linear** cade nel secondo ciclo **3 volte su 3**.
I due segnali concordano: *il passo scelto dice quali modelli stavano ancora
imparando*, e a una epoca stava ancora imparando solo la probe lineare.

### Cosa è uscito — F13, le combinazioni

| Combinazione | chunk-AUC | vs il suo reparam | vs il suo prompt | ore H100 |
|---|---|---|---|---|
| DoRA r=32 + VPT-shallow n=8 | **0,9405 ± 0,0016** | −0,0017 | +0,0044 | 31,1 |
| LoRA r=32 + VPT-deep n=8 | 0,9387 ± 0,0068 | −0,0005 | +0,0070 | 23,8 |
| DoRA r=32 + VPT-deep n=8 | 0,9354 ± 0,0011 | −0,0068 | +0,0037 | 31,4 |
| LoRA r=32 + VPT-shallow n=8 | 0,9350 ± 0,0052 | −0,0042 | −0,0011 | 23,7 |
| *(riferimento)* DoRA r=32, 1 epoca | **0,9422 ± 0,0017** | — | — | **10,5** |

**I due prior sono ridondanti, non complementari, ed è la risposta che C7
cercava.** Una combinazione si comporta come il suo **reparam**, leggermente
peggiorato: contro il reparam il delta è negativo su **15 righe metriche su 16**,
contro il prompt è positivo quasi ovunque (fino a **+0,0412** di AP_a). Il
componente che porta il risultato è il reparam; il prompt non aggiunge, e dove si
nota sposta in basso.

**Nessuna combinazione batte DoRA r=32 da sola a una epoca** (0,9422), che resta
**prima** su chunk-AUC e su chunk-AP nella classifica di tutti e 28 i modelli. La
migliore combinazione la eguaglia su AUC_a (0,8674 contro 0,8669) e su AP_a
(0,8313 contro 0,8305), che sono **pareggi dentro una σ**, e per farlo costa
**tre volte** il training.

⚠️ **E il prezzo non è solo tempo.** Una composizione **non è fondibile**
(`can_merge()` falso, ADR-064), quindi perde la proprietà che E5b ha misurato
valere **+42,8%** e **+83,7%** di latenza a inferenza — e la fusione **parziale**
che ADR-064 introduce non è stata profilata, quindi la colonna del costo di
inferenza per queste quattro righe resta **vuota** invece di riportare il tempo di
un'altra configurazione.

⚠️ **E3-C non è a parità di parametri**, e si dichiara: la combinazione aggiunge
fra **+0,06%** (VPT-shallow n=8) e **+1,39%** (VPT-deep n=8) ai parametri del suo
reparam. Il confronto «combinazione contro reparam» è quindi **generoso verso la
combinazione**, e perde lo stesso.

### Il fatto di metodo, che nessuno dei due blocchi cercava

**La validazione diventa ottimista quando il budget cresce.** Il divario fra
l'AUROC di validazione al passo selezionato e la chunk-AUC sul test, mediana sulle
celle, escludendo le probe lineari (che hanno un divario **negativo** per un'altra
ragione — la testa lineare è così debole che la validazione la sottostima):

| Regime | celle | mediana val − test |
|---|---|---|
| **1 epoca** (la matrice) | 51 | **+0,0030** |
| **3 epoche** (E3-M) | 15 | **+0,0058** |
| **3 cicli da zero** (E3-C) | 12 | **+0,0155** |

Ed è **appaiato**: su LoRA r=32 il divario passa da +0,0091 a +0,0148 e su DoRA da
+0,0048 a +0,0114 aggiungendo solo epoche. Peggio, su **tre celle E3-M su sei** i
due segnali sono **discordi** — la validazione sale e il test scende (VPT-shallow
+0,0016/−0,0122, LoRA +0,0051/−0,0006, DoRA +0,0028/−0,0037).

⚠️ **Il criterio non è rotto**: su tutte e 84 le celle la correlazione fra AUROC di
validazione e chunk-AUC resta alta (Pearson **0,957**, Spearman **0,760**). Quello
che degrada è la sua **risoluzione** proprio nella zona dove si decide, cioè fra
modelli a 0,003 di distanza.

⚠️ **Il numero è misurato, il meccanismo no.** La spiegazione naturale è che le
8383 finestre di validazione servono **due volte** — scelgono il passo *e*
fissano τ — e che più passi significano più occasioni di adattarvisi; ma questo
esperimento non l'ha isolata, e non gliene si attribuisce una. Il fatto che si
porta in tesi è il divario e la sua monotonia, non la sua causa.

### Cosa concludono, e cosa no

**Concludono:**

1. **Il verdetto di E3 non dipende dal budget**: a tre epoche l'ordinamento fra
   famiglie è lo stesso, e il migliore resta DoRA r=32 a **una** epoca.
2. **Il verdetto di E3 non dipende dalla composizione**: mettere insieme i due
   prior non li somma. C7 si legge quindi come un'**alternativa**, non come due
   contributi additivi.
3. **La probe lineare era sotto-addestrata**, e il suo numero di E2 (0,8473) va
   letto sapendolo: a tre epoche sale a **0,8574** con **+0,0385 di AP**. ⚠️ Resta
   comunque **sotto ogni altro metodo**, quindi il verdetto «la linear non regge»
   di E2 **non cambia**.
4. **Il costo di questi due blocchi è la loro conclusione**: ≈ 639 h di H100 per
   scoprire che la configurazione più economica già misurata è ancora la migliore.
   È un risultato negativo, ed è quello che rende difendibile la scelta.

**Non concludono:**

* ⚠️ **Non dicono che più epoche facciano danno.** Su quattro dei sei modelli il
  delta è dentro la dispersione fra semi; solo su **VPT-shallow n=8** il segno si
  ripete 0/3 su tutte le metriche.
* ⚠️ **Non separano «addestrare insieme» da «sommare i due prior».** È la domanda
  di **E3-S** (F13-bis, ADR-065), che addestra le stesse coppie **in fila** nelle
  due direzioni e che **non è stata eseguita**.
* ⚠️ **Non misurano il costo a inferenza di una combinazione.** E5b non l'ha
  profilata, e la fusione parziale di ADR-064 è implementata ma non misurata.
* ⚠️ **Il livello evento non discrimina, e qui si vede bene**: la F1 di evento
  mette in testa la **probe lineare a tre epoche** (0,7020) davanti a tutto il
  reparam, mentre la sua chunk-AP è **0,5173** contro 0,79. È la stessa cosa che
  E-strati aveva già trovato, e la ragione per cui la F1 di evento non si usa per
  ordinare i metodi.

<a id="versione-riportata-20260925"></a>

### Postilla del 25 settembre — una versione per modello nelle classifiche

Dal 25 settembre `runs/classifiche/` e il foglio del confronto con lo stato
dell'arte (`docs/incontri/metriche_confronto-sota.xlsx`) riportano **una** versione
di ciascuno dei sei modelli, sotto il nome semplice e senza «(3 ep.)». Decisione di
Antonio: fra le due si tiene quella con la chunk-AUC più alta sul **test** di
FS-UCF-Crime al seme 42, salvo la probe attentive, riportata a tre epoche come la
lineare perché le due baseline stiano al budget esteso. La regola ha una sede sola
(`scripts/classifiche_modelli.py`, `VERSIONE_RIPORTATA` e `modelli_riportati()`),
che classifiche, per-classe e ranking dei chunk leggono; `--tutte-le-versioni`
rimette in tabella entrambe.

| Modello | chunk-AUC test s42, 1 → 3 ep | AUROC val s42, 1 → 3 ep | chunk-AUC media tre semi, 1 → 3 ep | Riportata |
|---|---|---|---|---|
| Linear probe | 0,8473 → **0,8562** | 0,7787 → **0,7924** | 0,8473 → **0,8574** | **3 epoche** |
| Attentive probe | **0,9385** → 0,9339 | 0,9370 → **0,9376** | **0,9310** → 0,9289 | **3 epoche** (per decisione) |
| VPT-shallow n=8 | **0,9394** → 0,9347 | 0,9320 → **0,9405** | **0,9360** → 0,9238 | 1 epoca |
| VPT-deep n=8 | **0,9344** → 0,9311 | **0,9332** → 0,9299 | **0,9317** → 0,9292 | 1 epoca |
| LoRA r=32 | 0,9393 → **0,9451** | 0,9466 → **0,9547** | **0,9392** → 0,9386 | **3 epoche** |
| DoRA r=32 | **0,9441** → 0,9411 | 0,9467 → **0,9522** | **0,9422** → 0,9385 | 1 epoca |

⚠️ **Il criterio non è quello del protocollo, e va scritto.** Il protocollo sceglie
sulla validazione (`docs/04` §5.3); qui la scelta fra due checkpoint dello stesso
modello guarda il test su quattro righe e una decisione sulle altre due. Con
l'AUROC di validazione al seme 42 sarebbero state tre epoche ovunque salvo VPT-deep:
le due regole differiscono su **VPT-shallow n=8** e **DoRA r=32**. Il confronto con lo
stato dell'arte (§E4) resta su DoRA r=32 a una epoca e non dipende da questa scelta:
quel modello è stato scelto il 15 settembre sulla media a tre semi.

⚠️ **Sulla media a tre semi, che è ciò che le classifiche riportano, la versione a
tre epoche sta sotto quella a una ovunque salvo la probe lineare**: anche le due
scelte a tre epoche, Attentive (0,9289 contro 0,9310) e LoRA r=32 (0,9386 ± 0,0064
contro 0,9392 ± 0,0015), dentro la dispersione. «La versione migliore» vale al seme
42 del test, non sulla media.

**Perché una versione per nome regge.** Ogni versione riportata è un punto della
stessa traiettoria a tre epoche: la cella a una epoca è quella da cui la
continuazione di E3-M riparte (`init_from`). ⚠️ Non vale per il resto delle
classifiche: le ablation di rango e di token (E3-R.2/E3-P.2) sono a **una** epoca,
le combinazioni (E3-C, E3-S) a tre cicli, e le ore di E3-M in `classifiche.json` sono
quelle della sola continuazione.

**Cosa cambia negli artefatti**, nessun numero di un run: le classifiche passano da
36 modelli e 108 celle a **30 e 90**; in `classifiche.md` «Linear probe», «Attentive
probe» e «LoRA r=32» sono le celle E3-M e **non** coincidono con le righe della
Tavola §A; nella §6b (seme 42, AUC media sui due dataset) i primi tre non cambiano e
LoRA r=32 a tre epoche è quinto, 0,9226. `per_class.csv` e
`ranking_chunks_riepilogo.csv` seguono la stessa regola.

### Alimenta

`runs/classifiche/` (tutte e venti le tabelle, ora su **28 modelli e 84 celle**),
il deck della presentazione, e il §C7 della tesi.

---

<a id="e3-ablation-20260907"></a>

## E-MIVIA — Il modello su un dataset che non ha mai visto *(chiusa: l'ordinamento si trasferisce e perde 4 punti di AUC, la calibrazione vale 0,021 di F1)*

**Data:** 10 settembre 2026 · **ADR-066** · **Stato:** **chiusa** (job 560576, `aiq`,
19 min 04 s). Il criterio qui sotto e' stato scritto **prima** dei numeri e non e'
stato toccato dopo.

### La domanda

Il laboratorio ha consegnato **MIVIA-Crime**, il test set privato con cui misura il
proprio metodo, e ha chiesto di misurarci le nostre prestazioni
(`docs/varie/incontro-presentazione_0909.md`, Esperimenti 3). La domanda è
**cross-dataset**: *quanto di ciò che una cella ha imparato su UCF-Crime
sopravvive a un altro dominio di sorveglianza?*

⚠️ **Non è un run.** Non si addestra niente, non si sceglie nessun iperparametro,
non si stima nessuna soglia: si prendono i pesi **selezionati** di una cella già
chiusa — `e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42`, il miglior DoRA a seme 42 — e si
fa **una passata di inferenza**, con la stessa geometria (`edp_v2`: finestra 8 s,
hop 4 s, 32 campioni a 384 px `squash`).

### Il criterio, scritto prima dei numeri

**Cosa si riporta, e in quest'ordine di autorevolezza.**

1. **chunk-AUC e chunk-AP** su MIVIA-Crime, accanto a quelle su UCF-Crime della
   stessa cella (0,9424 · 0,7737 di riferimento nella Tavola). Sono
   **indipendenti dalla soglia**, quindi misurano quanto il modello *ordina* su
   un dominio nuovo, ed è l'unico confronto che non ha bisogno di ipotesi.
2. **AUC_a e AP_a**, ristrette ai 135 video anomali: la versione difficile della
   domanda, la stessa che su UCF-Crime ha mostrato che l'adattamento aiuta
   **dentro** il video anomalo e non a distinguerlo da uno normale.
3. **P/R/F1 a livello chunk, video ed evento**, alla soglia **trasferita**.

**Le due letture ammesse, e la terza che è vietata.**

* **Se AUC e AP tengono e P/R/F1 crolla**, la conclusione è che il modello
  **ordina** ancora bene e la sua **calibrazione** non si trasferisce. È il caso
  interessante per la tesi, e ha una conseguenza di deployment: su un dominio
  nuovo la soglia va ristimata su una manciata di video etichettati.
* **Se anche AUC e AP crollano**, la rappresentazione stessa non si trasferisce, e
  il numero va scritto così com'è.
* ⚠️ **Vietato**: leggere `soglia_oracolo` come un risultato. Quel blocco è il
  punto di Youden calcolato **su MIVIA-Crime stesso**, cioè guardando i video che
  poi giudica: è un **limite superiore** che nessun sistema schierato potrebbe
  raggiungere, e serve **solo** a separare la perdita di ordinamento da quella di
  calibrazione — la differenza fra la sua F1 e quella a τ trasferita *è* la parte
  di calibrazione.

**Perché τ si trasferisce e non si ristima.** Il protocollo dice che la soglia è
il punto di Youden sulla **ROC di validazione** (ADR-047 §3). MIVIA-Crime una
validazione non ce l'ha: sono 270 video tutti di valutazione. Ristimarla lì
sarebbe lo stesso difetto che `eval_split: val` dichiara da sé nella matrice.
`metrics.json` la registra come **`fissa`**, con `trasferita_da` e
`regola_originale` accanto.

### Cosa si può concludere, e cosa no

⚠️ **Un modello, un seme.** Si valuta la cella **s42** di DoRA r=32: qualunque
differenza rispetto a UCF-Crime è di **quel** modello su **quel** seme, e non
autorizza una frase sulla famiglia. Estendere costa una passata per cella (~30
minuti di H100 ciascuna, non ore: non si addestra).

⚠️ **Il confronto è cross-dataset e va scritto come tale.** Di `edp_v2` si riusa
la **geometria** — la sola condizione sotto cui le due tabelle stanno una accanto
all'altra — **non** lo split, che qui non esiste. Le due colonne non sono
«prima e dopo»: sono due popolazioni diverse.

⚠️ **Le due popolazioni differiscono anche in cose che non c'entrano col metodo**,
e vanno dichiarate accanto a qualunque delta: MIVIA-Crime ha **50% di video
anomali** (135 su 270) contro il 48% di UCF-Crime, ma è fatto di video **molto più
corti** — **3485 finestre** contro 9116 su un numero simile di video, e 4,0 ore
contro ~40. Un tasso di positivi diverso sposta l'AP di suo.

### I fatti della consegna che pesano sulla lettura

Misurati con `scripts/inventario.py`, decodificando ogni file:

| | UCF-Crime | MIVIA-Crime |
|---|---|---|
| video | 1900 (290 di test) | **270** (135 anomali) |
| risoluzione | 320×240, una sola | **47 distinte**, da 320×240 a 3840×2160 |
| fps | 30, uno solo | **da 6 a 60** (30,00: 107 · 25,00: 53 · 29,97: 35) |
| header `nb_frames` | esatto **1900/1900** | ⚠️ **sbagliato su 53/270**, fino a 1,285× |
| finestre (griglia test) | 9116 | **3485** |
| intervalli | 1831 | 205, di cui **25 clampati** a `num_frames-1` |

⚠️ **Il terzo è quello che vale la pena aver scoperto prima e non dopo.**
`num_frames` decide quante finestre ha un video e **dove cadono**: non è un
metadato, è la geometria. `PyAVDecoder.probe` leggeva il conteggio **dichiarato**
dal container, che su UCF-Crime coincide sempre — e coincide anche con
`durata × fps`, 1900/1900 — quindi in tre mesi non aveva mai sbagliato. L'ha
fatto emergere **il dimostratore e non un test**: la guardia per-finestra di
ADR-055 ha rifiutato di rigiocare `abuse_001` (4 finestre nella griglia
ricostruita, 3 nel file) invece di disegnare una timeline plausibile con gli score
appaiati alle finestre sbagliate. Ora `probe` confronta i due campi e conta
decodificando **solo se discordano**. ⚠️ Contare i **pacchetti** non basta ed è
stato provato: il container di `abuse_001` ne porta 541, cioè esattamente il
numero che mente.

⚠️ **Il vincolo di questa passata è la RAM, non la GPU.** Un item del loader è la
finestra decodificata a risoluzione **nativa**, prima del ridimensionamento a
384: ~7 MB su UCF-Crime, **~800 MB** su un video 4K. In volo ci stanno
`num_workers × prefetch × batch` item, e la prova in locale (13,7 GiB) è morta con
`av.error.MemoryError` a 8 worker. Il job chiede **96G** e 8 worker.

### Come si esegue

    EVD_ALLOW_BF16=1 EVD_ESTERNO_RUN=e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42 \
        hpc/invia.sh -p aiq -t 02:00:00 -c 14 -g 1 -m 96G -J evd-mivia \
        evd-vfm/scripts/slurm/valuta_esterno.sbatch

Prerequisito: l'inventario di MIVIA-Crime in `$EVD_DATA_DIR/mivia_crime_inventory/`
(`EVD_INVENTARIO_DATASET=mivia` su `scripts/slurm/inventario.sbatch`, scan +
merge). Il risultato è una **cartella di run vera** in
`runs/{run_id}__mivia/` — `config.yaml`, `meta.json`, `metrics.json`,
`scores.jsonl` — più la riga di registry: il dimostratore la rigioca in
`--modo replay` e `eval/report.py` ne disegna le curve.

### Risultati — job **560576**, `aiq` (`ainode02`), **19 min 04 s**

Cella `e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42`, pesi del passo **5520** (la selezione,
non il 5524), τ = **0,086323** trasferita, **270 video · 3485 chunk**. Artefatti in
`runs/e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42__mivia/`.

| | UCF-Crime (test) | MIVIA-Crime | Δ |
|---|---|---|---|
| **chunk-AUC** | 0,9441 | **0,9029** | **−0,0412** |
| chunk-AP | 0,7971 | 0,8335 | +0,0364 ⚠️ |
| **AUC_a** | 0,8704 | **0,8293** | **−0,0411** |
| AP_a | 0,8300 | 0,8813 | +0,0513 ⚠️ |
| chunk P / R / F1 | 0,587 / 0,829 / 0,687 | 0,802 / 0,648 / **0,717** | +0,030 di F1 |
| video P / R / F1 | 0,918 / 0,964 / 0,941 | 0,855 / 0,874 / **0,865** | −0,076 di F1 |
| evento P / R / F1 | 0,513 / 0,928 / 0,660 | 0,742 / 0,756 / **0,749** | +0,089 di F1 |
| chunk positivi | 16,36% (1491/9116) | **30,65%** (1068/3485) | ×1,87 |

⚠️ **Le due AP e le due AP_a NON si sottraggono, ed è il primo numero da leggere
con la sua nota.** L'AP ha come pavimento il **tasso di positivi**, e su
MIVIA-Crime è **quasi il doppio** (30,65% contro 16,36%): un classificatore
casuale prende AP 0,31 qui e 0,16 là. Normalizzando — AP diviso il pavimento —
il **lift** passa da **4,87×** a **2,72×**, cioè nella direzione dell'AUC. La
stessa causa gonfia la **precision** a ogni granularità. **L'unico confronto che
si legge senza correzioni è l'AUC**, che è invariante al tasso di positivi.

### Cosa concludono questi numeri

**(1) L'ordinamento si trasferisce, e perde 4 punti.** chunk-AUC **0,9029** contro
0,9441, e AUC_a **0,8293** contro 0,8704: ⚠️ **la caduta è la stessa, −0,041 su
entrambe, alla terza cifra**. Non è la capacità di distinguere un video anomalo da
uno normale a degradare più dell'altra: degradano insieme e della stessa quantità.
Un modello addestrato su UCF-Crime resta **ben sopra il caso** su un dataset che
non ha mai visto, con la stessa geometria e senza toccare un peso.

**(2) Anche la calibrazione regge, e vale 0,021 di F1.** È il numero che il blocco
`soglia_oracolo` esiste per dare: il punto di Youden calcolato **su MIVIA-Crime
stesso** cade a **τ = 0,00573** — quindici volte più in basso della τ trasferita —
e porta la chunk-F1 da **0,7167** a **0,7380**. ⚠️ **Quel +0,021 è un limite
superiore** — quella soglia guarda i video che poi giudica — quindi il costo vero
di non ricalibrare è **al più** due punti di F1. **La direzione però va detta**:
τ trasferita è **troppo alta** per questo dominio e il sistema **sotto-allarma**
(recall 0,648 contro 0,798 all'oracolo, precision 0,802 contro 0,687). Chi
schierasse il modello su un dominio nuovo dando peso al recall abbasserebbe la
soglia; chi tiene alla precision la lascia dov'è.

**(3) Il livello video è quello che perde di più, e non è una contraddizione.**
video-F1 **0,865** contro 0,941, −0,076, mentre la chunk-F1 *sale*. La regola del
livello video è **asimmetrica** (ADR-047 §4.2): su un video anomalo conta solo una
finestra che l'evento lo contenga davvero. Con τ troppo alta i video anomali con
eventi brevi non producono nessuna finestra sopra soglia, e si perdono interi —
recall video 0,874 contro 0,964. È lo stesso sotto-allarme del punto 2, visto alla
granularità dove costa di più.

**(4) `Shoplifting` è di nuovo il collo di bottiglia, e su un altro dataset.**
AUC **0,7509**, l'unica classe sotto 0,75 e la peggiore anche qui — su UCF-Crime
era 0,8350, di nuovo l'ultima. È la conferma **cross-dataset** del risultato di
**E-strati**: non è una proprietà dello split di UCF-Crime, è la classe. Le altre
due che cadono sono `Stealing` (0,9747 → **0,8185**) e `Shooting`
(0,9516 → 0,8875); ⚠️ tre classi **migliorano** — `Robbery` (0,9241 → **0,9608**),
`Arrest` (0,9177 → 0,9388) e `Vandalism` — e con `n_pos` fra 74 e 108 non sono
righe degeneri.
⚠️ `Abuse` su UCF-Crime aveva **6** chunk positivi e un'AP di 0,011: qui ne ha
**56**, quindi la sua riga qui è una misura e là non lo era. Le due colonne
`n_pos` vanno lette insieme alle metriche, sempre.

**(5) A livello evento: 155 eventi rilevati su 205**, con 186 allarmi di cui 138
veri. ⚠️ Il salto di F1 rispetto a UCF-Crime (+0,089) è quasi tutto **precision**
(0,513 → 0,742) e viene dalla stessa causa del punto 1 — più eventi in meno ore di
video — non da un sistema che sbaglia meno.

### Gli artefatti, gli stessi delle altre celle

`runs/e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42__mivia/` ha **tutto** ciò che ha una
cella della matrice tranne i cinque pezzi che appartengono al **training**, che
qui non c'è stato: `checkpoint/`, `history.jsonl`, `training.png`, `lr.png` — e
`explain/`, che non è un artefatto standard (esiste su **un** run solo, quello
scelto per l'analisi qualitativa).

Ci sono quindi: **8 figure dalle metriche** (`roc`, `pr`, `confusion`, `event`,
`per_class`, `threshold`, `score_hist`, `strata`), le **quattro figure di strati**
per asse, `strata.json`, `latenza.json`, `per_class.csv` +
`per_class_provenienza.json`, `per_video.csv`, `per_video_classi/` (13 file, uno
per classe) e i **cinque** `ranking_chunks*.csv` (3485 chunk: **TP 692 · TN 2246
· FP 171 · FN 376**) con la loro provenienza.

⚠️ **Tre di questi passaggi ricalcolano le metriche dagli score e le confrontano
con `metrics.json`, e tutti e tre passano** — `per classe verificato`,
`ranking chunk verificato`, e la guardia di `ricalcola_latenza.py` che riproduce
la regola storica. Non è una formalità: la geometria viene dal **sampler** e non
dal file (ADR-055), quindi è la prova che la rilettura del run cross-dataset è
fedele. È anche la verifica end-to-end più forte che `exp/esterno.py` abbia.

**Due script hanno dovuto imparare a vedere questo run**, e la ragione per cui
non lo vedevano è una **guardia che funziona**: il `run_id` di una valutazione
esterna è `{run_id}__{dataset}`, quindi `RunId.parse` lo rifiuta e
`trova_celle` lo scarta (`rpartition("_s")` dà `42__mivia`, che non è un numero).
⚠️ **Quel filtro non si allarga**: è ciò che impedisce a una cella misurata su un
altro dataset di comparire nelle classifiche di UCF-Crime come una riga
plausibile e falsa. Entrambi gli script prendono invece un **`--run` esplicito**,
che vuol dire «questa qui, e so cosa sto facendo»; e `classifiche_modelli.py`
**rifiuta** di girare con `--run` e l'`--out` predefinito, che riscriverebbe le
classifiche vere con una tabella a una riga.

### Cosa NON concludono

⚠️ **Un modello, un seme.** È la cella **s42** di DoRA r=32. Nessuna frase sulla
famiglia, nessuna sul confronto fra famiglie: per quelle servono le altre celle, e
costano **~20 minuti di H100 ciascuna** (non ore: non si addestra).

⚠️ **Non è un confronto «prima e dopo»**, sono due popolazioni: 30,65% di chunk
positivi contro 16,36%, **3485 finestre contro 9116** su un numero simile di
video, 4,0 ore contro ~40. Ogni metrica che dipende dal pavimento — AP, precision
— va riportata **con il tasso di positivi accanto**, o dice il contrario di quello
che sembra.

⚠️ **Il leakage di UCF-Crime non si applica qui**, ed è l'unico aspetto in cui
questo numero è *più pulito* di quello della matrice: le dieci coppie
byte-identiche che attraversano le partizioni di `edp_v2` (ADR-047) sono un
problema di quello split, e MIVIA-Crime è un dataset a sé.

<a id="e-mivia2-20260922"></a>

### E-MIVIA.2 — tutti i modelli a seme 42, e l'ordinamento non si trasferisce

**Data:** 22 settembre 2026 · **Stato:** **chiusa**, 36 celle · campagna
`experiments/20260921-mivia-s42`, ~9 ore di GPU su 5 job in parallelo, mediana
15 min a cella. Ogni cella è una passata di inferenza sui pesi **selezionati**,
con la geometria del run e τ **trasferita**: nessun addestramento, nessuna
ristima.

⚠️ **Correzione del 25 settembre: le quattro righe `fondi` di questa voce erano
sbagliate, e ora sono rifatte** (`docs/09` R24). La valutazione esterna passava
da `exp/demo.motore_live`, che non rileggeva `preadattamento`, quindi quelle
celle erano state valutate **senza** il reparam fuso nel backbone. Rifatte col
motore corretto e verificate su UCF-Crime (§E3-S), fanno **0,9009–0,9056** di
chunk-AUC su MIVIA-Crime invece di 0,8056–0,8211. Cambiano tre numeri della voce,
e con loro la sua frase di sintesi:

| | prima (sbagliato) | dopo la correzione |
|---|---|---|
| Spearman / Pearson su 36 celle | +0,395 / +0,545 | **+0,638 / +0,964** |
| delta di chunk-AUC su 36 celle | mediana −0,040, `fondi` a −0,12…−0,13 | **mediana −0,039, σ 0,0067, da −0,054 a −0,027** |
| posizione su MIVIA del primo su UCF (`vptshallow-n8-su-dora-r32_s42`) | 33ª su 36 | **14ª su 36** |

Quindi: i **valori** si trasferiscono (Pearson 0,96) e l'**ordine** solo in parte
(Spearman 0,64) — perché trenta modelli stanno in 0,04 di chunk-AUC e bastano
spostamenti piccoli per rimescolarli, non perché qualcuno crolli. **Non c'è
nessun «confronto controllato» fra `fondi` e `congela`**: le due direzioni
perdono allo stesso modo. Il testo sotto resta com'era, per storia, e le sue
righe `fondi` si leggono con questa tabella accanto.

**La domanda.** E-MIVIA aveva misurato **una** cella. La scelta del modello però
si fa su una classifica, e la domanda è se quella classifica **sopravvive al
cambio di dominio**: se il primo in casa è il primo anche fuori, la selezione
fatta su UCF-Crime regge; se no, va rifatta, e va detto.

**Il risultato, in una riga: no.**

| | Spearman | Pearson |
|---|---|---|
| tutte e 36 le celle | **+0,395** | +0,545 |
| senza le quattro `fondi` di E3-S | **+0,693** | +0,966 |

⚠️ **Il primo su UCF-Crime è 33° su 36 su MIVIA-Crime.** È
`e3s_..._vptshallow-n8-su-dora-r32_s42`: **0,9477** di chunk-AUC in casa, prima
fra tutte e 36, e **0,8163** fuori. Il primo su MIVIA-Crime è
`e3c_..._dora-r32-vptdeep-n8_s42` (**0,9084**), che in casa è 0,9355, cioè a metà
classifica. ⚠️ Il Pearson alto **senza** gli outlier dice che i *valori* si
trasferiscono bene; lo Spearman più basso dice che l'**ordine** no, e non è una
contraddizione: trenta modelli stanno in 0,04 di chunk-AUC, quindi bastano
spostamenti piccoli per rimescolare le posizioni.

**Perdono tutti, e quasi tutti allo stesso modo.** Escludendo le quattro celle
`fondi`, il delta di chunk-AUC ha mediana **−0,0395** con **σ 0,0068** ed
escursione da −0,0271 a −0,0540: una tassa di dominio quasi costante, che
**non** dipende dalla famiglia PEFT. Le mediane per blocco stanno tutte fra
−0,036 e −0,045.

⚠️ *Paragrafo superato da R24 (nota in testa alla voce): i numeri `fondi` qui sotto sono di un modello senza il suo reparam.*

⚠️ **Le quattro combinazioni `fondi` di E3-S crollano, e sono 4 su 4.**

| direzione | celle | delta chunk-AUC | chunk-AUC su MIVIA |
|---|---|---|---|
| `congela` (reparam sopra prompt congelati) | 4 | −0,0340 … −0,0423 | 0,8913 … 0,9040 |
| `fondi` (VPT sopra un reparam **già fuso**) | 4 | **−0,1202 … −0,1314** | **0,8056 … 0,8211** |

**È un confronto controllato**, ed è ciò che lo rende leggibile: le due
direzioni hanno lo **stesso budget** (16.572 passi, tre cicli), le **stesse due
famiglie** combinate e lo stesso seme. Cambia l'ordine in cui si addestrano, e
solo una delle due direzioni cade. ⚠️ **Il meccanismo non è isolato e non
gliene si attribuisce uno**: la spiegazione naturale — in `fondi` il backbone è
fisso e fuso, l'unica capacità di adattamento sono 8 token di prompt, quindi
tutto il resto del fitting finisce nella testa — è plausibile e non misurata.

⚠️ **Su un dominio nuovo le `fondi` valgono quanto una probe lineare.** La
fascia bassa di MIVIA-Crime è: `e2c` lineare **0,7933**, `vptdeep-su-dora`
0,8056, `e3m` lineare **0,8117**, `vptshallow-su-dora` 0,8163,
`vptdeep-su-lora` 0,8165, `vptshallow-su-lora` 0,8211. Quattro combinazioni che
in casa stavano fra 0,936 e 0,948 finiscono mescolate alle due probe lineari,
che in casa stavano a 0,847 e 0,856.

**Le altre due letture, invariate rispetto a E-MIVIA.1 e ora su 36 celle.**
Il **lift** dell'AP scende da ~4,8 a **2,69** di mediana: MIVIA-Crime ha il
**30,65%** di chunk positivi contro il 16,36%, quindi un'AP più alta è pavimento
e non merito. La **calibrazione costa poco**: il Youden calcolato sul dataset
stesso (`soglia_oracolo`, **non protocollo**) alza la chunk-F1 di **+0,0166** in
mediana, al più +0,0433 — quindi la perdita è **ordinamento**, non taratura. ⚠️
Su una cella il costo è **negativo** (−0,0055), e non è un errore: Youden
massimizza *J*, non *F1*. τ trasferita resta **troppo alta** e il sistema
**sotto-allarma**, con recall mediana **0,662**.

**Un difetto trovato per strada, e la sua forma.** Due celle su 36 — le probe
**lineari** — sono morte tre volte ciascuna con
`ValueError: invalid literal for int() with base 10: '???'`. Il `???` di
`head.n_classes` **non è un residuo**: `ExperimentRunner._n_classes` lo risolve
dall'obiettivo e dal dataset proprio perché il config non deve essere una
seconda fonte. Il difetto è che **il valore risolto non finisce in nessun
artefatto** — `config.yaml` conserva il segnaposto, `meta.json` e `metrics.json`
non hanno il campo — quindi chi ricostruisce il modello da lì non ha modo di
saperlo, e l'unico che lo sa è il **checkpoint**. Ora `motore_live` lo legge da
lì, cercando il classificatore per **forma** (`[n, embed_dim]`) e non per nome,
perché i nomi sono già tre. ⚠️ Il difetto era invisibile su 34 celle su 36:
tutte le altre teste sono `attentive`, che `n_classes` lo dichiara.

**Limiti, e sono due.** ⚠️ **Un seme solo** (42): questi delta non hanno barre
d'errore, e la dispersione fra semi su UCF-Crime vale fino a 0,016 di chunk-AUC
— più piccola dei −0,12 delle `fondi`, dello stesso ordine dei −0,04 di tutte le
altre. ⚠️ **E3-S non è chiusa**: al 22 settembre tre delle quattro celle
`congela` non hanno ancora il seme 104, quindi le loro righe stanno su **due**
semi invece di tre, e la colonna `n` delle classifiche lo dice.

⚠️ **E3-S è entrata in `runs/classifiche/` il 22 settembre**, con otto modelli
nuovi sotto la famiglia **`sequenziale`** — che non è `combinazione` di
proposito: metterle sotto la stessa etichetta di E3-C deciderebbe graficamente
la domanda che ADR-065 pone. La tabella passa da 28 modelli e 84 celle a **36 e
105**. ⚠️ Le quattro celle `congela` portano lo **stesso** `adapt` di quelle di
E3-C, quindi `e3s` è dovuta entrare anche in `ESPERIMENTI_ESTESI` o
`trova_celle` avrebbe sollevato «due cartelle per lo stesso (modello, seme)»; un
test lo difende con la contro-prova. ⚠️ La colonna della **latenza resta vuota**
per E3-S: E5b non l'ha profilata, e una cella `fondi` somiglia alla riga
`vptshallow-n8` senza esserlo, perché il backbone non è il congelato.

**Artefatti:** `runs/{run_id}/{run_id}__mivia/` per ogni cella — **dentro** la
cartella del run sorgente dal 22 settembre 2026, con dentro `scores.jsonl` (3485
righe, una per finestra, stesso schema di UCF-Crime), `metrics.json`,
`config.yaml` e `meta.json` — più le sezioni **6** e **6b** di
`runs/classifiche/classifiche.md` e `runs/classifiche/cross_dataset_mivia.csv`.

⚠️ **Il `run_id` resta `{run_id}__mivia`**, cioè il nome della cartella, non il
percorso: `exp/demo.apri_run` lo legge da lì, quindi il dimostratore rigioca una
valutazione esterna in `--modo replay` esattamente come prima e la riga di
registry continua a puntare alla cosa giusta.


## E3-R.2 / E3-P.2: ranghi e token su tre semi (concluse il 7 settembre)

**36/36 celle concluse**, 12 configurazioni sui semi 42, 67, 104; 6 celle
r32/deep n576 riusate e 30 nuove. Tutti i 20 job finali delle repliche
COMPLETED, exit 0:0; ultima fine alle 19:34:45 CEST.
Recuperati e verificati 459 file tramite SHA256; 20 nuove righe aggiunte al
registry locale. Checkpoint conservati sul cluster.

[Dossier principale](15_e3_ablation_tre_semi.md),
[strati completi](15_e3_ablation_strati.md),
[manifest di provenienza](varie/e3-ablation-provenienza-20260907.json).

Medie e SD campionarie sui tre semi. Le config di ogni ablation differiscono
solo per il seme. I conteggi dei parametri riguardano il solo adattatore.
La probe e' un riferimento storico: pin_memory/persistent_workers differiscono
fra s42 e s67/104. Non si seleziona un nuovo vincitore sul test.

| Metodo | Param. adattatore | AUC | AP | AUC_a | AP_a |
| --- | --- | --- | --- | --- | --- |
| Probe attentive | 0 | 0.9310 ± 0.0080 | 0.7632 ± 0.0092 | 0.8355 ± 0.0053 | 0.7873 ± 0.0041 |
| LoRA r8 | 3.538.944 | 0.9385 ± 0.0037 | 0.7868 ± 0.0119 | 0.8537 ± 0.0070 | 0.8137 ± 0.0101 |
| LoRA r16 | 7.077.888 | 0.9375 ± 0.0027 | 0.7848 ± 0.0068 | 0.8569 ± 0.0059 | 0.8188 ± 0.0096 |
| LoRA r32 | 14.155.776 | 0.9392 ± 0.0015 | 0.7889 ± 0.0055 | 0.8627 ± 0.0027 | 0.8270 ± 0.0059 |
| LoRA r64 | 28.311.552 | 0.9382 ± 0.0033 | 0.7885 ± 0.0099 | 0.8619 ± 0.0020 | 0.8289 ± 0.0024 |
| VPT deep n8 | 196.608 | 0.9317 ± 0.0024 | 0.7622 ± 0.0035 | 0.8382 ± 0.0019 | 0.7882 ± 0.0014 |
| VPT deep n64 | 1.572.864 | 0.9281 ± 0.0076 | 0.7455 ± 0.0172 | 0.8262 ± 0.0122 | 0.7733 ± 0.0113 |
| VPT deep n192 | 4.718.592 | 0.9244 ± 0.0019 | 0.7319 ± 0.0005 | 0.8127 ± 0.0016 | 0.7577 ± 0.0024 |
| VPT deep n576 | 14.155.776 | 0.8882 ± 0.0083 | 0.6279 ± 0.0201 | 0.7549 ± 0.0073 | 0.6779 ± 0.0141 |
| VPT shallow n8 | 8.192 | 0.9360 ± 0.0030 | 0.7690 ± 0.0081 | 0.8378 ± 0.0068 | 0.7901 ± 0.0060 |
| VPT shallow n64 | 65.536 | 0.9331 ± 0.0008 | 0.7668 ± 0.0099 | 0.8333 ± 0.0032 | 0.7887 ± 0.0075 |
| VPT shallow n192 | 196.608 | 0.9335 ± 0.0055 | 0.7625 ± 0.0182 | 0.8353 ± 0.0115 | 0.7859 ± 0.0156 |
| VPT shallow n576 | 589.824 | 0.9333 ± 0.0041 | 0.7635 ± 0.0164 | 0.8354 ± 0.0115 | 0.7888 ± 0.0179 |

**Interpretazione.** LoRA r32 ha la media maggiore, ma supera r8 solo
a s42: delta medio +0,00075 AUC e +0,00210 AP, senza ordinamento stabile.
VPT-deep n8 supera n576 in tutti i semi (+0,04346 AUC e +0,13431 AP).
VPT-shallow n8 supera la probe in tutti i semi (+0,00505 AUC e +0,00573 AP),
con il limite operativo della baseline dichiarato sopra. A pari parametri,
shallow n192 e deep n8 non si separano in modo consistente fra semi.

L'ordinamento storico reparam/probe/prompt descrive la matrice principale
a budget comparabile, non ogni configurazione VPT. Deep n576 conserva il
recall piu' alto su Shoplifting (0,858), da leggere insieme alla precisione
e alle soglie diverse. Il dossier conserva tutti i risultati appaiati.

**Verifica numerica:** score, etichette e griglia riletti; AUC/AP, Youden,
evento e latenza storica riprodotti su 39/39 celle (36 ablation + 3 probe).
Latenza per intersezione aggiunta come diagnostica: mediane medie
2,73-2,97 s, escluse decodifica e calcolo. Non e' una misura end-to-end.

Gli errori operativi delle riprese sono risolti e conservati nello storico.
I job finali non misurano da soli il costo totale di training.
Nessun test di suite, training o inferenza per questa analisi.

<a id="analisi-20260905"></a>

## Analisi del 5 settembre — complementarità, FAR e profili su due GPU

**Stato:** complementarità e FAR concluse su 15 celle; profili schema 2
completati il 4 settembre e recuperati il 5. Nessun nuovo training per queste
analisi. Le ablation n64/r8 hanno una sottomissione, non un esito documentato.

**Fonte per la stesura:** [dossier 14](14_analisi_complementari_stesura.md),
con tutte le tabelle, metodo, interpretazioni ammesse, limiti e provenienza.
I numeri sono conservati in `docs/`, oltre che negli artefatti gitignored.

### Domande e risultati

| Analisi | Risultato | Limite che deve accompagnarlo |
|---|---|---|
| Complementarità chunk | Accordo attentive/VPT **0,846 ± 0,038**; LoRA/DoRA **0,967 ± 0,006** | Decisioni alle rispettive soglie Youden; l'accordo non isola le rappresentazioni |
| Complementarità evento | Recall attentive **0,882 ± 0,015**, VPT **0,937 ± 0,037**, unione **0,963 ± 0,019** | Precision e F1 dell'unione non misurate; nessun ensemble validato |
| Shoplifting | Recall medio attentive **0,575**, VPT **0,858**, unione **0,867**, su **40 eventi** | Gli stessi eventi su tre semi; il recupero varia da 3 a 16 eventi fra semi |
| FAR, target val 1 allarme/h | FAR test: attentive **1,276 ± 0,254**, VPT **0,888 ± 0,673**, LoRA **6,435 ± 0,508**, DoRA **5,825 ± 0,725** | Ore di video completamente normali; val e test hanno video e griglie diversi |
| Profili schema 2 | **24/24 celle valide su A100 e 24/24 su H100**, batch 1/2/4/8 | Modello su input sintetici; non è una misura multi-camera |
| LoRA fusa, batch 1 | **131,71 ms A100**, **77,70 ms H100** | Una passata del benchmark per GPU; non attribuire significato alle differenze minime dal congelato |

Media ± deviazione standard **campionaria sui tre semi** `[42, 67, 104]` per
le analisi di score; i tempi hardware sono medie sulle iterazioni del profiler,
non sui semi di training. Le due dispersioni non sono intercambiabili.

### Metodo e portata

La rilettura usa `scripts/analisi_complementarita_far.py` e verifica per tutti
i 15 run score, etichette, span e conteggi evento contro gli originali. Produce
30 confronti chunk e 45 punti FAR, con soglie scelte soltanto sulla validazione.
Il FAR conta gli allarmi fusi sui video senza eventi per ora di durata nativa.
Poiché fondere allarmi rende il conteggio non monotono, il vincolo si applica
all'inviluppo superiore dei FAR a tutte le soglie candidate maggiori.

La complementarità sostiene un'analisi degli errori; non dimostra il vantaggio
di un ensemble distribuibile. I punti FAR misurano il trasferimento di una
soglia; non modificano il protocollo ufficiale e non garantiscono un tasso di
allarmi in deployment. L'oracolo chunk usa le etichette di test e va presentato
soltanto come limite superiore diagnostico.

I profili `e5b2_efficienza_gpuq` e `e5b2_efficienza_aiq` hanno registry `ok`,
commit `b508f860` e `profile_schema: 2`; includono percentili, energia GPU
campionata e avvio a freddo. Il campo storico `params_da_spedire` esclude la
testa: non usarlo per quantificare il payload totale di distribuzione.

### Aperto

- VPT-deep n64, seme 42: job **558264**, ripresa **558266**.
- LoRA r8, alpha16, seme 42: job **558265**, ripresa **558267**.

Stato riportato dalla verifica del 5 settembre: sottomessi e in coda. LR 1e-4,
5524 passi, sei validazioni, budget originale. Sono ablation esplorative a un
seme; qui non si attribuiscono loro accuratezza o costi misurati. Percorsi,
config e criteri di chiusura nella [scheda operativa](varie/esperimenti-20260905.md).

## E-strati — Dove il sistema va meglio e peggio *(chiuso: il bordo della finestra costa 3,8× di AP, e il collo di bottiglia ha un nome — Shoplifting)*

**Data:** 4 settembre 2026 · **costo:** zero ore di GPU, ~4 minuti di portatile ·
**ADR-060** · artefatti: `runs/*/strata.json`, `runs/*/strata.png`,
`runs/figures/strata_tavola.{csv,md}`.

**Cos'è.** Non un run: una **ri-analisi** delle predizioni già pagate. Le quindici
celle di `edp_v2` chiuse fra il 1 e il 3 settembre portano in `scores.jsonl` uno
score per finestra (ADR-051); `scripts/analisi_strati.py` li rilegge e li
verticalizza lungo i quattro assi di ADR-060. È la promessa di ADR-051 usata per
ciò per cui era stata scritta: 15/15 celle, nessuna GPU, secondi ciascuna.

⚠️ **È diagnostica, non protocollo.** Nessun numero della Tavola cambia. In tesi
va presentata come analisi degli errori, non come risultato confrontabile coi
numeri del laboratorio.

**Media ± deviazione su tre semi** (42, 67, 104) per tutte e cinque le colonne.

### 1. Il risultato più netto: il **bordo** della finestra

AP dei chunk positivi contro tutti i negativi, per frazione della finestra coperta
dall'evento:

| copertura | n_pos | e2c linear | e2d attentive | e3p VPT | e3r LoRA | e3r DoRA |
|---|---|---|---|---|---|---|
| ≤25% | 192 | 0,0596 ± 0,0006 | 0,1728 ± 0,0243 | 0,1008 ± 0,0121 | 0,2027 ± 0,0158 | **0,2051 ± 0,0166** |
| 25–50% | 234 | 0,0955 ± 0,0009 | 0,3491 ± 0,0205 | 0,2075 ± 0,0300 | 0,4145 ± 0,0093 | **0,4174 ± 0,0140** |
| 50–75% | 186 | 0,0867 ± 0,0014 | 0,3622 ± 0,0130 | 0,2414 ± 0,0194 | **0,4239 ± 0,0123** | 0,4189 ± 0,0170 |
| >75% | 879 | 0,4267 ± 0,0071 | 0,7452 ± 0,0069 | 0,5707 ± 0,0194 | 0,7754 ± 0,0097 | **0,7831 ± 0,0054** |

**Su DoRA l'AP passa da 0,205 a 0,783 fra il bordo e il pieno: 3,8×.** L'AUC si
muove molto meno (0,89 → 0,96 su E2d), ed è la ragione per cui questo asse va
letto in AP: l'AUC resta alta perché i negativi restano facili, mentre l'AP dice
che *fra le finestre che il sistema segnala* quelle di bordo sono le sbagliate.

⚠️ **La forma è la stessa per tutti e cinque i metodi**, il che la rende una
proprietà del **protocollo** e non del metodo: è la regola «intersezione
qualunque» (ADR-031 punto 3) che dichiara positivo un chunk toccato da un solo
frame, e nessun adattamento la compensa. Il pavimento a 0,1 di `w_overlap`
(invariante 6) non basta a renderli imparabili.

⚠️ **Il linear ha recall più alto dell'attentive in ogni fascia** (0,854 contro
0,627 sul bordo) e AP tre volte più bassa: è la sua soglia bassa, non una
capacità. Va letto con la precision di chunk aggregata — **0,280** contro 0,568.

### 2. La classe: **Shoplifting** è il collo di bottiglia

Recall a livello di evento (frazione degli eventi annotati che almeno una
finestra sopra soglia interseca):

| classe | n_eventi | e2d attentive | e3p VPT | e3r LoRA | e3r DoRA |
|---|---|---|---|---|---|
| **Shoplifting** | 40 | **0,575 ± 0,090** | 0,858 ± 0,118 | 0,617 ± 0,161 | 0,658 ± 0,088 |
| Arrest | 14 | 0,905 ± 0,083 | 0,976 ± 0,041 | 0,952 ± 0,041 | 0,929 ± 0,000 |
| Burglary | 27 | 0,914 ± 0,021 | 0,963 ± 0,037 | 0,926 ± 0,037 | 0,901 ± 0,021 |
| Shooting | 46 | 0,928 ± 0,033 | 0,957 ± 0,022 | 0,877 ± 0,045 | 0,891 ± 0,022 |
| RoadAccidents | 24 | 0,861 ± 0,064 | 0,819 ± 0,064 | 0,972 ± 0,024 | **0,986 ± 0,024** |
| Fighting | 10 | 1,000 | 1,000 | 1,000 | 1,000 |

**Shoplifting è a 0,58 dove ogni altra classe sta sopra 0,86**, ed è la stessa
classe che E3 aveva già segnalato come «la più difficile per tutti» — l'unica
dove VPT-deep *migliorava* sulla probe congelata. Qui si vede perché quella
osservazione non era un dettaglio: **VPT prende 0,858 dove l'attentive prende
0,575**, cioè +0,28 di recall sulla classe peggiore, pur essendo il metodo con la
chunk-AP più bassa dei quattro. ⚠️ Non lo si spiega e non gliene si attribuisce
una spiegazione: il fatto è misurato, il meccanismo no.

**Dove il reparam guadagna davvero**: RoadAccidents (+0,125 di DoRA sull'attentive)
e Robbery (+0,042). Sono classi **brevi** — RoadAccidents ha durata mediana 2,9 s,
la seconda più corta del test.

### 3. La durata: monotona, e il taglio è a 32 s

Recall di evento per fascia di durata, su E2d (le altre colonne hanno la stessa
forma):

| durata | n_eventi | recall |
|---|---|---|
| ≤4 s | 89 | 0,839 ± 0,017 |
| 4–8 s | 77 | 0,866 ± 0,015 |
| 8–32 s | 79 | 0,899 ± 0,046 |
| >32 s | 31 | **1,000 ± 0,000** |

**Gli eventi oltre i 32 s si trovano tutti, su tutti i metodi e tutti i semi**
(VPT fa 0,989). Sotto, il recall scende monotonicamente. La lettura è la geometria:
a hop 4 s e span 7,75 s le finestre che intersecano un evento lungo `d` sono
circa `(d + 7,75)/4`, quindi **due o tre** sotto i 4 s e **almeno dieci** sopra i
32 — e per essere rilevato ne basta una.

⚠️ **`n_eventi` ha deviazione zero su ogni fascia e ogni metodo** (89/77/79/31 =
276): la stratificazione della verità non dipende dal modello, ed è il controllo
che rende leggibile il resto. La somma torna anche sui rilevati — 247 su E2d s42,
cioè esattamente il `n_eventi_rilevati` di `event_prf`.

### 4. La rarità: **non** è monotona, ed è per metà un'altra cosa

| quota anomala del video | n_eventi | e2d attentive | e3r DoRA |
|---|---|---|---|
| ≤5% | 17 | 0,902 ± 0,090 | 0,961 ± 0,034 |
| **5–20%** | 91 | **0,736 ± 0,022** | **0,802 ± 0,040** |
| 20–50% | 118 | 0,955 ± 0,018 | 0,961 ± 0,005 |
| >50% | 50 | 0,967 ± 0,023 | 0,967 ± 0,012 |

L'ipotesi «più raro = più difficile» **non si verifica**: la fascia peggiore è
quella *intermedia*, e la più rara (≤5%) è fra le migliori — su soli 17 eventi.

⚠️ **E per metà è Shoplifting.** L'incrocio classe × fascia lo dice: **27 dei 91
eventi della fascia 5–20% sono Shoplifting** (il 30%), e 27 dei 40 Shoplifting
stanno lì. Escludendo la classe, il recall della fascia sale di ~0,10 —
**0,736 → 0,828** su E2d s42, **0,802 → 0,906** su DoRA s42 — **ma la fascia resta
la peggiore** (contro 0,96–0,98 delle altre) su tutti e sei i run controllati.
Quindi: metà dell'effetto è la classe, metà no, e nessuna delle due frasi da sola
sarebbe vera.

### 5. Un fatto sul metodo di valutazione, non sul modello

Le F1 a livello di **evento** di linear e attentive sono **0,6775** e **0,6778** —
indistinguibili — mentre la chunk-AP differisce di **0,28** (0,479 contro 0,763).
Il livello evento è quindi molto più **grosso** del livello chunk: perdona un
rilevatore temporalmente impreciso, perché gli basta un frame di intersezione
(ADR-048). Va detto in tesi accanto ai numeri di evento, altrimenti la coppia
sembra dire che la linear regge — e §E2 ha già stabilito che non regge.

### Cosa NON è misurato

- ⚠️ **Nessuno di questi assi è controllato per gli altri**: durata, classe e quota
  anomala sono correlate (Shoplifting ha durata mediana 4,9 s — quinta più corta
  su tredici — *e* sta per due terzi nella fascia 5–20%), e
  l'unico incrocio calcolato è quello del punto 4. Separarli chiederebbe un
  disegno fattoriale che questi dati non hanno.
- ⚠️ **Le fasce sono una scelta**, dichiarata e ancorata alla geometria; i bordi
  viaggiano in `strata.json` e si cambiano da riga di comando.
- ⚠️ **`Abuse` ha 2 eventi e 6 chunk positivi**: la sua riga non è riportabile da
  sola, e lo stesso vale per la fascia ≤5% di quota (17 eventi).

---

## E5b — L'efficienza a inferenza, la metà mancante di C7 *(chiusa: senza il merge il reparam è il più caro dei tre)*

**Chiusa il 3 settembre 2026** · `src/evd/exp/e5b.py`,
`scripts/e5b_efficienza.py`, `scripts/slurm/e5b.sbatch` · job **542380**,
`gnode03`, **A100-SXM4-80GB**, `gpuq`, **79 secondi** · run_id
`e5b_efficienza`, artefatto `efficiency_inference.json` · claim **C5** e **C7**.

⚠️ **La partizione è A100 e non è un dettaglio**: le latenze di H100 e A100 non
sono confrontabili fra loro, e i costi di *training* di E3-X.1 sono misurati su
A100 — la tabella congiunta è quindi omogenea per costruzione. Le colonne di
**overhead** sono rapporti interni alla stessa esecuzione e resterebbero
confrontabili anche fra macchine: è quello il numero di C7.

> E3-X.1 ha risposto alla metà **accuratezza** di C7 e ha consegnato il costo di
> **training** misurato dai log. Resta la metà su cui il PdD fonda il claim:
> «le famiglie prompt-based e reparameterization hanno trade-off **opposti**
> (training vs inferenza)». Fin qui l'asimmetria a inferenza era
> un'affermazione strutturale — reparam si riassorbe nei pesi, i prompt no — e
> questa voce esiste per trasformarla in un numero.

### Cosa chiede

Sei varianti, per batch 1 e 2, alla geometria di `edp_v2` (32 campioni, 384 px,
layer 23, bf16):

| variante | perché c'è |
|---|---|
| `frozen` | il pavimento: nessun adattatore, nessun token in più. È il **riferimento** di ogni colonna di overhead |
| `vptdeep-n576` | la cella di E3-P.1 |
| `lora-r32` · `dora-r32` | le due celle di E3-R.1, **non fuse**: è il modello così com'è dopo il training |
| `lora-r32-fusa` · `dora-r32-fusa` | lo stesso modello dopo `merge_()` + `unload_()`, cioè **quello che si manderebbe in produzione**. Non sono run: sono la forma di deployment di due run |

⚠️ **VPT non ha una riga fusa, e non è una dimenticanza**: `can_merge()` è falso
per costruzione — non esiste un peso in cui una sequenza più lunga si riassorba —
e quell'assenza **è** il risultato che C7 cerca.

### Le tre metriche, e cosa ognuna può dire

1. **Latenza per finestra a batch 1** — l'unica che parla di *deployment*: è il
   tempo di risposta che un sistema online paga per una finestra sola. Da qui il
   **fattore real-time**: in streaming ogni finestra copre `hop_s = 4 s` di video
   nuovo, quindi sopra 1,0 il sistema sta al passo. ⚠️ È il **modello soltanto**.
2. **Throughput a batch 2** — il batch con cui E3 ha valutato, quindi la riga
   confrontabile con quei log.
3. **FLOP dichiarati** — non per sostituire i secondi ma per **spiegarli**, e
   perché correggono un numero che `docs/05` §6 riporta e che è facile leggere
   male: 576 token su 9216 sono il **+6,25% della sequenza**, ma a quella
   lunghezza il termine quadratico dell'attenzione (`4·T²·D`) è **più grande** di
   quello lineare (`24·T·D²`), quindi il lavoro cresce di **~+10,2%**. Il 6,25%
   è vero e non è l'overhead di costo.

Più una quarta colonna che nessuna riga di latenza mostra: **quanti parametri si
spediscono**. Non fuso si manda il delta (14,2 M); fuso la separazione fra base e
delta non esiste più e si manda il **backbone intero** (~300 M). È il prezzo
nascosto del merge, e va nella tabella accanto al suo beneficio.

### Come è costruito, e le quattro scelte che lo rendono confrontabile

1. **Si cronometra `HeadScorer`**, l'oggetto che `InferenceEngine` chiama per ogni
   finestra e che ha prodotto i numeri di E3 — non un forward scritto per
   l'occasione, che misurerebbe sé stesso (lezione di E0b.2 punto 1). Per la
   stessa ragione la precisione è **bf16**: `metrics.json` di tutte e cinque le
   celle dice `eval_precision: bf16`.
2. **Le varianti si costruiscono dai config veri di `configs/adapt/`**, e un test
   parametrizzato verifica che ogni nome punti a un file esistente e costruibile.
   È il rimedio di ADR-056: un finto scritto oggi replica l'assunzione sbagliata
   di ieri.
3. **I pixel sono sintetici, il decoder è fuori.** ⚠️ **È il limite dichiarato di
   E5b**: dice quanto costa il **modello**, non il **sistema**. Il costo del
   decoder è misurato da E0d (2,4 fin/s per worker, 25,1 a otto), quindi
   l'end-to-end si **compone** dai due termini con i termini in vista, invece di
   essere rimisurato.
4. **Il merge si verifica prima di essere cronometrato**, in fp32 per separare
   l'algebra dal rumore di bf16. Un merge veloce che cambia l'uscita non è
   un'ottimizzazione: è un altro modello.

### Due difetti trovati costruendolo, e vale la pena averli scritti

**(a) `merge_()` scrive in place, e il profilo di E0b.2 non aveva questo
problema perché non fondeva mai niente.** Dopo una cella fusa il backbone **non
è più** quello di partenza: la variante successiva misurerebbe pesi corrotti, e
non solleverebbe nulla — *un tempo plausibile su un modello sbagliato ha
esattamente lo stesso aspetto di un tempo giusto*. Il rimedio è un ripristino da
una copia su CPU **verificato con un digest**, non dato per fatto.

**(b) La verifica del merge, com'era scritta, non poteva fallire** — e a
mostrarlo è stata la prima esecuzione, non una rilettura. I config di E3 usano
`init: kaiming_b_zero`: all'avvio `B = 0`, quindi ΔW è **esattamente zero**,
l'adattatore è l'identità, e fondere uno zero riesce sempre. Il sintomo che l'ha
tradita: LoRA e DoRA davano lo **stesso** scarto al settimo decimale (3,874e-07),
che è il modo in cui due cose diverse si somigliano quando nessuna delle due sta
facendo niente. Ora i parametri dell'adattatore si **perturbano** con rumore
seminato prima del controllo, e la riga registra `effetto_adattatore` — di
quanto l'adattatore sposta l'uscita — così che uno scarto piccolo significhi
«merge fedele» e non «non c'era niente da fondere». Una riga esce `sospetta` in
**due** casi diversi, ed è giusto distinguerli: il merge ha cambiato l'uscita,
oppure il merge non è stato messo alla prova.

> **La forma generale, che vale oltre il caso**: un controllo di fedeltà eseguito
> su un'inizializzazione che rende l'operazione un no-op verifica il no-op. Vale
> per ogni `init` che parte dall'identità — ed è la scelta giusta per il
> training, quindi il conflitto è strutturale e si ripresenterà.

### Verificato prima di sottomettere

Su ViT-B/224 in locale (numeri **non** di tesi, rule 4 del workspace): il merge è
fedele su **entrambe** le famiglie e con scarti **diversi** fra loro —
`max_abs` 1,49e-08 (LoRA) e 6,41e-07 (DoRA), entrambi all'epsilon di fp32 — con
l'adattatore che sposta l'uscita del 5,3% e del 3,3%. Il digest del backbone
torna identico dopo tutte e sei le varianti. È la prova che il meccanismo regge
sull'architettura **vera**, con la QKV fusa, che `TinyBackbone` non esercita.

### Cosa è uscito *(job 542380, `gnode03`, A100-SXM4-80GB, 3 settembre, 79 secondi)*

Le deviazioni standard stanno fra **0,06 e 0,15 ms** su latenze di 131–459 ms,
cioè ≤ 0,06%: i divari riportati qui sotto valgono centinaia di deviazioni, e
nessuna conclusione poggia sul rumore.

| variante | batch | latenza (ms) | **overhead** | FLOP previsti | token | VRAM | da spedire |
|---|---:|---:|---:|---:|---:|---:|---:|
| congelato | 1 | 131,4 ± 0,09 | — | — | 9216 | 1,91 GB | 0 |
| VPT-deep n=576 | 1 | 133,6 ± 0,11 | **+1,68%** | +10,23% | 9792 | 1,97 GB | 14.155.776 |
| LoRA r32 | 1 | 187,6 ± 0,14 | **+42,79%** | +0,00% | 9216 | 2,04 GB | 14.155.776 |
| LoRA r32 **fusa** | 1 | 131,8 ± 0,15 | **+0,34%** | +0,00% | 9216 | 1,91 GB | **304.680.960** |
| DoRA r32 | 1 | 241,4 ± 0,10 | **+83,71%** | +0,00% | 9216 | 2,27 GB | 14.376.960 |
| DoRA r32 **fusa** | 1 | 131,8 ± 0,08 | **+0,33%** | +0,00% | 9216 | 1,91 GB | **304.680.960** |

A batch 2 l'ordine non cambia: +0,63% · +44,73% · +0,10% · +83,05% · +0,16%.

### Cosa vuol dire

**1. Il trade-off di C7 esiste, ma non è quello che il PdD descrive — e senza il
merge punta dall'altra parte.** Il claim dice che i prompt costano a inferenza e
il reparam no. Misurato: **VPT-deep costa +1,68%**, cioè praticamente nulla,
mentre il reparam **non fuso** costa **+42,8%** (LoRA) e **+83,7%** (DoRA). Un
adattatore reparam lasciato com'è dopo il training è, a inferenza, il metodo
**più caro dei tre**. La frase «reparam non costa nulla a inferenza» è vera
**solo dopo `merge_()`**, e quella chiamata non è un dettaglio di
implementazione: è la condizione del claim.

**2. Il merge mantiene la promessa, e ha un prezzo che nessuno mette in
tabella.** Fuso, il reparam torna **sul congelato**: 131,8 contro 131,4 ms,
+0,3%, che a 0,1 ms di deviazione è ai margini della misurabilità — e la fedeltà
è verificata, non assunta (scarto 8,94e-08 per LoRA e 4,77e-07 per DoRA, con
l'adattatore che sposta l'uscita del 30,1% e del 4,9%). Il prezzo è che il delta
**smette di esistere come oggetto separato**: si spediscono 304,7 M parametri
invece di 14,2 M, cioè **21,5×**. Chi deve distribuire aggiornamenti, o tenere
più adattatori sullo stesso backbone, paga lì ciò che risparmia in latenza.

**3. Il modello FLOP sovrastima la penale dei prompt di 6,1×.** I 576 token sono
il **+6,25%** della sequenza e il **+10,23%** del lavoro aritmetico — perché a
9216 token il termine quadratico dell'attenzione supera quello lineare — ma la
latenza misurata cresce dell'**1,68%**. La lettura di `docs/05` §6 («il
trade-off del PdD si assottiglia su sequenze video») è quindi giusta nella
direzione e **prudente** nella misura. ⚠️ **Perché il divario sia così grande
questo esperimento non lo stabilisce**, e non gli si attribuisce una causa: a
batch 2 l'overhead di VPT **scende** a +0,63% invece di salire, che è l'opposto
di ciò che un regime compute-bound produrrebbe. Il fatto è misurato, la
spiegazione no.

**4. Real-time con ampio margine, e il collo di bottiglia è il modello.** Con
`hop_s = 4 s`, ogni variante digerisce da **16,6×** (DoRA non fusa) a **30,4×**
(congelato) il tempo reale sul solo modello. ⚠️ **Composizione con E0d**, che è
il modo dichiarato di ottenere l'end-to-end senza rimisurarlo: il decoder dà
**25,1 finestre/s** a otto worker contro le **7,61** del modello congelato,
quindi con la decodifica sovrapposta al forward il sistema resta limitato dal
**modello** e il fattore real-time sopravvive alla composizione. È il conto che
sostiene **C5**.

### Il verdetto congiunto — la tabella di E3-X.1 con l'efficienza accanto

| metodo | chunk-AUC | AP | s/passo | ore/cella | latenza | overhead | da spedire |
|---|---:|---:|---:|---:|---:|---:|---:|
| congelato (E2d) | 0,9385 | 0,7737 | 1,74 | 2,67 | 131,4 ms | — | 0 |
| VPT-deep n=576 | **0,8968** | **0,6509** | 6,59 | 10,11 | 133,6 ms | +1,68% | 14,2 M |
| LoRA r32 | 0,9400 | 0,7910 | 8,85 | 13,58 | 187,6 ms | +42,8% | 14,2 M |
| LoRA r32 **fusa** | 0,9400 | 0,7910 | 8,85 | 13,58 | **131,8 ms** | **+0,34%** | 304,7 M |
| DoRA r32 | 0,9424 | 0,7968 | 11,48 | 17,62 | 241,4 ms | +83,7% | 14,4 M |
| DoRA r32 **fusa** | 0,9424 | 0,7968 | 11,48 | 17,62 | **131,8 ms** | **+0,33%** | 304,7 M |

Letta tutta insieme, la riga che conta è questa: **il reparam fuso domina VPT su
ogni asse tranne uno.** È più accurato (+0,043 di chunk-AUC, +0,14 di AP), costa
**meno** a inferenza (+0,3% contro +1,7%), e perde solo su due cose — le ore di
training (13,6–17,6 contro 10,1) e i byte da spedire (304,7 M contro 14,2 M).
Il trade-off «prompt contro reparam» che il PdD dava per opposto **non si
verifica su questo backbone e a questa geometria**: si riduce a un compromesso
fra *ore di GPU una volta sola* e *megabyte a ogni distribuzione*, che è una
domanda di deployment e non di metodo.

### Aperto

1. ⚠️ **Perché la penale dei prompt sia 6,1× più piccola di quanto i FLOP
   prevedono, questo esperimento non lo dice** — e l'anomalia che lo rende
   interessante è che a batch 2 l'overhead **scende** invece di salire. Serve un
   profilo per kernel, non una riga in più in questa tabella.
2. **Una macchina sola.** I numeri sono A100; su H100 la penale di VPT potrebbe
   cambiare di segno relativo, perché dipende da quanto il forward è
   compute-bound. Il payload registra `gpu`, quindi il confronto si rifarà
   quando servirà, ma **non si estrapola**.
3. **L'end-to-end non è misurato**, per scelta dichiarata (punto 3 sopra). Chi lo
   volesse passa da `InferenceEngine.stream()`, che è la forma live-like già
   usata dal dimostratore: è una gamba in più, non una correzione di questa.

<a id="e5b2-20260904"></a>

## E5b.2 — Dal costo del modello al costo del sistema *(la composizione è chiusa: il ritardo dell'allarme è per il 93% griglia, non calcolo)*

**Aperta il 4 settembre 2026** · `src/evd/eval/deployment.py`,
`src/evd/eval/deployment_plots.py`, `src/evd/exp/e5b_deploy.py`,
`scripts/e5b_deployment.py` · artefatti `deployment.json`, `deployment.md` e
quattro PNG in `runs/e5b_efficienza/` · ADR-059 · claim **C5** e **C7**.

⚠️ **Nota del 24 settembre, alla fusione del branch: la latenza composta qui è
quella della regola STORICA.** `exp/e5b_deploy.py` legge `detection_latency` da
`metrics.json`, cioè il predicato «endpoint dentro l'evento». La sera stessa,
sulla linea principale, **ADR-061** ha stabilito che in tesi si riporta
`detection_latency_intersezione`, che fa rientrare fra 60 e 71 eventi per cella e
alza la mediana di ~0,45 s (E2d s42: 2,53 → 2,97 s). Il **2,57–2,68 s** e il **93%
griglia** di questa voce sono quindi coerenti con la regola vecchia. La stessa
composizione con la regola **nuova** esiste già sulla linea principale, in
`runs/classifiche/classifiche.md` §3 (media sui tre semi invece della cella
mediana): sulle sei varianti di E5b il ritardo è **2,91–3,14 s** e la quota di
calcolo **5,5–9,0%**. **La conclusione regge** — il ritardo è quasi tutto griglia, e
accelerare il modello non rende il sistema più reattivo — mentre i numeri puntuali
di questa voce e di D-bis non sono quelli da citare. `exp/e5b_deploy.py` legge
ancora il campo storico: allinearlo è una modifica che cambia artefatti, e non si
fa alla fusione.

⚠️ **Dove stanno gli artefatti, e dove staranno.** Quelli di oggi sono in
`runs/e5b_efficienza/`, accanto al profilo schema 1 che li ha alimentati (job
542380). Il job schema 2 scrive in `runs/e5b2_efficienza/` — cartella **diversa**
di proposito, perché la cartella di un run è la sua identità su disco e riusare
il nome vecchio distruggerebbe la sorgente di §E5b — e compone lì il suo studio.
I due resteranno affiancati, ed è giusto: descrivono due misure diverse.

**Stato: CHIUSA su DUE macchine.** La composizione ha i suoi numeri dal 4
settembre mattina; i due profili **schema 2** sono girati il pomeriggio dello
stesso giorno su **A100** (job 557837, `gnode03`) e **H100** (job 557836,
`ainode01`), e il confronto fra le due sta in `runs/figures/`.

> E5b ha misurato il modello e lo ha dichiarato: *«dice quanto costa il modello,
> non quanto costa il sistema»*. Questa voce chiude quella frase — e la chiude
> **componendo misure già pagate** invece di rimisurarle, che è la ragione per
> cui la parte più importante dello studio non è costata un secondo di coda.

### La domanda, e perché una latenza di forward non le risponde

Un capitolo di deployment deve saper dire tre cose che 131 ms non dicono:
*quanto tarda l'allarme*, *quante telecamere regge una GPU*, *quanto costa
masticare un archivio*. E deve dirle sapendo che il sistema non è solo il
modello: c'è un decoder davanti e una griglia di finestre attorno.

**Tre regimi, perché sono tre domande** — e un solo numero di «throughput» ne
risponderebbe a una:

1. **online, una telecamera** — sta al passo? qual è il tempo di risposta?
2. **online, molte telecamere** — quante ne regge una GPU, e chi è il vincolo?
3. **offline, archivio** — quanto ci mette a masticare un'ora di registrato?

### Da dove vengono i numeri, e il limite dichiarato

| termine | sorgente | dettaglio |
|---|---|---|
| **modello** | `e5b_efficienza` (job 542380) | A100-SXM4-80GB, bf16, batch 1 |
| **decoder** | `e0d_..._throughput-aug` (job 540018) | **8 worker**, `augment: none`, **40,8 ms/finestra** |
| **risposta all'evento** | `detection_latency` di E2/E3 | seme mediano per chunk-AUC, 3 semi per cella |
| **accuratezza** | `chunk_auc`/`chunk_ap`/`event_prf` idem | media ± deviazione su **3 semi** (ADR-058) |
| **verifica** | `registry.jsonl` | sei passate di test su `ainode02` |

⚠️ **Il decoder è la gamba `throughput-aug` e non `throughput`**, e la scelta
conta: la prima gira su un **nodo con GPU** con 8 CPU, cioè la configurazione in
cui un job vero decodifica; la seconda su un nodo CPU a 32 core, che dà 54
finestre/s su una macchina che non è quella del deployment. E `augment: none`
perché l'augmentation è di training (ADR-028) e a inferenza non gira — la riga
`official_nocrop` esiste nell'artefatto e dice di quanto il numero cambierebbe
se girasse (20,45 contro 25,33 fin/s).

⚠️ **I due termini di calcolo vengono da macchine diverse**, e nessuno dei due
include acquisizione RTSP, buffering di rete o dispatch dell'allarme. La
conclusione qui sotto sopravvive **per un margine e non per un pareggio**: il
modello domina il decoder di un fattore ≥ 3, quindi cambiare il termine incerto
del 50% non cambia chi comanda.

**Somma e massimo hanno due nomi.** Decoder e modello stanno su device diversi:
in regime stazionario si sovrappongono e il tempo per finestra è il **massimo**
(la stessa ragione per cui `num_workers` esiste, ADR-038); la latenza di **una**
risposta è la **somma**, perché quella finestra lì va prima decodificata e poi
valutata. Confonderli è il modo classico di sbagliare un conto di pipeline.

### 1. Sta al passo? *(online, una telecamera)*

| variante | modello | decoder | **risposta** | margine su hop | real-time | fin/s |
|---|---:|---:|---:|---:|---:|---:|
| congelato | 131,4 ms | 40,8 ms | **172,2 ms** | 3.828 ms | 30,4× | 7,61 |
| VPT-deep n=576 | 133,6 | 40,8 | **174,4** | 3.826 | 29,9× | 7,48 |
| LoRA r32 | 187,6 | 40,8 | **228,4** | 3.772 | 21,3× | 5,33 |
| LoRA r32 **fusa** | 131,8 | 40,8 | **172,6** | 3.827 | 30,3× | 7,58 |
| DoRA r32 | 241,4 | 40,8 | **282,2** | 3.718 | 16,6× | 4,14 |
| DoRA r32 **fusa** | 131,8 | 40,8 | **172,6** | 3.827 | 30,3× | 7,59 |

Anche la variante peggiore usa **il 7% del budget** che l'hop le concede: 282 ms
su 4000. Il decoder è sempre il termine minore, quindi il **collo di bottiglia è
il modello** — ed è il conto che sostiene **C5**.

### 2. Quante telecamere per GPU?

| variante | tetto modello | tetto decoder | **sistema** | vincolo |
|---|---:|---:|---:|---|
| congelato · LoRA fusa · DoRA fusa | 30,4 · 30,3 · 30,3 | 98,1 | **30** | modello |
| VPT-deep n=576 | 29,9 | 98,1 | **30** | modello |
| LoRA r32 | 21,3 | 98,1 | **21** | modello |
| DoRA r32 | 16,6 | 98,1 | **17** | modello |

⚠️ **È un limite superiore aritmetico**: nessuno scheduling, nessuna contesa,
nessun I/O di rete, e a batch 1 sequenziale. Sostituisce l'estrapolazione che
E5b dichiarava di non aver misurato — con **la stessa cautela**, perché resta
aritmetica: il numero *misurato* arriverà dallo sweep dei batch (sotto).

### 3. Quanto tarda l'allarme? — **il risultato di questa voce**

| variante | griglia (s) | + calcolo | **totale mediana** | totale p90 | **quota di calcolo** | eventi con latenza |
|---|---:|---:|---:|---:|---:|---:|
| congelato | 2,47 | 0,172 | **2,64 s** | 4,88 s | **6,5%** | 64% |
| VPT-deep n=576 | 2,40 | 0,174 | **2,57** | 4,14 | **6,8%** | 74% |
| LoRA r32 | 2,43 | 0,228 | **2,66** | 4,38 | **8,6%** | 68% |
| LoRA r32 **fusa** | 2,43 | 0,173 | **2,61** | 4,33 | **6,6%** | 68% |
| DoRA r32 | 2,40 | 0,282 | **2,68** | 4,25 | **10,5%** | 66% |
| DoRA r32 **fusa** | 2,40 | 0,173 | **2,57** | 4,14 | **6,7%** | 66% |

**Il ritardo con cui il sistema segnala un evento è per il 93% geometria e per
il 7% calcolo** — e sui metodi *non cambia*: 2,57–2,68 s, cioè un'escursione del
4% fra il metodo più veloce e il più lento, mentre le loro latenze di calcolo
differiscono del **64%**. Tre conseguenze, e la prima è la più importante.

**(a) Accelerare il modello non accorcia la risposta del sistema.** Portare il
calcolo a zero — GPU infinitamente veloce — farebbe scendere il ritardo da 2,64 a
2,47 s: il **6,5%**. Chi volesse un sistema più reattivo deve toccare l'**hop**
della griglia, non l'adattatore. È la riga che dice dove va speso lo sforzo di
ottimizzazione, e nessuna tabella di latenza da sola la contiene.

**(b) La scelta del metodo è quindi libera sull'asse della reattività.** DoRA non
fusa costa +83,7% di latenza e arriva **0,04 s dopo** il congelato: sul ritardo
percepito è indistinguibile. Il trade-off di C7 si gioca altrove — accuratezza,
ore di training, byte da spedire — e *non* sulla prontezza dell'allarme.

**(c) La copertura va letta accanto alla mediana, e le due si muovono in
direzioni opposte.** VPT ha la mediana più bassa (2,40 s) *e* la copertura più
alta (74%), ma è il metodo meno accurato: la latenza qui è calcolata sui soli
eventi **misurabili**, e una mediana sui soli misurabili *migliora* quando il
sistema ne manca di più (ADR-057). Le due colonne stanno insieme per questo.

### 4. Archivio, e cosa costa distribuirlo

| variante | s per ora di video | ×tempo reale | test set | chunk-AUC (3 semi) | AP |
|---|---:|---:|---:|---:|---:|
| congelato | 118 s | 30,4× | 20,0 min | 0,9310 ± 0,0080 | 0,7632 |
| VPT-deep n=576 | 120 | 29,9× | 20,3 | **0,8882 ± 0,0083** | 0,6279 |
| LoRA r32 | 169 | 21,3× | 28,5 | 0,9392 ± 0,0015 | 0,7889 |
| LoRA r32 **fusa** | 119 | 30,3× | 20,0 | 0,9392 ± 0,0015 | 0,7889 |
| DoRA r32 | 217 | 16,6× | 36,7 | **0,9422 ± 0,0017** | 0,7947 |
| DoRA r32 **fusa** | 119 | 30,3× | 20,0 | **0,9422 ± 0,0017** | 0,7947 |

⚠️ **Le colonne di accuratezza vengono dalla [Tavola](#-tavola-dei-numeri--tutto-ciò-che-si-può-scrivere-in-tesi)
A e coincidono con essa**; stanno qui solo per poter leggere costo e qualità
sulla stessa riga. **Il confronto fra metodi non si fa su queste medie**: si fa
**appaiato**, seme per seme, in **A-bis** — con tre semi un t ha 2 gradi di
libertà e solo gli effetti grandi concludono. Sottrarre due medie qui sarebbe
buttare via proprio ciò che i tre semi comprano.

⚠️ E la lettura di **stabilità** — σ del congelato e di VPT cinque volte quella
di LoRA e DoRA — è già nella Tavola, **punto E.4**, con il caveat che le serve:
con n=3 le σ sono stime grossolane e **non è un test**. Qui si nomina solo
perché è l'unica proprietà di questa tabella che parla di deployment senza
passare per un confronto: dice quanto un risultato dipende dal seme, cioè quanto
si può contare di ritrovarlo rifacendo il training.

**Cosa costa distribuirlo — e qui E5b aveva un numero da correggere.**

| variante | delta | **+ testa = aggiornamento** | MB fp32 | MB bf16 | **primo deploy** | MB fp32 |
|---|---:|---:|---:|---:|---:|---:|
| congelato | 0 | 49.339.393 | **188,2** | 94,1 | 354.020.353 | **1.350,5** |
| VPT-deep n=576 | 14.155.776 | 63.495.169 | **242,2** | 121,1 | 368.176.129 | 1.404,5 |
| LoRA r32 | 14.155.776 | 63.495.169 | **242,2** | 121,1 | 368.176.129 | 1.404,5 |
| DoRA r32 | 14.376.960 | 63.716.353 | **243,1** | 121,5 | 368.397.313 | 1.405,3 |
| LoRA / DoRA **fusa** | 304.680.960 | 354.020.353 | **1.350,5** | 675,2 | 354.020.353 | **1.350,5** |

⚠️ **Il rapporto fuso/non-fuso non è 21,5×, è 5,6×.** E5b contava il solo delta
lato backbone; ma la **testa attentiva** (49.339.393 parametri) è addestrata in
*ogni* cella e non sta nel checkpoint base, quindi accompagna sempre il delta —
`trainable_params` di `metrics.json` la conta già (63.495.169 = 14.155.776 +
49.339.393). Su un **aggiornamento** a un dispositivo che ha già il backbone il
rapporto è 1.350,5 / 242,2 = **5,6×**; e su un **primo deploy il segno si
inverte**, perché la variante fusa non porta un delta a parte: 1.350,5 MB contro
1.404,5. Sono due colonne perché sono due scenari, e con una sola si sceglierebbe
male in uno dei due.

### La verifica indipendente — e questa è la ragione per crederci

Una composizione aritmetica è plausibile finché qualcosa non la mette alla prova.
Le celle di E3 hanno eseguito **passate di test vere** su 9116 finestre, e il
registry ne registra la durata. Sei di quelle passate stanno sullo **stesso host**
(`ainode02`, H100), due per metodo:

| variante | previsto da E5b (ms/fin) | osservato (ms/fin) | atteso dal fit | residuo |
|---|---:|---:|---:|---:|
| VPT-deep n=576 | 126,1 | 107,0 *(16,4 · 16,1 min)* | 106,1 | **−0,79%** |
| LoRA r32 | 181,3 | 141,4 *(21,6 · 21,4 min)* | 143,2 | **+1,29%** |
| DoRA r32 | 229,3 | 176,5 *(26,8 · 26,9 min)* | 175,5 | **−0,55%** |

Il fit è `osservato = a + b · previsto`, e **né `a` né `b` sono imposti**:

* **ordinamento rispettato** — VPT < LoRA < DoRA, esattamente come E5b prevede;
* **overhead fisso: 21,4 ms/finestra** — decodifica non sovrapposta, I/O, collate;
  è circa metà dei 40,8 ms che E0d misura per il decoder da solo, cioè coerente
  con una pipeline che ne sovrappone la maggior parte;
* **rapporto hardware implicito: 1,49×** — H100 contro A100, *a inferenza*. E3-X.1
  aveva misurato **1,71×/1,69×** sul **training**, ed è ragionevole che
  l'inferenza guadagni meno, essendo meno compute-bound;
* **residuo massimo 1,29%**, e le due repliche per metodo distano ≤ 0,3 min.

⚠️ **Non è un test statistico e non va presentato come tale**: tre punti e due
parametri lasciano un grado di libertà. Ciò che conta è l'ordinamento e la
piccolezza dei residui — e il fatto che due parametri liberi abbiano assunto
valori **interpretabili** invece che arbitrari.

> **La previsione che i due job mettono alla prova, scritta prima dei numeri.**
> Quel **1,49×** è un rapporto hardware *dedotto*: nessuno ha mai cronometrato lo
> stesso profilo sulle due macchine. I job del 4 settembre lo misurano
> **direttamente**, ed è un controllo vero perché le due strade sono
> indipendenti — una passa da sei wall-clock di passate di test con dentro I/O,
> decodifica e un filesystem condiviso; l'altra da un forward sintetico su clip
> in memoria. Se il rapporto misurato cadesse vicino a 1,49×, la composizione
> avrebbe predetto una grandezza che non aveva visto. Se cadesse lontano,
> l'interpretazione dell'intercetta come «overhead fisso» sarebbe da rivedere —
> e sarebbe comunque un'informazione, perché quei due parametri sono
> identificabili solo insieme.

> **E la seconda domanda, che è quella di C7.** ``docs/10`` §E5b lascia aperto
> che «su H100 la penale di VPT potrebbe cambiare di segno relativo, perché
> dipende da quanto il forward è compute-bound». I due job la chiudono: se la
> **graduatoria** dei metodi per overhead è la stessa sulle due macchine, il
> verdetto di C7 non dipende dall'hardware; se cambia, va dichiarato che vale per
> una macchina sola. ⚠️ La domanda si decide sugli **overhead**, non sulle
> latenze: quelle di due GPU non si confrontano, e `confronta_macchine()` non
> espone da nessuna parte una loro differenza — solo il rapporto.

⚠️ **Perché solo le celle PEFT.** E2c ed E2d valutano **leggendo il feature
store** (`cache_hit_rate` 1,0): non eseguono il backbone, quindi la loro durata
misura il filesystem parallelo e non il modello — mettere quel punto nel fit
sposterebbe la pendenza per una ragione che non ha niente a che vedere con
l'hardware.

**E la contesa non tocca il fit, ed è un controllo che si può fare a posteriori
sugli orari del registry.** `e3r_lora_s67` (01:47–02:08) è girata **da sola** e
`_s104` (03:55–04:17) si **sovrappone** per undici minuti a `e3r_dora_s104`:
danno **21,6** e **21,4** minuti. Stessa cosa per DoRA — `_s67` da sola,
`_s104` contesa — **26,8** e **26,9**. Le celle PEFT sono **compute-bound**,
quindi due job sullo stesso nodo non se le pestano, ed è la ragione per cui le
sei durate si comportano come una funzione della sola latenza del modello.

⚠️ **Un'osservazione che invece NON regge, ed è giusto dirlo qui.** Verrebbe
voglia di leggere le durate di E2d — **27,2 e 29,1 minuti** per rileggere feature
cachate — contro i **16,1–16,4** che servono a VPT per **ricalcolarle** su H100,
e concluderne che su questo cluster la cache di valutazione non fa risparmiare
tempo. Il confronto **è confuso**: le due passate di E2d sono girate
**contemporaneamente** sullo stesso nodo (04:04 e 04:06, entrambe chiuse alle
14:33:24), e sono le uniche due I/O-bound del gruppo — cioè esattamente quelle a
cui la contesa fa male. Il sospetto resta interessante e **la misura per
chiuderlo non c'è**: servirebbe una passata di E2d da sola. Si annota fra le cose
aperte invece di essere riportato come risultato.

⚠️ **La composizione predice la passata sul cluster, non la ratifica.** Il fatto
che il modello spieghi il wall-clock a meno dell'1,3% dice che il profilo
sintetico descrive il carico vero — non che l'end-to-end su RTSP sia misurato.

### Cosa resta da misurare, e il **solo** job che serve

E5b passa a **`profile_schema: 2`**, e le quattro cose che aggiunge sono
esattamente quelle che la composizione **non può dedurre**:

1. **Percentili** (p50/p90/p95/p99, min, max), con **200 iterazioni al batch 1** e
   30 sopra. La distinzione non è comodità: al batch 1 la latenza è un **tempo di
   risposta** e la sua coda dimensiona un buffer; sopra è un **throughput**, di
   cui la media è l'estimatore. Un `p99` su 30 campioni è il terzo campione
   peggiore.
2. **Sweep dei batch 1/2/4/8**, che rende **misurata** la capacità in telecamere
   della tavola 2 — oggi aritmetica su due punti.

   > **La previsione, scritta prima dei numeri.** I due punti che ci sono già
   > dicono che da batch 1 a batch 2 il throughput sale del **5%** (7,61 → 7,98
   > fin/s sul congelato) mentre la latenza del forward **raddoppia** (131 → 251
   > ms). Cioè a 9216 token la GPU sembra **già satura a batch 1**, e il batching
   > comprerebbe quasi niente pagandolo tutto in attesa. Se è così, batch 4 e 8
   > aggiungeranno pochi punti percentuali e il tetto resterà intorno ai **30
   > stream** che la tavola 2 riporta oggi. Se invece il throughput salisse in
   > modo apprezzabile, vorrebbe dire che a batch 1 la GPU è **sotto-occupata** —
   > e sarebbe la stessa anomalia che E5b ha già registrato senza spiegarla,
   > l'overhead di VPT che *scende* passando da batch 1 a 2. Le due letture sono
   > distinguibili dai numeri, ed è per questo che la previsione si scrive
   > adesso invece che dopo.
3. **Energia**, campionata da NVML con fallback su `nvidia-smi` e **sorgente
   scritta nella riga**: un `null` senza motivo somiglia a un guasto. Non si
   deduce dalla latenza. ⚠️ È la potenza **della scheda**, e vale perché il job
   chiede la GPU intera.
4. **Avvio a freddo**, scomposto in caricamento dei pesi e prima passata — si
   riducono in due modi diversi, e in sorveglianza è la finestra di cecità dopo
   un riavvio. Più il **costo di `merge_()`**, che è la procedura che rende vero
   il claim di C7 e di cui nessuno aveva mai detto quanto dura.

**Verificato in locale prima di sottomettere** (RTX 4060, ViT-B/224, numeri
**non** di tesi, rule 4 del workspace): schema 2 completo su sei varianti × due
batch, avvio a freddo **10,7 s + 1,6 s**, energia via il fallback `nvidia-smi`
(pynvml non è installato in locale), `merge_()` in **43 ms** (LoRA) e **65 ms**
(DoRA), digest del backbone identico dopo tutte e sei le varianti. ⚠️ E un dato
che si vedrà solo lì: su una GPU laptop il **p99 sta il 217% sopra il p50** al
batch 1 — clock dinamici e compositor — mentre su A100 le deviazioni di E5b erano
≤ 0,06%. È esattamente il numero che la coda serve a misurare, e la ragione per
cui non si assume.

**Il job**, ~10 minuti su A100 invece di 79 secondi (il grosso è il batch 8):

```
EVD_ALLOW_BF16=1 hpc/invia.sh -p gpuq -t 01:00:00 -c 4 -g 1 -J evd-e5b2 \
    evd-vfm/scripts/slurm/e5b.sbatch
```

⚠️ **La partizione va scelta e dichiarata**: E5b è su **A100** (`gpuq`), e le
latenze di H100 e A100 non sono confrontabili fra loro. Rifarlo su `aiq`
darebbe una tabella diversa e non sostituibile a questa. Le colonne di
**overhead** restano invece confrontabili, perché sono rapporti interni alla
stessa esecuzione.

Lo `.sbatch` esegue **anche la composizione** subito dopo il profilo, e
⚠️ **non fa fallire il job** se quella parte fallisce: il profilo è ciò che costa
coda, e farlo rifare per un'aritmetica sarebbe il conto sbagliato.

### Cosa è uscito sulle due macchine *(job 557837 `gnode03` A100, 648 s; job 557836 `ainode01` H100, 382 s — 4 settembre)*

Profilo **schema 2** completo su entrambe: sei varianti × quattro batch,
percentili su 200 iterazioni a batch 1, energia campionata via `nvidia-smi`,
avvio a freddo e costo di `merge_()`.

⚠️ **La ripetibilità, prima di tutto il resto.** Il congelato a batch 1 su A100
dà **131,7 ms** contro i **131,4** del job 542380 — altro giorno, altro job,
stesso nodo `gnode03`: **0,2%** di scarto. È il pavimento su cui poggia tutto il
resto della tavola, ed è misurato invece che sperato.

| | **A100** (`gpuq`) | **H100** (`aiq`) |
|---|---:|---:|
| congelato, batch 1 | **131,7 ms** | **77,9 ms** |
| VPT-deep n=576 | +1,51% | **+0,20%** |
| LoRA r32 | +42,58% | **+44,56%** |
| LoRA r32 **fusa** | +0,04% | −0,28% |
| DoRA r32 | +83,40% | **+87,85%** |
| DoRA r32 **fusa** | −0,02% | +0,05% |
| coda p99/p50, batch 1 | 0,27–0,98% | **2,71–8,93%** |
| energia, congelato b1 | 50,4 J/finestra | **43,5 J/finestra** |
| stream/GPU, b1 → b8 | 30,4 → **34,1** | 51,3 → **60,0** |
| avvio a freddo | 3,02 + 1,10 s | 2,66 + 0,54 s |

#### 1. Delle due previsioni, una ha tenuto e una **no**

**La saturazione del batch: prevista e confermata.** La previsione scritta prima
dei numeri diceva che a 9216 token la GPU è già satura a batch 1, quindi batch 4
e 8 avrebbero aggiunto pochi punti percentuali. Misurato: **+12%** di throughput
su A100 passando da batch 1 a batch **8** (30,4 → 34,1 stream) e **+17%** su
H100. Il batching, a questa geometria, **non compra capacità** — e la paga tutta
in attesa: la latenza del forward passa da 132 a 937 ms.

**Il rapporto hardware: previsto 1,49×, misurato 1,733×.** La previsione **non ha
tenuto**, e sbaglia del 16%. Veniva dal fit `osservato = a + b · previsto` sui
wall-clock delle passate di test, dove `a` e `b` sono identificabili solo
*insieme*: con tre punti la retta ha scelto una coppia che fittava bene (residui
≤ 1,29%) ma che spartiva male il merito fra overhead e hardware.

⚠️ **E la misura diretta scioglie la degenerazione, in una direzione che vale la
pena di guardare.** Rifacendo il fit con la pendenza **imposta** al valore
misurato (1/1,733), l'intercetta diventa **38,4 ms/finestra** invece di 21,4 — e
38,4 ms è, entro il 6%, il costo del decoder che E0d misura da solo (**40,8
ms**). Cioè: la soluzione fisicamente sensata è quella con il rapporto vero, e
dice che in quelle passate di valutazione **la decodifica non si sovrapponeva
quasi per niente** al forward. I residui salgono da 1,29% a ~4%, ed è il prezzo
onesto di aver tolto un parametro libero a un fit che ne aveva troppi.

> **La forma generale, che vale oltre il caso**: due parametri identificabili
> solo insieme non si interpretano uno per volta, per quanto buono sia il fit —
> e il modo di scoprirlo è **misurarne uno per un'altra strada**. Qui la strada
> è costata dieci minuti di coda.

#### 2. Il verdetto di C7 **non si ribalta, e su H100 è più netto**

Il punto «Aperto #2» di §E5b chiedeva se su H100 la penale di VPT potesse
«cambiare di segno relativo». La risposta è **no**, e la direzione è l'opposta di
quella temuta: passando ad hardware più recente la penale dei prompt **si
riduce** (da +1,51% a **+0,20%**, cioè dentro il rumore) mentre quella del
reparam non fuso **cresce** (+42,58 → +44,56 e +83,40 → +87,85). Il claim del
PdD — prompt cari a inferenza, reparam gratis — è quindi **più smentito su H100
che su A100**.

⚠️ **La graduatoria in senso stretto cambia, e va detto come.** Su A100 VPT
sta in una classe sua, sopra le due fuse; su H100 finisce **dentro** la stessa
classe delle fuse, perché +0,20% e ~0% non si distinguono. Non è un'inversione:
è la penale dei prompt che smette di essere misurabile. L'ordine dei tre gruppi
— *fusi ≈ VPT* ≤ *LoRA* < *DoRA* — è lo stesso su entrambe.

#### 3. Tre cose che solo lo schema 2 poteva dire

**(a) La latenza dell'A100 non ha coda; quella dell'H100 sì.** A batch 1 il p99
sta fra **+0,27% e +0,98%** sopra il p50 su A100, e fra **+2,71% e +8,93%** su
H100. Il nodo `aiq` ospita otto GPU e il job ne chiede una: la coda misura la
**contesa del nodo**, non il modello. Per un dimensionamento è la differenza fra
un buffer da un millisecondo e uno da sette.

**(b) Il merge non è solo un'ottimizzazione di latenza, è un risparmio di
energia.** Fondere DoRA porta da **81,3 a 52,6 J/finestra** su A100: **−35%**.
LoRA −23%. In un impianto che guarda 24 ore di video al giorno per telecamera,
sono ~0,7 kWh per telecamera al mese sul solo adattatore.

**(c) L'H100 costa più watt e meno joule.** Assorbe 520–690 W contro i 337–470
dell'A100, ma per finestra spende **43,5 J contro 50,4**: **−14%**. La macchina
più affamata è anche la più efficiente per unità di lavoro, che è il conto che
conta quando il lavoro è fissato.

#### 4. Cosa **non** cambia fra le due macchine

Il ritardo dell'allarme. Su H100 il calcolo per finestra scende da **172,4 a
118,7 ms** (−31%) e la quota di calcolo nel ritardo totale dal **6,53% al
4,59%**; ma il ritardo passa da **2,639 s a 2,585 s**, cioè **−2,0%**.

**Un hardware 1,73× più veloce compra il due per cento di reattività**, perché il
93% del ritardo è la griglia e la griglia non sa che GPU ha sotto. È la
conferma più forte che questo studio potesse ricevere, e non l'ha comprata con
un argomento: l'ha comprata cambiando davvero la macchina. La frase «per un
sistema più pronto si cambia l'hop, non la GPU» regge anche quando la GPU la si
cambia sul serio.

⚠️ **Dove invece l'hardware paga eccome**: gli **stream per GPU** passano da
**30 a 51** (+69%) e l'energia per ora di video da **12,6 a 10,9 Wh** (−14%).
Comprare una macchina più veloce serve a guardare **più telecamere**, non a
rispondere prima — ed è una frase che nessuna delle due tabelle, da sola, poteva
dire.

### Due difetti trovati facendo girare la composizione

Nessuno dei due solleva da sé, ed entrambi sono la famiglia dell'**assenza
legittima che copre un difetto** (ADR-049).

**(a) `e0i_ucf_vjepa21l_lora-r32-qkvomlp_s1` porta lo stesso `adapt` di
`e3r_ucf_vjepa21l_lora-r32-qkvomlp_s42`.** Sono le celle della ricerca del
learning rate: girano su `edp_v1`, si fermano alla selezione (`select_only`,
ADR-037) e non hanno mai valutato. Senza filtro **ogni variante risultava a nove
semi invece di tre**, e — peggio — la cella «mediana» da cui si legge la latenza
veniva scelta in una lista per due terzi fatta di `NaN`. Nel fit sul wall-clock
le loro durate, **zero minuti**, dimezzavano le medie e producevano un rapporto
hardware di **3,4×**, un numero che non significa niente e che nulla avrebbe
segnalato. I filtri sono ora due affermazioni distinte: *l'esperimento sta nella
matrice* e *la cella ha valutato la griglia giusta* — quest'ultima confrontata
con `Geometria.n_chunk_valutazione` invece che con un letterale.

**(b) Un test di E5b passava solo grazie all'ordine degli altri test.**
`test_e5b._pezzi` costruiva `TinyBackbone` dal RNG **globale**, quindi i pesi
dipendevano da cosa avesse girato prima:
`test_una_cella_fusa_fuori_tolleranza_si_marca_invece_di_passare` passava dentro
la suite e **falliva da solo**, perché su certe inizializzazioni il merge esce
*bitwise esatto* e `merge_max_abs` è **0,0**. Il test asseriva, senza dirlo, che
l'arrotondamento fp32 debba per forza essere diverso da zero. La fixture è ora
seminata.

> **La forma generale, che vale oltre i due casi**: un `run_id` **non è**
> un'identità sufficiente quando due esperimenti condividono un campo, e un test
> il cui esito dipende da chi ha girato prima non è un test — è una misura del
> suo vicino.

### Aperto

1. ✅ **Chiuso**: lo sweep dei batch è girato, e ha confermato la previsione — a
   9216 token la GPU è satura a batch 1 e il batching compra il 12–17% di
   throughput pagandolo con 7× di latenza.
2. ✅ **Chiuso**: i numeri **non** sono più di una macchina sola. A100 e H100
   girano lo stesso profilo, e il verdetto di C7 si conserva.
3. ⚠️ **Multi-stream vero: ancora non misurato, e non lo si estrapola.** I 30
   stream su A100 e i 51 su H100 restano un tetto aritmetico — nessuno
   scheduling, nessuna contesa di decoder fra sorgenti, nessun I/O di rete. Lo
   sweep dei batch ha stretto il tetto ma non lo ha sostituito: servirebbe un
   servizio con più sorgenti concorrenti, che è un altro esperimento.
4. **L'acquisizione RTSP e il dispatch dell'allarme restano fuori** da ogni riga.
   Non sono trascurabili in un impianto vero; sono trascurabili rispetto ai 2,6 s
   di griglia, ed è questo che il capitolo può dire.
5. ⚠️ **Il decoder è misurato su UNA configurazione sola**, e ora è il termine più
   debole della composizione: 40,8 ms/finestra da un nodo H100 a 8 CPU (E0d, job
   540018), usati **anche** per comporre la riga A100 — che gira su un nodo con
   un'altra CPU. Non cambia le conclusioni (il modello domina di ≥ 3×, e su H100
   di 2,9×) ma è la misura da rifare se il margine si assottigliasse. Il conto
   indiretto del §«verifica» la corrobora: a pendenza imposta l'intercetta esce
   **38,4 ms**, entro il 6% dal valore misurato.
6. ⚠️ **La coda dell'H100 non è del modello ma del nodo.** Il p99 sta fino al
   **8,93%** sopra il p50 su `ainode01`, contro lo 0,98% di `gnode03`: il nodo
   `aiq` ha otto GPU e il job ne chiede una. Per un numero di **modello** servirebbe
   un nodo esclusivo; per un numero di **deployment** questa è la condizione
   realistica, ed è per questo che si riporta invece di essere corretta.
5. ⚠️ **La cache di valutazione conviene ancora?** Le passate di E2d (27,2 e 29,1
   min a *rileggere* feature da `/shared`) sono più lente di quelle di VPT (16,1 e
   16,4 a *ricalcolarle* su H100) — ma sono girate **in concorrenza fra loro**, e
   sono le uniche due I/O-bound del gruppo. Chiuderlo costa **una** passata di E2d
   da sola, cioè ~30 minuti di coda; il sospetto vale la pena perché ADR-049 aveva
   acceso quella cache stimando un risparmio del 40% del forward, e da allora non
   è mai stata rimisurata contro un backbone che nel frattempo gira su H100.

## E3 — Le tre famiglie a parità di budget, e la lettura congiunta E3-X.1 *(chiusa: reparam ≥ probe congelata > prompt)*

**Aggiornamento del 7 settembre:** questa voce conserva la matrice principale.
Le [ablation a tre semi](#e3-ablation-20260907) mostrano la dipendenza di VPT
da profondita' e token; non estendere il titolo all'intera famiglia.

**Chiusa il 3 settembre 2026** · campagne `e3p_vpt`, `e3r_lora`, `e3r_dora` ·
cinque celle, protocollo `edp_v2`, `split_sha 5e82820a…` per tutte · claim **C7**
(il centro del PdD) e **C2**.

| cella | run_id | semi | job | nodo |
|---|---|---|---|---|
| E3-P.1 | `e3p_ucf_vjepa21l_vptdeep-n576_s42` | 42 | 541600→541601 | `gnode02` (A100) |
| E3-R.1 LoRA | `e3r_ucf_vjepa21l_lora-r32-qkvomlp_s42` | 42 | 541603→541604→541616→542019 | `gnode02/13/04` (A100) |
| E3-R.1 LoRA | `…_s67` | 67 | 542088→542089 | `ainode02` (H100) |
| E3-R.1 DoRA | `e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42` | 42 | 541605→541606→541607→542025 | `gnode02/14`, poi `ainode02` |
| E3-R.1 DoRA | `…_s67` | 67 | 542091→542092 | `ainode02` (H100) |

> **E3-X.1 non è un run** (`docs/05` §9): è la lettura congiunta di E2d, E3-P.1 e
> E3-R.1, e le griglie sono state progettate perché lo sia. La si genera con
> **una sola invocazione** dal registro — `python scripts/make_figures.py --exp
> e2d --exp e3p --exp e3r --metric chunk_auc --metric chunk_ap --metric
> chunk_auc_anomalous --metric chunk_ap_anomalous --per-run` — che è ciò che
> ADR-014 chiede: le figure nascono dal registro, non da numeri ricopiati. Il
> `--exp` ripetibile è nato qui, e l'alternativa scartata era tre tabelle cucite
> a mano.

### La parità di budget, che è la premessa e non un risultato

Il confronto ha senso solo se i tre metodi possono spendere **lo stesso**, e su
questo la matrice è più stretta di quanto il piano chiedesse:

| | bersagli | parametri dell'adattatore | token in più | testa |
|---|---:|---:|---:|---:|
| E2c/E2d congelati | 0 | **0** | 0 | 49.339.393 |
| VPT-deep `n=576` | 24 | **14.155.776** | 576 | 49.339.393 |
| LoRA `r32` qkv+o+mlp | 96 | **14.155.776** | 0 | 49.339.393 |
| DoRA `r32` qkv+o+mlp | 96 | **14.376.960** | 0 | 49.339.393 |

⚠️ **VPT e LoRA coincidono al parametro** — 14.155.776 entrambi, non
«approssimativamente» — e DoRA aggiunge **+221.184** (+1,56%), che è il vettore
di magnitudine: uno scalare per colonna di ciascuno dei 96 bersagli. La testa è
**identica** in tutte e cinque le celle. Quindi la differenza fra le colonne che
seguono **non è capacità**: è il prior induttivo di *dove* quei parametri vivono.

### Cosa è uscito

Test: 9116 finestre su 290 video; τ è il punto di Youden sulla ROC di
validazione (8383 finestre), una sola soglia per chunk e video.

| cella | semi | chunk-AUC | AP | **AUC_a** | **AP_a** | chunk-F1 | video-F1 | passo scelto |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E2c frozen-linear | 42 | 0,8473 | 0,4833 | 0,6535 | 0,5266 | 0,441 | 0,823 | 4600 |
| E2d frozen-attentive | 42 | 0,9385 | 0,7737 | 0,8409 | 0,7909 | 0,688 | 0,923 | 4600 |
| **E3-P.1** VPT-deep 576 | 42 | **0,8968** | **0,6509** | 0,7632 | 0,6940 | 0,588 | 0,891 | **3680** |
| **E3-R.1** LoRA r32 | 42 · 67 | **0,9400** ± 0,0009 | **0,7910** ± 0,0059 | 0,8642 ± 0,0006 | 0,8292 ± 0,0062 | 0,665 · 0,669 | 0,932 · 0,912 | 5520 · 4600 |
| **E3-R.1** DoRA r32 | 42 · 67 | **0,9424** ± 0,0024 | **0,7968** ± 0,0004 | 0,8673 ± 0,0044 | 0,8322 ± 0,0031 | 0,687 · 0,676 | 0,941 · 0,917 | 5520 · 4600 |

Il costo, **misurato** e non stimato: mediana degli intervalli fra i checkpoint
ogni 100 passi, **scartando** quelli che contengono una validazione. Gli
intervalli sono ripetibili al centesimo su fette e nodi diversi (LoRA 8,82 ·
8,85 · 8,86; DoRA 11,46 · 11,48 · 11,50).

| metodo | s/passo A100 | s/passo H100 | training (A100) | 6 validazioni | cella intera |
|---|---:|---:|---:|---:|---:|
| frozen-linear | 1,40 | — | 2,15 h | 0,16 h | **2,67 h** |
| frozen-attentive | 1,74 | — | 2,67 h | 0,23 h | **3,41 h** |
| VPT-deep 576 | 6,59 | — | 10,11 h | 1,89 h | **12,44 h** |
| LoRA r32 | 8,85 | 5,17 | 13,58 h | 2,74 h | **16,90 h** |
| DoRA r32 | 11,48 | 6,81 | 17,62 h | 3,42 h | **21,48 h** |

H100/A100 vale **1,71×** su LoRA e **1,69×** su DoRA, dentro l'intervallo
1,68–1,81× che E0b.3 aveva misurato — e la previsione di E0b.3 per DoRA su H100,
**6,82 s/passo**, coincide col misurato a **6,81**.

### Cosa vuol dire

**1. VPT-deep non fallisce di poco: sta sotto la probe congelata.** −0,042 di
chunk-AUC e **−0,123 di AP** rispetto a E2d, pagando **3,6×** il tempo di
training. È sovradattamento e si vede in tre punti indipendenti: la `val_loss`
**sale** (0,259 al passo 920 → 0,329 al 5520) mentre la loss di training cala del
75%; la selezione prende il **3680** e non l'ultimo passo; e il per-classe dice
che il danno è **globale, non di una classe** — peggiora su **12 classi su 13**,
e l'unica dove migliora è `Shoplifting`, che è la più difficile per tutti
(0,8345 → 0,8462). ⚠️ Questo è un **risultato**, non un guasto: la cella è andata
5524/5524 passi e la sua provenienza è pulita.

**2. Il reparam paga, ma quanto dipende da quale metrica si guarda.** Su
chunk-AUC LoRA aggiunge **+0,0015** e DoRA **+0,0039** sopra la probe congelata:
è poco, e con un seme solo su E2d non si può affermare che sia fuori dal rumore.
Su **AP_a** — l'AP ristretta ai 140 video anomali, cioè «dato che l'evento c'è,
sai dire *quando*» — la distanza è **+0,038** e **+0,041**, e su AUC_a **+0,023**
e **+0,026**. La lettura che ne segue, ed è la frase da portare in tesi: il
guadagno dell'adattamento **non sta nel distinguere un video anomalo da uno
normale**, che la testa attentiva su feature congelate fa già bene, sta **dentro
il video anomalo**. Il per-classe è coerente: i guadagni si concentrano su
`RoadAccidents` (+0,025…+0,036), `Vandalism` (+0,024) e `Stealing`
(+0,014…+0,023), cioè eventi con un confine temporale netto.

**3. LoRA e DoRA non si separano, ed è la domanda che il seme 67 doveva
chiudere.** Il divario a s42 (**+0,0048** di AUC per DoRA) **non si replica** a
s67 (**+0,00004**). Il divario fra le medie è **+0,0024**, cioè *esattamente* la
deviazione standard di DoRA, e l'escursione di DoRA fra i due semi (**0,0034**) è
**più grande del divario stesso**. La direzione è consistente — DoRA non è mai
dietro, su nessuna delle quattro metriche e su nessuno dei due semi — la
magnitudine no. E DoRA costa **+30%** di training. ⚠️ **Non è un confondimento
hardware**: le celle s42 sono girate quasi tutte su A100 e quelle s67 su H100, ma
il non-determinismo fra le due macchine è stato misurato su E2d e vale **1,8e-7**
di AUC, sette ordini di grandezza sotto il divario in questione.

**4. La cache di valutazione è rimasta chiusa in tutte e tre le famiglie**, ed è
la verifica di ADR-056 sugli artefatti invece che sul ragionamento:
`cache_hit_rate` è `None` in E3-P.1 e in tutte e quattro le celle E3-R.1, contro
`1.0` di E2d. ⚠️ E3-P.1 è girata su `6f1341bc`, cioè **prima** della correzione:
si è salvata sulla seconda condizione (`extra_tokens() == 576`), come ADR-056
aveva argomentato — e il `None` nel suo `metrics.json` è la **prova**, non
l'ipotesi. Le quattro celle reparam sono girate su `cc6ff802` e `7b8fcdb`,
entrambi successivi alla correzione.

### Decisioni prese

1. **VPT-deep `n=576` resta com'è, e il suo numero entra in tesi così.** Non si
   ritocca il metodo per farlo somigliare a un successo: la cella risponde alla
   domanda «il prompt tuning regge il confronto a questo budget» e la risposta è
   no. Le ipotesi su *perché* — 576 token su tutti e 24 i layer, senza
   regolarizzazione propria — sono materiale per E3-P.2 (`n_tokens`), che è già
   in seconda fascia (`docs/05` §7).
2. **Il criterio di scelta fra LoRA e DoRA non è l'accuratezza**, perché
   sull'accuratezza i due sono indistinguibili a due semi. Resta il costo, e su
   quello LoRA vince di **+30%**. ⚠️ Quale dei due sia «il miglior reparam» nella
   tabella di E3-X.1 è una **decisione di Antonio** e non è ancora presa: fino ad
   allora si riporta la coppia.

### Alimenta

* **C7** (`docs/01`): la tabella di E3-X.1 è completa per la parte accuratezza.
* **E5b**: le cinque righe di costo di training qui sopra sono misurate e
  riusabili; il **costo di inferenza** no — vedi «Aperto».
* `docs/06` **M5**: la DoD «curve PEFT complete per entrambe le famiglie» è
  soddisfatta; «ablation prioritarie chiuse» no.

### Aperto

1. ⚠️ **VPT resta a un seme, LoRA e DoRA ne hanno due.** Il confronto **dentro**
   la famiglia reparam ha una dispersione; quello **fra famiglie** — che è
   E3-X.1 — poggia su un solo seme per il braccio prompt e per la probe
   congelata, e in tesi **si dichiara** (R22, ADR-054). Chiuderlo costa una cella
   VPT a seme 67: ~12,4 h di A100 o ~7,3 h di H100.
2. **Il costo di inferenza non è misurato in modo controllato.** La passata di
   test sta fra 22 e 35 minuti su tutte e cinque le celle **senza** un ordine che
   segua il metodo: è dominata dalla decodifica video e dall'I/O di un cluster
   condiviso, non dall'adattatore. Il fatto strutturale che E5b deve misurare:
   LoRA e DoRA sono **fondibili** nei pesi base e a inferenza non costano nulla,
   mentre VPT-deep porta **576 token in più per layer** su 9216, che è un costo
   reale e permanente. Nessuno dei due lati è misurato qui.
3. La ripresa da checkpoint costa **~38–40 minuti** di avvio fisso per fetta, ed
   è una voce che nessuna stima della matrice conteneva: va aggiunta al conto
   quando si dimensiona una catena.

## E2 — La baseline congelata su `edp_v2` *(chiusa a un seme: la linear non regge)*

**Chiusa il 1 settembre 2026** · campagne `e2_frozen` e `e2c_linear`, **seme 42
soltanto** · run_id `e2d_ucf_vjepa21l_frozen-attentive_s42` (job **541395**,
`gnode11`) e `e2c_ucf_vjepa21l_frozen-linear_s42` (job **541396**, `gnode12`),
entrambe **A100-SXM4-80GB** · protocollo `edp_v2`, `split_sha 5e82820a…` ·
claim **C2**.

> ⚠️ **È la prima voce di questo registro prodotta su `edp_v2`**, e quindi la
> prima i cui numeri sono candidati a entrare in tesi. Tutte le voci sotto sono
> `fsp_v1` o `edp_v1` e **non** si confrontano con questi.

### Cosa è stato eseguito

Due probe congelate sul **layer 23**, stesso budget (`edp_v2`: 5524 passi = una
epoca, batch effettivo 16, `lr 1e-4`, `cosine`, `wd 0.01`, `photometric`,
`train_sampling: static`), stessa selezione (`val_auroc` ogni 920 passi) e
stessa soglia (**Youden** sulla ROC di validazione). Differiscono per il **solo
gruppo `head`**: attentive pooler (**49.339.393** parametri addestrabili) contro
linear (**3.073**). È il disegno che rende E2c una diagnosi e non una seconda
baseline.

**I job sono ripartiti dal checkpoint** scritto dai job morti di ADR-050
(`resumed_from: 5524`, `selected_step 4600`): il training era già finito, e la
ripresa è andata **dritta alla valutazione**: **22 min 07 s** (E2c) e **30 min
50 s** (E2d) sulle A100 di `gpuq`, contro **17 min 25 s** e **22 min 04 s** sulle
H100 di `aiq` — cioè **1,27× e 1,40×**, in linea col rapporto 1,68–1,81× misurato
da E0b.3 sul solo forward, e più basso perché qui una parte del lavoro è I/O
sullo store e non calcolo. È la verifica misurata della frase di ADR-050,
«rimediare costa minuti, non ore», e vale perché lo snapshot dei pesi del passo
selezionato sopravvive alla ripresa.

> ### ⚠️ Le riprese sono state DUE, e il registry le ha entrambe
>
> Il registry ha **una riga per esecuzione, non per `run_id`** (invariante 2), e
> per queste due celle ne ha **quattro**:
>
> | esecuzione | nodo | GPU | git | finita (UTC) |
> |---|---|---|---|---|
> | 1ª | `ainode01` | **H100** | `8ebb3d8` | 19:11:57 (E2c) · 19:16:38 (E2d) |
> | 2ª | `gnode12` / `gnode11` | **A100** | `8ad0cf8` | 22:27:55 (E2c) · 22:36:41 (E2d) |
>
> Fra i due sha c'è un commit solo, ed è `8ad0cf8` — *«`seed_schedule` torna al
> valore consumato: un `config_sha` invalidato costa ore»*. Cioè: la prima
> ripresa è girata mentre `seed_schedule` portava i semi nuovi
> `[42, 67, 104, 88, 1320]`, che **entrano nel `config_sha`**, quindi l'identità
> del run non era più quella del checkpoint da riprendere. Ripristinato il valore
> consumato, le celle sono state rifatte.
>
> **Ciò che vale è la seconda**, ed è quella i cui artefatti stanno in
> `runs/<run_id>/` (`scores.jsonl`, figure, `metrics.json`). La prima resta nel
> registry perché il registry **non si riscrive** (regola 6).
>
> **E il confronto fra le due è un risultato gratuito che vale la pena avere:**
> stesso codice, stessi pesi, stesso protocollo, **hardware diverso** (H100
> contro A100). Le decisioni sono **identiche** — `n_pred_pos` 1973, F1 di chunk
> 0,6882217090069285 alla sedicesima cifra — e gli score no: chunk-AUC
> **0,9384909676639068** contro **0,9384911435828083** (Δ ≈ **1,8e-7**), AP
> 0,7737100778546919 contro 0,7737112643167765, e la soglia di Youden
> 0,14687518775463104 contro 0,14687739312648773 (Δ ≈ 2,2e-6). È la misura
> diretta di quanto vale il non-determinismo hardware su questa pipeline: **sette
> ordini di grandezza sotto** il pavimento di rumore di riesecuzione di una probe
> (1,3 pp, E0c), e **nessun chunk** cambia lato della soglia. In tesi si riporta
> come quello che è — una verifica di riproducibilità, non un'incertezza.

### I numeri

| | **E2d** attentive | **E2c** linear | Δ |
|---|---|---|---|
| **chunk AUC** *(primaria)* | **0,9385** | 0,8473 | **−0,0912** |
| **chunk AP** *(primaria)* | **0,7737** | 0,4833 | **−0,2904** |
| AUC_a *(140 video anomali)* | 0,8409 | 0,6535 | −0,1874 |
| AP_a | 0,7909 | 0,5266 | −0,2643 |
| chunk P / R / F1 @ τ | 0,604 / 0,799 / **0,688** | 0,288 / 0,934 / 0,441 | −0,248 |
| video P / R / F1 @ τ | 0,910 / 0,936 / **0,923** | 0,739 / 0,929 / 0,823 | −0,100 |
| evento P / R / F1 @ τ | 0,601 / 0,895 / **0,719** | 0,524 / 0,938 / 0,673 | −0,046 |
| τ (Youden su val) | 0,1469 | 0,1267 | — |
| AUROC di validazione al passo scelto | 0,9370 | 0,7787 | — |
| loss finale (training) | 0,0375 | 0,3642 | — |

Denominatori, identici per le due celle: **9116** chunk di valutazione su **290**
video (1491 positivi, 16,36%), **8383** score di validazione per τ, **276**
eventi annotati, `n_eventi_irraggiungibili` **0**.

### Cosa dicono, e cosa no

1. **La linear non regge, ed è il risultato che E2c esiste per produrre.** Il
   divario non è ai margini: **−0,091 di AUC** e soprattutto **−0,290 di AP**. La
   semantica di questo dominio **non è linearmente separabile** sul layer 23, e
   quindi l'attentive pooler non è un ornamento — sta facendo lavoro. È anche la
   lettura opposta al rischio che E2c presidia (collasso: se la linear avesse
   retto quasi quanto l'attentive, la testa non starebbe imparando niente).
   ⚠️ Con **un solo seme** non c'è dispersione, e sotto ~1,5 pp un confronto non
   concluderebbe (E0c); qui il divario è **9 pp e 29 pp**, cioè ben fuori.
2. **La P/R a soglia della linear è un caso di scuola.** τ di Youden le assegna
   **4826 predizioni positive su 9116** — il 53% del test — per catturare 1491
   veri: recall 0,934 e precision 0,288. Non è un difetto della soglia: è la ROC
   che, a quella separazione, non offre un punto migliore. Il corollario da
   riportare: **il recall a livello evento non è una misura di qualità senza la
   sua precision** — E2c «rileva» **259** eventi su 276 contro i **247** di E2d,
   pur essendo peggiore su ogni metrica di ranking, semplicemente perché emette
   più allarmi (267 contro 328, di cui veri 140 contro 197).
3. **L'80% del budget compra mezzo punto.** L'AUROC di validazione di E2d va da
   **0,9315** (passo 920) a **0,9370** (passo 4600), e la selezione prende il
   4600 su 5524 in **entrambe** le celle. Un'epoca è il budget che il protocollo
   dichiara e non si tocca, ma la curva va mostrata: è l'argomento con cui in
   tesi si dice che il risultato non dipende dal budget speso.
4. **La cache di valutazione ha pagato.** `cache_hit_rate 1.0`, **9116 hit e 0
   miss** su entrambe: ADR-049 — la griglia che arriva fino all'estrazione — è
   verificata sul percorso vero, e non da un test. Era il punto in cui uno `0.0`
   sarebbe stato l'unico sintomo di chiavi disgiunte.
5. **La generalizzazione val→test tiene su E2d e non su E2c.** 0,9370 → 0,9385
   per l'attentive (+0,0015), 0,7787 → 0,8473 per la linear (+0,069). Il secondo
   scarto è grande, e la spiegazione più semplice è che le due partizioni hanno
   tassi di positivi diversi (11,86% val, 16,36% test): dove la separazione è
   debole l'AUC è più sensibile alla composizione. Va dichiarato, non spiegato
   via.
6. **Le righe per-classe con `n_pos` a una cifra non sono misure.** `Abuse` ha
   **6** chunk positivi su 9116: la sua AP vale 0,025 su E2d e 0,0009 su E2c, e
   la differenza non significa niente. La tabella per-classe **deve riportare
   `n_pos`** accanto ad AUC e AP, sempre. Le classi con abbastanza massa dicono
   invece qualcosa: su E2d il minimo è **Shoplifting 0,8345** e il massimo
   **Fighting 0,9923** — cioè il modello non fallisce uniformemente.
7. **La sensibilità alla soglia è piatta su E2d.** F1 di chunk vale 0,663 · 0,690
   · 0,686 · 0,641 · 0,565 ai quantili 0,80 · 0,90 · 0,95 · 0,975 · 0,99, contro
   **0,688 a Youden**: la scelta di τ **non porta il risultato**, e il q=0,90
   darebbe 0,0023 in più, cioè niente. Conferma ADR-051 sul run vero.
8. **Latenza:** mediana **2,53 s** e p90 **4,95 s** su E2d (2,37 / 3,87 su E2c).
   ⚠️ Il campo vale **93** su E2d contro i **29** eventi persi da `event_prf`:
   **non è una contraddizione, sono due domande diverse** — la latenza pretende
   un chunk che finisca *dentro* l'intervallo, `event_prf` si accontenta
   dell'intersezione (ADR-048). Il nome lo nascondeva, ed è stato corretto il
   2 settembre: `n_mai_rilevati` → **`n_senza_latenza_misurabile`** (ADR-057).
   ⚠️ Nei `metrics.json` scritti **prima** di quella data il campo si chiama
   ancora `n_mai_rilevati`, e va letto col significato nuovo.

### Cosa resta aperto

- ✅ **«Quanti semi» ha risposta dal 1 settembre** (ADR-054): per ora **uno, il
  42**, e le sei sweep ufficiali sono passate da cinque celle a una. La riserva
  per replicare è `[42, 67, 104, 88, 1320]`, **in quest'ordine**. ⚠️ Resta vero
  che senza almeno tre non c'è nessuna barra d'errore e nessun confronto E3 può
  essere letto come appaiato: quando comprarli è una decisione di ore di cluster,
  da prendere dopo che E3 ha detto quale metodo vince.
- **E2a resta opzionale** (`docs/01` §3bis D7) ed **E2b resta un'ablation**: il
  layer è deciso.
- **Il costo per cella congelata su A100 non è ancora misurato end-to-end**:
  questi due run hanno pagato la sola valutazione. Il training di 5524 passi è
  stato cronometrato dai job morti di ADR-050 su **H100** (2h55m e 2h21m).
- ✅ **`CACHE_RAM_MB` vale 0 dal 1 settembre** (ADR-054): la voce «da abbassare
  prima di E3» è chiusa. La cache RAM del `WindowDataset` non paga e sopra ~500 MB
  per worker costa fino al **16%** (voce E0d-cache, chiusa lo stesso giorno).
  Questi due run congelati non ne avevano risentito — leggono dallo store, non
  decodificano — ma le celle PEFT decodificano tutto, ed è lì che il 16% si
  sarebbe pagato. Zero e non 166 perché fra i due il throughput non sceglie
  (0,991×), quindi sceglie la RAM. ⚠️ Il **beneficio** resta un limite inferiore
  (3000 pescaggi contro gli 88.384 di un'epoca): chi la riaccende rifà la misura
  nel regime vero.
- ✅ **«Quanti semi» è deciso** (ADR-054): per ora **uno, il 42**, e le sei sweep
  ufficiali sono passate da cinque celle a una. La riserva per replicare è
  `[42, 67, 104, 88, 1320]`, **in quest'ordine** — tre semi sono `[42, 67, 104]`.
  ⚠️ Finché il seme è uno, **questi numeri non hanno dispersione** e nessun
  confronto E3↔E2 si può presentare come significativo (R22, `docs/09`).

---

## E0d-cache — Quanto costa, e quanto rende, `CACHE_RAM_MB` *(chiuso: non rende)*

**Aperto e chiuso il 1 settembre 2026** · run_id
`e0d_ucf_vjepa21l_frozen-none_s1__cache_ram` (job **541415**, `fnode01`, `fatq`,
CPU) e `e0d_ucf_vjepa21l_frozen-none_s1__cache_benefit` (job **541451**, stesso
nodo, 08:03→09:29 UTC, **85 min 36 s**).

> **L'esito in una riga: la cache RAM del `WindowDataset` non paga, e sopra
> ~500 MB per worker fa danno.** Si adotta `CACHE_RAM_MB` **basso o zero** —
> ✅ **attuato il 1 settembre, e il valore è 0** (ADR-054). Zero e non 166 perché
> fra i due il throughput non sceglie (0,991×), quindi sceglie la RAM.

### La gamba «costo» — fatta, e non risponde alla domanda

L'asse `--cache-mb` di `scripts/e0d_throughput.py` attraversa quattro budget
(0 · 166 · 2000 · 8000 MB) su 800 finestre, `augment: none`, **0 worker** (un
processo solo, per isolare la cache di *un* worker):

| `cache_mb` | throughput a regime | RSS di picco | Δ cache |
|---|---|---|---|
| 0 | 3,091 fin/s | 1011 MB | +229 MB |
| 166 | 3,165 fin/s | 1113 MB | +328 MB |
| 2000 | 3,214 fin/s | 2895 MB | +2113 MB |
| 8000 | **3,271 fin/s** | 8619 MB | **+7833 MB** |

**+5,8% di throughput per 7,8 GB di RSS.** ⚠️ **Quel numero non è il beneficio
della cache**, ed è così per costruzione: il profilo passa **una volta sola** su
un elenco fisso di finestre, quindi ogni indice del dataset è un **miss** e la
cache non ha mai occasione di essere colpita una seconda volta. Il piccolo
guadagno che si vede è riuso *dentro* una finestra, non *fra* finestre. Preso
per il beneficio, direbbe di spegnere una cache che nel training vero potrebbe
valere molto di più — o molto di meno.

### La gamba «beneficio» — fatta, e dice di no

Il training vero pesca **con reimmissione** (`WeightedRandomSampler(replacement=True)`)
per un'epoca intera, e i pesi sono `w_video = 1/n_chunk`: un video corto ha peso
alto e i suoi indici possono tornare più volte dentro la quota di *un* worker. È
l'unico regime in cui la cache ha un effetto da misurare invece che da stimare.

`scripts/e0d_cache_benefit.py` pesca **3000** indici con reimmissione dai pesi
**reali** della partizione di training (88.371 chunk) e attraversa **la stessa
sequenza** a ogni budget candidato, `augment: photometric` — così l'unica
differenza fra le righe è il budget, non il campione.

| `cache_mb` *(per worker)* | throughput | speedup vs 0 |
|---|---|---|
| **0** | **3,743 fin/s** | **1,000** |
| 166 | 3,711 | 0,991 |
| 500 | 3,702 | 0,989 |
| 1000 | 3,487 | 0,932 |
| 2000 | 3,333 | 0,891 |
| 4000 | 3,143 | **0,840** |

**Monotonicamente in giù.** Non c'è un ginocchio, non c'è un ottimo interno: più
cache, meno throughput. E la causa sta nello stesso artefatto, che è la ragione
per cui `n_indici_unici` viene registrato accanto ai tempi: **2772 indici unici
su 3000 pescaggi**, cioè un **tasso di ripetizione del 7,6%**. La cache quasi non
colpisce mai, quindi di essa si paga solo la contabilità — e, ai budget alti, la
pressione sulla memoria.

Vale la pena dire perché il 7,6% è *coerente*, invece di trattarlo come una
sorpresa: 3000 pescaggi da una popolazione di **88.371** danno ~1,7% di
ripetizione anche uniformi. I pesi la portano a 7,6%, cioè la concentrano di un
fattore ~4,5 — che è la prova che `w_video` fa il suo lavoro — ma restano 3000
estrazioni da 88 mila.

### ⚠️ Il limite di questa misura, dichiarato

**3000 pescaggi non sono un'epoca.** Un'epoca di `edp_v2` ne fa **88.384**, cioè
~29× la popolazione, dove la ripetizione sarebbe enormemente più alta e la cache
avrebbe occasioni che qui non ha avute. Quindi:

- il **costo è misurato** e vale come misura — è monotono e arriva al 16%;
- il **beneficio è ancora sottostimato**, e questa misura è un suo **limite
  inferiore**, non una stima.

Rifarlo a `--n-draws 88384` costerebbe ~6,5 h di `fatq` per cella e ~39 h per le
sei: **non si fa adesso**, e la ragione è che la decisione non cambierebbe nella
direzione che conta. Ai due budget piccoli (166 e 500) la penalità misurata è
**dentro il rumore** (0,9-1,1%), quindi il rischio di sbagliare adottando un
valore basso è limitato dall'alto; adottarne uno grande costerebbe **il 16%** su
ogni cella di E3, cioè ore, contro un beneficio che nessuno ha visto.

⚠️ Due cose che questa misura **non** dice, e vanno sapute prima di riusarla:
i candidati sono un budget **PER WORKER** (per tradurli in `CACHE_RAM_MB` si
moltiplica per `num_workers`, che su `edp_v2` è **12**); e le colonne di **RSS
sono inaffidabili** in questo run — `cache_rss_delta_mb` esce **negativo** su due
righe (−128 e −39 MB) e `rss_base_mb` oscilla fra 786 e 4983 MB fra celle che
allocano lo stesso. Il campionamento della RSS di un processo singolo su un nodo
condiviso misura anche ciò che il nodo fa intorno: i **tempi** di questo run si
usano, le **memorie** no.

---

## E0i — Il learning rate, scelto una volta sola *(aperto: criterio scritto prima dei numeri)*

**Aperto il 28 agosto 2026** · sweep `e0i_lr` (9 celle) + `e0i_lr_frozen` (3
celle) · run_id `e0i_ucf_vjepa21l_{vptdeep-n576|lora-r32-qkvomlp|dora-r32-qkvomlp|frozen-attentive}_s{1|2|3}`
· **prima campagna fallita per intero** (28 agosto, job array `539643`), **rieseguita e chiusa il 29 agosto** dopo ADR-038 · decisione: ADR-036 punto 5; ridisegno: ADR-038.

> Stesso avvertimento di E0g, E0h e E2b-ML: questa voce esiste prima dei numeri
> apposta. Il criterio del §«Criterio di decisione» è datato 28 agosto e non si
> riscrive dopo aver visto i risultati. Se lo stai leggendo dopo, non è quello
> che ricordi.

### Cosa chiede

**Un solo numero per famiglia: il learning rate.** E fino al 28 agosto lo chiedeva
*ogni run*. `_seleziona_lr` addestrava un modello per punto della griglia e poi
rifaceva il fit finale con la LR scelta — N+1 fit per cella. Su E3, tre metodi per
cinque semi, con N=3 significava spendere il **300%** del costo della campagna per
riscegliere ogni volta lo stesso numero; e riscegliendolo *rumorosamente*, perché
E0c ha misurato che il pavimento di rumore di una riesecuzione non è trascurabile
(1,3 pp dove l'accuratezza è alta, 8,4 pp dove è bassa).

ADR-036 stacca la scelta dai run: si paga una volta qui, poi `lr_grid` scende a un
punto e `select_lr` restituisce `griglia_a_un_valore` senza addestrare niente.

**Non produce numeri di tesi.** Non tocca lo split di test — `select_only` ferma la
cella subito dopo la selezione — e ciò che lascia su disco è una ricevuta: i cinque
`trials` `(lr, val_loss)` per cella, e la LR che ne è uscita.

**Le quattro celle che aspettano questo numero**, cioè le quattro `lr_grid: ???`
del repository: `configs/train/probe_edp.yaml` (E2d, la baseline congelata contro
cui si legge tutto E3) e le tre famiglie PEFT che condividono
`configs/train/peft_budget_a.yaml`.

### Il disegno, e perché è questo

| | scelta | perché |
|---|---|---|
| **griglia** | 5 punti a mezza decade: `1e-4 · 3,16e-4 · 1e-3 · 3,16e-3 · 1e-2` | l'estremo alto è il `lr` del riferimento ufficiale per le attentive probe di V-JEPA 2.1; il basso è due decadi sotto, dove vive il percorso prompt. Cinque e non tre: a decade piena l'ottimo cade fra due punti abbastanza spesso da rendere la mediana sui semi una scelta fra valori distanti 10× |
| **famiglie** | 4, ciascuna con la **sua** LR | su APT il gradiente misurato è 3–6 ordini di grandezza sotto VPT (job 533071/533073). Una LR unica handicapperebbe una famiglia per costruzione, cioè deciderebbe C7 prima di misurarlo. Quello che l'invariante 5 vincola è la **procedura**, e quella è identica: stesso file `configs/train/lr_search.yaml` per tutte e quattro |
| **semi** | 3, non 5 | cinque semi servono a mettere un intervallo attorno a un numero che va in tesi. Qui si sceglie **un punto di una griglia a cinque valori** — una statistica d'ordine, non una media — e la mediana di tre è già robusta a un seme che sbaglia. I due semi in più sarebbero 30 fit, cioè un terzo della campagna E3 vera |
| **budget** | 500 passi, un sesto dei 3000 di ADR-036 | non è statistica, è coda: una cella sono 5 prove **in sequenza** e le prove della griglia non hanno checkpoint per costruzione (`ExperimentRunner._checkpointer`), quindi devono stare tutte in una fetta |
| **tutto il resto** | identico a E2/E3 | batch effettivo 16, `augment: official_nocrop`, `weight_decay: 0,0`, `lr_schedule: constant`, layer 23, `edp_v1`. Una LR scelta su un'altra dimensione di batch o senza augmentation non è trasferibile ai run che la useranno |

### Come si esegue

Due campagne e non una, per una ragione meccanica: E2d usa `head: attentive` e le
tre celle PEFT usano `detection`, e gli assi di una campagna sono un prodotto
cartesiano — un asse `head` accanto ad `adapt` produrrebbe otto celle di cui
quattro senza senso.

```powershell
python scripts/campagna.py piano  e0i_lr          # 9 celle, indici 0-8
python scripts/campagna.py valida e0i_lr          # risolve le config e controlla lo split
python scripts/campagna.py invia  e0i_lr          # stampa i comandi, NON sottomette
python scripts/campagna.py piano  e0i_lr_frozen   # 3 celle
```

**Su `aiq` prima che su `gpuq`** (richiesta di Antonio, 28 agosto), e si decide con
`sbatch --test-only`: `aiq` ha 16 H100 in tutto il cluster, quindi una coda lunga
può mangiarsi lo speedup di 1,7×.

`fetta_massima: 9` è quasi il tetto dell'account (10 job in coda, e ogni task di un
array conta): è la scelta dichiarata di fare **una sola ondata**, e il suo prezzo è
che finché la campagna è in coda non si sottomette altro. `paralleli: 3` invece è
prudente: con 3 la campagna dura ~16 h, con 9 ~6 h, e alzarlo è una decisione di
Antonio.

### Costo, e dove è il rischio — ⚠️ **PREVISIONE, FALSIFICATA DALL'ESECUZIONE**

> ⚠️ **Quanto segue è la stima scritta il 28 agosto *prima* di lanciare, e la campagna > l'ha smentita: i costi veri sono 2,5-3× questi (vedi *Cronaca*).** Resta qui perché > era una **previsione**, non un risultato — la regola 6 del registro vieta di correggere > un numero misurato, non di marcare una stima sbagliata — e perché la sua correttezza > aritmetica è parte della diagnosi: era una stima del **calcolo**, e il calcolo non era > il collo di bottiglia. Con `num_workers: 8` (ADR-038) questi numeri tornano applicabili.

Aritmetica sui numeri **H100** di E0b.3 (VPT 5,01 fin/s, LoRA 3,62, DoRA 2,68 a
batch 2; forward congelato 13,90 a batch 1). Una cella sono 5 prove, e ogni prova è
500 passi a batch effettivo 16 più **una** passata di validazione sui 13.551 chunk:

| famiglia | s/passo | 500 passi | + validazione | **cella (5 prove)** |
|---|---:|---:|---:|---:|
| frozen-attentive | 1,15 | 10 min | +16 min | **2,2 h** |
| VPT-deep n=576 | 3,19 | 27 min | +16 min | **3,6 h** |
| LoRA r=32 | 4,42 | 37 min | +16 min | **4,4 h** |
| DoRA r=32 | 5,97 | 50 min | +16 min | **5,5 h** |

Totale ~45 h-H100 per le 12 celle. Contro le ~79 h della sola campagna E3, è il
16% — ed è il conto che rende E0i conveniente: senza, quelle 79 h diventerebbero
~240 con una griglia a tre punti dentro ogni run.

**Il rischio è DoRA, e ha un nome: TIMEOUT.** 5,5 h stimate in una fetta da 6:59
sono un margine del 20%, e le prove della griglia non hanno checkpoint — un
TIMEOUT butta la cella intera, non i suoi ultimi passi. Se succede, il ripiego è
`train.max_steps=400` sulla sola fetta 6-8: E0i è un pilota, il suo budget non è
protocollo, e un budget diverso fra le famiglie **non** si può usare (romperebbe la
parità della procedura) — quindi se si scende a 400, si scende per tutte e si
rilancia tutto.

**Una riga che decide se questa campagna costa 45 h o 1.100.** `eval_every: 500`.
`_seleziona_lr` passa **sempre** il `val_loader` alle prove della griglia — è da lì
che esce il punteggio con cui si sceglie — e il `Trainer` valuta ogni `eval_every`
passi. Col default `1`, ogni prova farebbe 500 passate sui 13.551 chunk di
validazione, ~16 min l'una: **130 ore per cella**. Con `eval_every = max_steps` la
passata è una, all'ultimo passo, ed è esattamente il numero che serve.

### Criterio di decisione — scritto il 28 agosto, prima di lanciare

Eseguito da `scripts/aggrega_lr.py`, testato in `tests/unit/test_aggrega_lr.py`.
Il codice esiste **prima** dei numeri per la stessa ragione di E0g: un criterio che
vive solo in un documento viene riletto, inevitabilmente, dopo averli visti.

1. **La scelta dentro la cella** è già fatta e non si rivede: è l'argmin della loss
   di validazione allo stato finale dei 500 passi, cioè ciò che `select_lr` scrive
   in `metrics.json → lr_selection`.
2. **La LR della famiglia è la mediana delle tre scelte.** Mediana e non media: la
   media di `3,16e-4`, `1e-3` e `3,16e-3` è `4,5e-3`, un learning rate che nessuna
   prova ha eseguito — cioè il numero inventato che l'invariante 1 esiste per
   impedire. E non minimo: il minimo insegue il seme più fortunato.
   *(Con un numero pari di semi — un rilancio parziale — si prende il più basso dei
   due centrali, `median_low`, che è misurato e conservativo nel verso giusto: vedi
   il limite dichiarato al punto 5.)*
3. **Guardia di bordo: una mediana su `1e-4` o `1e-2` NON si adotta.** Un estremo
   della griglia non è una risposta, è la notizia che l'ottimo sta fuori dalla
   banda: la risposta giusta è allargarla e rifare, non prendere il bordo perché è
   quello che c'è. Questo punto è qui perché dopo aver visto i numeri è
   esattamente quello che si è tentati di saltare.
4. **La concordanza si riporta sempre**, ma non decide: 3/3 semi sulla stessa LR
   dice che la scelta è stabile; 1/3 — tre semi, tre risposte — dice che a 500
   passi la superficie è piatta, e va scritto accanto al numero perché significa
   che la scelta pesa meno di quanto un valore singolo faccia credere.
5. **Il limite, dichiarato adesso e non dopo:** si sceglie la LR migliore a **500**
   passi e la si usa a **3000**. La distorsione ha un verso noto — a orizzonte
   corto le LR grandi sono avvantaggiate, perché hanno avuto il tempo di muoversi e
   non quello di divergere — quindi una famiglia che sceglie `3,16e-3` o più si
   legge con quel sospetto. Non si corregge a mano: correggere sarebbe
   reintrodurre il numero inventato. Si dichiara, e in tesi si scrive così.
6. **Le famiglie non si mediano fra loro.** Quattro numeri, quattro destinazioni.

### Dove finisce il numero

Il valore **non** va in `configs/train/peft_budget_a.yaml`: quel file è condiviso
dalle tre famiglie e l'invariante 5 gli vincola la **cardinalità** della griglia —
un punto — non il suo valore, che è per famiglia. Il `???` resta lì a pretendere
che ogni esperimento lo dichiari.

| famiglia | destinazione |
|---|---|
| `frozen-attentive` | `configs/train/probe_edp.yaml` → `lr_grid` *(un solo consumatore: E2d)* |
| `vptdeep-n576` | `configs/experiment/e3p_vpt.yaml` → `train.lr_grid` |
| `lora-r32-qkvomlp` | `configs/experiment/e3r_lora.yaml` → `train.lr_grid` |
| `dora-r32-qkvomlp` | `configs/experiment/e3r_dora.yaml` → `train.lr_grid` |

### Cosa questo pilota **non** dice

- **Non dice qual è la LR migliore a 3000 passi**, e nemmeno che l'ordinamento
  fra i cinque punti sia lo stesso lì. Dice qual è la migliore a 500, con la
  distorsione del punto 5 dichiarata.
- **Non dice niente su `weight_decay` né sullo schedule di LR.** Quelli sono E0h,
  che è un pilota diverso e va ridisegnato: qui restano allo status quo
  (`0,0`, `constant`) perché far variare due assi insieme non risponde a nessuna
  delle due domande.
- **Non è un confronto fra famiglie.** Le loss di validazione di VPT e DoRA non si
  mettono in tabella l'una accanto all'altra: a 500 passi non vogliono dire niente,
  e il confronto è E3-X, a budget pieno e su metriche di chunk.

### Aperto

- **Il quarto braccio è mio, non di Antonio.** Il piano del 28 agosto diceva «3
  metodi × 3 semi = 9 job»; `probe_edp.yaml` ha però la stessa `lr_grid: ???`, e
  senza la gamba congelata E2d resta bloccata da Hydra. `e0i_lr_frozen` è la
  campagna che chiude quel buco, Antonio l'ha approvata, e **non è mai partita**.
- **La banda potrebbe essere da spostare in basso.** Le 21 prove salvate danno
  `1e-4` — il bordo — vincente in 7 casi su 9. Non si anticipa: la guardia di
  bordo è pre-dichiarata e deciderà sui numeri della campagna nuova, non su
  questi. Ma se scatta, la risposta è allargare la griglia verso il basso e
  rifare, non prendere il bordo.

### Alimenta

`configs/train/probe_edp.yaml` e i tre esperimenti di E3 (`lr_grid`), cioè il
prerequisito di E2d, E3-R.1 ed E3-P.1 — l'intera matrice Tier 1 e Tier 2.

### Esito — eseguita il 29 agosto, e **il criterio dice di non adottare nessuno dei quattro numeri**

**12 celle su 12 completate**, dopo la correzione del loader (ADR-038). Job:
`540019_0` · `540028_1..8` · `540167_0..2` (congelata) · `540197_6..8` (il
rilancio delle tre DoRA). Verdetto prodotto da `scripts/aggrega_lr.py`, cioè dal
criterio scritto il 28 agosto **prima** di lanciare:

```
frozen-attentive   semi [1e-4, 3.16e-4, 1e-4]  -> 1e-4  (2/3 concordi)  AL BORDO — non si adotta
vptdeep-n576       semi [1e-4, 1e-4,    1e-4]  -> 1e-4  (3/3 concordi)  AL BORDO — non si adotta
lora-r32-qkvomlp   semi [1e-4, 1e-4, 3.16e-4]  -> 1e-4  (2/3 concordi)  AL BORDO — non si adotta
dora-r32-qkvomlp   semi [1e-4, 1e-4, 3.16e-4]  -> 1e-4  (2/3 concordi)  AL BORDO — non si adotta
```

**Tutte e quattro le famiglie cadono sul bordo basso della griglia**, e la guardia
del punto 3 del criterio scatta per tutte. I quattro `???` restano dove sono.

Le loss di validazione, per esteso (più basso è meglio):

| famiglia · seme | 1e-4 | 3,16e-4 | 1e-3 | 3,16e-3 | 1e-2 |
|---|---:|---:|---:|---:|---:|
| VPT s1 | **0,2121** | 0,5097 | 0,4024 | 0,4387 | 0,4362 |
| VPT s2 | **0,2296** | 0,4574 | 0,4895 | 0,5109 | 0,5049 |
| VPT s3 | **0,2281** | 0,6705 | 0,5074 | 0,4974 | 0,4864 |
| LoRA s1 | **0,2096** | 0,2876 | 0,4024 | 0,4377 | 0,4358 |
| LoRA s2 | **0,1682** | 0,1856 | 0,5083 | 0,4326 | 0,5066 |
| LoRA s3 | 0,2335 | **0,2066** | 0,5655 | 0,4725 | 0,5807 |
| DoRA s1 | **0,2111** | 0,2176 | 0,4259 | 0,4366 | 0,4551 |
| DoRA s2 | **0,1683** | 0,1827 | 0,5199 | 0,4487 | 0,4368 |
| DoRA s3 | 0,2524 | **0,1826** | 0,5939 | 0,6301 | 0,5066 |
| frozen s1 | **0,1960** | 0,2226 | 0,2207 | 0,3124 | 0,4331 |
| frozen s2 | 0,1852 | **0,1787** | 0,5087 | 0,3556 | 0,5082 |
| frozen s3 | **0,1916** | 0,2126 | 0,5951 | 0,4766 | 0,4567 |

**Il riferimento che rende leggibile la tabella.** La partizione di validazione ha
il **12,86%** di chunk positivi, quindi un predittore *costante* sulla base rate
ha BCE **0,3838**. Da `1e-3` in su, **35 celle su 36** stanno sopra quel valore:
quei learning rate non producono un modello peggiore del migliore, ne producono
uno peggiore di **non addestrare affatto**. Sotto, le loss stanno fra 0,168 e
0,288. Non è una salita graduale verso un ottimo: è un **gradino** fra `3,16e-4`
e `1e-3`.

**Due cose che la tabella dice e che servono a disegnare il seguito.**

1. **Nessuna prova diverge.** Le loss di training partono tutte da ~0,70 (il caso)
   e rimbalzano fra 0,24 e 0,89 a *qualunque* learning rate — sono loss di un
   passo singolo a batch effettivo 16, cioè ~2 positivi per batch, quindi rumore
   quasi puro e non leggibili. Il segnale sta tutto nella validazione.
2. **`3,16e-4` non è morto per le famiglie reparam e per la congelata.** Su VPT il
   divario fra i due punti bassi è un abisso (0,21 contro 0,51); su LoRA, DoRA e
   frozen sono vicini e **un seme su tre inverte**. La superficie lì è piatta, il
   che è coerente con un ottimo **vicino** al bordo — non decadi sotto. È la
   ragione per cui la banda della seconda campagna va spostata di poco.

### Il braccio di sensibilità: **non è lo schedule**

Il sospetto ragionevole, sollevato da Antonio, era che il gradino fosse un
artefatto di `lr_schedule: constant` — che è una **nostra deviazione dal
riferimento**, dove ogni probe di V-JEPA 2.1 usa `WarmupCosineLRSchedule`, e che
E0i aveva ereditato come status quo perché E0h non è mai stato eseguito.

Misurato: `e0h_ucf_vjepa21l_vptdeep-n576-cosine_s1` (job `540186_0`, 2 h 42 min),
stessa cella di VPT s1 con `lr_schedule: cosine` e tre punti.

| lr | **cosine** | constant |
|---|---:|---:|
| 1e-4 | **0,2002** | 0,2121 |
| 1e-3 | 0,4164 | 0,4024 |
| 5e-3 | 0,4175 | 0,4387 *(a 3,16e-3)* |

**Il coseno non recupera i learning rate alti.** Vale ~0,012 di loss al punto
ottimo e lascia `1e-3` a 0,416, cioè sempre sopra il predittore costante.
L'ipotesi «mancava il decadimento» è **falsificata**, e il gradino è una proprietà
del problema. Non è la decisione di E0h — un braccio, un seme, tre punti — ma è la
prima misura che quel pilota abbia prodotto, e va citata quando lo si ridisegna.

### Costo reale, e il confronto con la previsione

| famiglia | previsto (28/8) | **misurato (29/8)** | prova |
|---|---:|---:|---:|
| frozen-attentive | 2,2 h | **3,0 h** | 36 min |
| VPT-deep | 3,6 h | **4,6 h** | 55 min |
| LoRA r=32 | 4,4 h | **6,2 h** | 74 min |
| DoRA r=32 | 5,5 h | **7,9 h** | 95 min |

Le previsioni restano ottimistiche del 25-45%, ma per una ragione diversa da
quella del 28 agosto: non è più il loader (ADR-038), è che i profili di E0b.3
misuravano clip sintetiche **già residenti sulla GPU**, senza il trasferimento
host→device di 113 MB per micro-batch né il loop vero. Lo scarto misurato su
VPT-deep è **0,86 s/passo costante**, e sommato al throughput GPU di E0b.3 predice
LoRA e DoRA entro il 5%. È il fattore da portarsi dietro per dimensionare E2/E3.

**Le tre celle DoRA sono andate in TIMEOUT a 6:59 con 4 prove su 5** — previsto, e
per la prima volta senza danno: la ricevuta di ADR-038 aveva salvato le quattro
prove finite, e il rilancio automatico (`540197`) ha eseguito **solo la quinta**,
in 1 h 35 min invece di 7,9 h. È il meccanismo che il 28 agosto non c'era e che è
costato 62 ore di H100.

Costo totale della campagna: **~70 ore di H100**, contro le ~45 previste.

### Cosa serve adesso

**Una seconda campagna con la banda spostata in basso**, che è ciò che il criterio
prescrive per un risultato al bordo. I dati dicono anche di quanto: poiché
`3,16e-4` compete con `1e-4` su tre famiglie su quattro, l'ottimo è **vicino** al
bordo e non decadi sotto. La banda proposta è **quattro punti a mezza decade,
`1e-5 · 3,16e-5 · 1e-4 · 3,16e-4`**, che scende di una decade e tiene dentro come
àncore i due punti già misurati.

Costo sui tempi veri: **~29 ore di H100** per le 12 celle a quattro punti (VPT
3,7 h · LoRA 5,0 h · DoRA 6,3 h · congelata 2,4 h), **tutte dentro una fetta**,
DoRA compresa. `lr_schedule` resta `constant`: il braccio dice che il coseno non
cambia il quadro.

⚠️ **Sono decisioni di Antonio** (`CLAUDE.md` §6): i quattro punti della banda sono
protocollo, e le ore di cluster sono sue. Finché non sono prese, `train.lr_grid`
resta `???` in `configs/train/probe_edp.yaml` e nei tre esperimenti di E3, e
Hydra ferma ogni cella di E2 e di E3 nominandolo.

### La seconda campagna: `e0i2` — banda decisa e sottomessa il 29 agosto

**La banda è `3e-5 · 5e-5 · 1e-4 · 2e-4`**, decisa da Antonio il 29 agosto, e non è
quella proposta qui sopra. Vale la pena scrivere entrambe le mosse, perché la
differenza è il punto:

* la proposta di questa voce (`1e-5 · 3,16e-5 · 1e-4 · 3,16e-4`) scendeva di una
  decade a passo di mezza decade;
* la proposta iniziale di Antonio era `3e-5 · 5e-5 · 7e-5 · 1e-4` — passo lineare,
  fitto dove i dati dicono che l'ottimo vive;
* **entrambe mettevano `1e-4` su un estremo della griglia**, ed è il valore che la
  prima campagna ha scelto quattro volte su quattro. Con `1e-4` sul bordo, la
  guardia del punto 3 del criterio sarebbe scattata di nuovo *sullo stesso
  numero*: ~29 ore di H100 per un secondo «non si adotta», e per costruzione, non
  per come fossero andate le misure. La banda adottata sposta l'estremo alto a
  `2e-4` e tiene `1e-4` **interno**, mantenendo il passo fitto in basso.

`2e-4` non c'è perché ci si aspetti che vinca — la prima campagna dice che a
`3,16e-4` si sta già peggio su VPT — ma perché senza di lui `1e-4` non sarebbe
adottabile nemmeno se fosse la risposta giusta. È il costo di un punto di griglia
per rendere la campagna capace di concludere.

**Cosa NON cambia**, e conta quanto la banda: il criterio di decisione (§«Criterio
di decisione», scritto il 28 agosto e mai riaperto), i tre semi, i 500 passi,
`eval_every = max_steps`, il batch effettivo 16, `augment: official_nocrop`,
`weight_decay: 0,0` e **`lr_schedule: constant`** — il braccio col coseno (job
`540186`) ha falsificato l'ipotesi «mancava il decadimento», quindi lo schedule
non è un asse di questa campagna. Si muove **una cosa sola**.

**Perché un `exp_id` nuovo e non un secondo giro di `e0i`.** Le due campagne
scelgono su griglie di **cardinalità diversa** (cinque punti, quattro), e
`scripts/aggrega_lr.py` rifiuta — è l'invariante 5 in forma eseguibile — un
esperimento le cui celle abbiano scelto fra un numero diverso di punti. Con lo
stesso `exp_id` i dodici `run_id` sarebbero coincisi, e sarebbero successe due
cose entrambe silenziose: il dedup di `campagna.py` (che chiede al **registry**, e
il registry le ha tutte e dodici) avrebbe dato ogni cella per già fatta, e un
`--tutte` avrebbe sovrascritto le ricevute della prima campagna, che sono l'unica
copia dei numeri su cui il criterio ha già deciso. È lo stesso espediente di
`e0h_..._-cosine`, portato un livello più su perché qui a cambiare non è una cella
ma la griglia di tutte. Il criterio di lettura resta questo, e `aggrega_lr.py` lo
applica alla campagna che gli si nomina:

```powershell
python scripts/aggrega_lr.py --esperimento e0i2
```

La guardia di bordo **legge gli estremi dalla griglia della cella**, non da due
costanti scritte nello script: `1e-4` era un bordo nella prima banda e in questa è
un punto interno, e la stessa riga di codice tratta correttamente entrambe.

**Come è stata sottomessa.** Due ondate, e non per prudenza ma per aritmetica: il
tetto dell'account è **10 job in coda** e ogni task di array conta, quindi le 9
celle PEFT e le 3 congelate non ci stanno insieme.

| ondata | sweep | celle | job | risorse |
|---|---|---:|---|---|
| 1ª | `e0i2_lr` | 9 (`0-8%9`) | `540229` (22:25 del 29/8) | `aiq`, 06:59:00, `-c 8 -g 1` |
| 2ª | `e0i2_lr_frozen` | 3 (`0-2%3`) | `540259` (00:12 del 30/8) | `aiq`, 04:30:00, `-c 8 -g 1` |

All'ingresso in coda SLURM ha avviato **sei** task su `ainode02` (le tre VPT e le tre
LoRA) e messo le tre DoRA in `PD (Resources)`: `%9` è un tetto, non una prenotazione, e su
`aiq` c'erano sei H100 libere delle sedici del cluster. La seconda ondata è partita alle **00:12 del 30 agosto**, appena chiuse le
tre celle VPT (COMPLETED in 3h35-3h38, contro le 3,7 h previste): due delle tre celle
congelate sono entrate subito, la terza in `PD (Resources)`. Nella prima campagna la gamba
congelata non è mai partita per esattamente questa aritmetica, e la voce lo
registrava come cosa lasciata aperta: qui è pianificata, non dimenticata.

**Costo atteso: ~29 h di H100** per le dodici celle, sui tempi **misurati** il 29
agosto (36 · 55 · 74 · 95 min a prova per congelata · VPT · LoRA · DoRA): VPT
3,7 h · LoRA 4,9 h · DoRA 6,3 h · congelata 2,4 h. **Tutte dentro una fetta**, DoRA
compresa — con il ~10% di margine, che è stretto e informato: dal 29 agosto un
TIMEOUT costa il rilancio della sola prova mancante (`exp/lr_trials.py`, ADR-038) e
non la cella intera.

### Esito di `e0i2` — un numero su quattro, e ADR-039 chiude gli altri tre

**12 celle su 12 COMPLETED**, nessun TIMEOUT (job `540229_0..8` e `540259_0..2`).
Costo reale **~52 h di H100** contro le ~29 previste: la previsione di questa voce
era di nuovo ottimistica, e va letta come tale.

Verdetto di `scripts/aggrega_lr.py --esperimento e0i2`, cioè del criterio scritto
il 28 agosto:

```
vptdeep-n576       semi [1e-4, 1e-4, 2e-4]  -> 1e-4  (2/3 concordi)  ok
frozen-attentive   semi [2e-4, 1e-4, 2e-4]  -> 2e-4  (2/3 concordi)  AL BORDO — non si adotta
lora-r32-qkvomlp   semi [2e-4, 1e-4, 2e-4]  -> 2e-4  (2/3 concordi)  AL BORDO — non si adotta
dora-r32-qkvomlp   semi [2e-4, 1e-4, 2e-4]  -> 2e-4  (2/3 concordi)  AL BORDO — non si adotta
```

Le loss di validazione, per esteso (più basso è meglio; **grassetto** = scelta):

| famiglia · seme | 3e-5 | 5e-5 | 1e-4 | 2e-4 |
|---|---:|---:|---:|---:|
| VPT s1 | 0,2333 | 0,2135 | **0,2102** | 0,2697 |
| VPT s2 | 0,2233 | 0,2237 | **0,2139** | 0,3763 |
| VPT s3 | 0,2874 | 0,2655 | 0,2800 | **0,2299** |
| LoRA s1 | 0,1937 | 0,2058 | 0,2119 | **0,1743** |
| LoRA s2 | 0,1822 | 0,1732 | **0,1686** | 0,1842 |
| LoRA s3 | 0,2830 | 0,2633 | 0,2355 | **0,2015** |
| DoRA s1 | 0,1936 | 0,2062 | 0,2072 | **0,1762** |
| DoRA s2 | 0,1821 | 0,1728 | **0,1685** | 0,1756 |
| DoRA s3 | 0,2866 | 0,2624 | 0,2390 | **0,2114** |
| frozen s1 | 0,2174 | 0,2066 | 0,2124 | **0,1806** |
| frozen s2 | 0,1945 | 0,1872 | **0,1817** | 0,1867 |
| frozen s3 | 0,2547 | 0,2166 | 0,2057 | **0,1975** |

**Due famiglie di comportamento, ed è il risultato che serve a disegnare il
seguito.** VPT ha una **curva con un picco**: `1e-4` vince su due semi e a `2e-4`
la loss esplode (0,2697 e 0,3763 contro ~0,21). Le altre tre sono **piatte**: lo
scarto fra il migliore e il peggiore dei quattro punti, dentro un seme, è ≤0,02,
mentre da seme a seme è ~0,07 — cioè a 500 passi la LR pesa meno del seme, e
l'argmin si sposta con il rumore. LoRA e DoRA sono, punto per punto,
**indistinguibili** (differenze ≤0,002 su nove celle su dodici): è un fatto da
riprendere quando si legge C7.

**Il rumore di riesecuzione, misurato senza spendere niente.** `1e-4` è l'unico
punto presente in **entrambe** le bande, quindi le due campagne danno 12 coppie
della stessa cella `(famiglia, seme, lr)` rieseguita a distanza di un giorno:
**|Δ| mediano 0,004**, massimo **0,052** (VPT s3). È la prima misura diretta del
pavimento di rumore di questa pipeline, e dice che quasi tutte le differenze della
tabella sono segnale. Nota collegata: proprio VPT s3 è l'outlier — a `1e-4` la
prima campagna gli dava 0,2281 contro 0,2800 di adesso; se avesse riprodotto, VPT
sarebbe 3/3 invece di 2/3.

**La decisione: `1e-4` per tutte e quattro (ADR-039).** Per VPT è l'uscita del
criterio; per le altre tre è una **deroga dichiarata**, presa da Antonio il 30
agosto invece di una terza campagna da ~52 h. Le ragioni stanno in ADR-039 e sono
quelle della tabella qui sopra: `1e-4` è il punto più eseguito dello studio, sulle
tre famiglie al bordo la scelta pesa meno del seme, ed è il verso conservativo
della distorsione dichiarata al punto 5 del criterio. `aggrega_lr.py` continua a
stampare «AL BORDO — non si adotta», ed è giusto così: il verdetto della macchina
e la decisione dell'umano restano due cose distinte.

### I costi per passo, misurati — e servono a dimensionare E2/E3

Dai timestamp dei log di `540229`/`540259`, a batch effettivo 16 su H100:

| famiglia | s/passo | passata di validazione | prova (500 passi) |
|---|---:|---:|---:|
| frozen-attentive | 1,76 | 21 min | 36 min |
| VPT-deep n=576 | 3,95 | 21 min | 54 min |
| LoRA r=32 | 5,43 | 29 min | 75 min |
| DoRA r=32 | 7,00 | 36 min | 95 min |

**La passata di validazione dipende dalla famiglia**, e questo la voce non lo
diceva: usa il forward **adattato**, quindi DoRA la paga 36 min dove la congelata
ne paga 21. È la ragione per cui la regola «+0,86 s/passo costante» scritta il 29
agosto prediceva bene VPT e male DoRA.

**Conseguenza per E2/E3**, a 3000 passi e con una sola valutazione finale: frozen
**1,8 h** · VPT **3,6 h** · LoRA **5,0 h** · DoRA **6,4 h**. DoRA sta in una fetta
da 6:59 con l'8% di margine — e `peft_budget_a.yaml` ha già `checkpoint_every: 100`,
quindi un TIMEOUT costa un periodo e non la cella.

### Cronaca — la prima campagna è fallita per intero, e la causa non era il calcolo

**28 agosto, job array `539643`, 9 celle su 9 FAILED, `ExitCode 0:10`, ~62 ore di
H100 spese e nessun risultato nel registry.** Questa sezione esiste per la regola
5 del registro — anche i risultati negativi si scrivono — e perché il difetto che
ha trovato pesa su tutta la matrice, non su questo pilota.

**Il costo per prova era sottostimato di 2,5–3×.** Misurato dai timestamp dei log
(`hpc/log/evd-e0i_lr-539643_{0,3,6,7,8}.out`), una prova — 500 passi più una
passata di validazione — è costata:

| famiglia | previsto (tabella sopra) | **misurato** | rapporto |
|---|---:|---:|---:|
| VPT-deep | 43 min | **2h12m · 2h08m · 2h09m** | 3,0× |
| LoRA r=32 | 53 min | **2h28m · 2h24m** | 2,8× |
| DoRA r=32 | 66 min | **2h46m · 2h48m** | 2,5× |

Cinque prove fanno **10,8 h · 12,1 h · 14,0 h** contro una fetta da 6:59. In
6h54m le celle VPT hanno chiuso 3 prove su 5, LoRA e DoRA 2 su 5.

Scomposto: **training ~8,5 s/passo** a batch effettivo 16, cioè ~1,9 finestre/s
contro le 5,01 previste per VPT; **validazione ~61 min** per 13.551 chunk, cioè
~3,7 finestre/s contro le 16,55 del forward congelato. La validazione è l'errore
singolo più grosso, **4×**.

**La causa: `ExperimentRunner._loader` aveva `num_workers: 0` cablato.** Ogni
loader del repository — training e validazione, prove della griglia e fit finale —
decodificava il video nel **processo principale**: il lavoro di CPU non si
sovrapponeva al forward, lo precedeva, e le 8 CPU chieste con `-c 8` non le usava
nessuno. E0d-throughput aveva già misurato i due estremi senza che nessuno
collegasse le due cose: **0 worker danno 4,11 finestre/s, 8 ne danno 18,64**.

Due segni che lo confermano, entrambi visibili nei log prima di guardare il
codice: il rapporto fra famiglie si è **appiattito** (previsto VPT/DoRA 1,87× dai
throughput GPU, osservato 1,28× — la firma di un limite fuori dalla GPU), e la
cache RAM si esauriva a `35/13551` finestre, perché una clip 384×384×32 in fp32
pesa ~56 MB.

⚠️ **Era scritto, con il suo rimedio, in `docs/10` §E0g dal 26 agosto**: «con
`num_workers > 0` si recupererebbe quasi tutto […] è però la voce di costo che
pesa su E3: va affrontata prima di E3-R.1, non dopo». Il rinvio è costato 62 ore
di H100. La lezione operativa non è sul loader ma sul registro: **un rinvio
dichiarato in un documento non è un promemoria**, e nulla lo ha riportato in
superficie quando è diventato bloccante.

**La causa immediata — il modo di morire — è un'altra, ed è indipendente.** Tutte
e nove le celle sono uscite con **segnale 10, `SIGUSR1`**, a `06:53:5x` di elapsed
contro un wall time di `06:59:00`: esattamente i 300 s di
`--signal=B:USR1@300`. Il gestore del segnale veniva installato **dentro**
`_checkpointer`, che esce con `None` prima di arrivarci quando
`checkpoint_every <= 0` — e `lr_search.yaml` ha `checkpoint_every: 0`, per una
ragione corretta. Senza gestore, l'azione predefinita di `SIGUSR1` termina il
processo: nessun traceback, `.err` con soli warning, `.out` troncato a metà di un
passo. Un campo faceva due lavori e del secondo non si era tenuto conto.

**Terza ipotesi, esclusa con una misura:** SLURM aveva messo 7 celle su
`ainode02` e 2 su `ainode01`, e la contesa sembrava la spiegazione ovvia. Non lo
è — stessa famiglia, prima prova: DoRA su un nodo con 7 celle **2h46m**, su un
nodo con 2 celle **2h49m** e **2h49m**. Il costo è intrinseco, e «parallelizzare
meno» non è un rimedio.

### Cosa dicono le 21 prove salvate — e perché **non** si adottano

Le celle non sono mai arrivate al registry, quindi l'unica copia è nei log
scaricati:

```
VPT   s1: 1e-4→0.2369  3.16e-4→0.4423  1e-3→0.4232
      s2: 1e-4→0.3037  3.16e-4→0.4140  1e-3→0.4856
      s3: 1e-4→0.2254  3.16e-4→0.4465  1e-3→0.5122
LoRA  s1: 1e-4→0.1783  3.16e-4→0.1929
      s2: 1e-4→0.1705  3.16e-4→0.2163
      s3: 1e-4→0.2109  3.16e-4→0.1801
DoRA  s1: 1e-4→0.1864  3.16e-4→0.2519
      s2: 1e-4→0.1712  3.16e-4→0.3201
      s3: 1e-4→0.2139  3.16e-4→0.1981
```

**`1e-4` vince in 7 casi su 9, ed è il bordo basso della griglia.** Anche a
campagna completa il punto 3 del criterio — la guardia di bordo, scritta **prima**
— direbbe «non si adotta»: la banda utile sta probabilmente **sotto** `1e-4`. Su
VPT il segnale è netto e monotono (3 semi su 3); su LoRA e DoRA due semi su tre
invertono fra i due punti misurati, quindi lì la superficie è piatta.

⚠️ Sono misure **a 500 passi con la griglia incompleta**, prodotte per giunta con
il difetto dell'augmentation descritto sotto: non entrano in tesi e non scelgono
niente. Servono a sapere dove guardare se la guardia di bordo scatterà di nuovo.

### Il secondo difetto, trovato correggendo il primo

`ExperimentRunner._augment` **non leggeva il campo `crop`** del gruppo Hydra:
`official_nocrop`, che dichiara `crop: false`, arrivava al loader con il default
`crop=True` della dataclass e `scale=(1,1) ratio=(1,1)` — la combinazione che
ADR-032 §3 aveva già riconosciuto come sbagliata, che su un frame 4:3 ricade sul
**ritaglio centrale 240×240**. Le nove celle di E0i hanno quindi addestrato con
il ritaglio che l'incontro del 27 agosto aveva rimosso, e lo stesso sarebbe
successo a **ogni** run di E2 e di E3. Dettaglio e correzione in ADR-038.

### Il ridisegno — cosa cambia e cosa no

**Il criterio di decisione non si tocca**, ed è la ragione per cui era stato
scritto prima. Non cambiano neppure la griglia, i tre semi, i 500 passi, il
`select_only` né la ripartizione delle destinazioni.

Cambiano tre cose, tutte in ADR-038: i budget di `edp_v1` dichiarano
**`num_workers: 8`**; il gestore di `SIGUSR1` si arma nel run, così una cella
senza checkpoint muore **dichiarandolo**; e le prove **finite** della griglia si
ricordano in `runs/{run_id}/lr_trials.json`, così un TIMEOUT costa una prova
invece di una cella. Con i worker accesi il conto della tabella *Costo* torna
applicabile — era una stima del calcolo, e il calcolo non era il collo di
bottiglia.

## E0h-3k — `constant` o `cosine`, deciso a 3000 passi *(**chiuso**: vince `cosine`, con due limiti dichiarati)*

**Aperto il 30 agosto 2026** · sweep `e0h_schedule_3k` (9 celle) · run_id
`e0h_ucf_vjepa21l_vptdeep-n576-{const-1e4|cos-1e4|cos-2e4}_s{1|2|3}` ·
decisione: ADR-040 (il meccanismo), Antonio (le ore).

> Questa voce esiste **prima** dei numeri, come E0g, E0i e E2b-ML. Il §«Criterio
> di decisione» è datato 30 agosto e non si riscrive dopo aver visto i risultati.

### Cosa chiede

**Se la matrice debba usare `constant` o `cosine`.** Ed è una domanda che finora
non era mai stata posta: `configs/train/peft_budget_a.yaml` e `probe_edp.yaml`
**non dichiarano affatto** `lr_schedule`, quindi ogni cella di E2 e di E3
erediterebbe `constant` dal default della dataclass `BudgetSpec`. Non è una
decisione, è un default — e `constant` è per giunta una **deviazione dal
riferimento**, dove ogni probe di V-JEPA 2.1 usa `WarmupCosineLRSchedule`.
Qualunque cosa esca, la riga va scritta esplicita in entrambi i file: è metà del
valore di questo pilota.

### Perché non basta quello che si sa già

Il braccio del 29 agosto (`e0h_..._vptdeep-n576-cosine_s1`, job `540186`) ha
misurato il coseno a **500** passi dentro il budget di E0i, e ha falsificato
l'ipotesi grossa — «il gradino sopra `3,16e-4` è mancanza di decadimento»: il
coseno vale ~0,012 al punto ottimo e lascia `1e-3` a 0,416. Ma non può dire quale
schedule usare nella matrice, e la ragione è la stessa per cui `lr_search.yaml`
aveva escluso il coseno dalla selezione: **il coseno decade su `max_steps`**,
quindi a 500 passi è un altro oggetto che a 3000. Questo pilota misura a 3000,
cioè dove la risposta serve.

### Il disegno, e perché è questo

| | scelta | perché |
|---|---|---|
| **bracci** | 3: **A** `constant`@1e-4 · **B** `cosine`@1e-4 · **C** `cosine`@2e-4 | con `warmup_steps = 0` e `final_lr = 0` il coseno ha LR **medio** `ref_lr/2` (`train/budget.py:lr_al_passo`), quindi B addestra a media `5e-5` contro il `1e-4` di A. A-vs-B da solo misura un coseno **handicappato**; A-vs-C da solo confonde schedule e picco. Servono entrambi per attribuire. `2e-4` non è scelto a occhio: è il picco che **appaia la LR media** ad A |
| **famiglia** | **VPT-deep n=576**, una sola | è l'unica delle quattro che in E0i2 ha una curva vera — a `2e-4` la loss esplode (0,27 e 0,38 su due semi contro ~0,21) — quindi è dove un effetto di schedule si vede. Sulle altre tre la superficie è piatta e il coseno potrebbe solo non farsi notare. È anche la più economica delle tre PEFT, e si concatena col braccio a 500 passi, che era sulla stessa famiglia |
| **semi** | 3 (1, 2, 3) | l'effetto atteso è dell'ordine di 0,01 e il rumore di riesecuzione **misurato** è \|Δ\| mediano 0,004 con massimo 0,052 (voce E0i). Con un seme solo non si distinguerebbe l'effetto dal rumore |
| **budget** | quello della **matrice**, `peft_budget_a.yaml` | 3000 passi, batch effettivo 16, niente validazione nel training, `checkpoint_every: 100`. È la riga che rende trasferibile la risposta: un pilota su un budget diverso risponderebbe per un orizzonte che nessuna cella percorrerà |
| **valutazione** | `eval_split: val` (ADR-040) | un pilota che decide un campo di budget di tutta la matrice non guarda i 290 video su cui la matrice si misura |
| **tutto il resto** | identico a E3-P.1 | `augment: official_nocrop`, `weight_decay: 0,0`, layer 23, `edp_v1` |

### ⚠️ Il limite di `eval_split: val`, dichiarato adesso

`_soglia` stima τ **sulla validazione** (`docs/04` §5.3). Valutando *sulla*
validazione, soglia e giudizio guardano gli stessi video: le metriche che ne
escono sono **ottimiste** e **non si confrontano** con quelle di E2/E3. Per
confrontare tre bracci a parità di regola il bias è identico in tutti e non
disturba l'**ordinamento**, che è la sola cosa che questo pilota deve produrre —
ma i suoi numeri non vanno in tesi come numeri, solo come verso. `metrics.json`
lo dichiara da sé con `threshold_on_eval_split: true`.

### Costo

> ⚠️ **La stima della valutazione qui sotto è stata FALSIFICATA dall'esecuzione**
> (job `540432`): ~45 min previsti, **oltre 2 h 15 min** misurati. La parte sul
> training regge invece al minuto. Resta scritta perché era una previsione e
> perché il punto dove sbagliava — l'unica riga che diceva «questa parte non è mai
> stata misurata» — è la diagnosi. Vedi *Cronaca*.

Sui tempi **misurati** il 30 agosto (voce E0i): VPT-deep **3,95 s/passo** a batch
effettivo 16, quindi 3000 passi sono **3,3 h**. La valutazione su `val` sono due
passate sui 312 video — `_valuta` e la stima di τ — stimate ~45 min, e **questa
parte non è mai stata misurata**: il numero noto (21 min) è la passata di *loss*
del training, un'altra cosa. Totale **~4,1 h per cella**, Wall Time `05:30:00`
con il 35% di margine, **~37 h di H100** per le nove celle, una sola ondata da 9.
Un TIMEOUT non è una catastrofe: `checkpoint_every: 100` fa riprendere dal
centesimo di passo.

### Cronaca — la prima sottomissione è morta nella valutazione, non nel training

**Job array `540432`, sottomesso alle 10:51 del 30 agosto, cancellato alle 16:23.**
Due celle in TIMEOUT (`540432_0` e `_1`, `constant` @1e-4, semi 1 e 2), due appena
avviate, cinque mai partite. ~11 h di H100 spese, **nessuna metrica prodotta**.

Il training non c'entra: ha chiuso i 3000 passi in **3 h 18 min**, contro le 3,3 h
previste dai 3,95 s/passo misurati il giorno prima. La previsione era giusta. È la
valutazione a essere costata 2,8× la stima:

| fase | previsto | **misurato** |
|---|---:|---:|
| training, 3000 passi | 3,3 h | **3 h 18 min** |
| valutazione su `val` (inferenza + soglia) | ~0,75 h | **> 2 h 15 min**, uccisa |

Dai timestamp, il numero che dice tutto:

| fase | finestre/s |
|---|---:|
| training (3000 passi × 16 finestre in 3 h 18 min) | **4,04** |
| inferenza su 312 video (13.551 finestre in 2 h 07 min) | **1,78** |

**Il forward puro era più lento, per finestra, del forward+backward.** La causa è
in ADR-041: `InferenceEngine.run_videos` valutava un video alla volta con la
decodifica nel processo principale, e `ExperimentRunner._soglia` faceva una
seconda passata seriale sulle stesse finestre. Nessuna delle due usava i worker
che il job aveva chiesto con `-c 8`. È la stessa diagnosi di ADR-038 sull'altro
ramo del sistema, trovata però nel modo più caro: **dopo** aver pagato tutto il
training.

**Cosa resta sul disco, e perché conta.** Le celle 0 e 1 hanno il checkpoint al
passo 3000 (`checkpoint_every: 100`). Poiché `eval.num_workers` **non entra nel
`config_sha`** (ADR-041 punto 5), quelle due celle riprendono da lì e pagano la
**sola** valutazione: il loro elapsed sarà quindi la misura diretta del costo
corretto, senza spendere un altro training. È così che si misura il guadagno
prima di rilanciare le nove — la lezione che questa cronaca esiste per non
ripagare una terza volta.

**Cosa NON è cambiato.** Il criterio qui sotto è quello scritto il 30 agosto prima
di lanciare, e il TIMEOUT non l'ha toccato: non è stato prodotto nessun numero da
cui rileggerlo.

### Criterio di decisione — scritto il 30 agosto, prima di lanciare

1. **La metrica primaria è la chunk-AUC sulla validazione**, che è la metrica
   primaria del protocollo (`docs/04` §4.3). La loss di validazione si riporta
   accanto perché è ciò su cui E0i ha scelto, e permette di leggere i due piloti
   nella stessa scala; se le due si contraddicessero, **decide la AUC** e la
   contraddizione si scrive.
2. **Le differenze sono appaiate per seme**, mai medie sottratte — stessa regola
   di E0g. Per ogni seme si calcola `Δ = braccio − A`.
3. **Si adotta `cosine` solo per una vittoria netta**: il migliore fra B e C deve
   battere A su **almeno 2 semi su 3** *e* avere una **Δ mediana ≥ +0,02 di AUC**.
   Sotto quella soglia si resta su `constant`.
4. **La soglia è alta di proposito, e la ragione è dichiarata adesso.** Se si
   adotta il coseno, `1e-4` **non è più il picco giusto** — è l'ottimo di un
   `constant` — e la LR si riapre, cioè un'altra campagna di selezione. Una
   vittoria marginale del coseno costerebbe più di quanto valga. La regola è
   quindi **asimmetrica** per costruzione, non per prudenza generica.
5. **Quale dei due picchi vince si riporta sempre.** Se C batte B, è la conferma
   diretta dell'argomento sulla LR media, e va scritto perché è anche
   un'informazione su ADR-039: `2e-4` è il punto che il criterio di E0i aveva
   scelto — al bordo — per le altre tre famiglie.
6. **Il limite di generalizzazione, dichiarato adesso.** Si deciderebbe per tutte
   e quattro le famiglie sulla base di una. Se il coseno **vince**, prima di
   adottarlo si conferma su `frozen-attentive` — la più lontana da VPT e la più
   economica di tutte: 6 celle da ~1,8 h, **~11 h**. Se il coseno **perde**, il
   limite non morde: si resta dove si è già, e la deviazione dal riferimento
   diventa una scelta misurata invece di un default ereditato.

### Esito — 31 agosto, criterio applicato com'era scritto

Nove celle su nove complete (`540760` su `aiq`, `540768` su `gpuq`, `ExitCode 0:0`),
tutte con **`eval_precision: bf16`** e `threshold_on_eval_split: True`.

**chunk-AUC sulla validazione**, la metrica primaria:

| braccio | s1 | s2 | s3 | media | ampiezza |
|---|---:|---:|---:|---:|---:|
| **A** `constant` @1e-4 | 0,916313 | 0,914533 | 0,916808 | 0,915885 | 0,00228 |
| **B** `cosine` @1e-4 | 0,930205 | 0,935689 | 0,943387 | **0,936427** | 0,01318 |
| **C** `cosine` @2e-4 | 0,927237 | 0,919991 | 0,936735 | 0,927987 | 0,01674 |

Differenze **appaiate per seme** (punto 2), `Δ = braccio − A`:

| | s1 | s2 | s3 | mediana | vince |
|---|---:|---:|---:|---:|---:|
| B − A | +0,013892 | +0,021156 | +0,026579 | **+0,021156** | **3/3** |
| C − A | +0,010924 | +0,005458 | +0,019926 | +0,010924 | 3/3 |

**Punto 3: si adotta `cosine`.** Il migliore è B, vince su 3/3 semi (ne servivano
2) e ha Δ mediana +0,021156 ≥ +0,02.

⚠️ **Il margine sulla soglia è +0,00116**, cioè **metà** dell'ampiezza fra semi del
braccio A. Sono due affermazioni distinte, ed entrambe vanno riportate: **l'effetto
è netto** — i tre Δ, da +0,0139 a +0,0266, stanno **6-12× sopra** la dispersione
fra semi di A — mentre **il superamento della soglia è fragile**, e con semi
diversi poteva non avvenire. Il punto 4 aveva dichiarato in anticipo che la soglia
è alta di proposito.

**Punto 5: vince il picco basso.** B batte C su 3/3 semi (Δ mediana +0,00665).
L'argomento della «LR media» — un coseno a picco `2e-4` ha media `1e-4` e dovrebbe
pareggiare A — **non regge**. È informazione diretta su ADR-039, dove `2e-4` era il
punto scelto al bordo per le altre tre famiglie.

**Punto 6: la conferma su `frozen-attentive` NON è stata fatta.** Antonio ha deciso
il 31 agosto di adottare senza. **È una deviazione dichiarata dal criterio**, non
un'omissione: sta in **ADR-046** e nei commenti di `peft_budget_a.yaml` e
`probe_edp.yaml`, accanto alla riga che modifica.

⚠️ **Il punto 1 è risultato inapplicabile.** Chiedeva di riportare accanto alla AUC
la **loss di validazione**, per leggere E0h e E0i nella stessa scala. Non esiste:
`metrics.json` ha solo `final_loss`, che è la loss dell'ultimo batch di *training*
(`train/trainer.py:476`), e il runner non calcola alcuna loss sulla validazione —
coerente con ADR-036, ma vuol dire che quel confronto **non si può fare**. Si
scrive invece di sostituirlo con un numero che gli somigli.

### Le due macchine, e perché non disturbano il confronto

`aiq` non si liberava (partenza stimata alle 02:16) mentre `gpuq` aveva 4 A100
libere e **zero job runnable** in coda. Le sette celle restanti sono state divise
**apposta per non allineare l'hardware al braccio**:

| braccio | H100 | A100 |
|---|---:|---:|
| `constant` @1e-4 | 2 (s1, s2) | 1 (s3) |
| `cosine` @1e-4 | 2 (s1, s2) | 1 (s3) |
| `cosine` @2e-4 | 1 (s1) | 2 (s2, s3) |

Due controlli dicono che non c'è spostamento sistematico: il seme **s3 è il più
alto in tutti e tre i bracci** — è un effetto di seme, non di scheda — e nel
braccio C, dove due celle su tre sono A100, l'A100 produce **sia il massimo (s3)
sia il minimo (s2)**.

**Il rapporto H100/A100 si conferma 1,77×** e va aggiunto ai numeri di E0b.3, che
lo davano a 1,68-1,81× su un altro carico: VPT-deep a batch effettivo 16 fa
**3,95 s/passo** su H100 e **6,89-6,99** su A100, misurato su intervalli
consecutivi di 100 passi in quattro celle. Elapsed reale: **3h41-3h44** su H100
(wall 5:30), **6h17-6h26** su A100 (wall 7:00).

### Cosa questo pilota **non** dice

- **Non dice niente su `weight_decay`**, che resta `0,0` in tutti e tre i bracci.
  È l'altro asse di E0h e va misurato a parte: farne variare due insieme non
  risponde a nessuna delle due domande.
- **Non produce numeri di tesi.** Vedi il limite di `eval_split: val` qui sopra.
- **Non ridisegna E0h.** La voce storica qui sotto — 80 celle a `k ∈ {1,8}` sulla
  cache di E0d — assumeva il regime few-shot e resta **da ridisegnare**; questo
  braccio ne stacca la metà che blocca la matrice.

### Alimenta

`configs/train/peft_budget_a.yaml` e `configs/train/probe_edp.yaml`, che dopo
questo pilota dichiareranno `lr_schedule` **esplicitamente** invece di ereditarlo
dal default di `BudgetSpec`.

---

## E0h — `weight_decay` e schedule di LR *(aperto: criterio scritto prima dei numeri)*

**Aperto il 26 agosto 2026** · sweep `e0h_wd_schedule` (80 celle) · run_id
`e0h_ucf_vjepa21l_frozen-attentive-wd{0.0|0.01|0.1|0.4}-{constant|cosine}_k{1|8}_s{1..5}`
· **nessun run eseguito** · decisione: ADR-030.

> Stesso avvertimento di E0g e di E2b-ML: questa voce esiste prima dei numeri
> apposta. Il criterio sotto è datato 26 agosto e non si riscrive dopo aver
> visto i risultati.

### Cosa chiede

Il confronto col codice ufficiale (`evals/video_classification_frozen`,
riferimento k400 `vitl-384`) trova due assi su cui `probe_e2*.yaml` non replica
il riferimento, e nessuno dei due era mai stato misurato:

- **`weight_decay`.** Il riferimento sweepa `wd ∈ {0,01 · 0,1 · 0,4 · 0,8}`
  (`configs/eval_2_1/vitl-384/k400.yaml`) e non usa **mai** zero. Le nostre
  celle di E2 sono a `0,0` dal primo giorno, senza che nessuno l'avesse deciso
  — era già segnalato aperto in ADR-028 senza essere chiuso.
- **Lo schedule del LR.** Il riferimento usa sempre `WarmupCosineLRSchedule`
  (`eval.py:490-515`, warmup a zero passi su ogni riga della griglia k400):
  decade da `lr` a 0 con un mezzo coseno. Il nostro `Trainer.fit` teneva il LR
  **costante** per l'intero fit — non era mai stato nemmeno implementabile,
  perché `BudgetSpec` non aveva un campo per uno schedule.

A 105 finestre di support (k=1) e con `weight_decay: 0.0` la probe memorizza il
support in meno di cento passi (E0g l'ha già misurato per caso, loss 0,0002):
è esattamente il regime in cui regolarizzazione e schedule contano di più, e
in cui il pilota deve guardare per primo.

### Come si esegue

Layer 23 fisso — lo stesso ancoraggio di E0g, per la stessa ragione: non è una
decisione di layer (la prende E2b) e questo pilota non si rifà quando E2b
sceglie altro. `augment: official` fisso, perché è già la catena decisa da E0g
per tutta E2 (invariante 5): un pilota che facesse variare due cose alla volta
non risponderebbe a nessuna delle due. Budget `train/probe_e2` per il resto
(passi fissi, 500 passi, batch 32, griglia LR a tre punti), con `weight_decay` e
`lr_schedule` come assi.

**Disegno:** `weight_decay ∈ {0,0 · 0,01 · 0,1 · 0,4}` (la griglia ufficiale
meno l'estremo 0,8 — il meno plausibile a 105–10 411 finestre di support) ×
`lr_schedule ∈ {constant, cosine}` × `k ∈ {1, 8}` (lo stesso schema di E0g: a
k=1 l'overfitting è massimo, a k=8 c'è abbastanza supervisione da vedere se
l'effetto si è già consumato) × 5 semi del calendario `fsp_v1` = **80 celle**.

### Criterio di decisione — scritto il 26 agosto, prima di lanciare

> ⚠️ Stessa clausola di E0g: se questo paragrafo sembra diverso da come lo si
> ricorda dopo aver visto i numeri, è la memoria a essere cambiata.

**Il confronto è un unico esperimento a 8 bracci**, non due assi decisi in
sequenza: le 8 combinazioni (`weight_decay`, `lr_schedule`) si confrontano
tutte contro lo **stesso status quo** — `(wd=0,0, constant)`, la config in
vigore oggi in `probe_e2*.yaml` — appaiate per seme, a ciascun k separatamente.
Decidere `weight_decay` e schedule in due passi separati nasconderebbe
un'eventuale interazione fra i due (per esempio: il coseno aiuta solo se il
wd non è zero), e trovarla dopo, dentro E2c o E2d, costerebbe una campagna
intera invece di un pilota.

**Metrica di decisione:** chunk AUC (ADR-024), la stessa di E0g e di E2b-ML.

**Regola.** Per ciascuno dei 7 bracci non-baseline, a ciascun k:

> Δ = chunk_AUC(braccio) − chunk_AUC(status quo), appaiata per seme.
> Il braccio **si candida** se mediana(Δ) ≥ **+0,01** (l'ordine di grandezza
> del rumore di riesecuzione misurato da E0c) **e** il segno è positivo in
> almeno **4 semi su 5**.

- **Se nessun braccio si candida a nessuno dei due k:** si tiene lo status quo
  (`wd=0,0`, `constant`) — lo stesso ripiego pre-dichiarato di Q26/E0f v2:
  cambiare un budget condiviso da E2b/E2c/E2d ha un costo, e deve guadagnarselo,
  non ereditarlo per default dall'essere «quello che fa il riferimento».
- **Se un solo braccio si candida a entrambi i k:** si adotta, e la mediana più
  alta fra i candidati a parità di condizione spezza un eventuale pareggio; a
  parità anche lì, vince il più semplice (constant su cosine, wd più piccolo).
- **Se un braccio si candida a un solo k (interazione con k):** si dichiara
  così, esattamente come è successo con l'augmentation in E0g — il budget può
  finire per dipendere da k, ma è una scoperta da riportare, non una scelta
  fatta in anticipo.

**Perché contro lo status quo e non contro «il valore del riferimento».**
A differenza di E0g — dove la catena ufficiale aveva un argomento a suo favore
indipendente dai nostri dati (è la ricetta che ha prodotto i numeri pubblicati)
— qui il riferimento sweepa `wd` e sceglie per **cella** con una validazione
che noi non stiamo replicando (griglia 5×4, selezione per K400 intero): non
c'è un singolo valore "ufficiale" da difendere per analogia. L'unica domanda
onesta è se cambiare la config **di oggi** paga, appaiato sugli stessi semi.

**Dove il pilota NON entra.** Sul support di training, come l'augmentation di
E0g: query `eval_center` e validazione episodica della selezione LR restano
deterministiche e fuori da questo asse.

### Costo

Il feature store è query-side (E0d): il support non ci è mai stato, quindi
questo pilota — come E0g — non aggiunge un solo forward rispetto a quello che
E2 già paga, e non invalida niente. `augment: official` fisso vuol dire però
che ogni cella eredita il costo per-passo già misurato da E0g.

| voce | valore |
|---|---|
| celle | 80 (40 a k=1, 40 a k=8) |
| passi per cella | ~500 a k=1 (nessuna selezione: si eredita `lr_grid[0]`, `docs/04` §5), ~2000 a k=8 (griglia LR a tre punti + re-fit) — identico a E0g, il `weight_decay`/schedule non cambia il numero di passi |
| costo per passo | eredita 2,46 s/passo della catena `official` (E0g, job 537519/537520) |
| durata di una cella | k=1: ~20 min. k=8: ~85 min — le stesse stime di E0g, stesso budget |
| totale | ~(40×20 min) + (40×85 min) ≈ **13,3 + 56,7 ≈ 70 GPU-ora** |
| wall clock atteso | dipende da quante fette il tetto dell'account concede in parallelo — vedi nota sotto |

**Il costo è più alto di E0g (~17 GPU-ora) perché le celle sono 2,67× di più**,
non perché ciascuna costi di più: `weight_decay` e `lr_schedule` non toccano
il numero di passi né la geometria d'ingresso. Prima di sottomettere le 80
celle intere vale la stessa regola imparata il 25 agosto (job 537517): **la
prima cella si sottomette da sola**, e qui in più con `lr_schedule="cosine"`
attivo — è la combinazione che il codice non aveva mai eseguito prima di
ADR-030.

### Aperto

- **Le risorse (`paralleli`, `fetta_massima`) restano al default prudente** di
  `campagna.py` (non il tetto pieno che E0g aveva avuto autorizzato il 25
  agosto): un'autorizzazione vale per la campagna per cui è stata data, non si
  eredita.
- **Nessun job è stato sottomesso.** Si sottomette quando Antonio lo chiede
  (`CLAUDE.md` §2 punto 8), a partire dalla cella singola di verifica.

---

## E2b-ML — Il braccio multilevel *(aperto: criterio e previsione scritti prima dei numeri)*

**Aperta il 26 agosto 2026** · sweep `e2b_multilevel` (5 celle) · run_id
`e2b_ucf_vjepa21l_frozen-attentive-ml4_k8_s{1..5}` · **nessun run eseguito** ·
decisione: ADR-029.

> **Questa voce esiste prima dei numeri, ed è il punto.** La regola 4 dice che le
> decisioni prese *guardando* i risultati si scrivono qui come mini-ADR; questa
> voce scrive invece cosa ci aspettiamo **prima**, perché un braccio che può
> facilmente «non concludere» è esattamente quello in cui la tentazione di
> raccontare l'esito a posteriori è più forte. Il criterio e la previsione sotto
> sono datati 26 agosto e non si riscrivono: si annotano.

### Cosa chiede

E2b (`e2b_layers`, 20 celle) chiede **quale layer** legge la testa. Questo braccio
chiede una cosa diversa: **i quattro layer gerarchici, letti insieme, rendono più
di uno solo?** È la strategia con cui V-JEPA 2.1 riporta Diving-48 e Jester —
`evals/video_classification_frozen/modelcustom/vit_encoder_multiclip_multilevel.py:123`
concatena i token dei quattro layer con `torch.cat(outputs, dim=1)`, asse **token**,
e la sonda attentiva fa attenzione sull'intera sequenza combinata. Su ViT-L 2.1
`out_layers` è `[5,11,17,23]` (`configs/eval_2_1/vitl-384/diving48.yaml:53`), cioè
esattamente i quattro che il feature store contiene.

Non sostituisce E2b: si legge **accanto** alle sue quattro curve, appaiato sugli
stessi cinque semi.

### Come si esegue

k=8 fisso, `train/probe_e2` (500 passi), `augment: official`, `head: attentive` con
`depth: 4` e `n_heads: 16` — tutto identico alle celle a layer singolo, che è la
condizione perché il confronto valga. L'unica differenza in config è
`layer: hierarchical` invece di un intero, e `use_activation_checkpointing: true`
sulla testa (non cambia i logit, solo la memoria).

**Cosa NON è identico, e va scritto accanto ai numeri:** la sequenza in ingresso
alla testa passa da 2048 a 8192 token. I `depth-1 = 3` blocchi di self-attention
del pooler sono quadratici, quindi ~16× i FLOP di attenzione; e l'I/O dallo store
è 4× — 16,8 MB per finestra invece di 4,2 in fp16 a 256 px. I **parametri
addestrabili sono identici**: la concat è sull'asse token, `embed_dim` non cambia,
e il classificatore ufficiale è costruito con `embed_dim=encoder.embed_dim` senza
nessun `* len(out_layers)` (`eval.py:151`). La parità di budget di `docs/05` §5 —
step, batch effettivo, griglia LR, augmentation — regge alla lettera; il costo
wall-clock finisce in E5b, dove `docs/05` §5 dice già che è «parte del risultato,
non un difetto».

### Criterio di lettura, dichiarato ora

Confronto **appaiato sui semi** fra il braccio multilevel e la cella a layer
singolo che E2b eleggerà (`scripts/confronta_predizioni.py`), su chunk AUC.

- Il braccio **conclude a favore** del multilevel se la mediana della differenza
  appaiata supera **+0,02** di chunk AUC **e** almeno 4 semi su 5 sono positivi.
- Sotto quella soglia l'esito dichiarato è **«non conclude»**, e si riporta così.

La soglia non è arbitraria: E0c ha misurato che il pavimento di rumore **non è una
costante** — 1,1 pp dove l'accuratezza è satura, **8,4 pp** dove non lo è — e su
UCF-Crime nessuno si aspetta il regime saturo. Cinque semi sono il minimo, e sotto
~2 pp non si distingue una vittoria da una fluttuazione.

### Previsione, scritta prima di lanciare

**Su ViT-L ci si aspetta che non concluda.** E la ragione per cui *non* ci si
aspetta di più merita di stare qui, perché contraddice l'intuizione con cui il
braccio è stato proposto:

L'argomento naturale era «gli intermedi sono più informativi su g/G, dove le
`norms_block` sono state addestrate, mentre su B/L sono a distanza zero
dall'inizializzazione (job 532484)». Quell'argomento **è già stato falsificato dai
nostri dati**: l'appendice (C) di E0c (18 agosto, righe 1496-1598) ha rifatto il
confronto fra layer su ViT-g e ViT-G e trova «l'ordinamento monotono e il salto
early/late … identici — **non è l'assenza di training della norma a spiegare il
crollo ai layer bassi**». Su g/G i layer intermedi, letti **da soli**, non rendono
meglio che su L.

Quindi il multilevel non scommette che gli intermedi siano *migliori*: scommette
che siano **complementari** — che una probe capace di attendere su tutti e quattro
insieme batta quella che ne vede uno. È un meccanismo diverso, ed è l'unico che
E0c non ha misurato, avendo misurato solo probe a layer singolo. Resta la ragione
per cui il braccio vale la pena; ma è una scommessa sul riferimento, non sui
nostri dati, e va detto in questi termini.

**«Non conclude» è un esito da riportare, non un fallimento** (regola 5): sarebbe
la metà negativa di un claim del riferimento, misurata su un dominio diverso da
quello dove il claim è stato fatto.

### Cosa alimenta

- La figura di E2b: le quattro curve single-layer più il punto multilevel con la
  dispersione sui cinque semi.
- **La potatura del feature store**, che resta sospesa finché questo braccio non
  ha un esito (`docs/09`, postilla del 26 agosto; R6).
- Se concludesse a favore, la scelta del layer per E2c/E2d/E3 va rifatta — ed è un
  mini-ADR da scrivere qui, allora.

### Aperto

- **Il wall time non è misurato.** Prima delle cinque celle se ne sottomette
  **una sola** (`seed=1`), che è la regola imparata il 25 agosto: dieci job che
  falliscono insieme costano dieci volte tanto. I tre numeri da guardare in quel
  run sono l'**hit rate della cache** (dev'essere 1,000000: sotto, il braccio sta
  pagando forward che credeva di leggere), la **VRAM di picco** e il tempo di
  valutazione, che a layer singolo è già l'86% del costo di una cella.
- **Nessun job è stato sottomesso.** Si sottomette quando Antonio lo chiede
  (`CLAUDE.md` §2 punto 8).

---

## E0g — Quale augmentation, per misura *(chiuso: si adotta la catena ufficiale)*

**Aperto il:** 25 agosto 2026 · **run_id:** `e0g_ucf_vjepa21l_frozen-attentive-aug{none|mild|official}_k{1|8}_s{1..5}` ·
**stato:** criterio congelato, campagna pronta e non ancora eseguita ·
**codice:** `evd/data/augment.py`, `evd/vendor/vjepa21_*`, `configs/augment/`,
`configs/experiment/e0g_augment.yaml`, `configs/sweep/e0g_augment.yaml` · **decisione:** ADR-028

### La domanda

Con quale augmentation si addestrano le probe di E2 e i run PEFT di E3. Non è una
domanda estetica: `docs/04` §7.2 D5 la classifica come **parte della parità di
budget** (invariante 5), perché a k=1 — 105 finestre di support, `weight_decay: 0.0`
— può contare più del metodo PEFT che si sta confrontando. Fino al 25 agosto la
risposta era «nessuna», ma per omissione: nel codice non esisteva.

Le due posizioni difendibili sono note e **nessuna delle due è misurata**:

* **`official` (la ricetta del riferimento).** La catena del probe V-JEPA 2.1 è
  cablata in `make_dataloader` (`eval.py:414-447`) e vale identica per k400,
  ssv2, coin, diving48, jester. Adottarla è la stessa disciplina con cui
  `configs/head/attentive.yaml` copia testa e profondità da `k400.yaml`, ed è
  ciò che rende i numeri assoluti leggibili accanto alla letteratura.
* **`none` o `mild` (l'argomento di dominio).** L'etichetta di una finestra è
  **temporale** — il segmento annotato la interseca — mentre il ritaglio è
  **spaziale**. A `scale = 0,08` su un frame 320×240 di CCTV si tengono circa
  78×78 pixel di sorgente: l'evento che definisce la classe può non esserci più,
  e il random erasing `p = 0,25` può cancellarlo. Su k400 l'azione riempie il
  fotogramma; su una telecamera fissa no.

### Il criterio di decisione — **scritto il 25 agosto, prima di lanciare**

> ⚠️ Questo paragrafo è la ragione per cui l'esperimento vale qualcosa. Se lo si
> rilegge dopo aver visto i numeri e sembra diverso da come lo si ricordava, è la
> memoria a essere cambiata, non il criterio.

**Disegno.** 3 catene × 2 budget (k ∈ {1, 8}) × 5 semi del calendario `fsp_v1` =
**30 celle**, probe attentiva congelata sul layer 23, budget `train/probe_e2`
(passi fissi, 500 passi, batch 32, griglia LR a tre punti). Passi fissi e non
epoche fisse: confrontando catene di augmentation il numero di **pescaggi
aumentati** deve essere identico, e a passi fissi lo è per costruzione a ogni k.

**Il confronto è appaiato.** A parità di seme le tre catene vedono lo stesso
support (stesso manifest), la stessa query e la stessa inizializzazione: la
differenza che resta è l'augmentation. Si legge quindi Δ = metrica(catena) −
metrica(`none`) **per seme**, mai come differenza di medie fra gruppi.

**Metrica di decisione:** **chunk AUC**, la primaria di ADR-024. AP secondaria,
riportata ma non decisoria.

**Regola, con l'asimmetria dichiarata.** Si adotta **`official`** *salvo prova
contraria*, dove «prova contraria» significa, **a entrambi i k**:

> Δ(`official` − `none`) ha mediana ≤ −0,01 di chunk AUC **e** segno negativo in
> almeno **4 semi su 5**.

Se la prova contraria scatta, si applica lo stesso test a `mild`: se `mild` lo
supera (cioè non è dimostrabilmente peggiore di `none`), si adotta `mild`;
altrimenti `none`.

**Perché asimmetrico, e perché non «vince il più alto».** Con cinque semi il
massimo di tre medie è quasi sempre rumore: E0c ha misurato che due esecuzioni
identiche della stessa probe distano **1,3 pp** dove l'accuratezza è alta e
**8,4 pp** dove è bassa, e su UCF-Crime siamo nel secondo regime. Un criterio
«vince chi fa il numero più grande» selezionerebbe il rumore e lo scriverebbe nel
protocollo. `official` è la ricetta del riferimento e non deve dimostrare
superiorità per essere usata: deve solo non essere **dimostrabilmente dannosa**.
È la stessa forma del ripiego pre-dichiarato di E0f v2 (Q26), dove nessun asse
concludeva e ha retto il default.

**La soglia di 0,01 non è un numero tondo scelto adesso**: è l'ordine di
grandezza del rumore di riesecuzione misurato in E0c, sotto il quale questo
disegno non ha potere. Una differenza più piccola non è «piccola», è
**invisibile a cinque semi**, e trattarla come reale sarebbe la stessa
sovrainterpretazione contro cui E0e-bis ha messo in guardia.

**Dove l'augmentation non arriva, e perche'.** Sul **support di training**, e
basta. La **query** resta `eval_center` deterministica (lo store E0d continua a
valere). La **validazione episodica** della selezione LR resta deterministica:
serve a misurare una configurazione, non ad addestrarla. E la **stima della
soglia** — che pure legge finestre di support — usa il preprocessing
deterministico: la soglia è un punto di lavoro che verra' applicato alle query, e
stimarlo su una distribuzione d'ingresso diversa lo sposterebbe di una quantita'
che nessuno ha misurato.

**Cosa NON decide questo pilota.** Il layer (lo decide E2b), il budget della
curva (deciso da ADR-027), e il `weight_decay`, che resta a 0,0 in tutte e 30 le
celle: cambiarlo qui confonderebbe due interventi. Che il riferimento non usi mai
`wd = 0` resta una domanda aperta per Antonio, separata (ADR-028, *Conseguenze*).

### Costo, e perché è basso

Il feature store è **query-side**: le 147.548 feature sono i 290 video di
`Anomaly_Test`, mentre il support viene da `Anomaly_Train` e non è mai stato
nella cache. Ogni finestra di support è quindi già oggi un miss, decodificata e
forwardata a ogni passo — accendere l'augmentation **non aggiunge un forward** e
**non invalida niente**. La query resta `eval_center` deterministica e continua a
leggersi dallo store.

| voce | valore |
|---|---|
| celle | 30 (15 a k=1, 15 a k=8) |
| passi per cella | ~500 a k=1 (sotto k=4 la LR si eredita: una prova sola), ~2000 a k=8 (tre punti di griglia + re-fit) |
| forward di support | ~16 mila a k=1, ~64 mila a k=8 |
| costo per passo, **misurato** il 25 agosto (job 537519 vs 537520, stessa cella k=1) | `none` **0,77 s**, `official` **2,46 s** — **3,2×** |
| durata di una cella | k=1: ~8 min (`none`) / ~20 min (`official`). k=8: ~30 min / ~85 min |
| totale | **~17 ore GPU**, in tre fette da 10 task (tetto pieno della QOS, autorizzato da Antonio) |
| wall clock atteso | ~2,2 h di calcolo, più la coda |

**Da dove viene il 3,2×, e perché va scritto qui invece che scoperto in E3.**
RandAugment lavora su immagini **PIL**, frame per frame, e il loader di training
gira con `num_workers: 0` (`ExperimentRunner._loader`): il lavoro di CPU non si
sovrappone al forward, lo precede. Con `num_workers > 0` si recupererebbe quasi
tutto — ma **il numero di worker entra nel seme dell'augmentation** (ogni worker
ha il suo contatore, e l'indice del worker è nel seme per non far ripetere lo
stesso ritaglio), quindi diventerebbe un campo che cambia i pixel e andrebbe
dichiarato nel budget come `batch_size`. Non si tocca dentro un pilota che deve
decidere un'altra cosa. **È però la voce di costo che pesa su E3**, dove il
forward non è congelato e le celle sono molte di più: va affrontata prima di
E3-R.1, non dopo.

**Il segnale che si vede già alla prima cella.** A 100 passi su k=1 la loss vale
**0,0002 senza augmentation e 0,2247 con** la catena ufficiale: con 105 finestre
di support e `weight_decay: 0.0` la probe memorizza il support in meno di cento
passi. Non dice ancora niente sull'AUC — è esattamente ciò che il pilota va a
misurare — ma dice che l'intervento morde dove ci si aspettava.

### Cronaca

**25 agosto, job 537517 — la prima cella, sottomessa da sola, è fallita.** Non per
l'augmentation: `configs/train/probe_e2.yaml` dichiara `early_stop_patience: 5`, e
il `Trainer` rifiuta una pazienza senza `val_loader`. Il **re-fit non ha una
validazione per costruzione** (`docs/04` §5), quindi sarebbe fallita **ogni cella
di E2, a ogni k, E2b compresa**. Corretto costruendo il budget del re-fit con
`early_stop_patience=None` e dichiarandolo nel log; la guardia del `Trainer` resta
dov'era, perché aveva ragione. Test di regressione con contro-prova: rosso senza
la correzione, verde con.

Vale la pena registrare **perché è costato cinque minuti invece di una coda
intera**: la prima cella della campagna era stata sottomessa **da sola** (`-a 0%1`)
prima delle altre ventinove, proprio perché nessun run di forma E2 aveva mai girato
su dati veri. Dieci job che falliscono insieme producono dieci log identici e
occupano il tetto della QOS mentre si legge il primo.

**25 agosto, cella 0 chiusa: `chunk_auc = 0,7895` (`none`, k=1, seme 1) — e un
numero che vale per tutta F7.** Training 6 min 24 s, **valutazione 39 min 33 s**.
L'86% del costo di una cella di E2 se ne va a leggere 36.887 feature dallo store,
con la GPU al 10-19%. `PIANO.md` e `configs/train/probe_e2.yaml` dicono entrambi
che una probe «costa minuti grazie alla cache»: è vero per il **forward** ed era
falso per la **lettura**.

La causa era `SafetensorsFeatureStore.get`, che apriva una connessione SQLite
**e** un file safetensors per ogni finestra — ~37 mila di ciascuna su filesystem
parallelo. È la stessa insidia che `__contains__` aveva già pagato il 25 agosto
(`CLAUDE.md` §4), rimasta nel percorso di lettura. Corretta: l'indice si legge
una volta e resta in memoria, e l'handle dell'ultimo file resta aperto (le
finestre di un video sono consecutive e stanno nello stesso file — 30 mila
chiavi su 276 file).

| | per finestra | su 36.887 finestre |
|---|---:|---:|
| prima (misurato dalla cella 0) | 64 ms | **39,5 min** |
| dopo (misurato sullo store vero, 400 letture) | 14,0 ms | **~8,6 min** |

Restano 14 ms perché a quel punto si è vicini al limite di I/O: un layer di una
finestra a 256 px sono 2048 token × 1024 × fp16 = **4,2 MB**, quindi valutare la
query di una cella significa leggere ~155 GB. Il margine che resta non sta più
negli handle.

**Cosa questo cambia, e cosa no.** Non cambia nessun numero — la lettura
restituisce gli stessi byte, e un test lo verifica bit per bit contro un lettore
appena aperto. Cambia il costo di E2: i due bracci di `e2d_curve` sono 70 celle,
e mezz'ora a cella di sola lettura erano **~35 ore di GPU quasi ferma**. Le celle
di E0g già eseguite hanno pagato il prezzo vecchio; quelle dopo il commit
`96e6a44` no, ed è irrilevante ai fini del confronto perché l'augmentation non
c'entra con la velocità di lettura.

### Esito — 26 agosto 2026, 30/30 celle, nessun fallimento

**La catena adottata è `official`**, per applicazione del criterio scritto sopra
prima dei run: la prova contraria richiede il danno a **entrambi** i k, e a k=8
non c'è. L'artefatto è `runs/e0g_aggrega-augment.json`, rigenerabile con
`scripts/aggrega_augment.py`.

| catena | k | mediana Δ vs `none` | semi negativi | verdetto |
|---|---:|---:|---:|---|
| `official` | 1 | **−0,0701** | 5/5 | danneggia |
| `official` | 8 | **+0,0205** | 2/5 | ok |
| `mild` | 1 | −0,0458 | 5/5 | danneggia |
| `mild` | 8 | −0,0298 | 4/5 | danneggia |

Differenze appaiate per seme (chunk AUC):

| seme | k=1 `none` | Δ `mild` | Δ `official` | k=8 `none` | Δ `mild` | Δ `official` |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0,7895 | −0,0733 | −0,1382 | 0,8445 | −0,0804 | −0,0275 |
| 2 | 0,7755 | −0,0898 | −0,0732 | 0,7732 | −0,0263 | **+0,0625** |
| 3 | 0,6850 | −0,0140 | −0,0701 | 0,7836 | −0,0351 | **+0,0205** |
| 4 | 0,6906 | −0,0317 | −0,0146 | 0,8234 | **+0,0317** | **+0,0276** |
| 5 | 0,7286 | −0,0458 | −0,0321 | 0,8179 | −0,0298 | −0,0418 |

**Il risultato vero non è «quale catena», è che l'effetto dipende da k**, e il
criterio non lo aveva previsto: era scritto per distinguere «danneggia» da «non
danneggia», non per un'interazione. A k=1 l'augmentation costa **7 punti di AUC**
in modo unanime; a k=8 la catena ufficiale ne guadagna 2 in mediana, con tre semi
su cinque positivi. La lettura che regge i numeri è quella che l'argomento di
dominio suggeriva, con un limite: con 13 segmenti di evento (105 finestre) un
ritaglio che tiene l'8% dell'area toglie più informazione di quanta varietà
aggiunga — l'etichetta della finestra è temporale, il ritaglio è spaziale, e su
una telecamera fissa l'evento occupa una frazione piccola della scena. Con 104
segmenti la varietà comincia a ripagare.

**Cosa si adotta, e a quale prezzo dichiarato.** `official` per tutta E2 ed E3,
come dice il criterio. Il prezzo è il punto k=1 della curva few-shot, che sarà
**circa 7 punti di AUC più basso** di quanto sarebbe senza augmentation, in modo
sistematico e misurato. Va scritto accanto alla curva, non scoperto in seduta: il
primo punto della curva è quello che il claim C2/C4 legge più da vicino.

⚠️ **Una scelta che il criterio non contemplava, e che quindi non prendo qui.**
Una catena **dipendente da k** — niente augmentation sotto k=4, ufficiale sopra —
è ciò che i numeri suggeriscono, ma non era fra le opzioni dichiarate prima dei
run, e sceglierla adesso sarebbe scegliere dopo aver visto i dati: esattamente
ciò che il pilota esisteva per evitare. Se si vuole, è una **decisione di
Antonio** e va dichiarata come tale in tesi, con la data in cui è stata presa e la
nota che nasce da E0g e non da un'ipotesi a priori. Il costo di non prenderla è i
7 punti a k=1; il costo di prenderla è un asse in più nella parità di budget
(l'augmentation smetterebbe di essere costante lungo la curva, e `docs/05` §5 va
riscritta di conseguenza).

**Due cose che il pilota ha prodotto oltre al verdetto**, entrambe registrate
sopra: il costo di valutazione di una cella di E2 (39,5 min → 8,6 min dopo la
correzione dello store) e il costo per passo dell'augmentation (3,2×, da
affrontare prima di E3-R.1). Nessuna delle due era in programma, ed entrambe
pesano su F7 più del verdetto.

## E0e — La geometria d'ingresso, per misura *(chiuso: esito negativo)*

**Aperto il:** 18 agosto 2026 · **run_id:** `e0e_ucf101mini_vjepa21l_frozen-attentive_kna_s1` ·
**stato:** **chiuso** il 18 agosto, esito negativo — la replica che ha deciso è E0e-bis · **codice:** `evd/exp/e0e.py`,
`scripts/e0e_geometria.py`, `scripts/slurm/e0e.sbatch`

> Questa voce è scritta **prima** dei risultati, ed è l'unico modo perché il criterio di §3
> conti qualcosa: un criterio scelto dopo aver visto i numeri non è un criterio, è una
> giustificazione. I risultati si aggiungono sotto, senza toccare ciò che c'è sopra.

### Cosa chiede

Quale ingresso vuole V-JEPA 2.1, dato che **il sorgente non lo dice**. Tutti e otto i config di
training 2.1 rilasciati (`configs/train_2_1/vit{b,l,g,G}16/`) sono a **256px** e campionano a
`fps: 4`; il README dichiara **384** per i checkpoint distillati, e a 384 girano tutti i config di
eval con `frame_step: 4` (≈2,13 s su una sorgente a 30 fps); **la ricetta dello stadio a 384 non è
nel repository**. In più la sorgente di UCF-Crime è **320×240 per tutti i 1900 video**, quindi 384
è un ingrandimento di 1,6× di un 240p, e la vista centrale della catena A butta via il **25% della
larghezza** — che in sorveglianza può contenere l'evento.

### Il piano

Fattoriale pieno, otto celle, probe congelata e budget identico (2000 passi, batch 32, lr 1e-3,
layer 23, hop 1 s, storage fp16, calcolo AMP BF16), sugli stessi 1363 video di E0c:

| asse | valori | perché questi |
|---|---|---|
| risoluzione | **256** · **384** | 256 è la griglia nativa della RoPE (`pretrained_grid_size = 256/16`) e la risoluzione di ogni config di training; 384 è quella dei checkpoint e degli eval. 2048 contro 4608 token per finestra |
| ritaglio | **`eval_center`** · **`squash`** | il primo è la catena A ufficiale e perde il 25% della larghezza; il secondo porta il frame intero a quadrato, e la distorsione 4:3→1:1 (rapporto 0,75) è il bordo inferiore di `random_resize_aspect_ratio` del training 2.1 |
| span | **fps 7,5** · **fps 4** | la geometria di eval contro il campionamento del pretraining. **Cambiare span non costa un token**: 16 frame restano 16 frame |

Due semi per cella (1, 2), sulle **stesse** feature: la loro escursione misura il rumore di
addestramento, ed è il denominatore con cui si legge il confronto fra celle.

> **Correzione al piano, trovata dalla stima (job 533775) prima di spendere una GPU.** La finestra
> a fps 4 copre **91 frame nativi** contro i 46 di fps 7,5, e su UCF101 questo esclude il **7,6%
> dei video di train e l'11,5% di quelli di test** (contro 0,2–0,3% a fps 7,5) perché più corti di
> una finestra. Due conseguenze, e la seconda è la grave: le quattro celle a fps 4 sarebbero
> **fallite** sul tetto del 5% di E0c; e anche senza tetto avrebbero misurato la top-1 su 332 video
> di test invece che su 375, quindi il confronto fra span avrebbe contenuto la differenza di
> *insieme* invece che quella di *geometria*. La correzione: l'ammissibilità si decide **una volta,
> sulla finestra più lunga del piano**, e le liste filtrate valgono per tutte le celle
> (`prepara_liste`). È il motivo per cui la stima viene prima dell'estrazione e non dopo.

### Il criterio, dichiarato adesso

1. Statistica primaria: **top-1 media sui semi**, per cella.
2. Pavimento di rumore: **la massima escursione fra semi dentro una cella**. Non è una costante
   (E0c: da 1,1 a 8,4 punti a seconda del regime), quindi si legge dai dati del pilota.
3. Vince la media più alta; le celle che le distano **meno del pavimento** sono pari merito, e fra
   quelle si prende la più economica: meno token per finestra, poi span più corto.

### Cosa questo pilota **non** dice

UCF101 è 320×240 come UCF-Crime, quindi si trasferisce la domanda «quanto sopravvive di un 240p
attraverso il preprocessing»; **non** si trasferisce il compito. E0e ordina geometrie, non predice
AUC su anomalie, e non è un numero di tesi (invariante 7). `three_views` è fuori dal piano perché
il feature store per finestra la rifiuta per contratto; resta misurabile dopo come delta sulla
geometria vincente.

### Risultati — **il pilota non risolve la domanda, e questo è il risultato**

**Chiuso il:** 18 agosto 2026 · **job:** stima 533775/533780 (`fatq`), celle 533781 (array 0–7,
`gpuq`), rilancio 533945 (celle 4–7), aggregazione 533988 · **commit:** `16f2160` (celle 0–3),
`b5a921a` (celle 4–7) · **costo:** 4 celle × 2h08 + 4 celle × (2h13 dopo il recupero del primo
seme) ≈ 17 GPU-ora su A100, più 8 store per 382 GB

| cella | top-1 media | min | max | escursione | token/finestra | finestre |
|---|---:|---:|---:|---:|---:|---:|
| `s256-eval_center-fps7p5` | **0,9970** | 0,9970 | 0,9970 | 0,00 pp | 2048 | 8129 |
| `s384-eval_center-fps7p5` | **0,9970** | 0,9941 | 1,0000 | 0,59 pp | 4608 | 8129 |
| `s256-eval_center-fps4` | 0,9941 | 0,9911 | 0,9970 | 0,59 pp | 2048 | 5881 |
| `s384-squash-fps4` | 0,9926 | 0,9882 | 0,9970 | 0,89 pp | 4608 | 5881 |
| `s256-squash-fps7p5` | 0,9778 | 0,9704 | 0,9852 | 1,48 pp | 2048 | 8129 |
| `s384-eval_center-fps4` | 0,9704 | 0,9467 | 0,9941 | 4,73 pp | 4608 | 5881 |
| `s256-squash-fps4` | 0,9615 | 0,9467 | 0,9763 | 2,96 pp | 2048 | 5881 |
| `s384-squash-fps7p5` | 0,9260 | 0,8935 | 0,9586 | **6,51 pp** | 4608 | 8129 |

**Il criterio di §3, applicato alla lettera, sceglie `s256-squash-fps7p5`** — e va scritto perché
è ciò che era stato dichiarato. Ma la sua *precondizione* è fallita: con un pavimento di rumore di
**6,51 pp**, sette celle su otto entrano nella fascia di pari merito, quindi il criterio degenera in
«prendi la più economica» e la geometria non è stata scelta dai dati. Riportare
`s256-squash-fps7p5` come risposta a Q21 sarebbe presentare un'assenza di segnale come una
decisione.

#### Il confronto appaiato dice perché, e dice qualcosa di più utile della tabella

A top-1 sature due scalari non distinguono «equivalenti» da «test senza potere» (E0c, appendice).
Sugli stessi 338 video, contando i **video che cambiano classe**:

| pavimento *dentro* la cella (seme 1 vs seme 2, **stesse feature**) | video discordi |
|---|---:|
| `s256-eval_center-fps7p5` · `s384-eval_center-fps7p5` | 2/338 (0,6%) |
| `s256-eval_center-fps4` | 4/338 (1,2%) |
| `s384-squash-fps4` | 5/338 (1,5%) |
| `s256-squash-fps7p5` | 13/338 (3,8%) |
| `s256-squash-fps4` · `s384-eval_center-fps4` | 18/338 (5,3%) |
| `s384-squash-fps7p5` | **40/338 (11,8%)** |

I contrasti *fra* celle, a un solo asse cambiato, vanno da 1 a 28,5 video di media: **tutti dentro
l'intervallo che il solo addestramento produce sulle stesse feature**. Nessun asse — risoluzione,
ritaglio, span — è separabile dal rumore su questo compito.

#### Cosa si è comunque imparato, e con che grado di fiducia

1. **Su UCF101 ridotto le geometrie sono indistinguibili in accuratezza.** Non è che 256 ≈ 384: è
   che il compito è saturo (tre celle fra 0,994 e 0,997, una a 1,0000) e non ha spazio per
   mostrare una differenza. Questo *è* un risultato: significa che una sanity a 10 classi non può
   essere lo strumento con cui si sceglie la geometria.
2. **Il segnale che emerge non è sull'accuratezza ma sulla stabilità**, ed è ordinato:
   `eval_center` a fps 7,5 è stabile a entrambe le risoluzioni (2/338), mentre `squash` e le celle
   a span lungo sono molto meno stabili (fino a 40/338). Se reggesse, sarebbe un argomento a
   favore di `eval_center` + fps 7,5. **Ma poggia su due semi**, quindi è un'ipotesi da verificare,
   non un esito.
3. **`s384-squash-fps7p5` è la cella peggiore e la più instabile insieme.** Il campo visivo intero
   a risoluzione ingrandita produce le feature su cui la probe converge peggio. Non so dire se
   sia una proprietà delle feature o dell'ottimizzazione.

#### L'errore di disegno, che è mio e va scritto

Il pavimento è stato definito come **il massimo, sulle otto celle, dell'escursione fra due semi**.
Con `n = 2` l'escursione è una statistica d'ordine estrema, e prenderne il massimo su otto celle la
gonfia per costruzione: il criterio era quasi garantito a inghiottire gli effetti prima ancora di
vedere i dati. L'appendice di E0c aveva già scritto entrambe le cose che servivano a evitarlo — che
il pavimento **non è costante** e che le accuratezze sature vanno confrontate **appaiate** — e il
pilota le ha ignorate usando uno scalare.

#### Cosa servirebbe perché il pilota decida davvero

- **Più semi.** Cinque per cella, non due: con due, il pavimento non è stimabile e il criterio non
  è applicabile.
- **Un compito con margine.** UCF101 a 10 classi è saturo. O tutte e 101 le classi, o un
  sottoinsieme scelto per essere difficile, o meno finestre di training — qualunque cosa riporti
  l'accuratezza lontano dal soffitto.
- **Statistica primaria appaiata**, non la top-1 media: i video discordi sugli stessi video, con
  il pavimento della cella accanto. L'infrastruttura c'è già (`scripts/confronta_predizioni.py`).
- Il costo stimato di una replica così: ~2,5× quella appena fatta, cioè ~45 GPU-ora.
  **Antonio l'ha decisa il 18 agosto** ed è E0e-bis, qui sotto — dove la stima si è rivelata
  larga: le tre correzioni sono di disegno e non di calcolo, gli otto store restano validi, e
  la replica costa **~9 GPU-ora**. Fino al suo esito Q21 e Q22 restano aperte per i
  supervisori invece che chiuse da un numero che non regge.

#### Nota operativa — due guasti, entrambi già pagati altrove

- **TIMEOUT delle quattro celle a 384** (job 533781, 4h esatte). Budget tarato sulle celle a 256
  (2h08) senza tener conto che a 384 la sequenza è di 4608 token contro 2048 e la probe costa il
  doppio. È lo stesso errore delle probe di g/G del 18 agosto, ripetuto. Le celle avevano
  completato estrazione e primo seme; i log erano **vuoti dopo il preambolo**, perché lo stdout
  resta nel buffer e il kill al Wall Time se lo porta via — lo stato vero l'hanno detto gli store e
  i file di predizione, non il log.
- **Rimedio strutturale:** `run_arm` scrive un record per seme e salta i semi già misurati
  (`b5a921a`). Senza, il rilancio avrebbe ripagato ~8 GPU-ora per risultati già ottenuti. È la
  stessa correzione della campagna g/G: l'artefatto su disco è la fonte di verità, non lo stato in
  coda.

## E0f — La stessa geometria, misurata sul dominio vero *(chiuso alla calibrazione)*

**Aperto il:** 19 agosto 2026 · **run_id:** `e0f_ucfcrime_vjepa21l_frozen-attentive_kna_s1` ·
**stato:** **chiuso** il 19 agosto — la calibrazione ha rifiutato il compito · **codice:** `evd/exp/e0f.py` (solo lo split),
`scripts/e0f_split.py`, `scripts/slurm/e0f.sbatch` — il fattoriale è quello di `evd/exp/e0e.py`

> Terza voce scritta **prima** dei risultati sulla stessa domanda. Qui serve più che altrove: due
> passate su UCF101 hanno già prodotto una risposta, e la tentazione di leggere questa come
> conferma o smentita di quella è esattamente ciò che un criterio dichiarato prima impedisce.

### Perché non basta E0e-bis

Il caveat che E0e-bis porta scritto dice cosa non trasferisce: UCF101 è 320×240 **come**
UCF-Crime, quindi si trasferisce la domanda «quanta informazione sopravvive al preprocessing di un
240p», **non il compito**. E per due dei tre assi il compito è precisamente ciò che conta:

1. **Il ritaglio.** `eval_center` scarta il 25% della larghezza. Su UCF101 l'azione è inquadrata
   al centro **per costruzione** — è un dataset di clip di azioni — quindi lì `squash` non aveva
   niente da guadagnare, e infatti l'effetto misurato è +0,96 pp a favore di `eval_center` con IC
   [−1,04, +2,96]. Su un video di sorveglianza quel quarto di scena può contenere l'evento, e
   allora l'asse ha un meccanismo per muoversi che su UCF101 non aveva.
2. **La risoluzione.** Gli eventi di sorveglianza sono piccoli e lontani; la tokenizzazione più
   fine di 384 (un patch ogni ~10 pixel sorgente invece di ~15) ha un motivo per contare che le
   azioni a tutto schermo di UCF101 non offrono.

Quindi E0f non è una replica: è la stessa domanda dove i meccanismi possono esprimersi.
E la decisione che ne dipende è la più cara della tesi — la geometria fissa il feature store che
E2, E3 e E4 leggeranno.

### Il vincolo che governa il disegno

**Lo split del pilota si costruisce da `Anomaly_Train.txt` e basta.** Scegliere il preprocessing
guardando `Anomaly_Test.txt` renderebbe *selezionato sui dati di test* ogni numero che la tesi
riporterà su quel test set: è l'invariante 6 aggirato di lato, la stessa cosa che Q18(a) proibisce
per le soglie. `costruisci_split` **solleva** se l'intersezione non è vuota, invece di filtrare in
silenzio — un filtro silenzioso avrebbe reso il vincolo invisibile alla prima modifica.

> **La scorciatoia che sembrava esistere non esiste.** UCF-Crime porta con sé degli split ufficiali
> di *action recognition* (`Action_Regnition_splits/`), che a prima vista sono il proxy perfetto:
> stesse classi, etichette pronte. **Intersecano `Anomaly_Test` in 171 video su 700** (verificato
> il 19 agosto). Usarli avrebbe contaminato esattamente come leggere il test set, solo in modo meno
> visibile.

### Lo split

| | |
|---|---|
| Sorgente | `Anomaly_Train.txt`, 1610 video |
| Filtri | `status != ok` (32 video: i duplicati di Q19, fra cui `Assault050`/`Robbery138`) · durata < **18 s** (103 video) |
| Tetto | **24 video per classe**, il massimo che la classe più povera consente (`Shooting`) |
| Split | **12 train + 12 val** per classe, disgiunti a livello di video, seme 1 |
| Geometria del campionamento | **hop 2 s**, 8 finestre per video nel support |
| Totale | **14 classi, 336 video: 168 train, 168 val** — tutti ammessi in tutte e otto le celle |
| Costo | 248 688 finestre, **1,7 TB** di store (job 534051) |

Il tetto non è pignoleria: senza, `Normal` (754 video ammissibili) e `Robbery` (143) deciderebbero
da soli la metrica, e il confronto fra geometrie misurerebbe soprattutto loro.

La ripartizione 12/12 pende verso la validazione più di quanto sembri, e la ragione viene da
E0e-bis: lì la calibrazione ha scelto **4 video per classe**, quindi un pool di training grande
sarebbe stato sprecato mentre ogni video di validazione compra precisione sulla metrica. Con 168
video di validazione un video vale 0,60 pp (in E0e-bis erano 275 video, 0,36 pp).

> **La durata minima non si indovina, si deriva — e ci è voluta una stima buttata per capirlo.**
> Un video entra nel pilota solo se regge `finestre_per_video` finestre, cioè se dura almeno
> `span + (W−1)·hop`. La prima stima (job 534050) girava con `hop = 10 s`, che porta la soglia a
> 73 s: **138 video su 364 sparivano**, e sparivano *dopo* che il tetto per classe era già stato
> applicato. Il filtro è uniforme fra le celle, quindi il confronto sarebbe rimasto valido — ma il
> bilanciamento per classe, che è l'unica ragione per cui il tetto esiste, era già rotto e in modo
> invisibile. Passando la soglia **derivata** al costruttore, il tetto si calcola sulla popolazione
> che sopravviverà davvero. È lo stesso errore dell'asse span di E0e, in un'altra veste: un
> filtro a valle che cambia la popolazione su cui si misura.

### Il rumore di etichetta: dov'è e dove non è

I video di training di UCF-Crime hanno **solo etichette a livello di video**. La mediana dura 262 s
e l'evento ne occupa pochi, quindi molte finestre di training portano un'etichetta che il loro
contenuto non giustifica.

- **Sul training il rumore c'è, e non si può togliere** senza annotazioni temporali che per i video
  di training non esistono (Sultani ha annotato solo il test set; quelle del laboratorio sono Q10,
  formato e licenza ancora ignoti). **Non invalida il confronto**: il rumore è identico in tutte le
  celle — stessi video, stesse posizioni delle finestre, stesse etichette — e fra due celle resta
  solo la geometria.
- **Sulla valutazione il rumore non c'è.** `run_e0c` media i logit di tutte le finestre di un video
  e classifica **il video**, e l'etichetta del video è vera. La metrica primaria è quindi una top-1
  a livello di video su 168 video con etichette pulite.

Il rischio che questo introduce è opposto a quello di E0e: non il **soffitto** ma il **pavimento**.
Se il compito è troppo rumoroso, tutte le celle stanno al livello del caso (7,1% a 14 vie) e non
c'è niente da ordinare. È esattamente ciò che la calibrazione intercetta.

### Il criterio, dichiarato adesso

Identico a E0e-bis, e lo è di proposito: cambiare criterio fra due misure della stessa domanda
renderebbe le due incomparabili.

1. **Statistica primaria: il segno.** Per ogni asse, 4 contrasti × 5 repliche = **20 confronti
   appaiati**; un asse conclude se un livello vince **≥ 16 su 20** non pari (p ≈ 0,006).
2. **Ripieghi, per ogni asse che non conclude: `256`, `eval_center`, `fps 7,5`.** Sono i valori
   **oggi in vigore** dopo E0e-bis: se E0f non distingue, non c'è ragione di cambiare la scelta
   esistente. È questo che li rende ripieghi e non una seconda scelta.
3. **Banda di calibrazione: [0,25 – 0,60]**, contro [0,70 – 0,90] di E0e-bis. Il caso è al 7,1% e
   il compito è rumoroso; una banda alta sarebbe irraggiungibile e una bassa starebbe nel rumore.
4. **`finestre_per_video` = 8** (era 2). Con W finestre equispaziate su 262 s e un evento di ~8 s,
   la probabilità che almeno una cada dentro l'evento è circa `W·8/262`: ~6% a W=2, ~24% a W=8.
   Sotto una certa soglia il support non contiene mai l'evento e non c'è niente da imparare.
   Costante dichiarata e **identica in tutte le celle**, quindi fuori dal confronto.

> **Cosa è lecito aggiustare dopo aver visto la calibrazione, e cosa no.** La calibrazione gira
> sulla **sola cella di riferimento** e non guarda le altre sette: se la banda risultasse
> irraggiungibile, ridichiararla — o allargare il pool di training — resta **dentro** il disegno,
> perché è una decisione sul *potere* della misura presa senza vedere il confronto. Toccare
> qualunque cosa dopo aver visto anche una sola cella diversa dalla riferimento, invece, è scegliere
> guardando il risultato. La distinzione è la stessa che ha permesso a E0e-bis di calibrare senza
> barare, e qui va scritta perché il rischio pavimento la rende più probabile da usare.

### Cosa questo pilota **non** dice

Ordina geometrie sul dominio giusto, ma il compito resta un **proxy**: è classificazione a 14 vie
con supervisione a livello di video, non anomaly detection temporale. Non produce AUC, non produce
AP, e non è un numero di tesi (invariante 7). Serve a decidere un iperparametro di preprocessing
prima di spendere il feature store, e a quello solo.

> **E c'è un limite più scomodo, che va scritto adesso e non quando si vedranno i numeri: questo
> proxy testa male proprio l'asse del ritaglio**, cioè quello con cui si era motivato E0f. Su
> UCF-Crime buona parte del segnale di classe sta nella **scena** — un `RoadAccidents` è una strada,
> uno `Shoplifting` è l'interno di un negozio — ed è il motivo per cui l'action recognition
> ufficiale su questo dataset sta al 23–28% invece che al 7,1% del caso. Ma la scena è distribuita
> su tutto il fotogramma, quindi tagliarne il 25% ne toglie poca: il proxy è poco sensibile
> esattamente all'effetto che `squash` dovrebbe catturare, che è «l'evento sta nel quarto di scena
> che il ritaglio butta via».
>
> Non c'è modo di fare meglio con i dati disponibili: servirebbero annotazioni **spaziali**, e né
> Sultani né gli split ufficiali ne hanno — sono tutte temporali. Quindi E0f va letto così:
> **risoluzione e span sono testati bene** sul dominio giusto, **il ritaglio resta debolmente
> testato da qualunque proxy costruibile senza contaminare il test set**. Questo è di per sé un
> esito da riportare, e la conseguenza pratica è che sull'asse del ritaglio il ripiego
> (`eval_center`, la catena ufficiale) pesa di più degli altri due.

### Esito — il pilota si ferma alla calibrazione, e la fermata è il risultato

**Chiuso il:** 19 agosto 2026 · **job:** stima 534050/534051 (`fatq`), calibrazioni 534052, 534079,
534110, 534147 (`gpuq`) · **commit:** `f4e8b5e` · **costo:** ~8 GPU-ora, **tutte in calibrazione**:
le otto celle non sono mai partite, e non partiranno.

**La calibrazione non ha trovato un budget che porti la cella di riferimento nella banda
[0,25 – 0,60], e questo chiude il pilota.** Non per un guasto: il compito costruito senza toccare
`Anomaly_Test` non ha abbastanza escursione per ospitare gli effetti da misurare.

| budget | finestre di training | passi (60 epoche) | top-1 |
|---:|---:|---:|---:|
| 1 | 112 | 420 | 0,065 |
| 2 | 224 | 840 | 0,101 |
| 4 | 448 | 1680 | **0,155** |
| 8 | 896 | 3360 | 0,149 |
| 12 | 1344 | 5040 | **0,155** |

Il caso vale 7,1% (14 vie), quindi il segnale c'è ed è **2,2× il caso** — il proxy funziona, non è
rumore. Ma la curva **si appiattisce a ~15,5%**: triplicando i dati da 4 a 12 video per classe non
guadagna nulla. Il tetto non è la quantità di dati, è il **rumore di etichetta**: i video di
training portano solo un'etichetta di video, la mediana dura 262 s e l'evento ne occupa pochi, e i
dati aggiunti portano lo stesso rumore in proporzione identica.

#### Il conto che rende la fermata una decisione e non una rinuncia

| | E0e-bis (UCF101) | E0f (UCF-Crime) |
|---|---:|---:|
| caso | 10,0% | 7,1% |
| raggiunto | 82,3% | 15,5% |
| **escursione utile** | **72,3 pp** | **8,3 pp** |

Gli effetti misurati da E0e-bis valgono 0,96 / 1,73 / 2,35 pp, cioè **1,3–3,2% dell'escursione
utile**. Con lo stesso peso relativo su E0f varrebbero **0,11–0,27 pp**. E la metrica di E0f si
muove a passi di **un video su 168 = 0,60 pp**.

**L'effetto atteso vale quindi fra 0,19 e 0,46 video su 168: meno di un video.** La metrica non ha
la risoluzione per rappresentarlo. Far girare le otto celle costerebbe ~15 GPU-ora e 1,7 TB per una
misura che *si può dimostrare in anticipo* incapace di risolvere ciò che deve misurare — che è
esattamente il fallimento della prima passata di E0e, riconosciuto stavolta prima di pagarlo
invece che dopo.

#### Cosa resta vero, e cosa questo cambia

1. **La geometria resta quella di E0e-bis: `s256-eval_center-fps7p5`**, con la sua base invariata —
   ripiego dichiarato, effetti limitati a ≤1,4 pp sulla risoluzione, costo 2,25× dei token come
   argomento decisivo. E0f non la conferma e non la smentisce: **non la misura**.
2. **La conferma sul dominio vero non è rimandata: è impossibile con i dati di oggi.** Non è una
   questione di budget di calcolo. Servirebbero annotazioni temporali sui video di *training*, e le
   uniche esistenti (Sultani) coprono solo il test set — usarle per scegliere il preprocessing
   contaminerebbe ogni numero che la tesi riporterà su quel test set. **Le annotazioni
   FS-UCF-Crime del laboratorio contengono intervalli anche sui video di training** (`CLAUDE.md`
   §7): se e quando arrivano, E0f diventa eseguibile *senza cambiare una riga* — lo split si
   ricostruisce con etichette per finestra invece che per video, e il rumore che chiude il pilota
   sparisce. È il primo uso concreto di quel file, e vale la pena dirlo nella mail al relatore.
3. **Le leve che sembravano restare, non restano.** Aumentare i video per classe: la curva è piatta
   su un fattore 3. Aumentare `finestre_per_video`: aggiunge finestre con la stessa proporzione di
   rumore, quindi agisce sullo stesso asse già saturo. Ridurre le classi per alzare l'escursione:
   si sceglierebbero le classi che fanno funzionare l'esperimento, che è la stessa colpa che il
   criterio dichiarato esiste per impedire.

#### Cosa è servito comunque

- **Il vincolo sul test set ha retto una tentazione concreta.** Gli split ufficiali di *action
  recognition* di UCF-Crime erano il proxy ovvio, con etichette pronte: **intersecano
  `Anomaly_Test` in 171 video su 700**, e `costruisci_split` li avrebbe rifiutati comunque perché
  solleva invece di filtrare in silenzio.
- **Quattro difetti trovati nell'infrastruttura**, tutti nel travaso della configurazione di E0e su
  un dataset diverso, tutti corretti con un test ciascuno: `calibra` non univa gli shard; un indice
  assente si dichiarava sano (`hit_rate = 1.0`); il budget non era verificato contro lo split; e la
  calibrazione variava dati ed epoche insieme. **L'ultimo è il più istruttivo**: la prima curva
  scendeva dopo il budget 4, e senza le epoche costanti l'avrei letta come «il proxy non regge» —
  la conclusione giusta per la ragione sbagliata.
- **`confronti_appaiati` e `scegli` hanno finalmente dei test.** Il criterio che sceglie la
  geometria d'ingresso di tutta la matrice era coperto solo da simulazioni lanciate a mano.

## E0f v2 — La stessa domanda, con etichette vere *(geometria congelata)*

**Aperto il:** 19 agosto 2026 · **run_id:** `e0f_ucfcrime_vjepa21l_frozen-attentive_kna_s2` ·
**stato:** **celle complete e geometria congelata al ripiego dichiarato (25 agosto); ratifica della lettura primaria del criterio ai supervisori (Q26)** · **codice:** `evd/exp/e0f.py`, `scripts/e0f_split.py`,
`scripts/slurm/e0f.sbatch` — il fattoriale resta quello di `evd/exp/e0e.py` ·
**commit:** `39f612d`, `bc576c7`

> Quarta voce scritta **prima** dei risultati sulla stessa domanda, e questa volta con
> un'aggravante: il pilota precedente si è chiuso con una fermata motivata, quindi la tentazione
> di leggere il nuovo come «avevo ragione a insistere» è concreta. Il criterio sta qui sotto,
> scritto prima che la prima cella parta.

### Perché si riapre — e non è un ripensamento

La v1 si è fermata su un conto esatto: escursione utile 8,3 pp, effetto atteso **0,19–0,46 video
su 168**, sotto la risoluzione della metrica. Quel conto è ancora giusto, e la sua causa era
dichiarata nella voce stessa: *«diventerà misurabile senza cambiare una riga quando arriveranno
le annotazioni FS-UCF-Crime, che hanno intervalli anche sui video di training»*.

Sono arrivate lo stesso giorno. Il pavimento era il **rumore di etichetta** — la mediana dei video
di training dura 262 s e l'evento ne occupa pochi, quindi quasi ogni finestra portava un'etichetta
che il suo contenuto non giustificava — e le annotazioni lo tolgono alla radice, non lo attenuano.

| | v1 | v2 |
|---|---|---|
| etichetta | del **video** | della **finestra** (regola dell'ultimo frame) |
| unità di valutazione | 168 video | **938 finestre**, 67 per classe |
| risoluzione della metrica | 0,60 pp (un video) | **0,107 pp** |
| celle | 8 | **4** (lo span esce dal gioco) |
| semi | 5 | **7** |
| soglia del test dei segni | 16/20 (p = 0,0059) | **12/14 (p = 0,0065)** |

### Il compito, e le tre scelte che lo definiscono

Classificazione a **14 vie per finestra** (13 anomalie + `Normal`), caso 7,1%.

1. **Positiva di classe c** = la finestra **termina** dentro un intervallo annotato `c`. È la
   regola dell'ultimo frame di `docs/04` §4.1, cioè la stessa con cui il sistema verrà valutato.
   La classe si legge **dall'intervallo, mai dalla cartella**: 216 intervalli su 1831 (11,8%)
   discordano, e su `Explosion`/`Shooting` la discordanza arriva al 20%.
2. **Normale** = l'intero **span** dista da ogni intervallo più di un margine di guardia di 1 s.
   Ciò che sta in mezzo — finestre che finiscono fuori dall'evento ma lo contengono ancora —
   **non riceve etichetta e non entra**: chiamarle normali sarebbe falso, chiamarle evento
   contraddirebbe la regola con cui si valuta.
3. Le finestre normali vengono dagli **stessi video di anomalia**, non da video normali separati.
   Costa zero (quelle feature si estraggono comunque) e toglie la **scorciatoia della scena**: se
   i normali venissero da altri video, una testa potrebbe separarli guardando il parcheggio invece
   dell'evento, e le geometrie si ordinerebbero su quanto bene ciascuna riconosce i parcheggi.

### Quattro celle e non otto, e la ragione non è il costo

**L'asse `span` non è in gioco.** Le annotazioni lo decidono con un argomento che il pilota non
può ribaltare (`docs/04` §4.3). **Questi conteggi appartengono alla griglia storica a stride
intero:** il livello chiamato fps 7,5 aveva span 2,03 s e **7** eventi su 1831 finivano prima
che esistesse una finestra piena; quello chiamato fps 4 aveva span 4,03 s e **31** eventi, e gli eventi
interamente contenuti in una sola finestra passano dal 16,7% al 33,3%. È una proprietà di ciò che
il protocollo può **rappresentare**, non di quanto bene l'encoder discrimini: rimisurarla
costerebbe il doppio e non potrebbe cambiare la scelta.

Restano in gioco i due assi su cui il dominio ha un meccanismo vero: la **risoluzione** (gli
eventi di sorveglianza sono piccoli e lontani) e il **ritaglio** (la vista centrale scarta il 25%
della larghezza, e in sorveglianza quel quarto può contenere l'evento).

`aggrega` registra lo span come **«fissato a priori»**, che è un esito diverso da «non conclude»:
scrivere che il pilota lo ha provato e non ce l'ha fatta sarebbe falso, e finirebbe così in tesi.

### Sette semi, e questa invece è statistica

Con lo span fuori gioco i contrasti per asse scendono da 4 a 2. A 5 semi sarebbero 10 confronti, e
su 10 la soglia equivalente a 16/20 sarebbe **10/10**, cioè l'unanimità: un criterio che nessun
rumore perdona. A 7 semi i confronti sono 14 e la soglia è **12/14 → p = 0,0065**, la stessa
asticella di E0e-bis. Le probe girano su feature già in cache: due semi in più costano minuti.

### Il criterio, dichiarato adesso

- Un asse **conclude** se un livello vince **≥ 12 dei 14** confronti appaiati.
- Un asse che non conclude prende il **ripiego dichiarato**: `256`, `eval_center`, `fps 7,5` — la
  configurazione oggi in vigore. È questo che li rende ripieghi e non scelte.
- La **banda di calibrazione è [0,35 – 0,80]**. Nella v1 era [0,25 – 0,60] perché il rischio era
  il pavimento; adesso le etichette sono pulite e il rischio torna a essere il **soffitto**. Se
  nessun budget della griglia `{1,2,4,8,12,16}` porta la cella di riferimento nella banda, si
  allarga la **griglia**, non la banda: la banda è il requisito di potere.

### Lo split, verificato prima di chiedere GPU

312 video (13 classi × 24: 16 di training e 8 di validazione), **zero** intersezione con
`Anomaly_Test`, 14 classi nella mappa delle etichette, **67 finestre di valutazione per classe**.
Il budget massimo del **pilota** è `video_per_classe_train = 16`: è stato verificato contando i
video che offrono almeno 3 finestre della loro classe, non assunto. Non è il `k` del protocollo
few-shot, che conta segmenti annotati e ha una griglia separata fino a 32.

**Perché la stessa mappa vale per tutte e quattro le celle.** Nel pilota già avviato tutti i video
selezionati hanno prodotto il profilo uniforme storico `s4`, verificato sulla mappa. L'identità
corrente generalizza quel formato: `s{stride}` per indici uniformi, `i{hash}` per passi nativi non
uniformi. In entrambi i casi non contiene risoluzione o ritaglio; a parità di video e griglia
temporale le celle vedono **le stesse finestre con le stesse etichette**.

### Costo, e perché le celle si cancellano dietro

Stima (job 534279, `fatq`, senza GPU): **51 445 finestre per cella**, 216 GB a 256 px e **485 GB a
384**. Quattro celle tenute insieme fanno 1,4 TB contro gli ~1,3 liberi di `/shared` — il tetto
vero è ~2,3 TB e `df` non lo dice. Quindi `--rimuovi-store`: il JSON della cella, con la top-1 di
tutti e sette i semi, si scrive **prima** della cancellazione ed è l'unica cosa che `aggrega`
legge. Per liberare spazio sono stati rimossi gli store di E0e, E0e-bis, E0c e E0f v1 (991 GB),
tutti di esperimenti chiusi e documentati qui.

### Cosa questo pilota **non** dice

- Non dice quale geometria dia la miglior **AUC few-shot**: la probe è congelata, il compito è a
  14 vie per finestra su uno split nostro, e il numero di tesi verrà dal test set ufficiale.
- Non dice niente su `three_views`, che resta fuori dal pilota per contratto del feature store.
- Non rimisura lo span, e la voce lo dichiara invece di lasciarlo dedurre da una tabella con una
  colonna in meno.

### La calibrazione — la banda non si raggiunge, e si registra così

**Chiusa il:** 20 agosto 2026 · **job:** 534281, 534384, 534399, 534406 (misure),
534481 (selezione) · **commit:** `fa5158e`

Il budget si esprime in **finestre per classe** e non nel rettangolo
`video × finestre`. Il rettangolo esiste per un caso preciso — con span diversi
fra le celle, lo stesso video offre un numero diverso di finestre, e prenderle
tutte farebbe addestrare celle diverse su quantità di dati diverse — ma qui lo
span è fissato e le quattro celle vedono finestre **identiche**: il rettangolo
protegge da una variabile che non varia, e in cambio buttava via il 70% dei dati
(672 finestre usate su 2296 disponibili nella classe più povera).

| budget | finestre di training | passi (60 epoche) | top-1 |
|---:|---:|---:|---:|
| 48 | 672 | 2 520 | 0,2456 |
| 96 | 1 344 | 5 040 | 0,2890 |
| **164** | **2 296** | **8 610** | **0,3158** |

**Nessun budget raggiunge la banda [0,35 – 0,80], e questo è il risultato che va
riportato, non nascosto.** `calibrazione.json` registra `banda_raggiunta: false`
con la motivazione accanto: la banda resta scritta, il suo fallimento pure, e chi
rilegge il file vede entrambi.

La banda era **mal specificata**, non è il pilota a essere rotto: l'avevo tarata
per analogia con la [0,70 – 0,90] di E0e-bis, che descriveva un compito a **10
vie di action recognition** su UCF101. Un compito a **14 vie su finestre causali
di 2 s di sorveglianza**, con encoder congelato, ha un punto di lavoro naturale
molto più basso — e la curva lo mostra: raddoppiando i dati da 672 a 1344
finestre si guadagnano 4,3 pp, e la metà successiva ne vale 2,7.

**Perché si prosegue lo stesso, e su quale conto.** La banda era un *proxy* del
requisito vero: che la metrica abbia spazio perché una differenza compaia in
entrambe le direzioni. Quel requisito è verificabile direttamente:

| | E0e-bis (UCF101) | E0f v1 | **E0f v2** |
|---|---:|---:|---:|
| caso | 10,0% | 7,1% | 7,1% |
| raggiunto | 82,3% | 15,5% | **31,6%** |
| **escursione utile** | **72,3 pp** | **8,3 pp** | **24,4 pp** |
| unità di valutazione | 3 783 clip | 168 video | **938 finestre** |
| risoluzione della metrica | 0,03 pp | 0,60 pp | **0,107 pp** |
| effetto atteso (1,3–3,2% dell'escursione) | 0,96–2,35 pp | 0,11–0,27 pp | **0,32–0,79 pp** |
| **effetto atteso, in unità di misura** | 36–89 clip | **0,19–0,46 video** | **3,0–7,4 finestre** |

Nella v1 l'effetto atteso valeva **meno di un video**: sotto la risoluzione della
metrica, e per questo il pilota si è fermato. Qui vale **3–7 finestre su 938**,
cioè 3–7 volte la risoluzione. La distanza dal pavimento è 24,4 pp e quella dal
soffitto 68 pp: c'è spazio in entrambe le direzioni, che è ciò che la banda
serviva a garantire.

**La regola di ripiego è dichiarata e non guarda l'esperimento.** Si prende il
budget con la top-1 massima, ed è la regola giusta quando la banda non si
raggiunge *dal basso*: lì il rischio è il pavimento e non il soffitto, e più dati
significa meno rumore di addestramento — che è proprio ciò di cui il confronto
appaiato ha bisogno. `scegli_budget` riceve **solo** le misure della cella di
riferimento, mai un confronto fra geometrie: un test lo verifica per
introspezione della firma, così la regola non può diventare una scelta *dentro*
l'esperimento.

> **Il precedente è di E0e, ed è deliberato seguirlo.** «Il criterio dichiarato,
> applicato alla lettera, degenera: si registra così, senza riscriverlo.» La
> banda non è stata allargata — il ripiego è opt-in, spento di default, e va
> chiesto esplicitamente con `E0F_RIPIEGO_FUORI_BANDA=1`.

### Quattro difetti trovati prima delle celle, e tre costati ore

I primi tre sono emersi **sul cluster**, che è il modo sbagliato di trovarli:

1. `calibra` leggeva il numero di classi da una chiave del rapporto di
   `verifica_budget`, che con l'unità nuova non esiste (job 534384, 7 s).
2. La guardia del rettangolo scattava anche sul budget in finestre: `subset_seed`
   valorizzato con gli altri due a `None` sembrava una configurazione incompleta
   (job 534399, 11 min).
3. La probe rileggeva le stesse feature da NFS **a ogni epoca**: 1,1 passi al
   secondo, 4 ore per la calibrazione e ~14 ore stimate per cella, cioè il doppio
   del limite per job. Il nodo assegna 2 TB di RAM e il job ne usava 1,2 GB.

Il quarto è stato trovato **in locale**, dai test scritti dopo i primi tre, ed è
il più pericoloso perché non avrebbe rotto niente: **le celle si sarebbero
addestrate per 4 epoche invece di 60.** `calibra` sceglie il budget misurando a
`epoche_calibrazione` epoche, `run_arm` usava un `max_steps` a parte; il pilota
avrebbe misurato geometrie sotto-addestrate credendo di misurarle nel punto
calibrato. Ora la formula è una funzione sola (`passi_per_epoche`) chiamata da
entrambi.

Da qui la rete di test che non c'era: `calibra`, `run_arm`, `aggrega` e `run_e0c`
girano tutti in locale su store sintetico, senza GPU, in due secondi.

### Il piano si riduce alla riga 256, e il criterio ridotto si dichiara adesso

**Dichiarato il:** 20 agosto 2026, 11:20 — **prima** che una qualunque delle due
celle abbia i suoi sette semi, e prima di ogni aggregazione. È questo che rende
la riduzione un criterio e non una scelta a posteriori.

**Cosa ha rotto il piano.** La cella 0 (`s256-squash-fps7p5`) è uscita in
**TIMEOUT** alle 11:04:30 dopo 6h59, con **3 semi su 7** salvati. Il fatto che
non era previsto: **`-R 3` non rientra in coda sul Wall Time.** Il requeue di
SLURM copre il fallimento del nodo e la prelazione, non il tempo scaduto — il
job viene marcato `TIMEOUT` e l'array passa al task successivo. Il ripristino
per seme funziona ([e0e.py:535](../src/evd/exp/e0e.py#L535)) e i tre semi non si
ripagano, ma il rilancio è esplicito, non automatico.

**Il costo residuo, misurato e non stimato.** Un seme a 256 costa **1h30**
(8610 passi, 0,63 s/passo); la preparazione di una cella nuova ne costa 2h17.
Le celle a 384 hanno **4608 token per finestra contro 2048**, cioè ~3× a seme:

| cella | semi mancanti | job da 7 h |
|---|---:|---:|
| 0 · `s256-squash-fps7p5` | 4 | 1 |
| 1 · `s256-eval_center-fps7p5` | ~3 | 2 |
| 2 · `s384-squash-fps7p5` | 7 + estrazione | ~5 |
| 3 · `s384-eval_center-fps7p5` | 7 + estrazione | ~5 |

≈ **78 ore di A100**, di cui due terzi sulla riga 384, e una settimana di coda
prima che E0d possa partire. **Decisione di Antonio, 20 agosto:** si chiude la
riga 256 e si rinuncia alla riga 384. I task 2 e 3 sono stati cancellati.

**Il criterio ridotto, per l'asse che resta.** Con una sola risoluzione in gioco
i contrasti dell'asse *ritaglio* scendono da 2 a 1, e i confronti appaiati da 14
a **7**. La soglia dichiarata è **7 su 7**:

| | confronti | soglia | p sotto H₀ |
|---|---:|---:|---:|
| E0e-bis | 20 | 16 | 0,0059 |
| E0f v2, piano pieno | 14 | 12 | 0,0065 |
| **E0f v2, piano ridotto** | **7** | **7** | **0,0078** |

Va detto per intero: 7/7 è **leggermente più permissiva** di 12/14 (0,0078
contro 0,0065), non equivalente. È l'unica soglia disponibile su 7 confronti che
resti nello stesso ordine di grandezza — la successiva, 6/7, vale 0,0625 e non è
un criterio.

**L'asse risoluzione è «chiuso senza misura», ed è un terzo esito.** Non è
«**non conclude**» (il pilota ha misurato e nessun livello ha vinto) e non è
«**fissato a priori**» (l'asse non è mai stato in gioco, come lo span, che ha
contro di sé un argomento di dominio che il pilota non può ribaltare). La
risoluzione *era* in gioco, con un meccanismo vero — gli eventi di sorveglianza
sono piccoli e lontani — ed è stata tolta **per costo**. Scriverla in uno degli
altri due modi sarebbe un campo di config che mente, che è il primo dei tre
errori di `CLAUDE.md` §5bis.7.

Il livello adottato è **256**, che è il ripiego già dichiarato prima di vedere
qualunque numero: prenderlo adesso non è una scelta a posteriori. Ma la frase
che va in tesi è **«256 perché il pilota non ha potuto misurarlo e il ripiego
dichiarato era 256»**, non «il pilota ha scelto 256». La differenza non è
formale: la seconda rivendica una prova che non esiste.

### Il piano pieno viene ripristinato, e anche questo si dichiara prima dei numeri

**Dichiarato il:** 20 agosto 2026, 12:05 — **prima** che una qualunque cella
avesse i suoi sette semi, cioè nella stessa finestra in cui è stata dichiarata la
riduzione. La sezione precedente **resta dov'è**: il registro deve mostrare che
il piano è stato ridotto e ripristinato nel giro di un'ora, non che 12/14 non è
mai stato in discussione.

**Perché si torna indietro.** La riduzione toglieva l'asse **risoluzione**, che
è il più difendibile dei due per una tesi di sorveglianza: gli eventi sono
piccoli e lontani, e 384 px danno **4608 token per finestra contro 2048**, cioè
più dettaglio spaziale esattamente dove il dominio lo chiede. Il ritaglio è una
domanda più stretta. Rinunciare a 384 significava non misurare l'ipotesi che più
plausibilmente conta, e la ragione era solo il prezzo.

**Perché il momento è pulito, e perché non lo sarebbe stato domani.** La
decisione è presa a **zero celle concluse**: nessun `celle/*.json` esiste, e i
tre semi della cella 0 e i primi della cella 1 non sono mai stati aggregati né
confrontati. Allargare il pilota **dopo** aver visto uscire la riga 256 sarebbe
stata una scelta informata dai risultati — un esperimento che si estende quando
il primo esito non piace è precisamente ciò che un criterio dichiarato in
anticipo serve a impedire. Deciderlo adesso non lo è, e questa riga esiste per
poterlo dimostrare con una data.

**Cosa torna in vigore:**

- Il criterio è di nuovo quello originale: **2 contrasti × 7 repliche = 14
  confronti appaiati per asse, soglia 12/14 → p = 0,0065.** La soglia 7/7 della
  sezione precedente non si applica a nessun numero.
- L'asse **risoluzione** torna misurato, e nessun asse è «chiuso senza misura».
  Lo span resta «fissato a priori», che non è cambiato.
- I ripieghi restano quelli dichiarati: `256`, `eval_center`, `fps 7,5`.

**Cosa resta valido di ciò che è già stato pagato.** Le celle sono indipendenti
e i semi sono appaiati per costruzione — lo stesso seme pilota il
sottocampionamento *e* l'inizializzazione in tutte e quattro le celle — quindi i
tre semi della cella 0 e quelli della cella 1 valgono esattamente come prima. Il
codice della probe è **identico**: `git diff` fra il commit che ha girato i semi
1–3 (`03bdf5e`) e HEAD è vuoto su `src/evd/exp/e0c.py`, sui modelli e su
`src/evd/data/windows.py`.

**Il codice del terzo esito resta, e non è un'astrazione orfana.** «Chiuso senza
misura» era stato scritto per un caso che è durato un'ora, ma quel caso non è
ipotetico: se le celle a 384 non dovessero rientrare — e il rischio è reale,
vedi sotto — il piano ricade sulla riduzione, e allora serve. Resta con i suoi
test e con la sezione qui sopra che dice per cosa è nato.

**Il costo, e il rischio che va detto adesso.** Ogni cella a 384 vuole la sua
estrazione (**485 GB**, stimati ~5 h) e sette semi che costano ~2,25× quelli a
256 per il solo conteggio dei token. Se un seme a 384 supera le **3 h 15**, in
un Wall Time da 6h59 ce ne sta **uno solo per job**, e una cella diventa ~8 job
invece di ~5. Il primo seme a 384 è la misura che decide, e il piano di
sottomissione si fa su quel numero invece che su questa moltiplicazione.
Lo spazio regge: le celle si cancellano dietro, quindi a riga 256 chiusa restano
~0,59 TB occupati e una cella a 384 ne aggiunge 0,485, su un tetto di ~2,3 TB.

### Le celle 0–2 rientrano, e il test set torna a 938 — derivando lo split, non rigenerandolo

**Dichiarato il:** 23 agosto 2026, ~12:30, mentre la cella 3 era in estrazione
(job `536023`). Nessun numero sotto è un esito inferenziale: il gate statistico
(§4 dell'handoff) resta aperto in attesa dei supervisori.

**La cella 2 (`s384-squash-fps7p5`) è chiusa.** La catena autonoma
`534105→535200→535227` ha concluso i sette semi alle 11:21 (s7 ripagato da capo,
come previsto senza checkpointing intra-seme), ha scritto `celle/s384-squash-fps7p5.json`
e rimosso lo store (~510 GB); l'anello di riserva è uscito in 5 s per idempotenza.

**La rigenerazione dello split bilanciato NON riproduceva lo split del 19 agosto,
e questo è un fatto da registrare.** Il job `536018` (fatq, 6 s) ha prodotto un
insieme a quota 67 uniforme, ma con **cinque video di valutazione diversi**
(Abuse044, Arrest044, Arson045, Explosion045, RoadAccidents103 entrano; Arson052,
Explosion005, RoadAccidents110, Robbery005, Shooting029 escono): il pool
candidati di `RoadAccidents` è passato **89→90** perché l'evoluzione di
`src/evd/data/windows.py` (F6) sposta i conteggi per video oltre la soglia di
ammissibilità, e l'estrazione a cascata cambia i video di cinque classi. Su un
filesystem e un RNG condivisi non c'è alcun bug: è il codice che è cambiato
sotto uno split già consumato. Conseguenza pratica: le predizioni delle celle
0–2 coprono il vecchio test set, quindi **83 unità del test rigenerato non
avrebbero predizioni**, e il ricalcolo a parità di insieme sarebbe stato
impossibile senza rieseguire tre celle (~1,1 TB di estrazioni).

**Decisione: lo split bilanciato si deriva dall'artefatto, non si rigenera.**
Stessi 312 video, stesse finestre per le 13 classi anomale, sole le finestre
`Normal` del lato val ridotte **98→67** con `bilancia_per_classe` (la stessa
funzione del generatore, importata da `evd.exp.e0f`). Tutte le guardie passano
by construction: 938 = 14 × 67; sottoinsieme esatto del 969; 31 rimosse tutte
`Normal`; etichette coerenti sugli ID comuni. La dir `$EVD_DATA_DIR/e0f_split_balanced`
contiene un `split.json` di provenienza che dichiara derivazione, motivo e SHA
dello split originale; SHA-256 della mappa derivata:
`dd4e09f16111e4916ab5f258543ecf1cc626658ebb866af9d0a00cd38efda3ad`. Il primo
tentativo di rigenerazione (job `536018`, mai consumato da nessun run) è stato
sostituito. **Lezione generale**: uno split consumato da run già pagati non si
rigenera col codice corrente «per comodità» — o si deriva dall'artefatto, o si
dichiara la riesecuzione.

**Dove vive la mappa, dal 25 agosto.** Fino a quel giorno l'unica copia locale
stava nel `tmp` di una sessione ormai chiusa: una cartella usa-e-getta, per
l'artefatto da cui dipende la geometria di tutta la matrice. Ora sono tre, con la
stessa SHA-256 `dd4e09f1…` verificata su ciascuna e contro quella del cluster:
`$EVD_DATA_DIR/e0f_split_balanced/` sul cluster e in locale — è il percorso che
`E0F_SPLIT_DIR` passa ai job — più una copia d'archivio in
`runs/<run_id>/split_balanced/`, che viaggia con gli artefatti del run e sta
sotto OneDrive. La mappa completa ha 31 114 chiavi (training e valutazione
insieme); le 938 sono il lato di valutazione dopo il bilanciamento.

**Ricalcolo delle celle sulla mappa a 938** (`confronta_predizioni.py
--window-labels --unita-attese 938`; artefatti in `runs/…/ricalcolo-938/`,
valori originali su 969 conservati come provenienza; per la cella 3 non c'è
valore originale — è nata direttamente sulla mappa derivata):

| cella | top-1 media su 969 | top-1 media su **938** | escursione fra semi |
|---|---|---|---|
| `s256-squash-fps7p5` | 0,2900 | **0,2779** | 4,8 pp |
| `s256-eval_center-fps7p5` | 0,2962 | **0,2845** | 4,5 pp |
| `s384-squash-fps7p5` | 0,3114 | **0,3000** | 7,5 pp |
| `s384-eval_center-fps7p5` | — (nata su 938) | **0,2865** | **11,2 pp** |

**L'ordine cambia quando entra la quarta cella**: `s384-squash` resta prima,
ma `s384-eval_center` scende terza dietro a `s256-eval_center`, e porta
l'escursione fra semi peggiore del fattoriale (11,2 pp). La lettura vera resta
quella appaiata per seme con il criterio approvato — questi sono numeri
descrittivi e il gate statistico è aperto.

**Cella 3 (`s384-eval_center-fps7p5`)**: sottomessa il 23 agosto alle ~12:08
(job array `536023`, `gpuq`, 06:59:00) con `E0F_SPLIT_DIR=$EVD_DATA_DIR/e0f_split_balanced`,
budget 164 finestre/classe e il **checkpointing intra-seme attivo**
(`E0F_CHECKPOINT_EVERY=1000`, `--signal=B:USR1@300`). La catena autonoma ha
usato **sei anelli `afterany`**, ognuno sottomesso dopo la verifica via `sacct`
del precedente, con ID mai indovinati:

| anello | job | uscita |
|---|---|---|
| 1 | `536023` | `FAILED 2:0` (uscita graziosa USR1), 19:02 del 23/8 — seme 1 ripreso e chiuso dall'anello 2 |
| 2 | `536217` | `FAILED 2:0`, 02:00 del 24/8 — semi 1–2 |
| 3 | `536229` | `FAILED 2:0`, 08:54 del 24/8 — semi 3–4 |
| 4 | `536347` | `FAILED 2:0`, 15:47 del 24/8 — semi 4–5 |
| 5 | `536860` | `FAILED 2:0`, 22:46 del 24/8 — semi 5–6 |
| 6 | `537152` | **`COMPLETED` alle 00:04 del 25/8** — seme 7, nessun TIMEOUT residuo |

Tutti i sette JSON di seme dichiarano `window_labels =
…/e0f_split_balanced/pilota-window-labels.json`; il ricalcolo con
`confronta_predizioni.py` riproduce i top-1 del JSON di cella esattamente.
Lo store `5f5bfc187ea4` si è auto-rimosso alla fine dell'ultimo anello.

### Esito

**Geometria congelata al ripiego pre-dichiarato: `256 / eval_center / fps 7,5`.**
Il gate statistico aperto il 22/8 è stato chiuso il 25 agosto con decisione di
Antonio: entrambe le letture del criterio sono registrate come *sensitivity
analysis* nell'artefatto `ricalcolo-938/aggrega-geometria-938.json` (con
provenienza completa), e la ratifica della lettura da usare come primaria —
proposta: l'aggregazione entro seme — passa ai supervisori come **Q26** di
`docs/12` v1.3. La decisione di geometria non dipende dalla ratifica:

| asse | lettura A — test dei segni 12/14 (pre-registrata) | lettura B — unanimità entro seme 7/7 (proposta) |
|---|---|---|
| risoluzione | `384` vince **10/14** → non conclude (serviva 12) | unanimità pro-`384` in **4/7** semi → non conclude |
| ritaglio | **7–7** fra i livelli → non conclude | unanimità fra abbinamenti: `eval_center` in 2 semi, `squash` in 2, misti 3 → **non conclude** |

Sotto alcuna delle due letture un livello raggiunge la soglia, quindi si applica
il ripiego dichiarato prima dei run: `256 / eval_center / fps 7,5`, la stessa
configurazione in vigore da E0e-bis. La verifica sul dominio vero non produce
prove per rovesciarla — ed è il ripiego più economico (216 GB contro 485 GB di
estrazione a parità di split).

**E sotto una terza lettura, che nessuno aveva chiesto.** La lettura B ha un
difetto da dire ai supervisori insieme alla proposta: «conclusione solo con 7/7»
non può concludere appena *un* seme è discorde, quindi il suo «non conclude» è
garantito per costruzione e non misura l'effetto — con 2 e 3 semi misti
osservati, non poteva uscire altro. L'aggregazione appaiata standard — media
della differenza dentro ogni seme, poi test dei segni sui sette semi — non ha
quel difetto, e dice la stessa cosa: risoluzione **5–2** per 384 (p = 0,2266),
ritaglio **3–4** (p = 0,50). Il ripiego regge sotto tutte e tre le letture, ed è
un argomento molto più solido di due letture di cui una cieca. La terza sta in
`evd.exp.e0e.media_entro_seme`, si ottiene con `--con-controllo-robustezza` e
**non è il criterio del pilota**: è materiale per Q26.

**L'aggregazione è rigenerabile, ed è stata rigenerata.**
`scripts/aggrega_geometria.py` compone le letture a partire dai soli file di
`ricalcolo-938/`; con `--verifica` rilegge l'artefatto in archivio e conferma che
coincide su letture, esito e valori per seme (provenienza esclusa: porta percorsi
assoluti e la data di generazione). Le tre letture camminano tutte sulle stesse
coppie, enumerate una volta sola in `_abbinamenti`, e il golden test delle top-1
pubblicate sta in `tests/unit/test_e0e.py`: un refactoring che sposta quei numeri
ha cambiato una conclusione di tesi, e adesso lo scopre la suite.

    python scripts/aggrega_geometria.py runs/<run_id>/ricalcolo-938 \
        --verifica runs/<run_id>/ricalcolo-938/aggrega-geometria-938.json

Quattro celle, sette semi ciascuna, tutte valutate sulle stesse **938 finestre
bilanciate** (67 per classe) derivate dall'artefatto del 19 agosto:

| cella | media top-1 su 938 | escursione fra semi |
|---|---|---|
| `s384-squash-fps7p5` | **0,3000** | 7,5 pp |
| `s384-eval_center-fps7p5` | 0,2865 | 11,2 pp |
| `s256-eval_center-fps7p5` | 0,2845 | 4,5 pp |
| `s256-squash-fps7p5` | 0,2779 | 4,8 pp |

Cosa dicono, in chiave descrittiva: **(a)** a 384 il ritaglio *squash* batte
l'*eval_center* e soprattutto lo fa con metà dell'escursione fra semi — la
cella peggior stabile è anche quella con la geometria più costosa; **(b)** le
due celle a 256 stanno entro mezzo punto l'una dall'altra; **(c)** la
differenza fra la prima e l'ultima cella è 2,2 punti, dello stesso ordine
dell'escursione interna alle celle a 384 — nessuna scelta di geometria aveva
prove statistiche per imporsi, ed è esattamente ciò che il ripiego dichiarato
governa. Dettaglio utile per la tesi: l'unico segnale direzionale (384 sopra
256, 10/14) vive quasi tutto nell'abbinamento *squash* (+2,21 pp di media) e
sparisce con *eval_center* (+0,20 pp) — interazione, non effetto principale.

## E0e-bis — La stessa domanda, con un disegno che può rispondere *(chiuso)*

**Aperto il:** 18 agosto 2026 · **run_id:** `e0ebis_ucf101mini_vjepa21l_frozen-attentive_kna_s1` ·
**stato:** **chiuso** il 19 agosto · **commit:** `59da74b`

> Anche questa voce è scritta **prima** dei risultati. Vale a maggior ragione qui: è la seconda
> passata sulla stessa domanda, e un criterio riscritto dopo aver visto due volte i numeri non
> varrebbe niente.

### Perché si rifà, e cosa cambia

La prima passata non ha concluso, e le tre cause stanno sopra. Questa versione ne corregge una
ciascuna. Nessuna delle tre è «più calcolo»: il piano di §«Cosa servirebbe» stimava ~45 GPU-ora,
questa replica ne costa **~9**, perché la correzione vera è di disegno.

| causa del fallimento | correzione | perché funziona |
|---|---|---|
| compito **saturo** (~99,7%) | budget di dati della probe ridotto, scelto da `calibra` | gli **otto store esistono già** (382 GB): l'estrazione costa zero e si tolgono dati, non se ne comprano |
| celle **non appaiate** sul support | `subset_seed = replica`, identico in tutte le celle | fra due celle resta solo la geometria: stessi video, stesso numero di esempi, stessi video di test |
| statistica **scalare** senza potere | **test dei segni** su 4 contrasti × 5 repliche = 20 confronti | con `n = 2` semi il pavimento era una statistica d'ordine estrema; il segno non ha quel difetto |

### Il budget di dati non lo scelgo io

`train_video_per_classe` lo fissa un passo di calibrazione che gira sulla **sola cella di
riferimento** — `s256-eval_center-fps7p5`, la più vicina alla ricetta ufficiale — e cerca la top-1
dentro una banda dichiarata prima, **[0,70 – 0,90]**; fra i budget in banda si prende il più
vicino al centro, cioè quello che lascia più spazio a una differenza in **entrambe** le direzioni.
Calibrare su una cella sola, e prima che le altre girino, è ciò che tiene la scelta del budget
**fuori** dal confronto: un budget scelto guardando le otto celle sarebbe una scelta dentro
l'esperimento. Se nessun budget cade in banda il passo **fallisce** invece di allargare la banda —
la banda è il requisito di potere, non un parametro da accomodare.

### La calibrazione, e cosa dice da sola

| video/classe | finestre di training | top-1 | in banda [0,70–0,90] |
|---:|---:|---:|:--|
| 1 | 20 | 0,469 | no |
| 2 | 40 | 0,582 | no |
| **4** | **80** | **0,818** | **sì — scelto** (più vicino al centro) |
| 8 | 160 | 0,876 | sì |
| 16 | 320 | 0,982 | no |
| *(prima passata)* | *8129* | *0,997* | *no* |

La curva spiega da sola perché la prima passata non poteva concludere: già a **16 video su 101** il
compito è saturo. Con tutti i video si lavorava su un tratto piatto, dove nessuna differenza di
feature può tradursi in differenza di accuratezza. Non era un problema di semi insufficienti — i
semi lo hanno solo reso invisibile.

### Il criterio, dichiarato adesso

1. Per ogni asse ci sono **4 contrasti** (le combinazioni degli altri due assi) × **5 repliche** =
   **20 confronti appaiati**. Ogni confronto è appaiato tre volte: stessa replica (quindi stesso
   support e stessa inizializzazione), stessi video di test, altri due assi fissati.
2. Un asse **conclude** se un livello vince **almeno 16 dei 20** confronti non pari. Sotto
   l'ipotesi nulla di nessuna differenza, P(X ≥ 16 | n = 20, p = 0,5) ≈ **0,006**.
3. Ogni asse che **non** conclude prende il suo **ripiego, dichiarato qui**: **256** (meno token),
   **`eval_center`** (è la catena ufficiale; lo `squash` è nostro) e **fps 7,5** (è la geometria
   dei config di eval). Sono ripieghi verso il riferimento e verso il costo minore. Fissarli adesso
   è l'unica cosa che li rende ripieghi e non scelte.

La configurazione finale è la combinazione dei tre esiti, asse per asse. **Un asse può concludere
mentre gli altri no**, e in quel caso si riporta esattamente così: la parte decisa dai dati e la
parte decisa dal ripiego, distinte.

### La verifica che il criterio funzioni, fatta prima di spendere la GPU

Il criterio è stato passato su due simulazioni locali. Con un segnale piantato su un asse:
`conclusi: ['risoluzione']`, 20/20 confronti a favore di 256. Con **rumore puro**:
`conclusi: []`, e si applicano i tre ripieghi. Il secondo caso è quello che conta — significa che
se questa passata non concludesse, saprei distinguerlo da «non ho potuto misurare», che è
esattamente ciò che mancava alla prima.

### Risultati — nessun asse conclude, ma stavolta si sa di quanto

**Chiuso il:** 19 agosto 2026 · **job:** calibrazione 533995, celle 534005 (array 0–7, `gpuq`),
aggregazione 534034 (`fatq`) · **commit:** `59da74b` · **costo:** 4 celle a 256 × ~0,52 h +
4 celle a 384 × ~1,07 h = **6,4 GPU-ora** su A100, zero estrazione (gli otto store della prima
passata sono stati riusati, identità di preprocessing invariata verificata nei log)

Le celle a 384 hanno impiegato **esattamente il doppio** di quelle a 256 (1h04 contro 0h31), che è
il rapporto dei token (4608 contro 2048). È la proporzione che il TIMEOUT della prima passata aveva
già insegnato, applicata prima invece che dopo.

| cella | top-1 media | min | max | escursione | token |
|---|---:|---:|---:|---:|---:|
| **`s256-eval_center-fps7p5`** | **0,8233** | 0,7418 | 0,8982 | 15,64 pp | 2048 |
| `s256-eval_center-fps4` | 0,8211 | 0,7855 | 0,8436 | 5,82 pp | 2048 |
| `s384-squash-fps7p5` | 0,8182 | 0,7600 | 0,8836 | 12,36 pp | 4608 |
| `s384-eval_center-fps7p5` | 0,8138 | 0,7273 | 0,8945 | 16,73 pp | 4608 |
| `s256-squash-fps4` | 0,7985 | 0,7164 | 0,8509 | 13,45 pp | 2048 |
| `s256-squash-fps7p5` | 0,7884 | 0,7636 | 0,8255 | 6,18 pp | 2048 |
| `s384-squash-fps4` | 0,7724 | 0,6982 | 0,8618 | 16,36 pp | 4608 |
| `s384-eval_center-fps4` | 0,7578 | 0,6764 | 0,8545 | 17,82 pp | 4608 |

#### Il criterio dichiarato, applicato: `s256-eval_center-fps7p5` per ripiego su tutti e tre gli assi

| asse | effetto appaiato | IC 95% | ρ appaiamento | test dei segni | esito |
|---|---:|:---:|---:|:---:|---|
| risoluzione (256 − 384) | **+1,73 pp** | [−1,41, +4,86] | 0,34 | 10/19 · p = 1,00 | non conclude → **256** |
| ritaglio (`eval_center` − `squash`) | **+0,96 pp** | [−1,04, +2,96] | 0,73 | 10/19 · p = 1,00 | non conclude → **`eval_center`** |
| span (fps 7,5 − fps 4) | **+2,35 pp** | [−0,17, +4,86] | 0,57 | 13/20 · p = 0,26 | non conclude → **fps 7,5** |

Nessun livello raggiunge le 16 vittorie su 20 richieste, quindi si applicano i tre ripieghi
dichiarati prima della calibrazione. **La configurazione scelta è `s256-eval_center-fps7p5`.**

#### Perché questo esito è diverso da quello della prima passata

La prima passata non concludeva e non sapeva dire perché. Questa lo sa, e produce tre affermazioni
quantitative che prima non esistevano.

1. **L'appaiamento ha rimosso varianza vera, e si misura quanta.** Se le celle fossero indipendenti
   la deviazione standard del delta appaiato sarebbe 8,23 pp (= 5,82 × √2). Osservata: 6,70 · 4,27
   · 5,37, cioè correlazioni di **0,34 · 0,73 · 0,57**. Il pairing funziona; non basta.
2. **Gli effetti sono limitati superiormente.** Non è «non so»: è che 384 **non può** battere 256 di
   più di 1,4 pp, e `squash` non può battere `eval_center` di più di 1,0 pp. Su una risoluzione che
   costa **2,25× i token**, questo basta a decidere per il costo a prescindere dalla significatività.
3. **Tutti e tre i punti stimati pendono verso il ripiego**, e la cella scelta ha anche la media più
   alta. La configurazione non è un default arbitrario: è dove i dati pendono su ogni asse.

#### Una tentazione a cui non si cede, e va scritta

L'intervallo di confidenza sullo span quasi esclude che fps 4 sia migliore, e un test t appaiato
darebbe p ≈ 0,066. **La primaria dichiarata è il test dei segni, e non conclude** (13/20, p = 0,26).
Riportare il t-test come esito perché è quello che dà il numero desiderato sarebbe scegliere la
statistica dopo aver visto i dati — la stessa classe di errore della prima passata, in direzione
opposta. L'IC resta qui come indizio *a sostegno del ripiego già dichiarato*, non come conclusione.
Che i due statistici puntino nella stessa direzione è comunque il caso migliore in cui trovarsi:
la risposta operativa non dipende da quale dei due si creda.

#### L'errore di disegno di questa passata, che è ancora mio

Ho scambiato un soffitto con un pavimento di rumore. Togliere il soffitto ha richiesto di scendere
a 80 finestre di training, e in quel regime l'escursione fra repliche dentro una stessa cella
arriva a 17,8 pp — contro effetti dell'ordine di 1–2,4 pp. La calibrazione mostra che il budget 8
(top-1 0,876) sarebbe stato probabilmente un compromesso migliore fra margine e varianza.

**Non è stato rilanciato con budget 8**, e la ragione è il punto di tutto il disegno: cambiare il
budget dopo aver visto che il criterio non conclude significa cercare il budget che fa concludere.
La curva di calibrazione era stata misurata prima, la scelta fatta con una regola dichiarata
(il più vicino al centro della banda), e quella regola ha scelto 4. Si tiene.

#### La risposta del proxy a Q21 e Q22

**`s256-eval_center-fps7p5`**: 256 px, catena di eval ufficiale a vista centrale, 16 frame a
7,5 fps target. ADR-024 ne fissa lo span canonico a **2,0 s**; il pilota usava ancora la
discretizzazione storica e non va rinominato retroattivamente. Da riportare in tesi **con la sua base**: scelta per ripiego dichiarato, non
per superiorità dimostrata, con gli effetti limitati a ≤ ~5 pp su un compito proxy e il costo
(2,25× token per 384) come argomento decisivo. Restano vere le due cose che il pilota non può
dire: UCF101 non è UCF-Crime, e ordinare geometrie non è predire AUC su anomalie.
Per questo ADR-024 ha chiuso la definizione temporale di Q22, mentre Q21 è stata riaperta e viene
verificata da E0f v2 sul dominio vero.

## E0c — Sanity sul noto *(forma ridotta)*

**Chiuso il:** 14 agosto 2026 · **run_id:** `e0c_ucf101_vjepa21l_frozen-none_kna_s1` ·
**commit:** `0f608ea` (estrazione), `921896e` (probe) · **macchina:** A100 `gpuq` ·
**job:** 532394 (feature), 532402 (merge), 532408 (probe)

### Cosa chiedeva

Una sola cosa, e non è un numero di tesi: **la pipeline è rotta?** `dati → finestre causali →
backbone congelato → feature store → testa attentiva → metrica` gira su un problema di cui si
conosce la risposta, così che un fallimento a valle su UCF-Crime non lasci il dubbio se sia il
metodo o l'infrastruttura.

### Come è stato eseguito

UCF101 split-1, **10 classi** — le prime due in ordine alfabetico per ciascuno dei cinque tipi
ufficiali (decisione di Antonio, 14 agosto): `ApplyEyeMakeup, ApplyLipstick, BabyCrawling,
BlowingCandles, BandMarching, Haircut, Drumming, PlayingCello, Archery, BalanceBeam`. 1363 video
(train 982 / test 381), inventario senza esclusi. Feature con lo **stesso** `estrai_feature.py`
degli altri blocchi: 224, `eval_center`, 7,5 fps, 16 frame, hop 1 s, storage fp16, calcolo AMP
BF16, layer 5/11/17/23 — 33.240 feature, 113 GB, replay hit rate 1,0. Probe attentiva sul layer
23, 2000 step, batch 32, lr 1e-3, logit mediati sulle finestre di ogni video.

Archivio verificato prima dell'uso leggendo **tutti i blocchi** (`hpc/tools/verifica_ucf101.py`):
13.320 clip, 101 classi, CRC validi; digest in `configs/data/ucf101.yaml`.

### Cosa è uscito

| | |
|---|---|
| top-1 (per video, 380 test) | **0,9947** |
| soglia dichiarata / delta | 0,90 ±10pp → **+9,47pp, dentro** |
| finestre train / test | 6192 / 2118 |
| video scartati perché più corti di una finestra | 3 (tutte `BlowingCandles`) |
| feature hit rate | 1,0 |
| loss finale | 4,32e-7 |

### Cosa vuol dire

**La pipeline non è rotta.** Ogni anello ha portato informazione fino in fondo: se il windowing
causale, la chiave di cache, il round-trip fp16 o l'aggregazione per video fossero sbagliati,
99,47% non uscirebbe. È il controllo che dà diritto a leggere i fallimenti futuri come fallimenti
del *metodo*.

**Non è un numero di tesi e non è confrontabile con la letteratura.** Il riferimento 0,90 è una
soglia di sanità dichiarata, **non** un ballpark pubblicato: V-JEPA 2 e 2.1 non valutano UCF101
(arXiv 2603.14482 — verificato il 14 agosto: riportano SSv2 77,7 · K400 87,7 · Diving-48 89,2 ·
IN1K 85,5). E la probe è su 10 classi contro le 101 della letteratura, con finestre causali contro
il protocollo multi-segmento × multi-vista. Invariante 7: mai commensurabili in silenzio.

La loss a 4,3e-7 dice che il budget ha **saturato molto prima** dei 2000 step: la probe ha 49,3 M
parametri contro 6192 finestre di training. Per un controllo di non-rottura è irrilevante; per E2
è un promemoria che il budget della testa va dimensionato, non ereditato da qui.

### Appendice — fp16 contro fp32 end-to-end, e il pavimento di rumore

*(15 agosto 2026, job 532450/532458 estrazione fp32 · 532459/532460/532461 probe)*

La domanda: le due scorciatoie (disco fp16 **e** calcolo BF16) cambiano le **decisioni**, non
solo i tensori? Estratto un secondo store UCF101 in fp32 pieno (226 GB, stesso tutto il resto) e
rieseguita la probe. Ma prima serviva il termine di paragone: l'attenzione memory-efficient non è
deterministica, quindi la probe è stata eseguita **due volte sulle stesse feature fp16**.

| Run | Feature | top-1 | Errori |
|---|---|---|---|
| originale (532408) | bf16/fp16 | 0,9947 | 2/380 |
| A (532459) | bf16/fp16 | 0,9921 | 3/380 |
| B (532460) | bf16/fp16 | **0,9816** | 7/380 |
| C (532461) | fp32/fp32 | 0,9921 | 3/380 |

Confronto **appaiato** sugli stessi 380 video:

| Coppia | Video con decisione diversa |
|---|---|
| A vs B — *stesse feature, solo rumore* | **6 / 380** |
| A vs C — *fp16 contro fp32* | **6 / 380** |
| B vs C | 10 / 380 |

**Il cambio di precisione è indistinguibile dal rumore di addestramento.** Sei video su 380 in
entrambi i casi; e la top-1 di C coincide con quella di A. Tre run sulle stesse identiche feature
coprono da 373/380 a 378/380: il **pavimento di rumore è 1,3 punti percentuali**, e qualunque
differenza sotto quella soglia non è attribuibile a niente.

Due dettagli che spiegano il perché e valgono oltre E0c. **Nessun video è sbagliato da tutti e
tre** i run (union 11, intersezione vuota): gli errori non sono proprietà dei video difficili, sono
proprietà del singolo addestramento. E si concentrano per **gruppo** UCF101 — B sbaglia tutte e
sette le clip `Archery_g01_*`, che vengono dallo stesso video sorgente. La dimensione campionaria
effettiva è quindi molto minore di 380, e la varianza molto maggiore di quanto suggerirebbe una
binomiale.

Lo scarto fra i logit (mediano 10 fra i due run fp16, 139 fra fp16 e fp32) **non è interpretabile**:
con la loss a ~1e-7 le due probe convergono a soluzioni di scala diversa, quindi i logit non sono
calibrati fra run. L'unico confronto valido qui è quello sulle decisioni.

### Decisioni prese

- **fp16 sullo store è confermato anche end-to-end**: non cambia le decisioni più di quanto le
  cambi rieseguire lo stesso identico addestramento. Insieme alla rel_L2 per tensore di E0d, la
  scelta è chiusa.
- **Nessun risultato di probe si riporta come numero singolo.** Vale da E2 in avanti: il pavimento
  di rumore misurato qui è 1,3pp su un compito saturo, e su UCF-Crime non c'è motivo di aspettarselo
  più piccolo. Servono più semi e una dispersione dichiarata, altrimenti si confrontano estrazioni
  a caso.
- **Il budget della testa va dimensionato.** 49,3 M parametri su 6192 finestre portano la loss a
  1e-7: la soluzione è sottodeterminata, ed è questa la causa prima dell'instabilità.

- **La spina dorsale è certificata**: E2 può partire senza sospetti sull'infrastruttura.
- **Il riferimento di E0c è una soglia dichiarata**, con la fonte che dice cosa non è. Il piano F4
  §F4.6 assumeva un ballpark pubblicato che per UCF101 non esiste: assunzione corretta qui.
- **Un video più corto di una finestra si scarta e si conta** (oltre il 5% ci si ferma). Vale da ora
  per ogni blocco, UCF-Crime compreso.

### Alimenta

Capitolo sull'infrastruttura sperimentale come prova di correttezza della pipeline; nota
metodologica sull'aggregazione per video. **Nessuna tabella di risultati**: non è un numero di tesi.

### Appendice — i quattro layer gerarchici a confronto

**Eseguita il:** 15 agosto 2026 · **job:** 532485–532490 (due repliche per layer, in catena) ·
**store:** `features_ucf101` fp16, nessuna estrazione nuova · **costo:** ~4,5 h totali ·
**artefatti:** `predizioni-L{5,11,17}-{a,b}.json`, `confronto-layer.json`.

Rilanciare la probe sui layer già estratti costava zero GPU-ora di encoder, e la domanda («il layer
23 è stato scelto senza confronto») era rimasta aperta. Antonio ha aggiunto il vincolo che rende il
risultato leggibile: **per ViT-B e ViT-L i layer intermedi non hanno una testa di lettura
addestrata** — verificato sul checkpoint lo stesso giorno (job 532484, `scripts/ispeziona_norms_block.py`):
`norms_block[0..2]` sono a distanza **esattamente zero** dall'inizializzazione.

| layer | replica a | replica b | media | escursione fra repliche |
|---|---|---|---|---|
| 5  | 0,6816 | 0,7658 | 0,7237 | **8,4 pp** |
| 11 | 0,8789 | 0,9132 | 0,8961 | 3,4 pp |
| 17 | 0,9632 | 0,9737 | 0,9684 | 1,1 pp |
| 23 | 0,9921 | 0,9816 | 0,9868 | 1,1 pp |

Confronto **appaiato** sugli stessi 380 video, contro `L23-a` (`scripts/confronta_predizioni.py`):

| contro L23-a | video discordi | rispetto al pavimento di rumore |
|---|---|---|
| L23-b *(sola riesecuzione)* | 6/380 = 1,6% | — è il pavimento |
| L17 | 13/380 = 3,4% | 2,2× |
| L11 | 44/380 = 11,6% | 7,3× |
| L5  | 120/380 = 31,6% | 20× |

**Tre cose.**

**(a) L'ordinamento è monotono e i salti sono enormi rispetto al rumore.** +17pp da L5 a L11, +7pp da
L11 a L17, +1,8pp da L17 a L23. Solo l'ultimo è dello stesso ordine dell'escursione fra repliche: a
tre quarti della profondità la rappresentazione è già quasi tutta lì, e le ultime sei layer aggiungono
poco su questo compito.

**(b) Il pavimento di rumore di 1,3pp non è una costante: è il rumore del regime saturo.** L'escursione
fra due repliche identiche passa da 1,1pp a L23 e L17 a **8,4pp a L5**. Ha senso — dove l'accuratezza
è alta i video borderline sono pochi — ma smentisce l'uso che se ne stava facendo: «servono più semi»
non è una quantità fissa, e il numero di ripetizioni va scelto **dove si misura**. Su UCF-Crime, dove
nessuno si aspetta il 99%, la dispersione sarà più vicina a quella di L5 che a quella di L23. Questa
è la conseguenza più importante dell'appendice, e vale per tutta E2 ed E3.

**(c) ⚠️ Cosa questo esperimento NON dice.** Non dice che i layer intermedi di V-JEPA 2.1 siano
rappresentazioni peggiori. Dice che *lette come le legge la nostra pipeline* rendono meno — e la
nostra pipeline le fa passare dalla loro `norms_block`, che su un ViT-L è una standardizzazione
**mai addestrata** (peso 1, bias 0). I blocchi intermedi sono addestrati eccome, perché l'uscita
finale dipende da loro; è la testa di lettura a non esserlo. Le due ipotesi — «l'attivazione intermedia
è meno utile» e «manca la testa che la renderebbe utile» — questo disegno non le separa. Il confronto
che le separa è lo stesso esperimento su **ViT-g o ViT-G**, dove `out_layers_distillation` coincide con
`hierarchical_layers` e tutte e quattro le uscite sono supervisionate. C'è anche un'asimmetria da
dichiarare: un layer *non* gerarchico (il 6, per dire) uscirebbe grezzo, quindi la pipeline tratta il
5 e il 6 in modo diverso per una ragione che nella rete non c'è.

### Decisioni prese *(appendice sui layer)*

- **Il layer 23 resta la scelta per E0c e per la spina dorsale**, ora con un confronto dietro invece
  che per default.
- **Il numero di ripetizioni si dimensiona sul regime, non una volta per tutte.** Da E2 in avanti la
  dispersione si dichiara insieme al numero, e dove l'accuratezza non è satura si prevedono più semi.
- **E2b confronta i layer su UCF-Crime, non su UCF101**, dove la metrica non è al soffitto; e va
  deciso (prima di E2) se legge le uscite `norms_block` o le attivazioni grezze — su B/L le due cose
  non coincidono. Proposta: misurarle entrambe su **un solo** layer, per sapere quanto pesa la
  differenza, e poi scegliere.

### Aperto

- ~~**Il confronto fra layer su ViT-g/ViT-G non è stato fatto**, ed è l'unico che separa le due
  ipotesi del punto (c).~~ — **chiuso il 18 agosto**, vedi appendice sotto.
- Budget della testa da dimensionare per E2 (qui saturava).

### Appendice — il confronto fra layer si conferma su ViT-g e ViT-G

**Eseguita il:** 18 agosto 2026 · **job:** estrazione 533429 (g, array 8 shard) e 533555 (G, array 8
shard) · merge 533565 (g) e 533569 (G) · probe 533605–533612 (g) e
533613/533614/533621/533622/533628–533631 (G) · **store:**
`features_ucf101/vjepa21giant/627d97333ce9` (147 GB) e
`features_ucf101/vjepa21gigantic/627d97333ce9` (173 GB) · stessi 1363 video, 224px, hop 1s,
`eval_center`, calcolo bf16/storage fp16 di E0c-L · **costo:** ~2h di estrazione (due array da 8
shard in parallelo) + ~16 GPU-ora di probe (16 run × 1h10–1h35 su A100 `gpuq`).

L'appendice del 15 agosto aveva lasciato aperta una domanda: il crollo di accuratezza sui layer
bassi di ViT-L è una proprietà della rappresentazione, o un artefatto delle prime tre
`norms_block` mai addestrate (§ punto (c))? Su ViT-g e ViT-G `n_output_distillation=4` invece di
1 (`vjepa21_source.py`, `ARCHS`): tutte e quattro le uscite gerarchiche sono supervisionate,
quindi è l'esperimento che separa le due ipotesi.

Stessa probe di E0c (10 classi, 2000 step, batch 32, lr 1e-3, 2 repliche per layer), sui layer
gerarchici di ciascuna arch: g `[9,19,29,39]`, G `[11,23,37,47]`.

⚠️ **Deviazione operativa dal budget di L.** Le prime 8 probe di g, lanciate col budget di tempo
di L (1h), sono andate in `TIMEOUT` a 1h esatta **senza scrivere output**: l'head attentiva scala
con `embed_dim` (1408/1664 contro 1024 di L) e a parità di 2000 passi costa di più in tempo. Rifatte
con budget 2h (completano fra 1h10 e 1h35); le due probe di G partite nel frattempo con budget 1h
sono state cancellate preventivamente. Nota a parte, non di risultato: l'account SLURM ha un tetto
di 10 job simultanei (coda+esecuzione) indipendente dalla partizione, e un primo script di
sottomissione controllava solo `squeue` per decidere se un job "era già fatto" — un job `COMPLETED`
sparisce dalla coda, quindi ha rilanciato 5 duplicati prima di essere corretto a controllare
l'esistenza di `predizioni-<label>.json`. Cancellati prima del completamento, nessun dato perso.

top1 medio e escursione fra le due repliche:

**g (embed_dim 1408)**

| layer | replica a | replica b | media | escursione |
|---|---|---|---|---|
| 9  | 0,6105 | 0,5921 | 0,6013 | 1,8 pp |
| 19 | 0,6974 | 0,7421 | 0,7197 | 4,5 pp |
| 29 | 0,8605 | 0,8026 | 0,8316 | 5,8 pp |
| 39 | 0,9737 | 0,9684 | 0,9711 | 0,5 pp |

**G (embed_dim 1664)**

| layer | replica a | replica b | media | escursione |
|---|---|---|---|---|
| 11 | 0,6158 | 0,6553 | 0,6355 | 4,0 pp |
| 23 | 0,8132 | 0,8184 | 0,8158 | 0,5 pp |
| 37 | 0,9132 | 0,9026 | 0,9079 | 1,1 pp |
| 47 | 0,9526 | 0,9763 | 0,9645 | 2,4 pp |

Confronto **appaiato** (`scripts/confronta_predizioni.py`) contro il layer più profondo di ciascuna
famiglia:

| contro g39-a | video discordi | rispetto al pavimento |
|---|---|---|
| g39-b *(sola riesecuzione)* | 17/380 = 4,5% | — è il pavimento |
| g29-a | 57/380 = 15,0% | 3,4× |
| g19-a | 121/380 = 31,8% | 7,1× |
| g9-a  | 152/380 = 40,0% | 8,9× |

| contro G47-a | video discordi | rispetto al pavimento |
|---|---|---|
| G47-b *(sola riesecuzione)* | 24/380 = 6,3% | — è il pavimento |
| G37-a | 30/380 = 7,9% | 1,2× |
| G23-a | 73/380 = 19,2% | 3,0× |
| G11-a | 134/380 = 35,3% | 5,6× |

**L'ordinamento monotono si conferma su entrambe le famiglie, con tutte e quattro le `norms_block`
addestrate.** Il salto fra il layer più superficiale e quello più profondo è enorme quanto su L
(g: +37pp da L9 a L39; G: +33pp da L11 a L47), con la stessa forma — crescita quasi lineare fino a
tre quarti della profondità, poi appiattimento. Le due ipotesi del punto (c) erano: (1) la
rappresentazione intermedia è davvero meno discriminativa per questo compito, o (2) sembra tale
solo perché su L la norma non è addestrata. **Il crollo ai layer bassi persiste identico dove la
norma è addestrata ovunque**, il che esclude che (2) sia la spiegazione principale: è la
rappresentazione stessa, letta con la testa attentiva di questa pipeline, a portare meno
informazione discriminativa nei blocchi iniziali — non l'assenza di training della standardizzazione.

**Il pavimento di rumore conferma quanto trovato il 15 agosto: non è una costante.** 4,5% per g e
6,3% per G, contro l'1,6% di L23 — più alto perché il top1 del layer più profondo di g/G (~0,97 e
~0,95–0,98) è meno saturo di quello di L23 (~0,99): più video restano vicino al bordo di decisione,
quindi due repliche identiche discordano di più. Coerente con l'appendice del 15 agosto, non
un'anomalia da spiegare a parte.

#### Decisioni prese

- **Il punto (c) dell'appendice E0c è chiuso.** Il layer più profondo resta la scelta motivata per
  la spina dorsale su ogni scala, e il motivo — i primi tre quarti della rete portano poca
  informazione discriminativa per un compito di classificazione di questo tipo, indipendentemente
  da come è normalizzata l'uscita — è ora verificato su tre scale (L, g, G) invece che assunto da
  una sola.
- **Il budget di tempo di una probe non si eredita fra scale.** Va dimensionato su `embed_dim`,
  non riusato dal profilo di L: 1h basta per L (43-51 min osservati), non per g/G (1h10-1h35).

#### Alimenta

Stessa sezione della tesi dell'appendice del 15 agosto; chiude la nota ⚠️ del punto (c) lì scritta.

#### Aperto

- Budget della testa da dimensionare per E2, su tutte e tre le scale (qui saturava anche su g/G).

| E0b *(parziale: E0b.1)* | 13 ago 2026 | 96/96 celle valide; AMP BF16 è il regime più veloce, nessun OOM fino a batch 32 | [E0b](#e0b--budget-di-calcolo-parziale-e0b1-forward-congelato) |
| E0b.2 | 16 ago 2026 | 56/56 celle, zero OOM: i prompt costano ~25% meno di reparam in addestramento, DoRA 1,46× la memoria di LoRA | [E0b.2](#e0b2--il-costo-di-addestrare-un-adattatore-per-famiglia) |
| E0b.2-scale | 16 ago 2026 | B 56/56, g 55/56, G 48/56; a 384px G non ammette batch 8 e DoRA non ammette batch 4 | [E0b.2-scale](#e0b2-scale--costo-e-integrabilità-su-b-g-e-g) |

## E0d — Pilota del feature store *(fedeltà chiusa; store query-side estratto)*

**Chiuso il:** 14 agosto 2026 (gamba numerica) · **estrazione query-side completata il:**
25 agosto 2026 · **run_id:** `e0d_ucf_vjepa21l_frozen-none_kna_s1` · **commit:** `9740b62`
(estrazioni), `7793d51` (asse `precision`), `9c8baf5` (append e merge del 25 agosto) ·
**macchina:** A100 `gpuq` per le estrazioni, `fatq` per i confronti e per il merge

### Cosa chiedeva

Il dtype con cui persistere il feature store, che a hop 1 s vale 470 GB in fp32 contro 235 in
fp16. E — domanda che non era in programma e che il pilota ha reso inevitabile — quanto costa in
fedeltà il regime **AMP BF16** scelto in E0b.1, dove era stato misurato solo in velocità.

### Come è stato eseguito

21 video di UCF-Crime, hop 1 s, 16 frame a 7,5 fps, geometria `eval_center`, batch 16, layer
5/11/17/23, entrambe le geometrie (224 e 384) — 7580 finestre per geometria. Quattro store:
riferimento `bf16/fp32`, candidato `bf16/fp16`, e un terzo `fp32/fp32` estratto apposta per
isolare la precisione di calcolo. Il confronto varia **un asse dichiarato** e pretende che tutto
il resto coincida; sull'asse `storage_dtype` la soglia si deriva da `finfo(fp16).eps / 2` con
margine 2, sull'asse `precision` **non esiste soglia derivabile** e il job misura senza dare un
verdetto.

La lettura del numero è stata calibrata a parte (`scripts/scala_feature.py`): quanto distano fra
loro finestre *diverse* dello stesso store, che è il denominatore senza il quale una rel_L2 non
significa niente.

### Cosa è uscito

| Asse | Geometria | rel_L2 media | rel_L2 max | Soglia | Esito |
|---|---|---|---|---|---|
| `storage_dtype` (fp16 vs fp32) | 224 | 1,784e-4 | 2,080e-4 | 9,766e-4 | passa, margine 4,7× |
| `storage_dtype` (fp16 vs fp32) | 384 | 1,793e-4 | 2,086e-4 | 9,766e-4 | passa, margine 4,7× |
| `precision` (BF16 vs fp32) | 224 | 2,292e-2 | 8,158e-2 | — | misura senza verdetto |
| `precision` (BF16 vs fp32) | 384 | 2,234e-2 | 1,477e-1 | — | misura senza verdetto |

7580/7580 finestre confrontate in ogni riga: zero mancanti, zero non finite, zero shape diverse
(hit rate 100%). Massimo assoluto delle feature 36,99–38,80, contro il tetto fp16 di 65504.

Calibrazione della perturbazione BF16 contro la separazione fra finestre diverse (224 / 384):

| Layer | Perturbazione | Stesso video (media) | 5° perc. | Video diversi | Perturbazione / 5° perc. |
|---|---|---|---|---|---|
| 5 | 1,60e-2 / 1,68e-2 | 0,253 / 0,256 | 0,126 / 0,137 | 0,479 / 0,484 | 12,7% / 12,2% |
| 11 | 1,76e-2 / 1,68e-2 | 0,308 / 0,308 | 0,173 / 0,176 | 0,498 / 0,509 | 10,2% / 9,5% |
| 17 | 1,96e-2 / 1,81e-2 | 0,328 / 0,324 | 0,202 / 0,203 | 0,517 / 0,525 | 9,7% / 8,9% |
| 23 | 3,78e-2 / 3,66e-2 | 0,670 / 0,672 | 0,401 / 0,411 | 1,108 / 1,132 | 9,4% / 8,9% |

### Cosa vuol dire

**Sullo store, il numero è quello giusto per la ragione giusta.** 1,78e-4 non è semplicemente
«piccolo»: è la firma dell'arrotondamento fp16 puro, che su mantisse distribuite uniformemente ha
media quadratica intorno a 1,8e-4. Il controllo che conta è un altro: se l'uscita dell'encoder
fosse rimasta in BF16, fp16 la rappresenterebbe **esattamente** (8 bit di mantissa stanno in 11) e
la rel_L2 sarebbe stata ~0 — un test che passa senza misurare niente. Non lo è, quindi l'uscita è
davvero fp32 (la LayerNorm finale sotto autocast promuove) e il confronto misura ciò che dichiara.
Il rischio vero di fp16 non è la precisione ma l'intervallo, e lì il margine è 1770×.

**Sul calcolo, 2,3e-2 va letto col denominatore.** Da solo direbbe poco; accanto alla scala dice
che la perturbazione BF16 vale circa un decimo della separazione più stretta fra due finestre
*dello stesso video* — cioè fra le cose che il frame-level AUC deve distinguere. Due fatti
rendono la lettura solida: il rapporto **non peggiora con la profondità** (12,7% → 9,4%), perché
l'errore cresce con la profondità ma la dispersione della rappresentazione cresce di più; e non
peggiora con la geometria, dato che la media è praticamente identica a 224 e 384. L'errore
accumula come atteso, non c'è nulla di anomalo da nessuna parte.

L'unica riserva sta nella coda: a 384 il massimo raddoppia (1,48e-1 contro 8,16e-2) a media
invariata. Qualche finestra devia molto più delle altre. Non compromette la media, ma è ciò che
andrà guardato se a valle comparisse una differenza inattesa proprio a 384.

### Decisioni prese

- **fp16 è il dtype dello store**, con evidenza su entrambe le geometrie. Dimezza 470 GB → 235 GB
  a hop 1 s. Il criterio era dichiarato prima della misura, non scelto dopo.
- **AMP BF16 resta il regime di calcolo**, ora con un costo misurato e non solo una velocità.
- **Il verdetto sulla precisione di calcolo resta a E2**, sulle metriche a valle. Il rapporto ~10%
  è un argomento a favore, non una prova: nessuna soglia è stata inventata per farlo sembrare tale.
- **`precision` è parte dell'identità dello store** e non si mescola dentro una campagna: due
  split estratti con regimi diversi darebbero alla probe uno scarto sistematico invisibile in
  entrambi gli store presi da soli.

### L'estrazione query-side, completata il 25 agosto

Congelata la geometria da E0f v2, lo store query-side `c6fe0f7cc26c`
(`256/eval_center/fps 7,5`, hop 1 s, fp16, layer 5/11/17/23) è stato portato a termine
aggiungendo gli **11 video** rimasti indietro alla quota del 15 agosto — 8 `Normal_Videos_*`
e 3 `Vandalism*`, cioè i più lunghi. Due ondate concatenate (`537186`, `537195`) sulla `gpuq`
e un merge dipendente (`537198`) sulla `fatq`; gli shard `index-4..11` dell'append non
collidono con gli originali `index-0..3` e l'indice letto è l'unito `index.sqlite`.

| grandezza | valore verificato |
|---|---|
| video | **290/290**, gli stessi su tutti e quattro i layer, con lo stesso numero di finestre per video |
| finestre per layer | 36 619 |
| feature indicizzate | **146 476** (= 4 × 36 619) |
| file | 1177, nessun temporaneo residuo |
| occupazione | **575 GiB apparenti** (617,2 GB, `du -sb` = 617 157 873 339 B); **648 GiB allocati** (`du -sh`), ~12,7% di overhead del filesystem parallelo |
| `replay_index_hit_rate` | 1,0 |

**Il `replay_index_hit_rate` non è una misura di completezza, e va letto per quello che è.**
Il merge è girato con `expected_count: null`: l'1,0 dice che **ogni chiave dell'indice unito
si risolve** su uno shard esistente — nessuna voce orfana — non che non manchi nessuna
finestra. La completezza poggia su un controllo separato, ed è quello riportato nella riga
«video» qui sopra: 290 file `.safetensors` per layer, insiemi di video identici fra i layer e
conteggi di finestre per video identici fra i layer. Le due cose vanno citate insieme; da sola
la prima non prova niente sull'estrazione.

**Il `code_version` è in drift, e la provenienza lo registra.** Lo `store_meta` porta
`f4e8b5e…` (il codice della prima quota, 15 agosto), gli shard 4–11 dell'append sono stati
scritti da `9c8baf5…`: dal 15 agosto `code_version` è provenienza e non identità
(vedi «Aperto» qui sotto), quindi le righe stanno in `provenienza.jsonl` e la riapertura non
solleva. Ciò che determina i valori — `ckpt_sha`, `preprocess_hash`, `precision`,
`storage_dtype` — è invariato fra le due quote.

**Il merge ha riusato il `run_id` della gamba di fedeltà, e ha sovrascritto due file.**
`meta.json` e `config.yaml` in `runs/e0d_ucf_vjepa21l_frozen-none_kna_s1/` descrivono ora il
job `537198` del 25 agosto: lo snapshot omonimo del 14 agosto (git sha, env, variabili
risolte) **non esiste più**, e `metrics.json` porta i conteggi dello store, non la fedeltà.
Non è andato perso nulla di sostanziale — i risultati della gamba numerica vivono nei file
`fidelity-*.json` e `scala-precision-*.json`, che hanno nomi distinti e sono intatti, e ogni
passo del 14 agosto ha la sua riga in `runs/registry.jsonl`, che è append-only — ma la
provenienza d'ambiente di quella giornata è recuperabile solo dai commit citati in testa a
questa voce. **Regola che ne esce:** una gamba nuova sotto un `run_id` già usato scrive
artefatti con nome proprio (`fidelity-*.json` fa così); `metrics.json`, `meta.json` e
`config.yaml` sono per-run e vengono riscritti senza avviso. Applicata subito alla gamba del
25 agosto, perché non tocchi a lei la prossima volta: i suoi tre file sono stati copiati
accanto agli originali come `store-completo-c6fe0f7cc26c.json`,
`meta-537198-store-completo.json` e `config-537198-store-completo.yaml`, sul cluster e nello
specchio locale.

### La copertura contro il protocollo — 25 agosto, hit rate 0,9927

Congelato `fsp_v1`, la domanda utile non è più «l'indice è coerente» ma **«ogni finestra che il
protocollo genera ha la sua chiave?»**. Sono due domande diverse, e la prima non implica la
seconda: `replay_index_hit_rate = 1.0` era stato misurato col merge a `expected_count: null`,
quindi diceva soltanto che nessuna voce d'indice punta nel vuoto.

`scripts/replay_store.py` ricostruisce le chiavi dalla griglia di `fsp_v1` e le confronta con
l'indice. Esito su `c6fe0f7cc26c`:

| grandezza | valore |
|---|---|
| finestre attese dal protocollo | **147 548** (290 video × 4 layer) |
| presenti nell'indice | 146 476 |
| **hit rate** | **0,992735** |
| video incompleti | **268 su 290** |
| finestre mancanti | 1072 = 268 × 4 layer |

**È esattamente una finestra per video, ed è sempre l'ultima.** `Abuse028_x264@1411x16s4` su un
video di 1412 frame; `Arrest001_x264@2373` su 2374; e così per tutti.

**La causa è una modifica al windowing entrata dopo l'inizio dell'estrazione.** Il commit
`2e6e30e` del **20 agosto** ha aggiunto in coda a `CausalWindowSampler._video_endpoints`:

```python
if endpoints[-1] != last_end:
    endpoints.append(last_end)
```

cioè *l'ultimo frame del video ha sempre la sua finestra*, anche quando cade fuori dalla griglia
di hop. Lo store è stato riempito a partire dal **15 agosto** (`f4e8b5e`), quando quella riga non
c'era; l'append degli 11 video del 25 agosto è girato col codice nuovo, ed è fra i 22 video
completi. I 268 restanti sono quelli in cui la griglia non cadeva già sull'ultimo frame.

**Perché conta più del numero.** Un miss non produce nessun errore: `WindowDataset` ricade sui
pixel e il run ricalcola in silenzio. Su 75 celle di E2 sarebbero 268 forward e 268 decodifiche
per cella — non una catastrofe, ma la DoD di F7 chiede *hit rate 100%*, e un cancello che si
accetta al 99% non è più un cancello. È anche la seconda volta che questo progetto paga la stessa
cosa: E0f v2 aveva scoperto il 23 agosto che «lo split consumato non si rigenera col codice
corrente», per l'evoluzione dello **stesso file**.

> **La regola che ne esce, e che vale per ogni artefatto derivato dal codice:** un artefatto
> costoso costruito da una funzione pura porta con sé *quale versione di quella funzione* l'ha
> generato, e chi lo riusa **verifica la copertura invece di assumerla**. Non serve versionare il
> codice nell'identità dello store — quella strada è già stata provata e abbandonata (`code_version`
> è provenienza dal 15 agosto): serve un controllo che confronti l'artefatto con la regola
> corrente, ed è quello che `replay_store.py` fa in dieci secondi.

### La copertura chiusa — 25 agosto, hit rate 1,000000

L'append è stato eseguito su autorizzazione di Antonio. **Job 537422**, quattro shard sulla
`gpuq` (20–30 min ciascuno), lista `$EVD_DATA_DIR/e0d_video_incompleti.txt`, indici `12..15`
scritti col nuovo `--store-shard`; **merge 537435** sulla `fatq` con `--expected-count 147548`.

| grandezza | valore |
|---|---|
| scritte per shard | **268** = 67 video × 4 layer, identico su tutti e quattro |
| `hit_rate_before_write` per shard | 0,9908 · 0,9911 · 0,9891 · 0,9885 |
| feature dopo il merge | **147.548**, uguale a `expected_count` |
| `replay_index_hit_rate` | 1,0 |
| **copertura contro `fsp_v1`** | **1,000000** su 147.548 finestre (`replay-copertura-fsp_v1.json`) |

Le due righe finali dicono cose diverse e vanno citate insieme: la prima che nessuna voce
d'indice è orfana, la seconda che **nessuna finestra del protocollo manca**. È la prima volta
che il merge gira con `expected_count` valorizzato, ed è ciò che rende il conteggio una
verifica invece di un numero.

I tre file per-run del merge sono stati copiati subito accanto agli originali
(`store-copertura-1.0-537435.json`, `meta-537435-copertura.json`,
`config-537435-copertura.yaml`), come impone la regola nata dal merge 537198.

### L'incidente del primo tentativo — job 537408, cancellato dopo 5 minuti

Il primo submit dell'append stava **riestraendo i 268 video per intero**, non la sola finestra
mancante: 122 GB di ricalcolo per shard. Il salto delle chiavi già presenti leggeva
`store.keys()`, cioè l'indice di **questo** shard — che in un append è vuoto per definizione.
Il difetto non produce nessun errore lungo la strada: si sarebbe visto solo a valle, in
`merge_shards`, che rifiuta le chiavi duplicate — a ore di GPU già spese.

Cancellato dopo 5 minuti guardando la dimensione di `index-12..15.sqlite`, che cresceva di
centinaia di KB al minuto invece che di decine. **Nessun dato perso:** `put_many` legge il file
esistente e ci fonde dentro i tensori nuovi, quindi le chiavi già presenti sono state riscritte
con lo stesso valore e il conteggio dei file è rimasto 1160. Rimossi i quattro indici e i
temporanei residui — fra cui due `.tmp` orfani del **19 agosto** da 2,7 GB, che la riga «nessun
temporaneo residuo» del merge 537198 non aveva visto perché sono dot-file. Store riverificato
**prima** di ripetere: replay 0,992735, identico a prima dell'incidente.

Correzione in `df1b791`: chi **scrive** un append usa `store.known_keys()`, l'unione di ogni
`index*.sqlite` della radice; `keys()` e `__contains__` restano l'indice di chi scrive, perché
rispondono a un'altra domanda — dove `get` sa andare a prendere il tensore. Il test ha la sua
contro-prova: togliendo `known_keys` fallisce con `n_hits = 0`.

> **La regola che ne esce**, gemella di quella sopra: `--shard-index` governava *due* decisioni
> indipendenti — quale fetta della lista e quale file d'indice — e finché coincidevano il
> difetto era invisibile. Un parametro (o un metodo) che risponde a due domande è un confine nel
> posto sbagliato, e si scopre quando una delle due cambia da sola.

### Alimenta

Capitolo sperimentale, sezione sul feature store; tabella di efficienza E5b (dimensione della
cache); materiale per l'incontro del 24 agosto insieme alle tabelle costo-vs-hop di E0b.1.

### Aperto

- ~~**Lo store non copre il 100% della griglia di `fsp_v1`**: mancano 1072 feature, l'ultima
  finestra di 268 video.~~ — **chiuso il 25 agosto** (job 537422 + merge 537435): copertura
  1,000000 su 147.548 finestre.
- **La gamba di throughput di E0d non è stata eseguita**: pipeline end-to-end e dimensione reale
  entro il 5% della stima restano da misurare. *(Il replay non è più fra le voci aperte, ma
  attenzione a cosa misura: vedi «L'estrazione query-side» qui sotto.)*
- ~~**E0d completa** resta bloccata finché geometria e hop non sono congelati dai supervisori.~~
  — **sbloccata il 25 agosto** per la parte d'estrazione: la geometria è congelata da E0f v2
  (`256/eval_center/fps 7,5`) e l'hop resta `1 s` come da `docs/12` §2. Resta aperta la sola
  gamba di throughput.
- **La coda a 384** (max 1,48e-1) non ha una spiegazione: servirebbe sapere se le finestre che
  deviano di più hanno qualcosa in comune.
- ~~**`code_version` fra gli invarianti dello store**~~ — **chiuso il 15 agosto.** È uscito
  dall'insieme verificato (`CAMPI_IDENTITA` in `evd/data/feature_store.py`) e diventato
  provenienza: riaprire uno store con un codice diverso non solleva più, ma scrive una riga in
  `provenienza.jsonl` accanto al meta, e `store.code_versions()` la riporta nel report del job.
  Il ragionamento: la versione del codice è un *proxy* di «il percorso di estrazione è lo
  stesso», ed è cattivo in entrambe le direzioni — cambia per un commit di soli documenti e non
  cambia per un working tree sporco. Ciò che determina davvero i valori resta verificato:
  `ckpt_sha`, `preprocess_hash`, `precision`, `storage_dtype`.

## E0b — Budget di calcolo *(parziale: E0b.1 forward congelato)*

**E0b.1 eseguita il:** 13 agosto 2026 · **run_id:**
`e0b_ucf_vjepa21l_frozen-none_kna_s1` (`runs/registry.jsonl`) ·
**commit:** `e11c0c07c6c5b7993516e67ebf59c84bd58e19be` · **manifest:** n/a (`kna`) ·
**macchina:** NVIDIA A100-SXM4-80GB, `gpuq`, `gnode04` · **job:** 532148

### Cosa chiedeva

E0b.1 doveva misurare il forward congelato di V-JEPA 2.1-L e quantificare gli effetti di
risoluzione, precisione, TF32, batch e numero di layer estratti. Questi numeri servono a portare
ai supervisori il costo delle alternative per geometria e hop, a dimensionare il pilota E0d e a
separare il costo dell'encoder da quello di LoRA/VPT, che sarà misurato in E0b.2.

### Come è stato eseguito

Il profilo ha coperto 96 celle:
`size={224,384} × precision={fp32,AMP-bf16} × tf32={off,on} × batch={1,2,4,8,16,32} ×`
`layer_set={final,hierarchical}`. `final` è il blocco `-1`; `hierarchical` estrae
`[5,11,17,23]` in un solo forward. Ogni cella usa 10 warmup e 50 iterazioni misurate con
sincronizzazione CUDA; gli input sono clip casuali già residenti sulla GPU. Il risultato misura
quindi **l'encoder**, non decoding, preprocessing, trasferimenti host-device o scrittura su disco.
La tabella dell'hop usa i 290 video del query set standard, 16 frame a 7,5 fps e i candidati
`{0,5; 1; 2; 4}` secondi.

Il primo tentativo, job 532145 sul commit `4992851`, è stato scartato. Dopo 12 min 37 s aveva
salvato le 24 celle FP32 a 224 px, poi la prima cella BF16 è fallita: la conversione integrale
dell'encoder a BF16 lasciava `v` in BF16 mentre l'interpolazione della 3D-RoPE promuoveva `q` e
`k` a FP32; SDPA richiede i tre tensori nello stesso dtype. Il run è stato archiviato come
`e0b_ucf_vjepa21l_frozen-none_kna_s1__failed-job532145` e nessuna sua cella è entrata nei
risultati. Il job 532148 usa invece il percorso Meta: pesi e input FP32 più AMP/autocast BF16.

Il job valido è terminato `COMPLETED`, exit `0:0`, in 1 h 18 min 10 s. Ha richiesto una GPU e
8 CPU, ma `sacct` registra 1 h 17 min di CPU su 10 h 25 min allocate: per altri profiler
sintetici basta 1 CPU. Commit e branch erano puliti (`main`); checkpoint SHA256
`7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6`.

### Cosa è uscito

Tutte le 96 celle sono finite con valori finiti: **96 `ok`, 0 OOM, 0 saltate**. La tabella
seguente usa, per ogni regime, il throughput massimo osservato; in tutti i casi cade a batch 32.
La VRAM è il picco a batch 32, riportato come `final / hierarchical`.

| size | regime | finestre/s | ms/finestra | VRAM GiB | speedup su FP32 |
|---:|---|---:|---:|---:|---:|
| 224 | FP32, TF32 off | 14,46 | 69,15 | 3,75 / 4,32 | 1,00× |
| 224 | FP32, TF32 on | 37,67 | 26,54 | 3,75 / 4,32 | 2,61× |
| 224 | AMP BF16 | 79,95 | 12,51 | 2,70 / 2,99 | 5,53× |
| 384 | FP32, TF32 off | 3,87 | 258,21 | 8,76 / 10,45 | 1,00× |
| 384 | FP32, TF32 on | 7,43 | 134,66 | 8,76 / 10,45 | 1,92× |
| 384 | AMP BF16 | 23,47 | 42,61 | 5,67 / 6,51 | 6,06× |

Il flag TF32 non modifica in modo apprezzabile AMP BF16: sulle 24 coppie la variazione assoluta
mediana del throughput è 0,025%. Il vantaggio di 7,28× misurato sul GEMM isolato non si trasferisce
all'intero ViT-L: qui TF32 vale 2,61× a 224 e 1,92× a 384. AMP BF16 resta 2,12× e 3,16× più
veloce dello stesso profilo FP32+TF32.

Estrarre quattro layer invece dell'ultimo aggiunge una latenza mediana dello 0,31% (massimo
1,30%). Il costo vero è lo storage: quattro uscite invece di una. A batch 16 AMP raggiunge già il
95,7% del throughput massimo a 224 e il 99,1% a 384, con picchi rispettivamente di 2,08 e 3,84 GiB
per i quattro layer; è il punto iniziale prudente per il pilota E0d.

La geometria pesa più della VRAM. A parità di AMP BF16, 384 px è 3,41× più lento di 224; la cache
è 2,94× più grande. Per il query set completo:

| hop | finestre | AMP BF16 224 | AMP BF16 384 | cache 4L fp16 224 | cache 4L fp16 384 |
|---:|---:|---:|---:|---:|---:|
| 0,5 s | 73.082 | 15,2 min | 51,9 min | 938,74 GB | 2.758,75 GB |
| 1 s | 36.608 | 7,6 min | 26,0 min | 470,23 GB | 1.381,91 GB |
| 2 s | 18.378 | 3,8 min | 13,1 min | 236,07 GB | 693,75 GB |
| 4 s | 9.269 | 1,9 min | 6,6 min | 119,06 GB | 349,89 GB |

I tempi AMP BF16 della tabella sono derivati dal miglior throughput `final` misurato; il report
automatico conserva anche le colonne FP32 e FP32+TF32. Tutti i numeri grezzi, comprese media,
deviazione standard e stima FLOP, sono in `runs/.../efficiency.json`. Lo SHA256 locale e remoto
di questo file coincide: `0f48042b3e9b3050531daafba6d12d440312e281e37a8ddaa76df4615c262482`.

### Cosa vuol dire

La A100 non è limitata dalla memoria nel forward congelato: anche la cella più pesante usa
10,45 GiB su 80 e batch 32 non va in OOM. Per le passate lunghe l'AMP BF16 con SDPA è il regime
di calcolo naturale; TF32 resta l'alternativa FP32 quando serve quella numerica. La raccolta dei
quattro layer gerarchici è quasi gratuita sul lato compute, quindi E2b può estrarli insieme e
rimandare la scelta del layer senza moltiplicare i forward.

Questi dati **non scelgono da soli** 224 contro 384 né l'hop: misurano il lato costo, mentre la
geometria cambia anche le feature e l'hop cambia la risoluzione temporale. Non misurano inoltre la
fedeltà di una cache fp16 rispetto a fp32, né il throughput end-to-end con PyAV e safetensors.

### Decisioni prese

- Per E0d pilota si parte da AMP BF16 + SDPA e batch 16; il pilota deve misurare il collo di
  bottiglia reale di decode e scrittura prima di promuovere il valore a default operativo.
- I quattro layer gerarchici si estraggono nello stesso forward. La potatura avverrà dopo E2b,
  perché il costo di calcolo non giustifica una shortlist anticipata.
- Nessuna decisione di protocollo è stata presa. `size`, `hop_s` e dtype persistente restano
  aperti; fp16 è il candidato per lo store, subordinato al controllo numerico del pilota E0d.

### Alimenta

Capitolo sperimentale sull'efficienza, tabella E5b, scelta del regime per E0d/E2 e discussione
del rischio R15. Le due tabelle costo-vs-hop entrano nel materiale per l'incontro del 24 agosto.

### Aperto

- ~~**E0b.2**~~ **eseguita il 16 agosto** — vedi la sezione qui sotto.
- **Protocollo:** scelta di geometria e hop con i supervisori. E0b.1 fornisce il costo, non la
  qualità o la risoluzione temporale desiderata.
- **E0d pilota:** throughput end-to-end, hit rate 100%, dimensione reale entro il 5% della stima e
  confronto numerico fp16/fp32.
- **Ripetibilità:** il run contiene 50 iterazioni per cella ma una sola esecuzione per regime e una
  sola A100. Le clip casuali non sono state seedate esplicitamente dal profiler; il contenuto non
  cambia il grafo del ViT, ma il prossimo profilo deve applicare `EVD_SEED` anche all'input.

## E0b.2 — Il costo di addestrare un adattatore, per famiglia

**Eseguita il:** 16 agosto 2026 · **run_id:** `e0b2_synth_vjepa21l_peft_kna_s0`
(`runs/registry.jsonl`) · **commit:** `c5d87d6` (working tree pulito) · **macchina:** A100-SXM4-80GB
`gpuq`, job 533078 · **regime:** AMP BF16 (pesi e input FP32) + TF32 + SDPA · **56/56 celle, zero
OOM**.

### Cosa misura, e cosa no

Il passo completo — forward, backward e aggiornamento — su `size × variante × batch`, con clip
sintetiche e la pipeline dati **fuori** dalla misura, come in E0b.1. Cronometra il `Trainer` vero
con un loader di un solo batch già in memoria: un profiler che riscrive il passo misura il
profiler, e la prima volta che il loop cambia le due strade divergono dentro un numero.

⚠️ **Non fissa `peft_budget_a`.** Dà il costo; quale budget di passi e quale batch scegliere è una
decisione di protocollo (`docs/05` §5, Q14), e questa tabella esiste per informarla.

### La tabella

Secondi per passo e picco di VRAM, 16 frame, AMP BF16 + TF32. `par` sono i parametri
dell'adattatore (la testa attentiva ne aggiunge 4,2 M, uguali per tutte).

**384px (4608 token)**

| variante | par | b1 | b2 | b4 | b8 | picco b8 | finestre/s b8 |
|---|---|---|---|---|---|---|---|
| LoRA `separate` | 3.145.728 | 0,24 | 0,34 | 0,66 | 1,26 | 30,55 GiB | 6,33 |
| LoRA `fused` | 2.359.296 | 0,21 | 0,31 | 0,61 | 1,17 | 27,11 GiB | 6,84 |
| DoRA `separate` | 3.244.032 | 0,31 | 0,43 | 0,81 | 1,55 | **44,61 GiB** | 5,16 |
| VPT-deep n8 | 196.608 | 0,18 | 0,25 | 0,48 | 0,94 | 26,99 GiB | 8,55 |
| VPT-shallow n8 | 8.192 | 0,18 | 0,25 | 0,47 | 0,92 | 26,98 GiB | 8,74 |
| APT n8 | 24.576 | 0,17 | 0,25 | 0,49 | 0,94 | **23,46 GiB** | 8,50 |
| DA-VPT n8 | 198.656 | 0,18 | 0,26 | 0,49 | 0,94 | 27,27 GiB | 8,50 |

**224px (1568 token)** — stesse tendenze, un terzo del costo: a b8 LoRA 0,39 s/passo e 11,73 GiB,
DoRA 0,50 e 16,78, i prompt 0,25–0,26 e 9,36–10,68.

### Cosa dice

1. **La famiglia prompt costa meno di reparam in addestramento, e il numero è ~25%.** A 384px e
   batch 8: 0,94 s/passo contro 1,26 di LoRA (+34% per reparam) e 3,5 GiB in meno. È la metà
   *attesa* del trade-off C7 — l'altra metà, il costo di **inferenza**, va nella direzione opposta
   ed è E5b. Questa tabella non conclude il confronto: ne fissa un lato.
2. **DoRA costa 1,46× la memoria di LoRA e 1,23× il tempo**, a fronte del 3% di parametri in più.
   Il prezzo non è nei parametri: è la norma per riga di `V`, che va materializzata a ogni forward.
   Va dichiarato in E3-R.4, perché «DoRA aggiunge 98.304 parametri» suggerisce un costo che non è
   quello vero.
3. **`fused` costa meno di `separate` anche in tempo e memoria**, non solo in parametri: −7% di
   passo e −11% di picco. Le due differiscono per la sola fattorizzazione, quindi il confronto
   E3-R.1 fra loro **non è a parità di costo** e la parità va imposta, non assunta.
4. **APT ha il picco più basso di tutti, e di parecchio**: 23,46 GiB contro i 27 di VPT, cioè −13%.
   È coerente con il meccanismo — APT non allunga il flusso residuo, quindi ogni blocco salva
   attivazioni su 4608 token invece che 4616, e la MLP non vede i prompt affatto. ⚠️ Il **fatto** è
   misurato; la spiegazione no: 8 token su 4608 sono lo 0,17%, non il 13%, quindi il grosso viene
   dai tensori che la concatenazione in sequenza fa salvare a ogni blocco per il backward. Non è
   stato isolato.
5. **Nessun OOM fino a batch 8 a 384px**, con il massimo a 44,6 GiB su una A100 da 80. C'è margine
   per batch 16 su reparam; per DoRA a 384 il margine è più stretto di quanto i parametri
   suggeriscano.

### Alimenta

`peft_budget_a` (`configs/train/`), la tabella E5b, e la discussione del 24 agosto su Q14 — dove il
punto 3 va portato esplicitamente: fissare il budget in parametri addestrabili non fissa il costo.

### Aperto

- **Una sola esecuzione per cella** (8 iterazioni dopo 2 di warmup, una A100). La dispersione fra
  ripetizioni non è misurata; per una tabella di tesi serve almeno un secondo run.
- **I FLOPs sono una stima**, non un conteggio: `3 ×` il forward analitico, con il fattore
  dichiarato in `flops_source`. Il rapporto fra varianti è affidabile, il valore assoluto no.
- **APT è misurata su tutti e 24 i blocchi**, che è la configurazione il cui primo banco non riceve
  gradiente (mini-fit 533077). Il costo non cambia, ma quando E3-P.3 sceglierà l'intervallo la riga
  andrà rimisurata su quello.
- **`peft_budget_a` resta da fissare**: è una decisione del 24 agosto, non di questa tabella.

## E0b.2-scale — Costo e integrabilità su B, g e G

**Eseguita il:** 16 agosto 2026 · **run_id:**
`e0b2_synth_vjepa21{b,giant,gigantic}_peft_kna_s0` · **commit:** `6c5d4cd` (working tree
pulito) · **macchina:** A100-SXM4-80GB `gpuq`, job 533126/533127/533130 · **regime:** identico
a E0b.2-L: AMP BF16, pesi e input FP32, TF32, SDPA, 224/384px, batch 1/2/4/8.

I primi tentativi G 533128/533129 sono falliti nel `TaskProlog` di `gnode09`, prima del codice;
533130 è la ripetizione invariata con quel nodo escluso. Non entrano nei risultati.

### Costo comune a 384px, batch 1

Secondi per passo e picco di VRAM (`s / GiB`). L è riportata dalla campagna 533078 per dare il
denominatore alla crescita di scala.

| variante | B | L | g | G |
|---|---:|---:|---:|---:|
| LoRA `separate` | 0,120 / 2,14 | 0,237 / 5,66 | 0,465 / 14,53 | 0,637 / 23,02 |
| DoRA `separate` | 0,160 / 2,89 | 0,315 / 7,77 | 0,578 / 19,59 | 0,801 / 30,49 |
| LoRA `fused` | 0,097 / 1,98 | 0,207 / 5,21 | 0,408 / 13,50 | 0,590 / 21,61 |
| VPT-deep n8 | 0,085 / 1,98 | 0,180 / 5,24 | 0,347 / 13,48 | 0,491 / 21,53 |
| VPT-shallow n8 | 0,081 / 1,98 | 0,176 / 5,23 | 0,340 / 13,47 | 0,484 / 21,52 |
| APT n8 | 0,079 / 1,80 | 0,175 / 4,74 | 0,337 / 12,47 | 0,497 / 20,09 |
| DA-VPT n8 | 0,085 / 2,00 | 0,182 / 5,28 | 0,348 / 13,52 | 0,498 / 21,59 |

### Frontiera di memoria

| scala | celle valide | limite osservato a 384px |
|---|---:|---|
| B | 56/56 | nessun OOM fino a batch 8 |
| g | 55/56 | DoRA `separate` OOM a batch 8; le altre sei varianti passano |
| G | 48/56 | tutte le sette varianti OOM a batch 8; DoRA OOM già a batch 4, quindi batch 8 è saltato |

L'informazione utile per il budget non è soltanto «G costa di più»: a batch 1 LoRA passa da
2,14 GiB su B a 23,02 su G, mentre DoRA arriva a 30,49 GiB; aumentando il batch, il margine di
DoRA scompare prima. Un unico `peft_budget_a` espresso solo in parametri addestrabili non rende
quindi confrontabili né le famiglie né le scale.

### Mini-fit F5 sulle dodici configurazioni

I job 533123/533124/533125 usano 32 finestre reali da 8 video UCF101, 384px e 12 passi. Tutte le
configurazioni reparam passano su B, g e G, compresi merge/unload e congelamento del backbone.

| scala | esito | osservazione |
|---|---:|---|
| B | 11/12 | fallisce solo APT full: 2 tensori su 24 fermi; `apt_late` passa 12/12 |
| g | 9/12 | VPT-deep e DA-VPT: 1 banco su 40 fermo; APT: 2 tensori su 80; `apt_late` passa 40/40 |
| G | 12/12 | tutte le verifiche passano, comprese le quattro configurazioni prompt |

Lo stato SLURM `FAILED` di 533123/533124 deriva dal valore di uscita del mini-fit quando anche una
sola configurazione non passa: entrambi hanno eseguito l'intera matrice e scritto gli artefatti.
Il controllo mirato 533132 ripete su g VPT-deep, APT e DA-VPT in FP32/TF32: gli stessi tensori
restano fermi. Non è quindi un semplice underflow BF16. Il fatto che G passi esclude anche un
errore generale del factory o del mapping dei 48 blocchi; resta un limite della configurazione
full-depth sul checkpoint g, da trattare scegliendo intervallo e learning rate in E3-P, non
allentando il test di integrazione.

### Limiti

- Una sola A100 e una sola ripetizione per cella: i tempi servono a dimensionare, non sono ancora
  la tabella finale di efficienza.
- I pixel di E0b.2 sono sintetici; il mini-fit verifica invece il percorso dati reale, ma non
  misura qualità few-shot e non dipende da `k`.
- Le celle OOM fissano la frontiera della griglia provata, non il massimo batch teorico fra due
  valori adiacenti.

## E0d-throughput — quante CPU servono davvero *(chiuso: 2,4 finestre/s per worker)*

**Eseguita il:** 28 agosto 2026 · **run_id:**
`e0d_ucf_vjepa21l_frozen-none_s1__throughput` (job **539567**, 16 min 29 s) ·
**commit:** `896fd21` · **macchina:** `fatq`, 32 CPU, **nessuna GPU** · 512
finestre per cella, estratte coi pesi del sampler dallo split sigillato.

### Cosa chiedeva

È la gamba di throughput di E0d, **aperta dal 14 agosto**. Tutti gli altri
profili del progetto — E0b.1, E0b.2, E0b.3 — girano su clip sintetiche già
residenti sulla GPU, apposta per togliere il decoder dalla misura: sul percorso
dati reale non esisteva un solo numero, e «quante CPU chiedere» era un'opinione.

### Cosa è uscito

| worker | fin/s (regime) | per worker | efficienza | avvio |
|---:|---:|---:|---:|---:|
| **0** | 4,11 | — | — | 431 ms |
| **1** | **2,48** ⚠️ | 2,48 | — | 953 ms |
| 2 | 4,83 | 2,42 | 100% | 854 ms |
| 4 | 9,59 | 2,40 | 99% | 946 ms |
| 8 | 18,64 | 2,33 | 96% | 1015 ms |
| 16 | 35,13 | 2,20 | 91% | 1181 ms |
| 32 | 64,21 | 2,01 | 83% | 1429 ms |

`persistent_workers` non sposta niente (4,83 contro 4,85; 18,64 contro 18,57):
→ vedi *Aperto*, perché questa misura **non può** vederne l'effetto.

### Cosa vuol dire

**Un worker decodifica ~2,4 finestre/s, e la scala è quasi ideale fino a 32.**
L'efficienza per worker scende solo dal 100% all'83% moltiplicando i worker per
sedici: il decoding di UCF-Crime — tutti e 1900 i video sono **320×240 H.264 a
25–30 fps** — è lavoro parallelo quasi puro. Da notare che una finestra costa
~233 frame nativi decodificati per tenerne 32: a `target_fps: 4` su un video a 30
fps, lo span di 7,75 s va comunque attraversato quasi tutto.

⚠️ **Un worker è più lento di zero** (2,48 contro 4,11 fin/s). Con un solo worker
si paga la serializzazione e il trasferimento di batch da 113 MB fra processi
senza guadagnare sovrapposizione: `-c 1` è il valore peggiore della griglia, e va
evitato per la stessa ragione per cui non si sceglie a caso.

**La stima a priori era sbagliata per eccesso.** Prima di misurare avevo scritto
«4–12 finestre/s per worker», ragionando che 320×240 è poco: sono **2,4**. La
direzione dell'errore è la stessa dell'estrapolazione di E0b.3 — ottimismo su un
percorso non misurato.

### Decisioni prese

**`-c 8` per ogni job con GPU** (training, valutazione, estrazione), e la ragione
non è il ginocchio ma il **consumatore**: la passata di valutazione è la cosa più
affamata della pipeline con **16,55 fin/s** su H100, e 8 worker ne danno 18,64.
Sotto, il loader diventa il collo di bottiglia: a `-c 4` una passata di
validazione passa da 13,6 a 23,5 minuti e una di test da 9,2 a 15,9, cioè
**+16,6 min per run**. Sopra, i core in più decodificano finestre che nessuno
aspetta.

**8 non è una richiesta avida, ed è il controllo che rende la scelta difendibile
davanti alla regola 8:** `aiq` ha 112 CPU per 8 GPU (**14 per scheda**) e `gpuq`
128 per 4 (**32 per scheda**). Chiedere 8 CPU per una GPU è il **57%** della
quota naturale del nodo su `aiq` e il **25%** su `gpuq`.

**Il training da solo ne vorrebbe 4** (serve 5,01 fin/s per VPT, il più vorace, e
4 worker ne danno 9,59) — ma un run addestra **e** valuta nello stesso job, e il
`-c` è uno solo.

### Alimenta

Il `-c` di ogni `.sbatch` della campagna E2/E3, il dimensionamento
dell'estrazione E0d, e la voce «CPU» di `hpc/README` §4.

### Aperto

- **La griglia non satura**, e questo esperimento non può dire dove lo farebbe: a
  32 worker il throughput cresce ancora. Non serve saperlo per la campagna — il
  vincolo è la GPU molto prima — ma servirebbe per un'estrazione massiva solo CPU.
- ⚠️ **`persistent_workers` non è stato misurato, anche se è nella griglia.** Il
  suo effetto è sul **riuso** del loader fra due iterazioni, e ogni cella qui lo
  itera **una volta sola**: l'asse è inerte per costruzione, ed è un difetto di
  disegno di questa misura, non un risultato. Nella pipeline vera la domanda è
  quasi vuota — il loader di training copre tutto il budget in una passata e
  quelli di valutazione girano una volta — quindi il campo resta **non
  impostato**, che è diverso da «misurato e scartato».
- ⚠️ **`pin_memory` non è misurabile su `fatq`**: senza CUDA PyTorch lo ignora. La
  cella che conta — il trasferimento host→device di un micro-batch da 113 MB —
  vuole una scheda, e si farà accanto al primo job GPU invece di occuparne una per
  un benchmark di decoding.
- **La misura è su `fatq` e legge da `/shared`**: stessa NFS di `aiq` e `gpuq`, ma
  CPU di modello diverso. Il numero va riletto se un job GPU risultasse
  loader-bound.

### Seconda esecuzione — con l'augmentation sull'asse *(29 agosto, job 540018)*

**run_id:** `e0d_ucf_vjepa21l_frozen-none_s1__throughput-aug` · **`aiq`,
`ainode01`, 1 H100, `-c 8`, 8 min 59 s** · commit `62b6b9d`, working tree pulito.

⚠️ **La prima misura descriveva un loader che nessun run monta.** Girava senza
`augment`, e i run di `edp_v1` montano `official_nocrop` — RandAugment
`rand-m7-n4-mstd0.5-inc1` su 32 frame a 384×384, applicato **frame per frame su
immagini PIL**. È la quarta volta in questo progetto che un rapporto misurato
fuori dal contesto vero non si trasferisce dentro; l'asse è stato aggiunto in
ADR-038, e la catena si costruisce con `ExperimentRunner._augment` e si passa al
loader con lo stesso nome che usa `_loader`.

Finestre/s in regime, 512 finestre estratte coi pesi del sampler dallo split di
training, `--device cuda`:

| worker | `none` | `official_nocrop` | costo dell'augmentation |
|---:|---:|---:|---:|
| 0 | 4,68 | **3,63** | 1,29× |
| 4 | 12,85 | 10,29 | 1,25× |
| 8 | **25,10** | **20,45** | 1,23× |

**Tre cose che questa tabella dice, e una che non dice.**

1. **L'augmentation costa 1,23×, non 3,2×.** Il 3,2× di `docs/10` §E0g (job
   537519 contro 537520) era un rapporto fra *tempi per passo* di un altro
   regime — geometria few-shot, support piccolo, forward congelato — e non
   descrive il costo di decodifica di questa geometria. Un rapporto si rimisura
   nel contesto in cui lo si usa.
2. **Il loader non è più il collo di bottiglia.** A 8 worker il percorso di
   training produce 20,45 fin/s contro le **5,01** che VPT-deep consuma (E0b.3) e
   le **16,55** della passata di valutazione, che gira senza catena a 25,1. La
   riga `workers_per_requisito` lo dice per esteso: **4 worker** bastano al
   training, **8** alla valutazione — cioè `-c 8` è dimensionato, non abbondante.
3. **La riga a zero worker è la misura diretta del difetto di ADR-038**: 3,63
   fin/s è, entro il rumore, ciò che la campagna E0i fallita ha esibito sul
   percorso di training (~1,9 fin/s **compreso** il forward, che ci si sommava
   invece di sovrapporsi).

Cosa **non** dice: dove satura. A 8 worker cresce ancora, e `ginocchio()` lo
dichiara (`NON satura`) invece di chiamare ginocchio il bordo della griglia.
`persistent_workers` resta inerte anche qui (≤0,9% su tutte le coppie), per la
stessa ragione di disegno della prima esecuzione: ogni cella itera il loader una
volta sola.

**Verifica end-to-end, che è ciò che conta davvero.** La cella 0 della campagna
E0i rilanciata (job `540019_0`, VPT-deep s1) ha fatto i primi 99 passi in **401
s: 4,05 s/passo**, contro gli **8,5-9,0** della campagna del 28 agosto. Il resto
della distanza dai 3,19 s/passo di E0b.3 è il trasferimento host→device (113 MB
per micro-batch, 8 micro-batch per passo) più il loop vero, che il profilo
sintetico di E0b.3 non conteneva.

## E0b.3 — I costi alla geometria EDP-v1 *(chiuso: 9.216 token misurati, non estrapolati)*

**Eseguita il:** 28 agosto 2026 · **run_id:** `e0b3_ucf_vjepa21l_frozen-none_s1`
(job **539362**, 14 min 57 s) e `e0b3_synth_vjepa21l_peft-edp_s1` (job **539363**,
2 min 10 s) · **commit:** `f15683f`, working tree pulito · **macchina:**
A100-SXM4-80GB, `gpuq`, `gnode01` e `gnode05` · **regime:** AMP BF16 (pesi e
input FP32) + TF32 + SDPA · **split:** `edp_v1` sigillato `e0a1e36db554`.

### Cosa chiedeva

Tutte le tabelle di costo del progetto — E0b.1, E0b.2, le stime di `PIANO.md` §3
— sono a **4.608 token** per finestra (384 px, 16 frame). ADR-034 ne ha fissati
**9.216** (384 px, 32 frame a 4,0 fps): 2× la parte lineare e 4× l'attenzione. La
domanda era quanto costa *davvero* la geometria in vigore, perché ogni stima di
ore era un'estrapolazione, e prenotare coda su un'estrapolazione sbaglia in una
direzione che non si conosce. **Non produce numeri di tesi: dimensiona il
budget.**

### Come è stato eseguito

Due job indipendenti, `scripts/slurm/e0b3.sbatch frozen` e `… peft`, che
**riusano** i profiler di E0b.1 (`e0_profilo.py e0b1`) ed E0b.2
(`profilo_peft.py`) invece di introdurne un terzo, e leggono geometria e
configurazioni PEFT dai file che le dichiarano — `configs/protocol/edp_v1.yaml` e
`configs/adapt/*.yaml` — verificando anche che `lora` e `dora` dichiarino gli
stessi rango, alpha e placement: la parità di budget di E3-X è un'assunzione di
questo profilo, e un profilo che la assume senza controllarla misurerebbe due
cose diverse chiamandole uguali.

Una modifica al profiler PEFT: `E0b2Config.objective`. La gamba PEFT gira con
**`objective: binary` e `n_classes: 1`**, cioè BCE su un logit — il loop che i
run veri eseguiranno (ADR-031 punto 4). Sul costo pesa quanto la testa, cioè
quasi nulla, ma E0b.2 è scritta sul principio «si misura il `Trainer`, non una
sua copia», e un profilo che ottimizza una loss che nessuno userà è una copia.

**Frozen:** 12 celle — 384 px × AMP BF16 × TF32 on × batch {1,2,4,8,16,32} ×
layer_set {`final`, `hierarchical` [5,11,17,23]}, 10 warmup e 50 iterazioni con
sincronizzazione CUDA, clip sintetiche già residenti sulla GPU. **12/12 ok, zero
OOM. Misura l'encoder**, non decoding, I/O o scrittura.
**PEFT:** 15 celle — le tre configurazioni fissate (`lora_r32_qkvomlp`,
`dora_r32_qkvomlp`, `vpt_deep_n576`) × batch {1,2,4,8,16}, passo completo
(forward, backward, `AdamW`) attraverso `Trainer.fit`, 2 warmup e 8 iterazioni.
**9 ok, 3 OOM, 3 saltate dopo l'OOM.**

### Cosa è uscito

**Forward congelato** (VRAM = picco della cella; il throughput satura a batch 16):

| batch | fin/s `final` | ms/finestra | VRAM GiB `final` / `hierarchical` |
|---:|---:|---:|---:|
| 1 | 8,25 | 121,25 | 1,45 / 1,50 |
| 4 | 9,06 | 110,33 | 2,29 / 2,50 |
| 8 | 9,25 | 108,11 | 3,42 / 3,84 |
| **16** | **9,33** | **107,14** | 5,67 / 6,51 |
| 32 | 9,33 | 107,21 | 10,17 / 11,86 |

Estrarre quattro layer invece dell'ultimo costa **+0,6%** di latenza mediana; il
prezzo resta lo storage. Nessun OOM a batch 32: su 80 GB il forward congelato
**non è limitato dalla memoria** nemmeno a 9.216 token.

**Passo completo, per famiglia** (AMP BF16 + TF32; `n_trainable_adapter` verificato):

| variante | par. adattatore | max batch | fin/s @ b=2 | ms/passo @ b=2 | VRAM GiB @ b=2 | primo OOM |
|---|---:|---:|---:|---:|---:|---:|
| **VPT-deep n=576** | 14.155.776 | **8** | **2,83** | 706 | 15,47 | 16 |
| **LoRA r=32** | 14.155.776 | **4** | 2,12 | 944 | 24,36 | 8 |
| **DoRA r=32** | 14.376.960 | **2** | 1,60 | 1252 | 40,77 | 4 |

**Le passate, sullo split sigillato** (90.432 chunk di training, 13.551 di
validazione, 9.136 di valutazione — ricontati con `enumerate_chunks`, cioè il
codice dei run):

| | forward congelato | VPT-deep | LoRA | DoRA |
|---|---:|---:|---:|---:|
| training (90.432) | **2,69 h** | 8,9 h | 11,9 h | 15,7 h |
| validazione (13.551) | 24,2 min | — | — | — |
| valutazione (9.136) | **16,3 min** | — | — | — |

### Cosa vuol dire

**L'estrapolazione sbagliava, e sbagliava per difetto.** ADR-034 stimava ~7,3
finestre/s scalando i 23,47 fin/s di E0b.1 con un blend lineare/quadratico; la
misura dà **9,33**, cioè il 28% in più. Il rapporto sul tempo è **2,51×** contro
i 4,5× sui token — l'attenzione quadratica non domina ancora a 9.216 token su
A100. Conseguenza pratica: una passata di estrazione sullo split di valutazione
costa **16 minuti**, non 21.

**Il trade-off centrale della tesi (C7) si vede già nel costo, e va nella
direzione attesa.** A parità di parametri addestrabili — 14.155.776 esatti per
VPT e LoRA, per costruzione (ADR-033 §4) — **VPT-deep è la più economica da
addestrare** (2,83 fin/s) nonostante allunghi la sequenza del 6,25%, e **DoRA la
più cara** (1,60 fin/s, il 43% in meno di VPT) con **2,6× la VRAM di LoRA a
parità di batch**: la norma per riga che DoRA materializza si paga in memoria
prima che in tempo. Il lato *inferenza* del trade-off è opposto e non è misurato
qui — è E5b, dove LoRA e DoRA si fondono nei pesi e VPT no.

**Il micro-batch massimo lo fissa DoRA, ed è 2.** È il vincolo che dimensiona
tutto: sotto parità di budget (invariante 5) ciò che deve coincidere è il batch
**effettivo**, non il micro-batch, quindi ogni famiglia può usare il proprio
massimo e compensare con `grad_accum` — ma il numero di passi che entrano in una
fetta di coda da 7 ore lo detta comunque la famiglia più lenta.

**Il tasso di positivi della validazione non è quello della valutazione.** Sulla
griglia statica: **12,86%** in validazione contro **16,35%** in valutazione (in
training 10,73% uniforme, 22,78% coi pesi di campionamento). Non è un difetto del
ritaglio — la validazione è un campione dei video di *training*, che sono meno
ricchi di eventi dell'`Anomaly_Test` ufficiale — ma la soglia τ si stima lì al
quantile 0,95, e un quantile è sensibile alla frequenza di base. Va dichiarato
accanto ai numeri a soglia fissa, e l'analisi di sensibilità già prevista su
`[0,80 · 0,90 · 0,95 · 0,975 · 0,99]` è il modo in cui si misura quanto pesa.


### E0b.3 su H100 — il rapporto vero è **1,7×**, non 2,7×

**Eseguita il:** 28 agosto 2026, subito dopo · **run_id:**
`e0b3_ucf_vjepa21l_frozen-none_s1__h100` (job **539468**, 7 min 57 s) e
`e0b3_synth_vjepa21l_peft-edp_s1__h100` (job **539470**, 1 min 25 s) ·
**commit:** `feff837` · **macchina:** NVIDIA H100 80GB HBM3, `aiq` · stessa
griglia, stesso regime, stesso codice.

**Perché è stata rifatta.** Dal 28 agosto ogni job del progetto si prova prima su
`aiq` (richiesta di Antonio). Tutta l'aritmetica del budget era però misurata su
A100, e il rapporto H100/A100 **sul nostro carico** non era mai stato preso: i
2,73× e 2,86× che giravano nei documenti vengono da `smoke_torch`, un
microbenchmark.

| | A100 | H100 | rapporto |
|---|---:|---:|---:|
| forward congelato, batch 16 | 9,33 fin/s | **16,55** | **1,77×** |
| forward congelato, batch 1 | 8,25 | 13,90 | 1,69× |
| VPT-deep n=576, batch 2 | 2,83 | **5,01** | 1,77× |
| LoRA r=32, batch 2 | 2,12 | **3,62** | 1,71× |
| DoRA r=32, batch 2 | 1,60 | **2,68** | 1,68× |
| passata di valutazione (9.136 chunk) | 16,3 min | **9,2 min** | 1,77× |

**Cosa vuol dire.** **(a)** Il microbenchmark sovrastimava del **55%**: è la terza
volta che un rapporto preso fuori dal modello non si trasferisce dentro (TF32
valeva 7,28× sul GEMM e 1,92–2,61× sul ViT). Il numero da usare per
dimensionare è **1,7×**. **(b)** **La VRAM è identica al byte** — stessi 80 GB,
stessi picchi, stesse frontiere di OOM — quindi `batch_size: 2` non cambia: il
micro-batch massimo lo fissa DoRA anche qui. **(c)** Il fatto operativo: a 3000
passi **ogni run sta in una fetta da 7 ore** su `aiq` (VPT 3,0 h, LoRA 4,1 h,
DoRA 5,4 h), quindi le catene `afterany` che il budget A100 imponeva non
servono, e la campagna scende da ~110 h-A100 a **~79 h-H100**.

### Decisioni prese

- **Il regime numerico non è più un asse.** AMP BF16 + TF32 + SDPA per tutto:
  E0b.1 ha già misurato i rapporti fp32/TF32/BF16 e rifarli qui avrebbe
  raddoppiato la coda per un numero pubblicato.
- **La cache lato training resta inesistente**, e la misura non la riapre: 90.432
  chunk × 18,0 MiB fanno **1,59 TB** per un layer, su un tetto di ~2,3 TB.
- **L'estrazione dello store di valutazione è dimensionata**: 172 GB, un layer,
  ~16 minuti di A100 più decoding e I/O, che questo profilo non misura.

### Alimenta

`configs/train/probe_edp.yaml` (`max_steps`, `batch_size`, `grad_accum`), la
tabella di efficienza E5b, il dimensionamento di E0d e la pianificazione di
F7/F8.

### Aperto

- **I tre `???` di `probe_edp.yaml` non sono stati sostituiti.** `batch_size` e
  `grad_accum` seguono dalla misura, ma `max_steps` è un **budget**, cioè quante
  ore di A100 la campagna E3 può costare: è una decisione di Antonio (`CLAUDE.md`
  §6), non un numero che questo profilo può prendere da sé. La conversione è
  scritta qui sopra e basta moltiplicare.
- **Una sola A100 e una sola esecuzione per cella**: i tempi dimensionano, non
  sono ancora la tabella di efficienza di E5b.
- **Le celle OOM fissano la frontiera della griglia provata**, non il massimo
  batch teorico fra due valori adiacenti: fra 4 e 8 per LoRA, fra 2 e 4 per DoRA,
  fra 8 e 16 per VPT c'è un valore non misurato.
- **I pixel sono sintetici e la pipeline dati è fuori dalla misura**, come in
  E0b.1/E0b.2: decoding e I/O restano una domanda di E0d.

## E-proiezioni — Le rappresentazioni dei chunk in due dimensioni *(chiuso: si raggruppano per VIDEO)*

**21 settembre 2026**, ADR-070. Quattro celle, il quadrato congelato/adattato x
UCF-Crime/MIVIA-Crime, sulla sola partizione di **test**. Un descrittore e' la
media dei 9216 token del layer 23, cioe' 1024 numeri per finestra. Pesi:
`e2d_ucf_vjepa21l_frozen-attentive_s42` (solo il backbone — il descrittore non
dipende dalla testa) e `e3r_ucf_vjepa21l_dora-r32-qkvomlp_s42`, pesi
**selezionati** al passo 5520.

| Cella | Sorgente | Finestre | Positive | Tempo |
|---|---|---|---|---|
| `congelato_ucf` | feature store, nessuna GPU | 9116 | 1491 | 2 min su `fatq` ⚠️ |
| `dora_ucf` | forward | 9116 | 1491 | 24 min 43 s su H100 |
| `dora_mivia` | forward | 3485 | 1068 | 14 min 47 s |
| `congelato_mivia` | forward | 3485 | 1068 | 13 min 36 s |

### Il risultato

**Le rappresentazioni di chunk si organizzano per VIDEO, non per anomalia, e
l'adattamento non lo cambia.** Su 12.601 finestre e 560 video per spazio:

| | congelato | DoRA r=32 |
|---|---|---|
| silhouette in alta dimensione, **etichetta** | +0,027 | +0,038 |
| silhouette in alta dimensione, **video** | **+0,623** | **+0,623** |
| PCA, prime due componenti | 22,4% | 21,3% |
| PCA, 50 componenti | 84,0% | 82,3% |

Il rapporto e' **ventitre' a uno**, ed e' *identico* nei due spazi: video dello
stesso sfondo, camera e compressione stanno vicini prima che per contenuto, e
DoRA non sposta questo di una cifra. Si vede a occhio nel quarto pannello di ogni
figura, dove ogni filamento e ogni blob compatto e' **un video**.

Fedelta' delle proiezioni (trustworthiness k=10, sottocampione 5000):
**t-SNE 0,995-0,997** > UMAP 0,961-0,980 > PCA 0,903-0,916. t-SNE riporta in 2D
anche la struttura per video (silhouette-video fino a +0,66); la PCA la distrugge
(+0,014 sul congelato, -0,118 sull'adattato).

### Limiti dichiarati

⚠️ **Il controllo di aggancio kNN NON funziona e i suoi numeri non si usano.** Da'
0,9742 (`congelato_ucf`) e 0,9779 (`dora_ucf`), sopra sia la probe lineare E2c
(0,8473) sia l'attentive E2d (0,9385). Non e' una scoperta: i vicini di una
finestra sono quasi tutti finestre **dello stesso video**, dove l'etichetta e'
quasi sempre la stessa, quindi il kNN misura lo stesso confondimento che la
silhouette denuncia. La versione corretta esclude i vicini co-video e **non e'
stata eseguita**.

⚠️ **Asimmetria numerica fra le due meta' dello spazio congelato**: `congelato_ucf`
viene da token salvati in **fp16** dallo store, `congelato_mivia` da un forward
fresco in bf16. E0d ha misurato rel_L2 1,8e-4 per la quantizzazione, e la media su
9216 token la abbassa ancora; `--sorgente forward` la elimina al prezzo di ~25
minuti di H100.

⚠️ **I 2 minuti della cella 0 non sono il costo a freddo**: lo stesso lavoro sullo
stesso nodo era costato **21 minuti** al tentativo precedente. La spiegazione
plausibile e' la page cache e **non e' stata misurata**.

⚠️ **Un seme solo** (42), come il resto dei confronti di settembre.

### Alimenta

La discussione di C7 (dove vive il guadagno del reparam) e la lettura di E-MIVIA:
lo shift di dominio si vede nel pannello `dataset`, ma va letto sapendo che la
geometria dominante e' quella dei video.

### Aperto

Il kNN leave-one-**video**-out; e quale dei tre metodi portare in tesi, che e' una
decisione di Antonio davanti alle 18 figure.

Artefatti: `runs/analisi_proiezioni_20260921/` (descrittori, `proiezioni.npz`,
`diagnostiche.json`, `figure/`, `LEGGIMI.md`).

---

## Template di voce

```markdown
## EXX — Titolo del blocco

**Chiuso il:** · **run_id:** da `<prefisso>` a `<prefisso>` (`runs/registry.jsonl`) ·
**commit:** `<sha>` · **manifest:** `<sha256>` · **macchina:** A100 `gpuq` / H100 `aiq` / CPU

### Cosa chiedeva
La domanda a cui il blocco doveva rispondere, e a quale claim (C1–C7) serve.

### Come è stato eseguito
Config, sweep dichiarato, budget, precisione (fp32/TF32 — e dichiarata), numero di run,
tempo di calcolo consumato. Le deviazioni dal protocollo, se ce ne sono state, con il perché.

### Cosa è uscito
I numeri, con le barre d'errore. Tabella breve; il resto sta nel registry e le figure le
genera `make_figures.py`.

### Cosa vuol dire
L'interpretazione: la parte che a settembre non si ricostruisce dai numeri. Cosa conferma,
cosa smentisce, cosa era atteso e cosa no.

### Decisioni prese
Le scelte fatte guardando questi risultati (mini-ADR), e cosa vincolano d'ora in poi.

### Alimenta
Quale sezione della tesi, quale figura, quale tabella.

### Aperto
Cosa resta da capire, e cosa servirebbe per capirlo.
```


## E-VTCD-SPATIAL ? Campagna spaziale UCF, avviata il 9 ottobre 2026

**Stato: in corso.** Autorizzata da Antonio dopo il piano del 9 ottobre. Solo VPT-shallow n=8 su LoRA r=32, step selezionato 12880. GT Forma: 140 video, 9.002 piani, 131.428 frame confermati, originale conservato. Cinque fold da 28 video; due video Abuse separati.

Il pilota reale ha riprodotto dodici finestre e verificato raster e un fit piccolo convergente. Sette VTCD esistenti hanno gia' tabelle numeriche; le nuove ablation e le repliche restano nella stessa campagna. Dodici test CPU mirati passati; suite completa non rieseguita. Nessun numero del detector cambia.

Archivio distinto `runs/valutazione_spaziale/vtcd-spatial-ucf-20261009/`; root cluster `~/evd/experiments/20261009-vtcd-spatial`, supervisore cron su lnode01. Metriche e log parziali sono consultabili; convergenza, conteggi, figure e rientro richiedono la ricevuta finale. Nessun rendering previsto. [Piano](varie/piano_ablation_vtcd_localizzazione_20261009.md), [avvio e artefatti](varie/avvio_vtcd_spatial_20261009.md), ADR-079.


## Chiusura verificata ? 9 ottobre 2026

Campagna completa: 25 configurazioni, 126 fit convergenti, 140 video e 9.002 piani. Metriche ricalcolate dai conteggi con scarto massimo 0,0. Dodici test CPU mirati passati. Nessun rendering.

Archivio di 308.218.643 byte, 15.747 file verificati, SHA256 `493c62282ef91827b47691c27213f8302049ec7cd743e97f49bf14b61f77d025`. La ricevuta locale e quella del cluster riportano `status=complete`, `local_pull=verified`.

Pagina consultabile: http://127.0.0.1:8768/dashboard.html. Sono disponibili tabelle globali/per classe/per video e istanze, selettori e fold, conteggi, predizioni, dizionari, baseline, differenze appaiate, strati, oracoli diagnostici, figura e log. Il massimo osservato fra le configurazioni principali s42 e' il globale con normali, R32 K16: macro-IoU pixel per classe/video 0,398539; micro-IoU per piano 0,508895, su maschere e unione di massimo due concetti, routing noto. E' un massimo esplorativo, non una stima indipendente del vincitore.


### Estensione richiesta: unione bbox, tutte le metriche e MIVIA

La IoU della forma boxes e' gia' l'IoU delle unioni, senza ID. Il 39,85% iniziale era della maschera; il massimo bbox nelle configurazioni principali s42 e' 36,3968% macro, 46,2012% micro per piano. La dashboard presenta tutte le metriche disponibili. GT esterna MIVIA congelata: 39 video, tre per classe, 678 piani, 9.051 frame; nessun tuning o fold MIVIA. Nuovi algoritmi bbox da discutere prima del lancio; proposta e condizioni in [estensione](varie/estensione_vtcd_box_mivia_discussione.md). Nessuna valutazione MIVIA nuova ancora eseguita.

## E-VTCD-SELECTED — Campagna ampliata e selezionata, 10 ottobre 2026

**Stato: in corso.** 114 configurazioni principali, 744 fit unici concretizzati, 19.148 chunk di riferimento training e 31.389 output chunk/R; riusati 4.594 output. Solo VPT-shallow n=8 su LoRA r=32 s42, checkpoint selezionato step 12880. I 25 assetti storici sono riportati separatamente. UCF: 140 video/9.002 piani; MIVIA: 39 video/678 piani esterni. Primaria IoU bbox-unione; fold annidati e tutte le metriche. Repliche e bbox dopo la selezione; MIVIA con parametri congelati. Nessun rendering.

Preflight 787606 completo, replay A100 787612 completo su dodici finestre, errore zero sui descrittori e logit. Diciotto test mirati passati. Supervisore isolato su lnode02, cinque job fatq e due gpuq massimi entro otto job utente; il training E3G dell'altro agente è preservato. Corretti i problemi di compatibilità dei manifest storici e di sincronizzazione delle ricevute; log conservati. Nessun nuovo risultato di qualità viene anticipato. La chiusura richiede convergenze, audit, archivio e rientro locale verificati. [Specifica](varie/vtcd_sottoinsieme_significativo_20261010.md), [avvio](varie/avvio_vtcd_selected_20261010.md), ADR-080.
