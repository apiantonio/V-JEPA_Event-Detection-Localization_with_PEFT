# Chunk score rankings

Each run contains ranking_chunks.csv and four separate CSV files:
ranking_chunks_TP.csv, ranking_chunks_TN.csv, ranking_chunks_FP.csv,
ranking_chunks_FN.csv. Test only, normal videos included: they also serve to
analyse TN and FP. The threshold is the original one of the run, estimated on
validation. A score equal to the threshold is a positive prediction.

TP: detected anomalous chunks, decreasing score (best positives first).
TN: negative chunks correctly rejected, increasing score (best negatives first).
FP: flagged negative chunks, decreasing score (clearest errors first).
FN: missed anomalous chunks, increasing score (clearest errors first).
The single CSV concatenates the TP, TN, FP, FN groups. posizione_esito restarts
from 1 in each group; ties are ordered by video_id, frame and window_id.

video_id and start_frame/end_frame identify the sequence, with inclusive bounds
in native frames. class_idx is the chunk annotation, -1 on negatives; it is not
a predicted class. video_anomalo also marks the negative chunks inside an
anomalous video. Scores are copied from scores.jsonl, without rounding.

The ranking points to segments to inspect; it does not measure the quality of a
whole video on its own: a video can appear in several groups. Scores are not
calibrated probabilities and rankings are read within a single run.
Recomposed counts and P/R/F1 match metrics.json. Hashes and ordering are in
ranking_chunks_provenienza.json. The sources are not modified.

Regenerate from the repository root:
python scripts/classifiche_modelli.py --solo-ranking-chunk
