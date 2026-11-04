# Detection comparisons

The main README presents metrics as rows, separately for the EDP-v2 8-second chunk
protocol and the OVAD row protocol. `results.json` retains unrounded thesis values,
run IDs, source receipt hashes and the supplied SOTA values used to generate the tables.

`sources/edp8-*.md` are the laboratory's supplied 8-second comparison tables.
`sources/ovad-supplied.md` is the supplied seven-method OVAD evaluation.
These are results supplied by the laboratory, not reruns performed in this repository.
The original supplied source documents are retained without translation or changed numbers.

OVAD uses different input-window durations (30 s for supplied SOTA, 8 s for thesis models),
and the supplied EDP event recall is not known to use the same event definition as our
EDP-v2 implementation. The README describes these limits alongside the tables.
Rows with different protocols or reporting versions must not be combined.
