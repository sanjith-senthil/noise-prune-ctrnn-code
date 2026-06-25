# Pruning Benchmark

This package contains the code needed to train and prune the CTRNNs used in the
Mod-Cog pruning paper.

Included methods:
- L-NP mask and L-NP rescale
- S-NP mask and S-NP rescale
- capped L-NP/S-NP rescale
- magnitude pruning
- random pruning
- OBS-compensated pruning

The public release is intentionally CTRNN- and Mod-Cog-focused. Earlier
exploratory methods such as OP-NP, WANDA, WoodFisher, trajectory-preservation
analyses, JSE analyses, and alternate RNN architectures are not part of this
release.

Run a suite with:

```bash
python -m pruning_benchmark --mode suite --config configs/task_preservation_tanh_h512_modcog_revised8_12k_seqbest_revised_task_only_p50_80.json
```
