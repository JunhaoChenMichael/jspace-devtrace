# Metacognitive Alignment M0 Baseline Gate

**Decision: GREEN**

## Decoupled reproduction gate

| Signal | Paper AUC | Observed AUC | Delta | |Delta| | Tolerance | Pass |
|---|---:|---:|---:|---:|---:|:---:|
| V | 0.337000 | 0.341540 | +0.004540 | 0.004540 | 0.050000 | yes |
| W_rr | 0.654000 | 0.654549 | +0.000549 | 0.000549 | 0.050000 | yes |

## Condition metrics

| Condition | Episodes | Candidates | V pooled | V within | Yes rate | W_rr pooled | W_rr within |
|---|---:|---:|---:|---:|---:|---:|---:|
| Explicit | 88 | 417 | 0.500988 | 0.486932 | 0.000000 | 0.509048 | 0.515530 |
| Evoked | 75 | 352 | 0.380168 | 0.243778 | 0.000000 | 0.620361 | 0.625889 |
| Decoupled | 68 | 335 | 0.341540 | 0.213480 | 0.000000 | 0.654549 | 0.665441 |
| Compositional | 52 | 261 | 0.326187 | 0.189423 | 0.000000 | 0.546559 | 0.566346 |

## Immutable inputs

- explicit: `/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/explicit.json` (SHA-256 `2189e6e5333d33afdd1b56d9e4a20307ce83c51ba0ed45d13cebf2877a306488`)
- evoked: `/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/evoked.json` (SHA-256 `ffdf2e154bed18a44ab8411bc74d3db72a9557a85aed65db008419f08daad8ed`)
- decoupled: `/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/decoupled.json` (SHA-256 `09179319ad4f255faf88f0863036be7271beb56790e665d77812e8712cbe62a8`)
- compositional: `/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/compositional.json` (SHA-256 `2dbc1faedda196a4248dec6d5efc6c31a0fe94230940efe593fb84f68abd69e7`)

Machine-readable report: `/rodata/azradonc_dev/m253405/jspace-devtrace/data/results/metacog_a100_3seed/seed2/m0/gate.json`

No training or GPU model execution is performed by this gate.
