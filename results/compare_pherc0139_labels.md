
### all (n = 256 slabs of the reference)

| Run | Precision | Recall | Count error | Phase MAE |
|---|---|---|---|---|
| published labels (37 meshes) | 0.440 (0.417 to 0.464) | 0.426 (0.402 to 0.449) | 0.405 (0.370 to 0.442) | 0.196 (0.182 to 0.209) |
| w042 and w046 removed (35) | 0.439 (0.420 to 0.458) | 0.419 (0.400 to 0.439) | 0.405 (0.371 to 0.441) | 0.188 (0.176 to 0.200) |
| renumbered: w042 and w046 removed, w043 and up shifted down by one | 0.399 (0.375 to 0.422) | 0.389 (0.365 to 0.411) | 0.635 (0.577 to 0.696) | 0.241 (0.222 to 0.263) |

### outside w040-w049 (n = 94 slabs of the reference)

| Run | Precision | Recall | Count error | Phase MAE |
|---|---|---|---|---|
| published labels (37 meshes) | 0.588 (0.556 to 0.620) | 0.573 (0.541 to 0.604) | 0.260 (0.223 to 0.299) | 0.116 (0.102 to 0.131) |
| w042 and w046 removed (35) | 0.558 (0.528 to 0.587) | 0.542 (0.512 to 0.571) | 0.281 (0.249 to 0.316) | 0.115 (0.103 to 0.129) |
| renumbered: w042 and w046 removed, w043 and up shifted down by one | 0.571 (0.542 to 0.599) | 0.556 (0.526 to 0.584) | 0.269 (0.240 to 0.298) | 0.112 (0.102 to 0.124) |

### touching w040-w049 (n = 162 slabs of the reference)

| Run | Precision | Recall | Count error | Phase MAE |
|---|---|---|---|---|
| published labels (37 meshes) | 0.375 (0.347 to 0.401) | 0.361 (0.333 to 0.388) | 0.488 (0.444 to 0.535) | 0.242 (0.225 to 0.258) |
| w042 and w046 removed (35) | 0.388 (0.367 to 0.408) | 0.368 (0.346 to 0.389) | 0.475 (0.432 to 0.522) | 0.229 (0.214 to 0.243) |
| renumbered: w042 and w046 removed, w043 and up shifted down by one | 0.323 (0.300 to 0.348) | 0.316 (0.293 to 0.340) | 0.850 (0.775 to 0.930) | 0.318 (0.291 to 0.343) |
