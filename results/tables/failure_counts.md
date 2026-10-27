# Errors at score >= 0.5 on DAWN sand test (IoU >= 0.5, same class)

A deployed detector uses one score threshold; these counts show what it would do at 0.5. Missed objects are broken down by COCO size (small < 32^2 px, medium < 96^2 px).

| Run | Correct | False detections | Missed | Missed small / medium / large | Recall | Precision |
|---|---|---|---|---|---|---|
| FRCNN-B0 | 1233 | 402 | 431 | 152 / 211 / 68 | 0.74 | 0.75 |
| FRCNN-B1 | 993 | 225 | 671 | 193 / 326 / 152 | 0.60 | 0.82 |
| FRCNN-A | 1093 | 403 | 571 | 152 / 302 / 117 | 0.66 | 0.73 |
| FRCNN-C | 1106 | 304 | 558 | 159 / 271 / 128 | 0.66 | 0.78 |
| FRCNN-B-cc | 1186 | 404 | 478 | 130 / 235 / 113 | 0.71 | 0.75 |
| FRCNN-B-dcp | 1044 | 260 | 620 | 177 / 303 / 140 | 0.63 | 0.80 |
| RTDETR-B0 | 858 | 103 | 806 | 256 / 389 / 161 | 0.52 | 0.89 |
| RTDETR-B1 | 1149 | 318 | 515 | 149 / 252 / 114 | 0.69 | 0.78 |
| RTDETR-A | 1034 | 279 | 630 | 174 / 315 / 141 | 0.62 | 0.79 |
| RTDETR-C | 1189 | 338 | 475 | 135 / 237 / 103 | 0.71 | 0.78 |
| RTDETR-B-cc | 1216 | 402 | 448 | 111 / 240 / 97 | 0.73 | 0.75 |
| RTDETR-B-dcp | 1146 | 335 | 518 | 145 / 255 / 118 | 0.69 | 0.77 |
