# AP50-95 per object size: clear BDD val vs real sandstorm (DAWN)

COCO buckets: small < 32^2 px, medium < 96^2 px, large above. The raw BDD-vs-DAWN gap is confounded by object size and image domain; comparing within a size bucket removes the size part.

| Method | Small BDD | Small DAWN | Medium BDD | Medium DAWN | Large BDD | Large DAWN |
|---|---|---|---|---|---|---|
| FRCNN-B0 | 0.120 | 0.207 | 0.356 | 0.334 | 0.583 | 0.654 |
| FRCNN-B1 | 0.150 | 0.149 | 0.411 | 0.252 | 0.606 | 0.466 |
| FRCNN-A | 0.145 | 0.149 | 0.378 | 0.254 | 0.609 | 0.501 |
| FRCNN-C | 0.144 | 0.171 | 0.404 | 0.257 | 0.599 | 0.485 |
| FRCNN-B-cc | 0.143 | 0.176 | 0.407 | 0.287 | 0.596 | 0.514 |
| FRCNN-B-dcp | 0.149 | 0.151 | 0.400 | 0.249 | 0.604 | 0.446 |
| RTDETR-B0 | 0.093 | 0.170 | 0.277 | 0.282 | 0.459 | 0.535 |
| RTDETR-B1 | 0.159 | 0.195 | 0.409 | 0.285 | 0.612 | 0.488 |
| RTDETR-A | 0.150 | 0.191 | 0.406 | 0.282 | 0.621 | 0.519 |
| RTDETR-C | 0.157 | 0.224 | 0.411 | 0.294 | 0.611 | 0.525 |
| RTDETR-B-cc | 0.151 | 0.210 | 0.396 | 0.270 | 0.602 | 0.521 |
| RTDETR-B-dcp | 0.153 | 0.190 | 0.395 | 0.296 | 0.615 | 0.488 |

Size mix of the ground truth (share of boxes small / medium / large):

- BDD clear val: 46% / 37% / 17%
- DAWN sand test: 20% / 48% / 32%
