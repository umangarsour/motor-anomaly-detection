# Before / after (same split: train cycles 1-4, test cycle 5)

| Metric | Original config | Final model | Change |
|---|---|---|---|
| Accuracy | 0.8140 | 0.8087 | -0.0053 |
| Balanced accuracy | 0.5755 | 0.6042 | +0.0288 |
| Macro F1 | 0.5610 | 0.5699 | +0.0090 |
| Weighted F1 | 0.8165 | 0.8154 | -0.0010 |
| Macro precision | 0.5540 | 0.5493 | -0.0047 |
| Macro recall | 0.5755 | 0.6042 | +0.0288 |
| CV macro F1 (GroupKFold 4) | 0.5632 | 0.5724 | +0.0092 |
| CV accuracy | 0.8219 | 0.8174 | -0.0046 |
| OOB score | 0.8212 | 0.8152 | |
| Features | 28 | 27 | |
| Training time (s) | 27.2 | 24.9 | |
| Batch inference, 20,000 rows (s) | 0.26 | 0.10 | |
| Single-row latency, median (ms) | 51.9 | 29.0 | |
| Model size (MB) | 529 (uncompressed, as originally saved) | 34.2 (compressed) | |

| Class | Recall before | Recall after | F1 before | F1 after | Support |
|---|---|---|---|---|---|
| Bearing_Wear | 0.068 | 0.182 | 0.091 | 0.160 | 44 |
| Electrical_Fault | 0.798 | 0.774 | 0.785 | 0.780 | 4951 |
| Misalignment | 0.197 | 0.246 | 0.182 | 0.176 | 61 |
| Multi_Fault | 0.860 | 0.851 | 0.769 | 0.773 | 1517 |
| Normal | 0.879 | 0.870 | 0.871 | 0.870 | 4614 |
| Overheating | 0.405 | 0.485 | 0.369 | 0.371 | 687 |
| Overload | 0.822 | 0.822 | 0.860 | 0.860 | 8126 |
