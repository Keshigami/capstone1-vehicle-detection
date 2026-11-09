# Results

Quick mode on an M4 Max (CPU), 4,500 images, 11 epochs, 307 s:

| metric | value | what it means |
|---|---|---|
| val accuracy | 73.0% | majority-class baseline 58.7%; the notebook reported 64.5% |
| val macro F1 | 0.47 | rare classes drag it down; bus 0.90, car 0.86, pickup 0.48 recall |
| mean box IoU | 0.245 | mean-box baseline 0.086; a real detector would be 0.6+ |
| boxes with IoU >= 0.5 | 11% | the box is on the right vehicle but loose |
| best-match IoU | 0.249 | almost the same, so misses are imprecision, not another vehicle |
| box MAE | 33 px at 224 | notebook: 202 px |

Part 2: 294 fatal events, 353 deaths, 2013 to January 2023; 73% in the USA, California 91 events;
2022 the peak with 93 events; Autopilot claimed in 35 events (38 deaths), verified in 16 (19
deaths, 5.4% of all deaths), 27 deaths once NHTSA SGO reports are added; model known for only 38%
of events.
