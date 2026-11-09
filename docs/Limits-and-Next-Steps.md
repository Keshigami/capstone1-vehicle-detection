# Limits and next steps

- One box per image, the largest vehicle; the data has up to a dozen vehicles per frame and a
  production model would predict all of them (anchor-based or DETR-style head, 512 px input).
- IoU 0.25 is honest for five minutes of CPU training; `just train-full` adds the remaining images
  and up to 30 fine-tune epochs, which should give a few points, not a new regime.
- Classes with fewer than a dozen validation images (motorcycle, single-unit truck, non-motorized
  vehicle, pedestrian) have near-zero recall; they need oversampling or more data.
- The MobileNetV2 weights download once from Keras on the first `just train`; after that the whole
  project is offline. Tests never touch the network.
- The Tesla sheet counts events with no miles-driven denominator, so it supports "how many" and
  "where", not "is Autopilot safer".
