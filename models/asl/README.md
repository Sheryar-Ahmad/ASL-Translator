# ASL Model Directory

The automatic downloader places downloaded ASL models here:

    models/asl/hf_model/

It also creates:

    models/asl/labels.txt

If automatic download fails, manually place a pretrained ASL ONNX model
in this folder and update config.yaml.

## Landmark model contract

Input shape:

    [1, 63]

or:

    [1, 42]

## Image model contract

Input shape:

    [1, 3, 224, 224]

or:

    [1, 224, 224, 3]

labels.txt line order must match model output class order.
