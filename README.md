# Acne Detection and Cross-Domain Classification

This project has two parts. Part 1 is acne lesion detection on ACNE04 using Faster R-CNN and YOLOv11n. Part 2 is whole-image acne classification on DermNet, where I trained a ResNet-50 on ACNE04 patches and tried to adapt it across domains with mosaic augmentation and histogram matching.

## Datasets

- **ACNE04** — 1450 facial images with bounding box annotations. Downloaded through the Roboflow API. Link: https://universe.roboflow.com/acne-vulgaris-detection/acne04-detection. This is automatically downloaded when running the training calls.
- **DermNet** — over 20 skin condition classes used for cross-domain evaluation. I downloaded this through Kaggle, storing it at `/kaggle/input/datasets/shubhamgoel27/dermnet`. Link: https://www.kaggle.com/datasets/shubhamgoel27/dermnet

## Setup

This project was created on Kaggle for 2x T4 GPUs. You must first clone the repo, install the required libraries, and input your Roboflow API Key before running. 

```bash
!git clone https://github.com/j-iliu/acne04_analysis.git
%cd acne04_analysis
!pip install -r requirements.txt
```

## Part 1 — Detection

### Faster R-CNN

I used a ResNet-50 + FPN backbone and decreased the size of the default anchors `(32, 64, 128, 256, 512)` by half to `(16, 32, 64, 128, 256)` because the lesion annotations had a mean size of ~40 pixels with a standard deviation of ~15. This is how to run the trainer:

```bash
!python train_faster_rcnn.py \
    --name="SGD_faster_rcnn" \
    --optimizer="SGD" \
    --lr=0.001 \
    --steps 10 \
    --epochs=15 \
    --batch_size=4 \
    --patience=20
```

This saves `SGD_faster_rcnn_best.pt` to `/kaggle/working`.

### YOLOv11n

I trained YOLOv11n with the AdamW optimizer, default learning rate of 0.01 lowered to 0.001 with cosine annealing, for 75 epochs with mosaic turned off for the last 15. This is how to run the trainer:

```bash
!python train_yolov11.py \
    --name="yolov11 \
    --epochs=50 \
    --batch_size=96 \
    --patience=30
```

### Evaluation

To compute mAP and the precision-recall, run the cell at the bottom of the notebook with `MODEL_PATH` pointing to your trained checkpoint. It outputs `mAP@50`, `mAP@50-95`, and saves a PR curve PNG to `/kaggle/working`.

## Part 2 — Classification

I trained this in two stages.

### Stage 1 — soft-label mosaic training

Stage 1 trains the model to predict the acne fraction of a 4-tile mosaic as a soft label. This is how you run it:

```bash
!python train_stage1.py \
    --name="mosaic" \
    --batch_size=64 \
    --lr=0.001 \
    --steps 1 7 9 \
    --epochs=13 \
    --optimizer="AdamW"
```

For the VGGFace2-pretrained variant I set the layer-4 learning rate to 0.1 of the inout learning rate to preserve more of the pretraining. To enable histogram matching, add `--use_histogram_matching`. If you use histogram matching for training, you must use it during inference. Patches created are cached upon first run to save time on later runs.

### Stage 2 — hard-label threshold training

Stage 2 trains a small `Linear(2, 2)` head on top of stage 1's frozen output to produce hard binary decisions. To speed this up I cached stage 1's features so each epoch runs in seconds.

```bash
python train_stage2_cached.py \
    --model_path=/kaggle/working/mosaic_best_stage_1.pt \
    --acne_ratio=0.08 \
    --use_histogram \
    --lr=0.01 \
    --epochs=30
```

`--acne_ratio=0.08` trains on mosaics with 8% acne (to roughly match DermNet's acne prevalancy). 
`--acne_ratio=0.5` is the default.
`--use_histogram` enables histogram matching for training.

## Evaluation

Use the evaluate_acne_model to evaluate your model on Acne04 patches and DermNet images.

```bash
!python evaluate_acne_model.py --model_path=MODEL_PATH
```
`--stage1` flag for using a stage1 model. Default evaluation is on a stage 2 model.
`--use_histogram_matching` enables histogram matching for evaluation. Must use if applied during training.


## Files

```
train_faster_rcnn.py             # Part 1 Faster R-CNN training
train_stage1.py                  # Part 2 stage 1 training
train_stage2_cached.py           # Part 2 stage 2 training (cached features)
faster_rcnn_dataset.py           # ACNE04 detection dataset wrapper
classification_dataset.py        # ACNE04 patch + mosaic dataset
domain_transfer_dataset.py       # DermNet acne/non-acne dataset
classification_loaders.py        # data loaders
classification_model_trainer.py  # classification Trainer (DDP)
trainer.py                       # detection Trainer (DDP)
load_models.py                   # ResNet-50 / VGGFace2 loading + stage-2 head
evaluation.py                    # mAP / classification metrics
histogram_matching.py            # histogram matcher + reference CDF builder
```

## Notes

- All training scripts use distributed data parallel across both GPUs. To run on one GPU, set `CUDA_VISIBLE_DEVICES=0` before launching.
- Patch and feature caches are written to `/kaggle/working`. They're reused automatically on re-run, so the first training pass is slow and subsequent ones are fast.
- For histogram matching, the reference CDFs are computed once from 20 DermNet train images (deterministic seed) and cached at `/kaggle/working/ref_cdfs.pkl`.
- When using Faster R-CNN after YOLOv11 or vice versa, you must redo the Robloflow download.
