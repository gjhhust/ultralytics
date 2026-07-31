# YOLOFT for XS-VID v2

This repository contains the legacy YOLOFT runtime used for the XS-VID v2
benchmark release. It supports the Detection, multiple-object tracking (MOT),
and single-object tracking (SOT) tracks. The released task checkpoints share a
temporal YOLOFT backbone and use separate Detection, MOT, and SOT heads.

The canonical dataset annotations, released checkpoints, and project page are
available from [XS-VID](https://gjhhust.github.io/XS-VID/). The implementation
is based on the [Ultralytics](https://github.com/ultralytics/ultralytics)
codebase and includes the operators required by the original YOLOFT model.

## Installation

The released checkpoints were validated with Linux, CUDA, and PyTorch 1.12.1.
Other recent PyTorch/CUDA combinations may work, but are not part of the
reproduction target.

```bash
conda create -n yoloft python=3.8 -y
conda activate yoloft
pip install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
git clone https://github.com/gjhhust/ultralytics.git yoloft
cd yoloft
git checkout xsvid-v2-legacy-release
pip install -e ".[xsvid]"
```

Build the two CUDA extensions once in the same environment:

```bash
cd ultralytics/nn/modules/ops_dcnv3 && python setup.py build install
cd ../alt_cuda_corr_sparse && python setup.py build install
cd ../../../..
```

Check that the three released task checkpoints load against this checkout:

```bash
python tools/xsvid/smoke_checkpoints.py \
  --model-package /path/to/YOLOFT-XSVID-v2 \
  --output /tmp/xsvid_checkpoint_smoke.json
```

## Dataset preparation

Download and extract the XS-VID v2 media and annotations into one directory.
The resulting root contains `images/` and `annotations/`; the distributed
release also includes portable `labels/` and `splits/` views. The canonical
`annotations/train.json` and `annotations/test.json` are the source of truth.
The dataset repository documents integrity checks and the optional development
split.

Create the small data YAML bridge expected by the legacy Detection loader:

```bash
python tools/xsvid/prepare_legacy_layout.py \
  --data-root /path/to/XS-VID-v2 \
  --detection-protocol paper
```

Use `--detection-protocol canonical` for new Detection benchmark results. The
`paper` view is provided solely for exact reproduction of the submitted
Detection table.

## Detection (VID)

Run temporal YOLOFT-L on the complete benchmark test partition:

```bash
python tools/xsvid/eval_vid.py \
  --config /path/to/YOLOFT-XSVID-v2/configs/yoloft-l-temporal.yaml \
  --weights /path/to/YOLOFT-XSVID-v2/checkpoints/yoloft-l-xsvid-v2-paper.pt \
  --data /path/to/XS-VID-v2/legacy_yoloft.yaml \
  --cfg config/train/default.yaml \
  --project /path/to/results --name vid_test --device 0

python tools/xsvid/score_detection.py \
  --gt /path/to/XS-VID-v2/annotations/protocols/paper_detection_test.json \
  --pred /path/to/results/vid_test/predictions.json \
  --output /path/to/results/vid_test/metrics.json
```

The expected Detection AP is 29.27 on the paper compatibility view and 29.33
on canonical `annotations/test.json`; both round to the 29.3 AP reported in
the paper.

## Multiple-object tracking (MOT)

MOT first extracts appearance embeddings from the released MOT checkpoint,
then associates the Detection predictions with the bundled BoT-SORT pipeline.
The embedding extraction is sharded by video and can be run in parallel.

```bash
python dump_omni_emb.py \
  --data-root /path/to/XS-VID-v2 \
  --annotation /path/to/XS-VID-v2/annotations/test.json \
  --det-json /path/to/results/vid_test/predictions.json \
  --config /path/to/YOLOFT-XSVID-v2/configs/yoloft-l-temporal.yaml \
  --ckpt /path/to/YOLOFT-XSVID-v2/checkpoints/yoloft-l-xsvid-v2-mot.pt \
  --out /path/to/results/mot/shard_0.pkl \
  --shard 0/3 --det-floor 0.001 --roi-grid 5 --gpu 0

# Run the equivalent command for --shard 1/3 and 2/3, then merge.
python tools/xsvid/merge_omni_dumps.py \
  --inputs /path/to/results/mot/shard_0.pkl \
           /path/to/results/mot/shard_1.pkl \
           /path/to/results/mot/shard_2.pkl \
  --output /path/to/results/mot/merged.pkl

python tools/xsvid/track_botsort.py \
  --data-root /path/to/XS-VID-v2 \
  --annotation /path/to/XS-VID-v2/annotations/test.json \
  --dump /path/to/results/mot/merged.pkl \
  --output /path/to/results/mot/predictions.json

python tools/xsvid/normalize_tao_tracks.py \
  --input /path/to/results/mot/predictions.json \
  --output /path/to/results/mot/predictions.json

python tools/xsvid/score_tao.py \
  --data-root /path/to/XS-VID-v2 \
  --pred /path/to/results/mot/predictions.json \
  --track-field paper_track_id \
  --output /path/to/results/mot/metrics.json
```

For new MOT results, use the canonical `track_id` field instead. The sparse
`paper_track_id` compatibility field falls back to `track_id` and is retained
only for reproducing the submitted paper protocol.

## Single-object tracking (SOT)

Run all 272 XS-VID SOT sequences and score the results:

```bash
python eval_sot_transt.py \
  --data-root /path/to/XS-VID-v2 \
  --out-root /path/to/results/sot \
  --config /path/to/YOLOFT-XSVID-v2/configs/yoloft-l-temporal.yaml \
  --ckpt /path/to/YOLOFT-XSVID-v2/checkpoints/yoloft-l-xsvid-v2-sot.pt \
  --param yoloft_sot --feat-mode fuse918 --gpu 0

python tools/xsvid/score_sot.py \
  --data-root /path/to/XS-VID-v2 \
  --predictions /path/to/results/sot/yoloft_sot \
  --output /path/to/results/sot/metrics.json
```

The released checkpoint reproduces 57.09 success AUC and 79.35 precision at
20 pixels on the 272 test sequences.

## Training task heads

The Detection checkpoint initializes the shared temporal backbone. The MOT and
SOT commands below freeze that backbone and fine-tune only their task-specific
heads for ten epochs. They are independent extensions, not a jointly trained
single checkpoint.

Prepare lightweight MOT and SOT training views from the canonical training
annotations once:

```bash
python tools/xsvid/prepare_unified_train_data.py \
  --data-root /path/to/XS-VID-v2
```

Fine-tune the MOT embedding head:

```bash
python train_omni_embed.py \
  --data-root /path/to/XS-VID-v2 \
  --det-ckpt /path/to/YOLOFT-XSVID-v2/checkpoints/yoloft-l-xsvid-v2-paper.pt \
  --config /path/to/YOLOFT-XSVID-v2/configs/yoloft-l-temporal.yaml \
  --out-dir /path/to/results/mot_train \
  --tag yoloft_mot --epochs 10 --emb-stride 4 --embed-dim 256 --gpu 0
```

Fine-tune the SOT head:

```bash
python train_sot_transt.py \
  --data-root /path/to/XS-VID-v2 \
  --det-ckpt /path/to/YOLOFT-XSVID-v2/checkpoints/yoloft-l-xsvid-v2-paper.pt \
  --transt-pretrain /path/to/YOLOFT-XSVID-v2/checkpoints/transt.pth \
  --config /path/to/YOLOFT-XSVID-v2/configs/yoloft-l-temporal.yaml \
  --out-dir /path/to/results/sot_train \
  --tag yoloft_sot --epochs 10 --feat-mode fuse918 --gpu 0
```

## Citation

```bibtex
@article{guo2024xsvid,
  title={XS-VID: An Extremely Small Video Object Detection Dataset},
  author={Guo, Jiahao and Xu, Ziyang and Wu, Lianjun and Gao, Fei and Liu, Wenyu and Wang, Xinggang},
  journal={arXiv preprint arXiv:2407.18137},
  year={2024}
}
```

## License and acknowledgements

The code retains the upstream Ultralytics license. Third-party BoT-SORT code
and its licenses are preserved under `third_party/boxmot/`. Please cite the
respective projects when using those components.
