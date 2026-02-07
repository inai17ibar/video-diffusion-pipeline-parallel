# ベンチマーク実験ガイド

GPU数を変えてパイプライン並列の効果を測定するための実験手順。

## 前提条件

### ハードウェア要件
- Ubuntu 20.04 / 22.04
- NVIDIA GPU x 7 (A5000推奨)
- CUDA 11.8以上

### ソフトウェア要件
```bash
pip install torch diffusers transformers accelerate
```

## クイックスタート

### 1. GPUの確認
```bash
nvidia-smi
```

### 2. 単一GPUでの動作確認
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    -m src.modes.production \
    --total-steps 25 \
    --latent-shape 1 4 14 64 64 \
    --num-samples 1
```

### 3. マルチGPUでの実行（7GPU）
```bash
torchrun --nproc_per_node=7 \
    -m src.modes.production \
    --total-steps 28 \
    --latent-shape 1 4 14 64 64 \
    --num-samples 10
```

## ベンチマークスクリプト

GPU数を変えて性能を比較するスクリプト:

```bash
#!/bin/bash
# benchmark.sh - GPU数を変えてパイプライン並列の効果を測定

TOTAL_STEPS=28  # 1, 2, 4, 7 で割り切れる
NUM_SAMPLES=10
SEED=42
LATENT_SHAPE="1 4 14 64 64"

# 結果を保存
RESULTS_FILE="benchmark_results.csv"
echo "gpu_count,total_time_sec,per_sample_sec" > $RESULTS_FILE

# GPU数を変えて実験 (1, 2, 4, 7)
for NGPUS in 1 2 4 7; do
    echo "=========================================="
    echo "Testing with $NGPUS GPU(s)..."
    echo "=========================================="

    # CUDA_VISIBLE_DEVICESで使用するGPU数を制限
    DEVICES=$(seq -s, 0 $((NGPUS-1)))

    START_TIME=$(date +%s.%N)

    CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
        --nproc_per_node=$NGPUS \
        -m src.modes.production \
        --total-steps $TOTAL_STEPS \
        --latent-shape $LATENT_SHAPE \
        --num-samples $NUM_SAMPLES \
        --seed $SEED

    END_TIME=$(date +%s.%N)
    ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)
    PER_SAMPLE=$(echo "scale=3; $ELAPSED / $NUM_SAMPLES" | bc)

    echo "$NGPUS,$ELAPSED,$PER_SAMPLE" >> $RESULTS_FILE
    echo "Completed: $NGPUS GPUs in ${ELAPSED}s (${PER_SAMPLE}s per sample)"

    sleep 5  # GPU冷却のための待機
done

echo ""
echo "Results saved to $RESULTS_FILE"
cat $RESULTS_FILE
```

### スクリプトの実行
```bash
chmod +x benchmark.sh
./benchmark.sh
```

## 実験パラメータ

### 必須パラメータ

| パラメータ | 説明 | 例 |
|-----------|------|-----|
| `--total-steps` | 拡散ステップ数（GPU数で割り切れる必要あり） | 28 |
| `--latent-shape` | 潜在変数の形状 (B C F H W) | 1 4 14 64 64 |

### オプションパラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `--num-samples` | 生成するサンプル数 | 1 |
| `--seed` | 乱数シード | 0 |
| `--model-id` | HuggingFace モデルID | stabilityai/stable-video-diffusion-img2vid-xt |
| `--fps` | フレームレート（条件付け用） | 6 |
| `--motion-bucket-id` | モーションバケットID (0-255) | 127 |

## 期待される結果

| GPU数 | 各GPUのステップ数 | 期待されるスループット |
|-------|------------------|----------------------|
| 1 | 28 | 基準値 |
| 2 | 14 | 約2倍 |
| 4 | 7 | 約4倍 |
| 7 | 4 | 約7倍 |

**注意**:
- 最初のサンプルはパイプライン充填時間がかかるため、複数サンプル（10以上）で測定することが重要
- 定常状態のスループットで比較すること

## トラブルシューティング

### NCCL初期化エラー
```bash
# 環境変数を設定
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # InfiniBandがない場合
```

### OOM (Out of Memory)
```bash
# バッチサイズを1に、解像度を下げる
--latent-shape 1 4 14 32 32
```

### モデルダウンロードが遅い
```bash
# HuggingFaceキャッシュを事前にダウンロード
python -c "from diffusers.models import UNetSpatioTemporalConditionModel; UNetSpatioTemporalConditionModel.from_pretrained('stabilityai/stable-video-diffusion-img2vid-xt', subfolder='unet')"
```

## シミュレータモード（GPUなし環境）

CPU環境でロジックを確認する場合:

```bash
torchrun --nproc_per_node=4 \
    -m src.modes.simulator \
    --total-steps 28 \
    --device cpu \
    --dtype fp32
```
