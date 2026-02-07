# video-diffusion-pipeline-parallel

複数のGPUに**拡散ステップ（時間軸）**を分散させることで、ビデオ拡散モデルの推論を高速化するパイプライン並列システム。

**A5000 × 7 GPU** 環境で、モデル並列やフレーム並列ではなく、**拡散ステップのパイプライン並列**によりビデオ生成を加速します。

---

## モチベーション

ビデオ拡散モデルは、拡散ステップを**順次実行**する必要があるため低速です。

GPU数を増やすだけではレイテンシは改善しません。本プロジェクトでは、**拡散ステップをGPU間でパイプライン化**することで、複数サンプル生成時のスループットを向上させます。

---

## 核となるアイデア

- 各GPUは**同じUNetモデル**を保持
- 拡散ステップをGPU間で分割
- 潜在変数テンソルをステップ順にGPU間で受け渡し
- これは**分散推論**であり、分散学習ではない

```
ノイズ
  ↓ GPU0 (step 0–4)
  ↓ GPU1 (step 5–9)
  ↓ GPU2 (step 10–14)
  ↓ GPU3 (step 15–19)
  ↓ GPU4 (step 20–24)
  ↓ GPU5 (step 25–29)
  ↓ GPU6 (step 30–34)
  → ビデオ
```

---

## アーキテクチャ

```
Rank 0 (Steps 0-4)  →  Rank 1 (Steps 5-9)  →  ...  →  Rank 6 (Steps 30-34)
   ↓                       ↓                              ↓
  UNet実行               UNet実行                       UNet実行
   ↓                       ↓                              ↓
 潜在変数を送信  →    潜在変数を送信  →  ...  →     最終出力
```

---

## 対象モデル

- Stable Video Diffusion (SVD) または同様のUNetベースビデオ拡散モデル
- fp16推論
- 14〜25フレーム、25〜35拡散ステップ

---

## 期待される性能（参考値）

| 構成 | 1サンプルあたりの時間 |
|------|----------------------|
| 単一GPU | 40〜60秒 |
| 7GPU（最初のサンプル） | 約35秒 |
| 7GPU（定常状態） | 8〜12秒 |

> 複数のビデオを連続生成する場合、スループットが大幅に向上します。

---

## 必要環境

### ハードウェア
- Linux (Ubuntu 20.04 / 22.04)
- NVIDIA GPU × 7 (A5000推奨)
- CUDA 11.8以上

### ソフトウェア
- Python 3.9以上
- PyTorch 2.x
- torch.distributed (NCCL)

### 依存関係のインストール
```bash
pip install torch diffusers transformers accelerate
```

---

## プロジェクト構造

```
video-diffusion-pipeline-parallel/
├── README.md                 # 英語版README
├── README_JP.md              # 日本語版README（本ファイル）
├── docs/
│   ├── context.md           # 設計の背景
│   └── benchmark.md         # ベンチマーク実験ガイド
├── src/
│   ├── distributed/
│   │   ├── backend.py       # nccl/gloo選択
│   │   └── setup.py         # torch.distributed初期化
│   ├── models/
│   │   ├── dummy_unet.py    # シミュレータ用軽量モデル
│   │   └── svd_unet.py      # Stable Video Diffusion UNetラッパー
│   ├── modes/
│   │   ├── simulator.py     # CPU/単一GPUテストモード
│   │   └── production.py    # マルチGPU本番モード
│   └── pipeline/
│       ├── pipeline.py      # パイプライン実行エンジン
│       └── step_assignment.py # ランク↔ステップ割り当て
└── LICENSE
```

---

## クイックスタート

### シミュレータモード（CPU環境でのテスト）

```bash
torchrun --nproc_per_node=4 \
    -m src.modes.simulator \
    --total-steps 28 \
    --device cpu \
    --dtype fp32
```

### 本番モード（マルチGPU）

#### 単一GPUでの動作確認
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    -m src.modes.production \
    --total-steps 25 \
    --latent-shape 1 4 14 64 64 \
    --num-samples 1
```

#### 7GPUでの実行
```bash
torchrun --nproc_per_node=7 \
    -m src.modes.production \
    --total-steps 28 \
    --latent-shape 1 4 14 64 64 \
    --num-samples 10
```

---

## ベンチマーク実験

GPU数を変えてパイプライン並列の効果を測定できます。詳細は [docs/benchmark.md](docs/benchmark.md) を参照してください。

### 簡易ベンチマーク

```bash
# GPU数を変えて比較（1, 2, 4, 7 GPU）
for NGPUS in 1 2 4 7; do
    echo "=== Testing with $NGPUS GPU(s) ==="
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPUS-1))) \
    torchrun --nproc_per_node=$NGPUS \
        -m src.modes.production \
        --total-steps 28 \
        --latent-shape 1 4 14 64 64 \
        --num-samples 10
done
```

---

## コマンドライン引数

### シミュレータモード (`src.modes.simulator`)

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--total-steps` | 拡散ステップ数 | 28 |
| `--world-size` | プロセス数 | 1 |
| `--device` | デバイス (cpu, cuda:0, mps) | cpu |
| `--dtype` | データ型 (fp32, fp16, bf16) | fp32 |
| `--backend` | 通信バックエンド (auto, gloo, nccl) | auto |

### 本番モード (`src.modes.production`)

| 引数 | 説明 | デフォルト |
|------|------|-----------|
| `--total-steps` | 拡散ステップ数（必須） | - |
| `--latent-shape` | 潜在変数形状 (B C F H W) | - |
| `--num-samples` | 生成サンプル数 | 1 |
| `--model-id` | HuggingFaceモデルID | stabilityai/stable-video-diffusion-img2vid-xt |
| `--fps` | フレームレート | 6 |
| `--motion-bucket-id` | モーションバケットID (0-255) | 127 |

---

## 開発モード

| モード | バックエンド | 用途 |
|--------|-------------|------|
| シミュレータ | gloo (CPU) | ロジック検証、デバッグ |
| 本番 | nccl (GPU) | 実際の推論、ベンチマーク |

---

## プロジェクトステータス

- [x] パイプライン設計
- [x] ステップ分割プロトタイプ
- [x] diffusers統合 (StableVideoUNet)
- [ ] マルチサンプルパイプライン充填最適化
- [ ] 性能プロファイリング

---

## 注意事項

- `total_steps`は`world_size`（GPU数）で割り切れる必要があります
- 初回実行時にモデルがHuggingFaceからダウンロードされます（約10GB）
- このプロジェクトは**システム設計と推論構造**に焦点を当てており、エンドツーエンドで最適化された本番システムと競合することを意図していません

---

## ライセンス

MIT
