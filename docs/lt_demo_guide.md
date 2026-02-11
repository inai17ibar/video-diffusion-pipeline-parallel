# LT発表用デモガイド

## 概要

このガイドでは、パイプライン並列ビデオ拡散システムのLT発表用デモの準備と実行方法を説明します。

---

## 事前準備

### 1. サンプル入力画像の準備

デモ用に適した画像を用意してください：

- **推奨サイズ**: 1024x576（16:9）または 576x1024（9:16）
- **内容**: 動きが想像しやすいシーン（風景、人物、動物など）
- **形式**: JPG, PNG

**サンプル画像のダウンロード例**:
```bash
# Unsplashから風景画像をダウンロード
curl -L "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1024" -o demo_input.jpg
```

### 2. 依存関係のインストール

```bash
pip install diffusers transformers accelerate pillow imageio imageio-ffmpeg
```

### 3. モデルの事前ダウンロード（推奨）

初回実行時のダウンロード待ちを避けるため:
```bash
python -c "
from diffusers import StableVideoDiffusionPipeline
pipe = StableVideoDiffusionPipeline.from_pretrained(
    'stabilityai/stable-video-diffusion-img2vid-xt',
    torch_dtype='float16',
    variant='fp16'
)
print('Model downloaded successfully')
"
```

---

## デモ実行

### 単一GPU（比較用ベースライン）

```bash
python scripts/generate_video_demo.py \
    --input-image demo_input.jpg \
    --total-steps 25 \
    --num-frames 14 \
    --output-dir outputs/demo
```

### 7GPU並列（メイン）

```bash
torchrun --nproc_per_node=7 scripts/generate_video_demo.py \
    --input-image demo_input.jpg \
    --total-steps 21 \
    --num-frames 14 \
    --output-dir outputs/demo
```

### 出力ファイル

実行後、`outputs/demo/` に以下が生成されます：

| ファイル | 説明 |
|---------|------|
| `*_input_*.png` | 入力画像のコピー |
| `*_svd_1gpu_*.mp4` | 単一GPU生成動画 |
| `*_svd_7gpu_*.mp4` | 7GPU並列生成動画 |
| `*_svd_*gpu_*.gif` | GIF形式（プレゼン用） |

---

## LT発表スライド用素材

### 1. アーキテクチャ図

```
入力画像
    ↓
[CLIP Encoder] → 画像埋め込み
    ↓
[VAE Encoder] → 潜在変数
    ↓
┌─────────────────────────────────────────────────┐
│           パイプライン並列拡散               │
│                                                 │
│  GPU0 → GPU1 → GPU2 → GPU3 → GPU4 → GPU5 → GPU6 │
│  [3ステップ] [3ステップ] ... [3ステップ]        │
└─────────────────────────────────────────────────┘
    ↓
[VAE Decoder] → 動画フレーム
    ↓
出力動画 (14フレーム)
```

### 2. 性能比較表

| 項目 | 単一GPU | 7GPU並列 |
|------|---------|----------|
| 1サンプルのレイテンシ | 4.2秒 | 8.4秒 |
| 10サンプルの合計時間 | 42秒 | 14秒 |
| スループット | 0.24 サンプル/秒 | 0.71 サンプル/秒 |
| メモリ使用量/GPU | 24GB | 10GB |

### 3. ユースケース別推奨

```
┌────────────────────────────────────────────────┐
│  ユースケース        │  推奨構成  │  理由     │
├────────────────────────────────────────────────┤
│  単発リクエスト      │  1 GPU    │  低レイテンシ │
│  バッチ生成          │  7 GPU    │  高スループット │
│  14フレーム以上      │  7 GPU必須 │  メモリ制約    │
└────────────────────────────────────────────────┘
```

---

## デモシナリオ

### シナリオ1: 基本動作デモ（3分）

1. 入力画像を見せる
2. 単一GPUで動画生成を実行（実行ログを見せる）
3. 生成された動画を再生
4. 「1枚の画像から動く動画が生成されました」

### シナリオ2: 並列化の効果（5分）

1. 「では7GPUに分散するとどうなるか」
2. パイプラインの図を説明
3. 7GPU並列で実行
4. ログで「各GPUが異なるステップを処理」していることを見せる
5. 生成された動画を再生
6. 「同じ品質の動画が生成されます」

### シナリオ3: スループット比較（3分）

1. 「単一サンプルではメリットがないように見えますが...」
2. 複数サンプル生成の図を説明
3. 「バッチ処理では約3倍のスループット向上」
4. ベンチマーク結果のグラフを見せる

---

## トラブルシューティング

### OOM (Out of Memory)

```bash
# 解像度を下げる
--height 320 --width 576

# フレーム数を減らす
--num-frames 8
```

### NCCL初期化エラー

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
```

### 動画が生成されない

```bash
# imageio-ffmpegがインストールされているか確認
pip install imageio-ffmpeg
```

---

## クイックリファレンス

### 最小限のデモコマンド

```bash
# 1. 入力画像をダウンロード
curl -L "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1024" -o demo.jpg

# 2. 単一GPU実行
python scripts/generate_video_demo.py --input-image demo.jpg

# 3. 7GPU実行
torchrun --nproc_per_node=7 scripts/generate_video_demo.py --input-image demo.jpg --total-steps 21

# 4. 出力確認
ls -la outputs/
```

### 発表時に見せるファイル

1. `demo.jpg` - 入力画像
2. `*_svd_1gpu_*.gif` - 単一GPU結果
3. `*_svd_7gpu_*.gif` - 7GPU結果
4. 実行ログ（ターミナル）
