#!/bin/bash
# benchmark_throughput.sh - マルチサンプル生成でスループットを測定
#
# パイプライン並列の本来のメリット（スループット向上）を実証するためのベンチマーク
#
# 使用方法:
#   chmod +x scripts/benchmark_throughput.sh
#   ./scripts/benchmark_throughput.sh
#
# 注意: 7GPUが利用可能な環境で実行してください

set -e

# 設定
TOTAL_STEPS=28          # 1, 2, 4, 7 で割り切れる
NUM_SAMPLES=10          # 複数サンプルでスループットを測定
SEED=42
LATENT_SHAPE="1 4 8 32 32"  # B=1, C=4, F=8, H=32, W=32（メモリ節約のため小さめ）

# 結果ファイル
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${RESULTS_DIR}/throughput_${TIMESTAMP}.csv"

# ディレクトリ作成
mkdir -p $RESULTS_DIR

echo "=============================================="
echo "パイプライン並列スループットベンチマーク"
echo "=============================================="
echo "設定:"
echo "  Total Steps: $TOTAL_STEPS"
echo "  Num Samples: $NUM_SAMPLES"
echo "  Latent Shape: $LATENT_SHAPE"
echo "  結果ファイル: $RESULTS_FILE"
echo "=============================================="
echo ""

# CSVヘッダー
echo "gpu_count,num_samples,total_time_sec,first_sample_sec,steady_state_per_sample_sec,throughput_samples_per_sec" > $RESULTS_FILE

# GPU数を変えて実験 (1, 2, 4, 7)
for NGPUS in 1 2 4 7; do
    echo "=========================================="
    echo "Testing with $NGPUS GPU(s), $NUM_SAMPLES samples..."
    echo "=========================================="

    # 使用するGPUを制限
    DEVICES=$(seq -s, 0 $((NGPUS-1)))

    # ログファイル
    LOG_FILE="${RESULTS_DIR}/log_${NGPUS}gpu_${TIMESTAMP}.txt"

    # 時間計測開始
    START_TIME=$(date +%s.%N)

    # ベンチマーク実行
    CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
        --nproc_per_node=$NGPUS \
        -m src.modes.production \
        --total-steps $TOTAL_STEPS \
        --latent-shape $LATENT_SHAPE \
        --num-samples $NUM_SAMPLES \
        --seed $SEED \
        --enable-memory-opt \
        2>&1 | tee $LOG_FILE

    # 時間計測終了
    END_TIME=$(date +%s.%N)
    TOTAL_TIME=$(echo "$END_TIME - $START_TIME" | bc)

    # メトリクス計算
    # 最初のサンプル時間（概算: 全GPU分の処理時間）
    FIRST_SAMPLE_TIME=$(echo "scale=3; $TOTAL_TIME * $NGPUS / ($NUM_SAMPLES + $NGPUS - 1)" | bc)

    # 定常状態のサンプルあたり時間
    if [ $NUM_SAMPLES -gt 1 ]; then
        STEADY_STATE_PER_SAMPLE=$(echo "scale=3; ($TOTAL_TIME - $FIRST_SAMPLE_TIME) / ($NUM_SAMPLES - 1)" | bc)
    else
        STEADY_STATE_PER_SAMPLE=$TOTAL_TIME
    fi

    # スループット（サンプル/秒）
    THROUGHPUT=$(echo "scale=3; $NUM_SAMPLES / $TOTAL_TIME" | bc)

    # 結果をCSVに記録
    echo "$NGPUS,$NUM_SAMPLES,$TOTAL_TIME,$FIRST_SAMPLE_TIME,$STEADY_STATE_PER_SAMPLE,$THROUGHPUT" >> $RESULTS_FILE

    echo ""
    echo "結果 ($NGPUS GPU):"
    echo "  合計時間: ${TOTAL_TIME}秒"
    echo "  初回サンプル時間（推定）: ${FIRST_SAMPLE_TIME}秒"
    echo "  定常状態サンプルあたり: ${STEADY_STATE_PER_SAMPLE}秒"
    echo "  スループット: ${THROUGHPUT} サンプル/秒"
    echo ""

    # GPU冷却のための待機
    sleep 10
done

echo "=============================================="
echo "ベンチマーク完了"
echo "=============================================="
echo ""
echo "結果サマリー:"
echo ""
cat $RESULTS_FILE | column -t -s,
echo ""
echo "詳細ログ: ${RESULTS_DIR}/log_*gpu_${TIMESTAMP}.txt"
echo "CSV結果: $RESULTS_FILE"
