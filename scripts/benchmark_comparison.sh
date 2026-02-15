#!/bin/bash
# benchmark_comparison.sh - パイプライン並列 vs データ並列 比較ベンチマーク
#
# 両方式を同じ条件（GPU数・サンプル数・ステップ数）で実行し、
# スループットを比較する。
#
# 使用方法:
#   chmod +x scripts/benchmark_comparison.sh
#   ./scripts/benchmark_comparison.sh
#
# 注意: 7GPUが利用可能な環境で実行してください

set -e

# 設定
TOTAL_STEPS=28          # 1, 2, 4, 7 で割り切れる
NUM_SAMPLES=14          # 全GPU数(7)で割り切れる数
WARMUP_SAMPLES=7        # ウォームアップ（7で割り切れる）
MODEL="dummy"           # dummy or svd
SEED=42
LATENT_HEIGHT=40
LATENT_WIDTH=72
LATENT_FRAMES=14
HIDDEN_CHANNELS=64

# 結果ディレクトリ
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="${RESULTS_DIR}/comparison_${TIMESTAMP}.csv"

mkdir -p "$RESULTS_DIR"

echo "=============================================================="
echo "パイプライン並列 vs データ並列 比較ベンチマーク"
echo "=============================================================="
echo "設定:"
echo "  Total Steps:     $TOTAL_STEPS"
echo "  Num Samples:     $NUM_SAMPLES"
echo "  Warmup Samples:  $WARMUP_SAMPLES"
echo "  Model:           $MODEL"
echo "  Latent:          1x4x${LATENT_FRAMES}x${LATENT_HEIGHT}x${LATENT_WIDTH}"
echo "  結果ファイル:    $CSV_FILE"
echo "=============================================================="
echo ""

# CSVヘッダー
echo "mode,gpu_count,total_steps,steps_per_gpu,num_samples,first_sample_s,avg_sample_s,throughput_sps" > "$CSV_FILE"

# 実行結果からJSONを抽出してCSV行を生成する関数
extract_and_append() {
    local mode=$1
    local ngpus=$2
    local log_file=$3

    local json_line
    json_line=$(grep "^BENCHMARK_JSON=" "$log_file" | tail -1 | sed 's/^BENCHMARK_JSON=//')

    if [ -z "$json_line" ]; then
        echo "  [WARNING] BENCHMARK_JSON not found in $log_file"
        return 1
    fi

    local steps_per_gpu first_s avg_s throughput
    steps_per_gpu=$(echo "$json_line" | python3 -c "import sys,json; print(json.load(sys.stdin)['steps_per_gpu'])")
    first_s=$(echo "$json_line" | python3 -c "import sys,json; print(json.load(sys.stdin)['first_sample_time_s'])")
    avg_s=$(echo "$json_line" | python3 -c "import sys,json; print(json.load(sys.stdin)['avg_sample_time_s'])")
    throughput=$(echo "$json_line" | python3 -c "import sys,json; print(json.load(sys.stdin)['throughput_samples_per_s'])")

    echo "${mode},${ngpus},${TOTAL_STEPS},${steps_per_gpu},${NUM_SAMPLES},${first_s},${avg_s},${throughput}" >> "$CSV_FILE"
    echo "  -> throughput: ${throughput} samples/s, avg: ${avg_s} s/sample"
}

# GPU数のリスト
GPU_COUNTS=(1 2 4 7)

for NGPUS in "${GPU_COUNTS[@]}"; do
    echo ""
    echo "=========================================="
    echo "GPU数: $NGPUS"
    echo "=========================================="

    DEVICES=$(seq -s, 0 $((NGPUS-1)))

    # --- パイプライン並列 ---
    echo ""
    echo "  [Pipeline Parallel] Running..."
    PP_LOG="${RESULTS_DIR}/pp_${NGPUS}gpu_${TIMESTAMP}.log"

    CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
        --nproc_per_node=$NGPUS \
        -m src.modes.benchmark \
        --total-steps $TOTAL_STEPS \
        --num-samples $NUM_SAMPLES \
        --warmup-samples $WARMUP_SAMPLES \
        --model $MODEL \
        --latent-height $LATENT_HEIGHT \
        --latent-width $LATENT_WIDTH \
        --latent-frames $LATENT_FRAMES \
        --hidden-channels $HIDDEN_CHANNELS \
        --seed $SEED \
        2>&1 | tee "$PP_LOG"

    extract_and_append "pipeline_parallel" "$NGPUS" "$PP_LOG"

    sleep 5

    # --- データ並列 ---
    echo ""
    echo "  [Data Parallel] Running..."
    DP_LOG="${RESULTS_DIR}/dp_${NGPUS}gpu_${TIMESTAMP}.log"

    CUDA_VISIBLE_DEVICES=$DEVICES torchrun \
        --nproc_per_node=$NGPUS \
        -m src.modes.benchmark_data_parallel \
        --total-steps $TOTAL_STEPS \
        --num-samples $NUM_SAMPLES \
        --warmup-samples $WARMUP_SAMPLES \
        --model $MODEL \
        --latent-height $LATENT_HEIGHT \
        --latent-width $LATENT_WIDTH \
        --latent-frames $LATENT_FRAMES \
        --hidden-channels $HIDDEN_CHANNELS \
        --seed $SEED \
        2>&1 | tee "$DP_LOG"

    extract_and_append "data_parallel" "$NGPUS" "$DP_LOG"

    sleep 5
done

echo ""
echo "=============================================================="
echo "比較結果サマリー"
echo "=============================================================="
echo ""
column -t -s, < "$CSV_FILE"
echo ""
echo "詳細ログ: ${RESULTS_DIR}/*_${TIMESTAMP}.log"
echo "CSV結果:  $CSV_FILE"
