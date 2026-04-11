#!/bin/bash
# ============================================================
# TWCC 安全檢查 — 確認沒有忘記關掉的資源
# ============================================================
# ⚠️ 你同學被扣 2 萬多就是因為「開發型容器」沒關
# 我們用的 sbatch 不會有這個問題，但跑完還是確認一下
#
# Usage:
#   bash twcc/check_billing_safe.sh    ← 跑完實驗後執行
# ============================================================

echo "============================================================"
echo "  🔍 TWCC 安全檢查 — 確認沒有殘留資源"
echo "  時間: $(date)"
echo "============================================================"

# ---- 1. 檢查是否有正在跑的 SLURM jobs ----
echo ""
echo "📋 [1/3] 你的 SLURM Jobs:"
JOBS=$(squeue -u $USER -h 2>/dev/null | wc -l)
if [ "$JOBS" -eq 0 ]; then
    echo "  ✅ 沒有任何 job 在跑 — 安全"
else
    echo "  ⚠️  還有 $JOBS 個 job："
    squeue -u $USER --format="  %10i %20j %8T %12M %12l %6D %R"
    echo ""
    echo "  如果這些都是你的實驗 → 正常，等它跑完"
    echo "  如果不認識 → 用 scancel <JOBID> 取消"
    echo "  全部取消 → scancel -u $USER"
fi

# ---- 2. 檢查最近完成的 jobs（確認有正常結束）----
echo ""
echo "📋 [2/3] 最近 24 小時完成的 Jobs:"
sacct -u $USER --starttime=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M 2>/dev/null || date -v-24H +%Y-%m-%dT%H:%M 2>/dev/null || echo "2026-04-10") \
    --format="JobID%15,JobName%20,State%12,Elapsed%10,MaxRSS%10" 2>/dev/null || \
    echo "  (sacct 不可用，跳過)"

# ---- 3. 最重要：提醒不要開開發型容器 ----
echo ""
echo "📋 [3/3] 計費安全提醒:"
echo "  ✅ sbatch 批次作業 → 跑完自動結束，不會持續扣款"
echo "  ✅ 我們的腳本都有 --time 限制 → 超時自動 kill"
echo "  ❌ 千萬不要在 TWCC 網頁開「開發型容器」→ 忘記關會持續扣款！"
echo "  ❌ 不要用 srun --pty bash → 忘記 exit 也會持續佔用資源"
echo ""

# ---- 4. 實驗完成狀態 ----
echo "📋 實驗進度:"
if [ -d "results/experiments/tmc" ]; then
    TOTAL=$(ls results/experiments/tmc/exp*.json 2>/dev/null | wc -l)
    P1=$(ls results/experiments/tmc/expA*.json results/experiments/tmc/expB*.json results/experiments/tmc/expC*.json 2>/dev/null | wc -l)
    P2=$(ls results/experiments/tmc/expD*.json 2>/dev/null | wc -l)
    P3=$(ls results/experiments/tmc/expE*.json 2>/dev/null | wc -l)
    echo "  Phase 1: $P1/33 完成"
    echo "  Phase 2: $P2/9 完成"
    echo "  Phase 3: $P3/12 完成"
    echo "  Total:   $TOTAL/54 完成"
else
    echo "  (尚未有結果)"
fi

echo ""
echo "============================================================"
echo "  檢查完畢。如果上面都是 ✅ → 你不會被扣冤枉錢 💰"
echo "============================================================"
