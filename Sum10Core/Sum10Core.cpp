#define SUM10CORE_EXPORTS
#include "Sum10Core.h"

#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>
#include <cstdint>

using std::int32_t;
using std::uint8_t;

// ========== 简单测试函数（可保留） ==========
int __stdcall SumBoard(const int32_t* boardValues, int32_t rows, int32_t cols)
{
    int32_t n = rows * cols;
    int32_t sum = 0;
    for (int32_t i = 0; i < n; ++i)
        sum += boardValues[i];
    return sum;
}

// ========== 内部结构与辅助函数 ==========

struct State
{
    std::vector<uint8_t> map;    // 1 = 有子，0 = 已消除
    std::vector<RectMove> path;  // 已走路径
    int32_t score = 0;           // 累计消掉的格子数
    double  hscore = 0.0;        // 启发式评分
};

// 计算孤岛数量：上下左右都没有邻居的孤立格子数
static int32_t CountIslands(const std::vector<uint8_t>& map,
    int32_t rows, int32_t cols)
{
    int32_t islands = 0;
    for (int32_t r = 0; r < rows; ++r)
    {
        for (int32_t c = 0; c < cols; ++c)
        {
            int32_t idx = r * cols + c;
            if (map[idx] != 1) continue;

            // 上
            if (r > 0 && map[(r - 1) * cols + c] == 1) continue;
            // 下
            if (r < rows - 1 && map[(r + 1) * cols + c] == 1) continue;
            // 左
            if (c > 0 && map[r * cols + (c - 1)] == 1) continue;
            // 右
            if (c < cols - 1 && map[r * cols + (c + 1)] == 1) continue;

            ++islands;
        }
    }
    return islands;
}

// 构建前缀和：src 是 rows*cols 的一维数组，P 大小为 (rows+1)*(cols+1)
static void BuildPrefix(const std::vector<int32_t>& src,
    int32_t rows, int32_t cols,
    std::vector<int32_t>& P)
{
    std::fill(P.begin(), P.end(), 0);
    int32_t stride = cols + 1;

    for (int32_t r = 0; r < rows; ++r)
    {
        int32_t row_sum = 0;
        for (int32_t c = 0; c < cols; ++c)
        {
            row_sum += src[r * cols + c];
            P[(r + 1) * stride + (c + 1)] =
                P[r * stride + (c + 1)] + row_sum;
        }
    }
}

// 用前缀和快速求矩形和
static inline int32_t RectSum(const std::vector<int32_t>& P,
    int32_t rows, int32_t cols,
    int32_t r1, int32_t c1,
    int32_t r2, int32_t c2)
{
    (void)rows; // 防止未使用警告
    int32_t stride = cols + 1;
    int32_t A = P[(r2 + 1) * stride + (c2 + 1)];
    int32_t B = P[r1 * stride + (c2 + 1)];
    int32_t C = P[(r2 + 1) * stride + c1];
    int32_t D = P[r1 * stride + c1];
    return A - B - C + D;
}

// 对应 Python _fast_scan_rects_v6：
// 利用前缀和，遍历所有 active cell 对，找出 sum=10 的矩形，记录其格子数
static void FastScanRectsV6(
    const std::vector<uint8_t>& map,
    const int32_t* vals,
    int32_t rows,
    int32_t cols,
    std::vector<RectMove>& outMoves,
    std::vector<int32_t>& outCounts
)
{
    outMoves.clear();
    outCounts.clear();

    const int32_t total = rows * cols;
    // 收集活跃格子索引
    std::vector<int32_t> active;
    active.reserve(total);
    for (int32_t i = 0; i < total; ++i)
        if (map[i] == 1) active.push_back(i);

    if (active.size() < 2) return;

    // 构建当前值/计数数组
    std::vector<int32_t> curVals(total, 0);
    std::vector<int32_t> curCnt(total, 0);
    for (int32_t i = 0; i < total; ++i)
    {
        if (map[i] == 1)
        {
            curVals[i] = vals[i];
            curCnt[i] = 1;
        }
    }

    // 前缀和
    std::vector<int32_t> Pval((rows + 1) * (cols + 1));
    std::vector<int32_t> Pcnt((rows + 1) * (cols + 1));
    BuildPrefix(curVals, rows, cols, Pval);
    BuildPrefix(curCnt, rows, cols, Pcnt);

    // 双重循环枚举所有活跃格子对
    const std::size_t nActive = active.size();
    for (std::size_t ia = 0; ia < nActive; ++ia)
    {
        int32_t idx1 = active[ia];
        int32_t r1_raw = idx1 / cols;
        int32_t c1_raw = idx1 % cols;

        for (std::size_t jb = ia; jb < nActive; ++jb)
        {
            int32_t idx2 = active[jb];
            int32_t r2_raw = idx2 / cols;
            int32_t c2_raw = idx2 % cols;

            int32_t min_r = std::min(r1_raw, r2_raw);
            int32_t max_r = std::max(r1_raw, r2_raw);
            int32_t min_c = std::min(c1_raw, c2_raw);
            int32_t max_c = std::max(c1_raw, c2_raw);

            int32_t sum = RectSum(Pval, rows, cols, min_r, min_c, max_r, max_c);
            if (sum != 10) continue;

            int32_t cnt = RectSum(Pcnt, rows, cols, min_r, min_c, max_r, max_c);

            RectMove m;
            m.r1 = min_r; m.c1 = min_c;
            m.r2 = max_r; m.c2 = max_c;

            outMoves.push_back(m);
            outCounts.push_back(cnt);
        }
    }
}

// 应用一个矩形操作，把区域内全部置 0
static void ApplyMove(const RectMove& m,
    std::vector<uint8_t>& map,
    int32_t rows,
    int32_t cols)
{
    (void)rows;
    for (int32_t r = m.r1; r <= m.r2; ++r)
    {
        int32_t base = r * cols;
        for (int32_t c = m.c1; c <= m.c2; ++c)
        {
            map[base + c] = 0;
        }
    }
}

// 对应 Python _evaluate_state（V6.2 赌徒版）
//
// h = score * 2000
//   - islands * wIsland
//   - center_mass * wFragment
//   + random() * noiseLevel
//
// noiseLevel = 50 常规
//   若 wIsland < 20 且 wFragment < 1，则视为 赌徒/狂战士，noiseLevel = 2000
static double EvaluateState(
    int32_t score,
    const std::vector<uint8_t>& map,
    int32_t rows,
    int32_t cols,
    int32_t wIsland,
    double  wFragment,
    std::mt19937& rng
)
{
    // 1. 基础分
    double h = static_cast<double>(score) * 2000.0;

    // 2. 孤岛惩罚
    if (wIsland > 0)
    {
        int32_t islands = CountIslands(map, rows, cols);
        h -= static_cast<double>(islands * wIsland);
    }

    // 3. 中心引力惩罚
    if (wFragment > 0.0)
    {
        int32_t center_r = rows / 2;
        int32_t center_c = cols / 2;
        int32_t center_mass = 0;

        for (int32_t r = 0; r < rows; ++r)
        {
            for (int32_t c = 0; c < cols; ++c)
            {
                if (map[r * cols + c] == 1)
                {
                    int32_t dist = std::abs(r - center_r) + std::abs(c - center_c);
                    center_mass += (20 - dist);  // 和 Py 一样，20 是经验常数
                }
            }
        }
        h -= static_cast<double>(center_mass) * wFragment;
    }

    // 4. 赌徒随机噪声
    double noiseLevel = 50.0;
    if (wIsland < 20 && wFragment < 1.0)
    {
        // 赌徒 / 狂战士：给 2000 分随机扰动
        noiseLevel = 2000.0;
    }
    std::uniform_real_distribution<double> dist01(0.0, 1.0);
    h += dist01(rng) * noiseLevel;

    return h;
}

// ========== 导出：V6.2 赌徒版 Beam Search 求解器 ==========

int __stdcall SolveBoardSimple(
    const int32_t* boardValues,
    int32_t rows,
    int32_t cols,
    int32_t beamWidth,
    int32_t maxDepth,
    int32_t mode,
    int32_t wIsland,
    double  wFragment,
    int32_t timeLimitMs,
    RectMove* outMoves,
    int32_t maxMoves,
    int32_t* outMoveCount
)
{
    if (!boardValues || !outMoves || !outMoveCount) return -1;
    if (rows <= 0 || cols <= 0) return -2;

    if (beamWidth <= 0) beamWidth = 32;
    if (maxDepth <= 0) maxDepth = rows * cols;
    if (timeLimitMs <= 0) timeLimitMs = 20000; // 默认 20 秒

    const int32_t total = rows * cols;

    // 初始 map：全部为 1（表示棋盘上都有数字）
    std::vector<uint8_t> startMap(total, 1);

    // RNG
    std::random_device rd;
    std::mt19937 rng(rd());

    // 初始状态
    State init;
    init.map = startMap;
    init.score = 0;
    init.path.clear();
    init.hscore = EvaluateState(init.score, init.map, rows, cols,
        wIsland, wFragment, rng);

    std::vector<State> beam;
    beam.reserve(beamWidth);
    beam.push_back(init);
    State best = init;

    // 时间截止点
    auto deadline = std::chrono::steady_clock::now()
        + std::chrono::milliseconds(timeLimitMs);

    std::vector<RectMove> moves;
    std::vector<int32_t>  counts;

    int32_t depth = 0;
    while (depth < maxDepth)
    {
        if (std::chrono::steady_clock::now() > deadline)
            break;

        std::vector<State> nextStates;
        nextStates.reserve(beamWidth * 4);
        bool foundAnyMove = false;

        for (const State& st : beam)
        {
            // 扫描所有和为 10 的矩形
            FastScanRectsV6(st.map, boardValues, rows, cols, moves, counts);

            if (moves.empty())
            {
                // 没有任何后续步，更新 best
                if (st.score > best.score) best = st;
                continue;
            }

            // 规则过滤 + 预筛选 60 个候选（按 count 从大到小）
            std::vector<int32_t> indices;
            indices.reserve(moves.size());

            for (std::size_t i = 0; i < moves.size(); ++i)
            {
                int32_t cnt = counts[i];
                bool rulePass = false;
                if (mode == 0)
                {
                    // classic: 只允许 2 格
                    rulePass = (cnt == 2);
                }
                else
                {
                    // omni: 允许 >= 2 格
                    rulePass = (cnt >= 2);
                }
                if (rulePass)
                    indices.push_back(static_cast<int32_t>(i));
            }

            if (indices.empty())
            {
                if (st.score > best.score) best = st;
                continue;
            }

            foundAnyMove = true;

            // 按 count 降序排，保留前 60 个（防止“沧海遗珠”）
            std::sort(indices.begin(), indices.end(),
                [&](int32_t a, int32_t b)
                {
                    return counts[a] > counts[b];
                });
            const int32_t window = 60;
            if ((int32_t)indices.size() > window)
                indices.resize(window);

            // 扩展这些候选
            for (int32_t idx : indices)
            {
                const RectMove& m = moves[idx];
                int32_t cnt = counts[idx];

                State ns;
                ns.map = st.map;
                ApplyMove(m, ns.map, rows, cols);

                ns.path = st.path;
                ns.path.push_back(m);

                ns.score = st.score + cnt;
                ns.hscore = EvaluateState(ns.score, ns.map, rows, cols,
                    wIsland, wFragment, rng);

                nextStates.push_back(std::move(ns));
            }
        }

        if (!foundAnyMove || nextStates.empty())
            break;

        // Beam 选择：按 hscore 降序取前 beamWidth 个
        std::sort(nextStates.begin(), nextStates.end(),
            [](const State& a, const State& b)
            {
                return a.hscore > b.hscore;
            });
        if ((int32_t)nextStates.size() > beamWidth)
            nextStates.resize(beamWidth);

        beam.swap(nextStates);

        // 更新全局 best（以纯 score 为准）
        if (!beam.empty() && beam[0].score > best.score)
            best = beam[0];

        ++depth;
    }

    // 将最佳路径写回 C#
    int32_t movesToCopy = static_cast<int32_t>(best.path.size());
    if (movesToCopy > maxMoves) movesToCopy = maxMoves;

    for (int32_t i = 0; i < movesToCopy; ++i)
        outMoves[i] = best.path[i];

    *outMoveCount = movesToCopy;
    return best.score;
}
