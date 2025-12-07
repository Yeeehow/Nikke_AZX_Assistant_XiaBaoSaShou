#pragma once
#include <cstdint>

#ifdef SUM10CORE_EXPORTS
#define SUM10_API __declspec(dllexport)
#else
#define SUM10_API __declspec(dllimport)
#endif

extern "C" {

    // 和 C# 侧 RectMove 一一对应
    struct RectMove
    {
        std::int32_t r1;
        std::int32_t c1;
        std::int32_t r2;
        std::int32_t c2;
    };

    // 测试函数（可选）——把盘面所有数字加起来
    SUM10_API int __stdcall SumBoard(
        const std::int32_t* boardValues,
        std::int32_t rows,
        std::int32_t cols
    );

    // 核心求解器（V6.2 赌徒版·单性格版）
    //
    // 参数：
    //   boardValues : rows*cols 的一维数组（按行展开）
    //   rows, cols  : 棋盘大小，直接填 16 和 10 就行
    //   beamWidth   : Beam 宽度，建议 32~128
    //   maxDepth    : 最大搜索层数（可以用 rows*cols）
    //   mode        : 0 = classic (只允许两个格子的矩形)，1 = omni (允许>=2格)
    //   wIsland     : 孤岛惩罚权重（性格参数）
    //   wFragment   : 中心引力惩罚（性格参数）
    //   timeLimitMs : 搜索时间上限（毫秒）
    //   outMoves    : 输出路径数组
    //   maxMoves    : outMoves 容量
    //   outMoveCount: 实际写入的步数
    //
    // 返回值：最佳得分（被消掉的格子数量）
    SUM10_API int __stdcall SolveBoardSimple(
        const std::int32_t* boardValues,
        std::int32_t rows,
        std::int32_t cols,
        std::int32_t beamWidth,
        std::int32_t maxDepth,
        std::int32_t mode,
        std::int32_t wIsland,
        double        wFragment,
        std::int32_t timeLimitMs,
        RectMove* outMoves,
        std::int32_t maxMoves,
        std::int32_t* outMoveCount
    );

} // extern "C"
