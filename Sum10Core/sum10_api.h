#pragma once
#include <cstdint>

#ifdef _WIN32
#define SUM10_API __declspec(dllexport)
#define SUM10_CALL __stdcall
#else
#define SUM10_API
#define SUM10_CALL
#endif

extern "C" {

    // 日志回调（WPF 用来显示 log）
    typedef void (SUM10_CALL* sum10_log_callback_t)(const wchar_t* msg);

    // 设置日志回调
    SUM10_API void SUM10_CALL sum10_set_log_callback(sum10_log_callback_t cb);

    // 截屏保存为 PNG
    SUM10_API int  SUM10_CALL sum10_capture_screen_png(const wchar_t* outPngPath);

    // OCR：输入截图路径 + 四角点（8 floats：TLx,TLy, TRx,TRy, BRx,BRy, BLx,BLy）
    // 输出：rows*cols 的 digits（int），avgConf（float），并输出 warped 图到 outWarpPngPath
    SUM10_API int  SUM10_CALL sum10_ocr_board(
        const wchar_t* screenPngPath,
        const float* corners8,
        int rows,
        int cols,
        const wchar_t* digitTemplateDir,
        int* outDigits,
        float* outAvgConf,
        const wchar_t* outWarpPngPath
    );

    // 求解器（最小闭环：贪心找 sum=10 的全激活矩形）
    // outMoves: (r1,c1,r2,c2) * moveCount，最大 maxMoves
    SUM10_API int  SUM10_CALL sum10_solve_greedy(
        const int* digits,
        int rows,
        int cols,
        int* outMoves,
        int maxMoves,
        int* outMoveCount,
        int* outScore
    );

    // 执行拖拽路径（SendInput）
    // offsetX/offsetY：鼠标点在 cell center 的偏移（像素）
    // delayMs：每步拖拽的停顿
    SUM10_API int  SUM10_CALL sum10_execute_path(
        const float* corners8,
        int rows,
        int cols,
        const int* moves,
        int moveCount,
        int offsetX,
        int offsetY,
        int delayMs
    );

} // extern "C"
