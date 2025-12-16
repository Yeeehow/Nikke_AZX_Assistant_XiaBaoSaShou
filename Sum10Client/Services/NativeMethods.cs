using System;
using System.Runtime.InteropServices;

namespace Sum10Client.Services
{
    internal static class NativeMethods
    {
        private const string DllName = "Sum10Core.dll";

        [UnmanagedFunctionPointer(CallingConvention.StdCall, CharSet = CharSet.Unicode)]
        public delegate void Sum10LogCallback([MarshalAs(UnmanagedType.LPWStr)] string msg);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Unicode)]
        public static extern void sum10_set_log_callback(Sum10LogCallback cb);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Unicode)]
        public static extern int sum10_capture_screen_png(string outPngPath);

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Unicode)]
        public static extern int sum10_ocr_board(
            string screenPngPath,
            float[] corners8,
            int rows,
            int cols,
            string digitTemplateDir,
            [Out] int[] outDigits,
            out float outAvgConf,
            [Out] float[] outCellConf,
            string outWarpPngPath
        );

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern int sum10_solve_greedy(
            int[] digits,
            int rows,
            int cols,
            [Out] int[] outMoves,
            int maxMoves,
            out int outMoveCount,
            out int outScore
        );

        [DllImport(DllName, CallingConvention = CallingConvention.StdCall)]
        public static extern int sum10_execute_path(
            float[] corners8,
            int rows,
            int cols,
            int[] moves,
            int moveCount,
            int offsetX,
            int offsetY,
            int delayMs
        );
    }
}
