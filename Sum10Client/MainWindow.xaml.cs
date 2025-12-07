using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;

namespace Sum10Client
{
    public partial class MainWindow : Window
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct RectMove
        {
            public int r1;
            public int c1;
            public int r2;
            public int c2;

            public override string ToString()
            {
                return $"({r1},{c1}) -> ({r2},{c2})";
            }
        }

        // 新版求解接口
        [DllImport("Sum10Core.dll", CallingConvention = CallingConvention.StdCall)]
        private static extern int SolveBoardSimple(
            int[] boardValues,
            int rows,
            int cols,
            int beamWidth,
            int maxDepth,
            int mode,
            int wIsland,
            double wFragment,
            int timeLimitMs,
            [Out] RectMove[] outMoves,
            int maxMoves,
            out int outMoveCount
        );

        public MainWindow()
        {
            InitializeComponent();
        }

        private void BtnCallCpp_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                // ✅ 现在直接用 16x10
                int rows = 16;
                int cols = 10;

                var parts = TxtBoard.Text
                    .Split(new[] { ',', ' ', '\n', '\r', '\t' }, StringSplitOptions.RemoveEmptyEntries);

                int[] values = parts.Select(p => int.Parse(p)).ToArray();

                if (values.Length != rows * cols)
                {
                    MessageBox.Show(
                        $"需要 {rows}x{cols} = {rows * cols} 个数字，你输入了 {values.Length} 个。\n" +
                        "（用空格、逗号或换行分隔都可以）",
                        "输入错误"
                    );
                    return;
                }

                RectMove[] path = new RectMove[512]; // 预留多一点
                int moveCount;

                int bestScore = SolveBoardSimple(
                    values,
                    rows,
                    cols,
                    64,                // beamWidth 建议 32~128
                    rows * cols,       // maxDepth
                    1,                 // mode: 1 = omni
                    24,                // wIsland → 对应“微醺赌徒 (24, 0.5)”
                    0.5,               // wFragment
                    20000,             // timeLimitMs = 20 秒
                    path,
                    path.Length,
                    out moveCount
                );

                var sb = new System.Text.StringBuilder();
                sb.AppendLine($"最佳得分（消掉格子数）: {bestScore}");
                sb.AppendLine($"路径步数: {moveCount}");
                for (int i = 0; i < moveCount; i++)
                {
                    sb.AppendLine($"{i + 1}. ({path[i].r1},{path[i].c1}) -> ({path[i].r2},{path[i].c2})");
                }

                LblResult.Text = sb.ToString();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "异常");
            }
        }
    }
}
