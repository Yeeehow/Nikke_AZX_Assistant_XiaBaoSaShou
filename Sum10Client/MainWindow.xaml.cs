using Sum10Client.Services;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace Sum10Client
{
    public partial class MainWindow : Window
    {
        private string _workDir;
        private string _screenPng;
        private string _warpPng;
        private string _digitsDir;

        // corners: TL,TR,BR,BL (8 floats)
        private float[] _corners = new float[8];
        private int _cornerCount = 0;

        private BitmapImage? _screenBitmap;
        private int _screenPixelW = 0;
        private int _screenPixelH = 0;

        private int[]? _digits;
        private int[]? _movesBuf;
        private int _moveCount = 0;

        // keep delegate alive
        private NativeMethods.Sum10LogCallback? _logCb;

        public MainWindow()
        {
            InitializeComponent();

            _workDir = AppDomain.CurrentDomain.BaseDirectory;
            _screenPng = System.IO.Path.Combine(_workDir, "screen.png");
            _warpPng = System.IO.Path.Combine(_workDir, "warped.png");
            _digitsDir = System.IO.Path.Combine(_workDir, "Assets", "Digits");

            Loaded += MainWindow_Loaded;
        }

        private void MainWindow_Loaded(object sender, RoutedEventArgs e)
        {
            // register native log callback
            _logCb = (msg) => Dispatcher.Invoke(() =>
            {
                TbLog.AppendText(msg + Environment.NewLine);
                TbLog.ScrollToEnd();
            });

            var baseDir = AppDomain.CurrentDomain.BaseDirectory;
            TbLog.AppendText("BaseDirectory = " + baseDir + Environment.NewLine);
            TbLog.AppendText("Has Sum10Core.dll = " + File.Exists(System.IO.Path.Combine(baseDir, "Sum10Core.dll")) + Environment.NewLine);
            TbLog.AppendText("Has opencv_world4120d.dll = " + File.Exists(System.IO.Path.Combine(baseDir, "opencv_world4120d.dll")) + Environment.NewLine);

            try
            {
                NativeMethods.sum10_set_log_callback(_logCb);
            }
            catch (DllNotFoundException)
            {
                MessageBox.Show("找不到 Sum10Core.dll。请把 DLL 放到 WPF 输出目录（bin\\x64\\Debug）或项目 Core 文件夹并复制到输出。", "Error");
            }
        }

        private int ReadInt(TextBox tb, int fallback)
        {
            if (int.TryParse(tb.Text.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v)) return v;
            return fallback;
        }

        private void BtnCapture_Click(object sender, RoutedEventArgs e)
        {
            int rc = NativeMethods.sum10_capture_screen_png(_screenPng);
            if (rc != 0)
            {
                MessageBox.Show($"capture failed: {rc}");
                return;
            }

            LoadAndShowImage(_screenPng);
            AppendInfo("Captured. Now click 4 points in order: TL -> TR -> BR -> BL");
            ResetPointsOnly();
        }

        private void LoadAndShowImage(string path)
        {
            if (!File.Exists(path)) return;

            // load into bitmap
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.UriSource = new Uri(path);
            bmp.EndInit();
            bmp.Freeze();

            _screenBitmap = bmp;
            ImgScreen.Source = bmp;

            _screenPixelW = bmp.PixelWidth;
            _screenPixelH = bmp.PixelHeight;

            Overlay.Children.Clear();
        }

        private void BtnResetPoints_Click(object sender, RoutedEventArgs e)
        {
            ResetPointsOnly();
            Overlay.Children.Clear();
            AppendInfo("Points reset. Click 4 points: TL -> TR -> BR -> BL");
        }

        private void ResetPointsOnly()
        {
            _cornerCount = 0;
            Array.Clear(_corners, 0, _corners.Length);
        }

        private void ImgScreen_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (_screenBitmap == null) return;
            if (_cornerCount >= 4) return;

            // get click position in Image control
            Point p = e.GetPosition(ImgScreen);
            double displayW = ImgScreen.ActualWidth;
            double displayH = ImgScreen.ActualHeight;

            if (displayW <= 1 || displayH <= 1) return;

            // map to actual pixel
            float px = (float)(p.X * _screenPixelW / displayW);
            float py = (float)(p.Y * _screenPixelH / displayH);

            int idx = _cornerCount * 2;
            _corners[idx] = px;
            _corners[idx + 1] = py;

            DrawPointOnOverlay(p, _cornerCount);
            _cornerCount++;

            string[] names = { "TL", "TR", "BR", "BL" };
            AppendInfo($"Captured {names[_cornerCount - 1]}: ({px:0},{py:0})");

            if (_cornerCount == 4)
            {
                AppendInfo("4 points ready. Click Run OCR.");
            }
        }

        private void DrawPointOnOverlay(Point p, int i)
        {
            // overlay uses same coordinate space as Image (display space)
            double r = 6;
            var el = new Ellipse
            {
                Width = r * 2,
                Height = r * 2,
                Fill = Brushes.Red,
                Stroke = Brushes.White,
                StrokeThickness = 2
            };
            Canvas.SetLeft(el, p.X - r);
            Canvas.SetTop(el, p.Y - r);
            Overlay.Children.Add(el);

            var txt = new TextBlock
            {
                Text = (i + 1).ToString(),
                Foreground = Brushes.Yellow,
                FontWeight = FontWeights.Bold
            };
            Canvas.SetLeft(txt, p.X + 6);
            Canvas.SetTop(txt, p.Y - 10);
            Overlay.Children.Add(txt);
        }

        private void BtnOcr_Click(object sender, RoutedEventArgs e)
        {
            if (_cornerCount < 4)
            {
                MessageBox.Show("请先在截图上点 4 个角点（TL->TR->BR->BL）。");
                return;
            }

            if (!Directory.Exists(_digitsDir))
            {
                MessageBox.Show($"找不到模板目录：{_digitsDir}\n请创建 Assets\\Digits 并放入 0.png~9.png（至少 1~9）。");
                return;
            }

            int rows = ReadInt(TbRows, 10);
            int cols = ReadInt(TbCols, 17);

            _digits = new int[rows * cols];

            int rc = NativeMethods.sum10_ocr_board(
                _screenPng,
                _corners,
                rows,
                cols,
                _digitsDir,
                _digits,
                out float avgConf,
                _warpPng
            );

            if (rc != 0)
            {
                MessageBox.Show($"OCR failed: {rc}");
                return;
            }

            TbOcrInfo.Text = $"OCR done. avgConf={avgConf:0.000}. warped saved: {_warpPng}";
            BuildBoardGrid(rows, cols, _digits);

            // optionally show warped as the main image (你可以改成另开一个窗口）
            if (File.Exists(_warpPng))
            {
                LoadAndShowImage(_warpPng);
                AppendInfo("Showing warped preview image.");
            }
        }

        private void BuildBoardGrid(int rows, int cols, int[] digits)
        {
            BoardGrid.Rows = rows;
            BoardGrid.Columns = cols;
            BoardGrid.Children.Clear();

            for (int i = 0; i < rows * cols; i++)
            {
                var tb = new TextBlock
                {
                    Text = digits[i].ToString(),
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center,
                    FontSize = 14
                };

                var border = new Border
                {
                    BorderBrush = Brushes.LightGray,
                    BorderThickness = new Thickness(0.5),
                    Padding = new Thickness(6),
                    Child = tb
                };
                BoardGrid.Children.Add(border);
            }
        }

        private void BtnSolve_Click(object sender, RoutedEventArgs e)
        {
            if (_digits == null)
            {
                MessageBox.Show("请先 Run OCR。");
                return;
            }

            int rows = ReadInt(TbRows, 16);
            int cols = ReadInt(TbCols, 10);

            int maxMoves = 2000;
            _movesBuf = new int[maxMoves * 4];

            int rc = NativeMethods.sum10_solve_greedy(
                _digits,
                rows,
                cols,
                _movesBuf,
                maxMoves,
                out _moveCount,
                out int score
            );

            if (rc != 0)
            {
                MessageBox.Show($"Solve failed: {rc}");
                return;
            }

            LbMoves.Items.Clear();
            for (int i = 0; i < _moveCount; i++)
            {
                int r1 = _movesBuf[i * 4 + 0];
                int c1 = _movesBuf[i * 4 + 1];
                int r2 = _movesBuf[i * 4 + 2];
                int c2 = _movesBuf[i * 4 + 3];
                LbMoves.Items.Add($"{i + 1:0000}: ({r1},{c1}) -> ({r2},{c2})");
            }

            AppendInfo($"Solved (greedy). moves={_moveCount}, score(removed cells)={score}");
        }

        private void BtnExecute_Click(object sender, RoutedEventArgs e)
        {
            if (_movesBuf == null || _moveCount <= 0)
            {
                MessageBox.Show("请先 Solve。");
                return;
            }

            if (_cornerCount < 4)
            {
                MessageBox.Show("需要四角点（TL->TR->BR->BL）来映射鼠标位置。");
                return;
            }

            int rows = ReadInt(TbRows, 16);
            int cols = ReadInt(TbCols, 10);
            int offX = ReadInt(TbOffX, 0);
            int offY = ReadInt(TbOffY, 0);
            int delay = ReadInt(TbDelay, 120);

            var confirm = MessageBox.Show("将开始自动拖拽执行。请切回游戏窗口并确保棋盘可见。\n继续？", "Execute", MessageBoxButton.YesNo);
            if (confirm != MessageBoxResult.Yes) return;

            // 2 秒倒计时（你可以改更久）
            AppendInfo("Execute starts in 2 seconds...");
            System.Threading.Thread.Sleep(2000);

            int rc = NativeMethods.sum10_execute_path(
                _corners,
                rows,
                cols,
                _movesBuf,
                _moveCount,
                offX,
                offY,
                delay
            );

            if (rc != 0)
            {
                MessageBox.Show($"Execute failed: {rc}");
                return;
            }

            AppendInfo("Execute completed.");
        }

        private void AppendInfo(string s)
        {
            TbLog.AppendText("[UI] " + s + Environment.NewLine);
            TbLog.ScrollToEnd();
        }
    }
}
