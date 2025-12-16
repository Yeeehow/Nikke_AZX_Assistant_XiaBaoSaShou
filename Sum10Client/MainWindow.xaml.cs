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

        // optional: dedicated execution corners (if user wants refined mapping)
        private float[] _execCorners = new float[8];
        private int _execCornerCount = 0;

        private bool _isDragSelecting = false;
        private bool _isPickingExecCorners = false;
        private Point _dragStart;
        private Rect? _selectedRectDisplay;
        private readonly List<Point> _execPointsDisplay = new();

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

            ResetSelection();
            ResetOcrAndMoves();
            LoadAndShowImage(_screenPng);
            AppendInfo("截图完成，请按截图工具习惯拖动框选棋盘区域。");
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

            RenderOverlay();
        }

        private void BtnResetPoints_Click(object sender, RoutedEventArgs e)
        {
            ResetSelection();
            AppendInfo("已重置选择，请在截图上拖动框选棋盘区域。");
        }

        private void ImgScreen_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (_screenBitmap == null) return;
            Point p = e.GetPosition(ImgScreen);

            if (_isPickingExecCorners)
            {
                CaptureExecCorner(p);
                return;
            }

            StartDragSelection(p);
        }

        private void ImgScreen_MouseMove(object sender, MouseEventArgs e)
        {
            if (!_isDragSelecting) return;
            Point p = e.GetPosition(ImgScreen);
            UpdateDragSelection(p);
        }

        private void ImgScreen_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
        {
            if (!_isDragSelecting) return;
            Point p = e.GetPosition(ImgScreen);
            FinishDragSelection(p);
        }

        private bool TryGetImageDisplaySize(out double w, out double h)
        {
            w = ImgScreen.ActualWidth;
            h = ImgScreen.ActualHeight;
            return w > 1 && h > 1 && _screenPixelW > 1 && _screenPixelH > 1;
        }

        private void StartDragSelection(Point p)
        {
            if (!TryGetImageDisplaySize(out _, out _)) return;

            _isDragSelecting = true;
            _isPickingExecCorners = false;
            _dragStart = p;

            _cornerCount = 0;
            Array.Clear(_corners, 0, _corners.Length);

            _execCornerCount = 0;
            _execPointsDisplay.Clear();

            ImgScreen.CaptureMouse();
            UpdateDragSelection(p);
        }

        private void UpdateDragSelection(Point current)
        {
            var rect = new Rect(_dragStart, current);
            _selectedRectDisplay = rect;
            RenderOverlay();
        }

            private void FinishDragSelection(Point end)
        {
            _isDragSelecting = false;
            ImgScreen.ReleaseMouseCapture();

            var rect = new Rect(_dragStart, end);
            if (rect.Width < 4 || rect.Height < 4)
            {
                AppendInfo("框选区域太小，请重新拖动。");
                return;
            }

            _selectedRectDisplay = rect;
            SetCornersFromRect(rect);
            RenderOverlay();
        }

        private void SetCornersFromRect(Rect displayRect)
        {
            if (!TryGetImageDisplaySize(out double w, out double h)) return;

            float x0 = (float)(displayRect.X * _screenPixelW / w);
            float y0 = (float)(displayRect.Y * _screenPixelH / h);
            float x1 = (float)((displayRect.X + displayRect.Width) * _screenPixelW / w);
            float y1 = (float)((displayRect.Y + displayRect.Height) * _screenPixelH / h);

            _corners[0] = x0; _corners[1] = y0;
            _corners[2] = x1; _corners[3] = y0;
            _corners[4] = x1; _corners[5] = y1;
            _corners[6] = x0; _corners[7] = y1;
            _cornerCount = 4;

            AppendInfo($"已框选区域：({x0:0},{y0:0}) - ({x1:0},{y1:0})，可直接 OCR。");
        }

        private void CaptureExecCorner(Point displayPoint)
        {
            if (_execCornerCount >= 4) return;
            if (!TryGetImageDisplaySize(out double w, out double h)) return;

            float px = (float)(displayPoint.X * _screenPixelW / w);
            float py = (float)(displayPoint.Y * _screenPixelH / h);

            int idx = _execCornerCount * 2;
            _execCorners[idx] = px;
            _execCorners[idx + 1] = py;
            _execCornerCount++;

            _execPointsDisplay.Add(displayPoint);
            RenderOverlay();

            string[] names = { "TL", "TR", "BR", "BL" };
            AppendInfo($"执行角点 {names[_execCornerCount - 1]} 已选择: ({px:0},{py:0})");

            if (_execCornerCount == 4)
            {
                _isPickingExecCorners = false;
                AppendInfo("执行用的 4 角点就绪，可点击 Execute。");
            }
        }

        private void DrawPointOnOverlay(Point p, int i, Brush fill, Brush textColor)
        {
            // overlay uses same coordinate space as Image (display space)
            double r = 6;
            var el = new Ellipse
            {
                Width = r * 2,
                Height = r * 2,
                Fill = fill,
                Stroke = Brushes.White,
                StrokeThickness = 2
            };
            Canvas.SetLeft(el, p.X - r);
            Canvas.SetTop(el, p.Y - r);
            Overlay.Children.Add(el);

            var txt = new TextBlock
            {
                Text = (i + 1).ToString(),
                Foreground = textColor,
                FontWeight = FontWeights.Bold
            };
            Canvas.SetLeft(txt, p.X + 6);
            Canvas.SetTop(txt, p.Y - 10);
            Overlay.Children.Add(txt);
        }

        private void DrawSelectionRect(Rect rect)
        {
            var box = new Rectangle
            {
                Width = Math.Abs(rect.Width),
                Height = Math.Abs(rect.Height),
                Stroke = Brushes.LimeGreen,
                StrokeThickness = 2,
                Fill = new SolidColorBrush(Color.FromArgb(40, 0, 255, 0))
            };

            Canvas.SetLeft(box, Math.Min(rect.X, rect.X + rect.Width));
            Canvas.SetTop(box, Math.Min(rect.Y, rect.Y + rect.Height));
            Overlay.Children.Add(box);
        }

        private void RenderOverlay()
        {
            Overlay.Children.Clear();

            if (_selectedRectDisplay.HasValue)
            {
                DrawSelectionRect(_selectedRectDisplay.Value);
            }

            for (int i = 0; i < _execPointsDisplay.Count; i++)
            {
                DrawPointOnOverlay(_execPointsDisplay[i], i, Brushes.DeepSkyBlue, Brushes.White);
            }
        }

        private void ResetSelection()
        {
            _cornerCount = 0;
            _execCornerCount = 0;
            _isDragSelecting = false;
            _isPickingExecCorners = false;
            _selectedRectDisplay = null;
            _execPointsDisplay.Clear();

            Array.Clear(_corners, 0, _corners.Length);
            Array.Clear(_execCorners, 0, _execCorners.Length);

            RenderOverlay();
        }

        private void ResetOcrAndMoves()
        {
            _digits = null;
            _movesBuf = null;
            _moveCount = 0;

            TbOcrInfo.Text = "-";
            LbMoves.Items.Clear();

            BoardGrid.Children.Clear();
            BoardGrid.Rows = 1;
            BoardGrid.Columns = 1;
        }

        private void BtnOcr_Click(object sender, RoutedEventArgs e)
        {
            if (_cornerCount < 4)
            {
                MessageBox.Show("请先在截图上框选棋盘区域。");
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
                _selectedRectDisplay = null;
                _execPointsDisplay.Clear();
                RenderOverlay();
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

            int cornerReady = (_execCornerCount == 4) ? 4 : _cornerCount;
            if (cornerReady < 4)
            {
                MessageBox.Show("需要有效的棋盘框选或执行角点来映射鼠标位置。");
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

            var cornersToUse = (_execCornerCount == 4) ? _execCorners : _corners;
            AppendInfo(_execCornerCount == 4 ? "使用单独选取的执行角点。" : "使用框选区域四角映射执行。");

            int rc = NativeMethods.sum10_execute_path(
                cornersToUse,
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

        private void BtnPickExecCorners_Click(object sender, RoutedEventArgs e)
        {
            if (_screenBitmap == null)
            {
                MessageBox.Show("请先 Capture Screen 并框选棋盘区域。");
                return;
            }

            if (_cornerCount < 4)
            {
                MessageBox.Show("请先用拖拽框选棋盘，再挑选执行角点。");
                return;
            }

            _isPickingExecCorners = true;
            _execCornerCount = 0;
            _execPointsDisplay.Clear();
            Array.Clear(_execCorners, 0, _execCorners.Length);
            RenderOverlay();

            AppendInfo("请在截图上依次点击执行用的四角点：TL -> TR -> BR -> BL");
        }

        private void AppendInfo(string s)
        {
            TbLog.AppendText("[UI] " + s + Environment.NewLine);
            TbLog.ScrollToEnd();
        }
    }
}
