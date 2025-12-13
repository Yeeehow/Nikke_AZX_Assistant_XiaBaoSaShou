#include "sum10_api.h"
#include <windows.h>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include <opencv2/opencv.hpp>

// ---------- internal logging ----------
static sum10_log_callback_t g_log_cb = nullptr;

static void Log(const std::wstring& s) {
    if (g_log_cb) g_log_cb(s.c_str());
}

static std::string WStringToUtf8(const std::wstring& ws) {
    if (ws.empty()) return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), nullptr, 0, nullptr, nullptr);
    std::string out(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws.c_str(), (int)ws.size(), out.data(), len, nullptr, nullptr);
    return out;
}

// ---------- screen capture ----------
static cv::Mat CaptureScreenBGRA() {
    int w = GetSystemMetrics(SM_CXSCREEN);
    int h = GetSystemMetrics(SM_CYSCREEN);

    HDC hScreen = GetDC(nullptr);
    HDC hMem = CreateCompatibleDC(hScreen);
    HBITMAP hBmp = CreateCompatibleBitmap(hScreen, w, h);
    HGDIOBJ old = SelectObject(hMem, hBmp);

    BitBlt(hMem, 0, 0, w, h, hScreen, 0, 0, SRCCOPY | CAPTUREBLT);

    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = w;
    bmi.bmiHeader.biHeight = -h; // top-down
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;

    std::vector<uint8_t> buf((size_t)w * (size_t)h * 4);
    GetDIBits(hMem, hBmp, 0, (UINT)h, buf.data(), &bmi, DIB_RGB_COLORS);

    SelectObject(hMem, old);
    DeleteObject(hBmp);
    DeleteDC(hMem);
    ReleaseDC(nullptr, hScreen);

    cv::Mat bgra(h, w, CV_8UC4, buf.data());
    return bgra.clone();
}

// ---------- perspective warp ----------
static cv::Mat UnwarpBoard(const cv::Mat& imgBgr, const float* c8) {
    cv::Point2f tl(c8[0], c8[1]);
    cv::Point2f tr(c8[2], c8[3]);
    cv::Point2f br(c8[4], c8[5]);
    cv::Point2f bl(c8[6], c8[7]);

    auto dist = [](cv::Point2f a, cv::Point2f b) {
        auto d = a - b;
        return std::sqrt(d.x * d.x + d.y * d.y);
        };

    int maxW = (int)std::max(dist(br, bl), dist(tr, tl));
    int maxH = (int)std::max(dist(tr, br), dist(tl, bl));
    maxW = std::max(maxW, 10);
    maxH = std::max(maxH, 10);

    std::array<cv::Point2f, 4> src = { tl, tr, br, bl };
    std::array<cv::Point2f, 4> dst = {
        cv::Point2f(0,0),
        cv::Point2f((float)maxW - 1, 0),
        cv::Point2f((float)maxW - 1, (float)maxH - 1),
        cv::Point2f(0, (float)maxH - 1)
    };

    cv::Mat M = cv::getPerspectiveTransform(src.data(), dst.data());
    cv::Mat warped;
    cv::warpPerspective(imgBgr, warped, M, cv::Size(maxW, maxH));
    return warped;
}

// ---------- template OCR ----------
struct DigitTemplates {
    // index 0..9, empty if missing
    cv::Mat t[10];
    bool loaded[10]{};
};

static DigitTemplates LoadTemplates(const std::wstring& dir) {
    DigitTemplates dt;
    for (int d = 0; d <= 9; d++) {
        std::wstring wpath = dir + L"\\" + std::to_wstring(d) + L".png";
        std::string path = WStringToUtf8(wpath);
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) continue;

        cv::Mat r;
        cv::resize(img, r, cv::Size(28, 28));
        r.convertTo(r, CV_32F, 1.0 / 255.0);
        dt.t[d] = r;
        dt.loaded[d] = true;
    }
    return dt;
}

static void PreprocessCell(const cv::Mat& cellBgr, cv::Mat& out28f) {
    cv::Mat gray;
    cv::cvtColor(cellBgr, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);

    cv::Mat bin;
    cv::threshold(blur, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 让数字为白色，背景黑色（若相反则取反）
    double meanVal = cv::mean(bin)[0];
    if (meanVal > 127) {
        cv::bitwise_not(bin, bin);
    }

    // 简单去噪
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
    cv::morphologyEx(bin, bin, cv::MORPH_OPEN, kernel);

    cv::Mat resized;
    cv::resize(bin, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA);

    resized.convertTo(out28f, CV_32F, 1.0 / 255.0);
}

static float CorrCoeff(const cv::Mat& a28f, const cv::Mat& b28f) {
    // a,b: 28x28 CV_32F
    cv::Mat ra = a28f.reshape(1, 1);
    cv::Mat rb = b28f.reshape(1, 1);

    // 归一化相关系数（手写版本，避免 matchTemplate 限制）
    cv::Scalar ma = cv::mean(ra);
    cv::Scalar mb = cv::mean(rb);

    cv::Mat da = ra - ma[0];
    cv::Mat db = rb - mb[0];

    double num = da.dot(db);
    double den = std::sqrt(da.dot(da) * db.dot(db)) + 1e-9;
    return (float)(num / den);
}

static int OcrBoardByTemplates(
    const cv::Mat& warpedBgr,
    int rows,
    int cols,
    const DigitTemplates& dt,
    std::vector<int>& outDigits,
    float& outAvgConf
) {
    outDigits.assign(rows * cols, 0);

    int W = warpedBgr.cols;
    int H = warpedBgr.rows;

    int cellW = W / cols;
    int cellH = H / rows;
    if (cellW <= 0 || cellH <= 0) return -10;

    float confSum = 0.0f;
    int confCount = 0;

    // 可调：裁掉网格线（留边）
    int marginX = (int)(cellW * 0.12f);
    int marginY = (int)(cellH * 0.12f);

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int x0 = c * cellW + marginX;
            int y0 = r * cellH + marginY;
            int x1 = (c + 1) * cellW - marginX;
            int y1 = (r + 1) * cellH - marginY;

            x0 = std::clamp(x0, 0, W - 1);
            y0 = std::clamp(y0, 0, H - 1);
            x1 = std::clamp(x1, x0 + 1, W);
            y1 = std::clamp(y1, y0 + 1, H);

            cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
            cv::Mat cell = warpedBgr(roi);

            cv::Mat cell28f;
            PreprocessCell(cell, cell28f);

            int bestD = 0;
            float bestScore = -1e9f;

            for (int d = 0; d <= 9; d++) {
                if (!dt.loaded[d]) continue;
                float s = CorrCoeff(cell28f, dt.t[d]);
                if (s > bestScore) {
                    bestScore = s;
                    bestD = d;
                }
            }

            outDigits[r * cols + c] = bestD;
            confSum += bestScore;
            confCount++;
        }
    }

    outAvgConf = (confCount > 0) ? (confSum / confCount) : 0.0f;
    return 0;
}

// ---------- greedy solver (sum==10, full-active rectangle) ----------
struct Move { int r1, c1, r2, c2; };

static int RectSum(const std::vector<int>& ps, int cols, int r1, int c1, int r2, int c2) {
    // prefix sum ps indexed (r,c) on (rows+1)x(cols+1)
    auto idx = [cols](int r, int c) { return r * (cols + 1) + c; };
    int A = ps[idx(r2 + 1, c2 + 1)];
    int B = ps[idx(r1, c2 + 1)];
    int C = ps[idx(r2 + 1, c1)];
    int D = ps[idx(r1, c1)];
    return A - B - C + D;
}

static void BuildPrefix(
    const std::vector<int>& digits,
    const std::vector<uint8_t>& active,
    int rows,
    int cols,
    std::vector<int>& psSum,
    std::vector<int>& psAct
) {
    psSum.assign((rows + 1) * (cols + 1), 0);
    psAct.assign((rows + 1) * (cols + 1), 0);

    auto idx = [cols](int r, int c) { return r * (cols + 1) + c; };
    auto cell = [cols](int r, int c) { return r * cols + c; };

    for (int r = 1; r <= rows; r++) {
        for (int c = 1; c <= cols; c++) {
            int rr = r - 1, cc = c - 1;
            int a = active[cell(rr, cc)] ? 1 : 0;
            int v = active[cell(rr, cc)] ? digits[cell(rr, cc)] : 0;

            psSum[idx(r, c)] = psSum[idx(r - 1, c)] + psSum[idx(r, c - 1)] - psSum[idx(r - 1, c - 1)] + v;
            psAct[idx(r, c)] = psAct[idx(r - 1, c)] + psAct[idx(r, c - 1)] - psAct[idx(r - 1, c - 1)] + a;
        }
    }
}

static int SolveGreedySum10(
    const int* digitsIn,
    int rows,
    int cols,
    std::vector<Move>& outMoves,
    int& outScore
) {
    std::vector<int> digits(rows * cols);
    for (int i = 0; i < rows * cols; i++) digits[i] = digitsIn[i];

    std::vector<uint8_t> active(rows * cols, 1);
    outMoves.clear();
    outScore = 0;

    while (true) {
        std::vector<int> psSum, psAct;
        BuildPrefix(digits, active, rows, cols, psSum, psAct);

        bool found = false;
        Move best{};
        int bestArea = 0;

        // O(R^2*C^2) 足够跑小棋盘
        for (int r1 = 0; r1 < rows; r1++) {
            for (int r2 = r1; r2 < rows; r2++) {
                for (int c1 = 0; c1 < cols; c1++) {
                    for (int c2 = c1; c2 < cols; c2++) {
                        int area = (r2 - r1 + 1) * (c2 - c1 + 1);
                        int actCnt = RectSum(psAct, cols, r1, c1, r2, c2);
                        if (actCnt != area) continue;

                        int s = RectSum(psSum, cols, r1, c1, r2, c2);
                        if (s != 10) continue;

                        // 贪心：优先选更大面积
                        if (area > bestArea) {
                            bestArea = area;
                            best = { r1, c1, r2, c2 };
                            found = true;
                        }
                    }
                }
            }
        }

        if (!found) break;

        // apply remove
        for (int r = best.r1; r <= best.r2; r++) {
            for (int c = best.c1; c <= best.c2; c++) {
                active[r * cols + c] = 0;
            }
        }

        outMoves.push_back(best);
        outScore += bestArea;

        if (outMoves.size() > 5000) break; // safety
    }

    return 0;
}

// ---------- automation (SendInput) ----------
static void EnsureDpiAware() {
    // Windows 10+：per-monitor v2
    HMODULE user32 = GetModuleHandleW(L"user32.dll");
    if (!user32) return;

    using Fn = BOOL(WINAPI*)(DPI_AWARENESS_CONTEXT);
    auto fn = (Fn)GetProcAddress(user32, "SetProcessDpiAwarenessContext");
    if (fn) fn(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
}

static void MoveMouseAbs(int x, int y) {
    int sw = GetSystemMetrics(SM_CXSCREEN);
    int sh = GetSystemMetrics(SM_CYSCREEN);
    x = std::clamp(x, 0, sw - 1);
    y = std::clamp(y, 0, sh - 1);

    INPUT in{};
    in.type = INPUT_MOUSE;
    in.mi.dwFlags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
    in.mi.dx = (LONG)std::lround(x * 65535.0 / (sw - 1));
    in.mi.dy = (LONG)std::lround(y * 65535.0 / (sh - 1));
    SendInput(1, &in, sizeof(INPUT));
}

static void MouseDown() {
    INPUT in{};
    in.type = INPUT_MOUSE;
    in.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &in, sizeof(INPUT));
}

static void MouseUp() {
    INPUT in{};
    in.type = INPUT_MOUSE;
    in.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &in, sizeof(INPUT));
}

static POINT CellCenterScreen(const float* c8, int rows, int cols, int r, int c, int offX, int offY) {
    cv::Point2f tl(c8[0], c8[1]);
    cv::Point2f tr(c8[2], c8[3]);
    cv::Point2f br(c8[4], c8[5]);
    cv::Point2f bl(c8[6], c8[7]);

    float u = (c + 0.5f) / (float)cols;
    float v = (r + 0.5f) / (float)rows;

    cv::Point2f top = tl + u * (tr - tl);
    cv::Point2f bot = bl + u * (br - bl);
    cv::Point2f p = top + v * (bot - top);

    POINT pt;
    pt.x = (LONG)std::lround(p.x) + offX;
    pt.y = (LONG)std::lround(p.y) + offY;
    return pt;
}

static void Drag(POINT a, POINT b, int delayMs) {
    MoveMouseAbs(a.x, a.y);
    Sleep(20);
    MouseDown();
    Sleep(25);
    MoveMouseAbs(b.x, b.y);
    Sleep(25);
    MouseUp();
    Sleep(std::max(delayMs, 0));
}

// ---------- exported API ----------
void SUM10_CALL sum10_set_log_callback(sum10_log_callback_t cb) {
    g_log_cb = cb;
    Log(L"[Native] log callback set.");
}

int SUM10_CALL sum10_capture_screen_png(const wchar_t* outPngPath) {
    try {
        cv::Mat bgra = CaptureScreenBGRA();
        cv::Mat bgr;
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);

        std::string path = WStringToUtf8(outPngPath ? outPngPath : L"screen.png");
        bool ok = cv::imwrite(path, bgr);
        if (!ok) return -2;

        Log(L"[Native] captured screen -> png");
        return 0;
    }
    catch (...) {
        return -1;
    }
}

int SUM10_CALL sum10_ocr_board(
    const wchar_t* screenPngPath,
    const float* corners8,
    int rows,
    int cols,
    const wchar_t* digitTemplateDir,
    int* outDigits,
    float* outAvgConf,
    const wchar_t* outWarpPngPath
) {
    try {
        if (!screenPngPath || !corners8 || !digitTemplateDir || !outDigits || !outAvgConf || !outWarpPngPath) return -1;
        if (rows <= 0 || cols <= 0) return -2;

        std::string sp = WStringToUtf8(screenPngPath);
        cv::Mat screen = cv::imread(sp, cv::IMREAD_COLOR);
        if (screen.empty()) return -3;

        cv::Mat warped = UnwarpBoard(screen, corners8);

        // save warped preview
        std::string wp = WStringToUtf8(outWarpPngPath);
        cv::imwrite(wp, warped);

        DigitTemplates dt = LoadTemplates(digitTemplateDir);
        std::vector<int> digits;
        float avgConf = 0.0f;

        int rc = OcrBoardByTemplates(warped, rows, cols, dt, digits, avgConf);
        if (rc != 0) return rc;

        for (int i = 0; i < rows * cols; i++) outDigits[i] = digits[i];
        *outAvgConf = avgConf;

        Log(L"[Native] OCR done.");
        return 0;
    }
    catch (...) {
        return -100;
    }
}

int SUM10_CALL sum10_solve_greedy(
    const int* digits,
    int rows,
    int cols,
    int* outMoves,
    int maxMoves,
    int* outMoveCount,
    int* outScore
) {
    try {
        if (!digits || !outMoves || !outMoveCount || !outScore) return -1;
        if (rows <= 0 || cols <= 0 || maxMoves <= 0) return -2;

        std::vector<Move> moves;
        int score = 0;
        SolveGreedySum10(digits, rows, cols, moves, score);

        int n = (int)std::min((size_t)maxMoves, moves.size());
        for (int i = 0; i < n; i++) {
            outMoves[i * 4 + 0] = moves[i].r1;
            outMoves[i * 4 + 1] = moves[i].c1;
            outMoves[i * 4 + 2] = moves[i].r2;
            outMoves[i * 4 + 3] = moves[i].c2;
        }

        *outMoveCount = n;
        *outScore = score;

        Log(L"[Native] solve greedy done.");
        return 0;
    }
    catch (...) {
        return -100;
    }
}

int SUM10_CALL sum10_execute_path(
    const float* corners8,
    int rows,
    int cols,
    const int* moves,
    int moveCount,
    int offsetX,
    int offsetY,
    int delayMs
) {
    try {
        if (!corners8 || !moves) return -1;
        if (rows <= 0 || cols <= 0 || moveCount < 0) return -2;

        EnsureDpiAware();
        Log(L"[Native] executing path...");

        for (int i = 0; i < moveCount; i++) {
            int r1 = moves[i * 4 + 0];
            int c1 = moves[i * 4 + 1];
            int r2 = moves[i * 4 + 2];
            int c2 = moves[i * 4 + 3];

            POINT a = CellCenterScreen(corners8, rows, cols, r1, c1, offsetX, offsetY);
            POINT b = CellCenterScreen(corners8, rows, cols, r2, c2, offsetX, offsetY);

            Drag(a, b, delayMs);
        }

        Log(L"[Native] execute done.");
        return 0;
    }
    catch (...) {
        return -100;
    }
}
