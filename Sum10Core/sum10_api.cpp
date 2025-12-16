#include "sum10_api.h"
#include <windows.h>
#include <string>
#include <vector>
#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <memory>
#include <thread>
#include <future>
#include <chrono>
#include <random>
#include <mutex>

#include <opencv2/opencv.hpp>
#if defined(HAVE_OPENCV_CUDAIMGPROC)
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#endif

// ---------- internal logging ----------
static sum10_log_callback_t g_log_cb = nullptr;

static void Log(const std::wstring& s) {
    if (g_log_cb) g_log_cb(s.c_str());
}

// ---------- cuda helpers ----------
#if defined(HAVE_OPENCV_CUDAIMGPROC)
static bool g_cuda_available = false;
static std::once_flag g_cuda_once;

static void DetectCudaOnce() {
    try {
        g_cuda_available = cv::cuda::getCudaEnabledDeviceCount() > 0;
        if (g_cuda_available) {
            Log(L"[Native] CUDA device detected; enabling GPU-accelerated paths.");
        }
        else {
            Log(L"[Native] CUDA modules available but no device detected; using CPU fallback.");
        }
    }
    catch (...) {
        g_cuda_available = false;
    }
}

static bool IsCudaAvailable() {
    std::call_once(g_cuda_once, DetectCudaOnce);
    return g_cuda_available;
}
#else
static bool IsCudaAvailable() { return false; }
#endif

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

// Small helper to avoid requiring C++17 std::clamp.
static inline int ClampInt(int v, int lo, int hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}

// ---------- template OCR ----------
struct DigitTemplates {
    // index 0..9, allow multiple templates per digit
    std::vector<cv::Mat> t[10];
    bool loaded[10]{};
#if defined(HAVE_OPENCV_CUDAIMGPROC)
    std::vector<cv::cuda::GpuMat> gpu_t[10];
    bool gpu_loaded[10]{};
#endif
};

static float ComputeMedian(const cv::Mat& gray) {
    CV_Assert(gray.type() == CV_8U);
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    cv::Mat hist;
    int channels[] = { 0 };
    cv::calcHist(&gray, 1, channels, cv::Mat(), hist, 1, &histSize, ranges, true, false);
    float cum = 0.0f;
    float half = (float)(gray.total() / 2.0);
    for (int i = 0; i < histSize; i++) {
        cum += hist.at<float>(i);
        if (cum >= half) return (float)i;
    }
    return 127.0f;
}

static std::vector<std::wstring> SplitTemplateDirs(const std::wstring& dirList) {
    std::wstringstream ss(dirList);
    std::wstring token;
    std::vector<std::wstring> dirs;

    while (std::getline(ss, token, L';')) {
        if (!token.empty()) dirs.push_back(token);
    }

    // also allow '|' separator for convenience
    if (dirs.empty()) {
        std::wstring t;
        std::wstringstream ss2(dirList);
        while (std::getline(ss2, t, L'|')) {
            if (!t.empty()) dirs.push_back(t);
        }
    }

    if (dirs.empty() && !dirList.empty()) dirs.push_back(dirList);
    return dirs;
}

static DigitTemplates LoadTemplates(const std::wstring& dirList) {
    DigitTemplates dt;
    auto dirs = SplitTemplateDirs(dirList);
    bool useCuda = IsCudaAvailable();

    for (const auto& dir : dirs) {
        for (int d = 0; d <= 9; d++) {
            std::wstring pattern = dir + L"\\" + std::to_wstring(d) + L"*.png";
            std::string spattern = WStringToUtf8(pattern);

            std::vector<std::string> files;
            cv::glob(spattern, files, false);
            for (const auto& path : files) {
                cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
                if (img.empty()) continue;

                cv::Mat r;
                cv::resize(img, r, cv::Size(28, 28));
                r.convertTo(r, CV_32F, 1.0 / 255.0);
                dt.t[d].push_back(r);
                dt.loaded[d] = true;

#if defined(HAVE_OPENCV_CUDAIMGPROC)
                if (useCuda) {
                    cv::cuda::GpuMat g;
                    g.upload(r);
                    dt.gpu_t[d].push_back(std::move(g));
                    dt.gpu_loaded[d] = true;
                }
#endif
            }
        }
    }

    std::wstring summary = L"Loaded digit templates from ";
    for (size_t i = 0; i < dirs.size(); ++i) {
        summary += dirs[i];
        if (i + 1 < dirs.size()) summary += L"; ";
    }
    Log(summary);

    for (int d = 0; d <= 9; ++d) {
        if (dt.t[d].empty()) continue;
        Log(L"  digit " + std::to_wstring(d) + L" templates: " + std::to_wstring(dt.t[d].size()));
    }

    return dt;
}

static bool HasAnyTemplate(const DigitTemplates& dt) {
    for (int d = 0; d <= 9; ++d) {
        if (!dt.t[d].empty()) return true;
    }
    return false;
}

struct PreprocessResult {
    cv::Mat mat28f;
    float fgRatio{ 0.0f };
    float quality{ 0.0f };
#if defined(HAVE_OPENCV_CUDAIMGPROC)
    cv::cuda::GpuMat gpu28f;
    bool hasGpu{ false };
#endif
};

static cv::Mat RefineBounding(const cv::Mat& bin) {
    cv::Mat nz;
    cv::findNonZero(bin, nz);
    if (nz.empty()) return bin;

    cv::Rect box = cv::boundingRect(nz);
    // small margin to avoid cutting strokes
    box.x = std::max(box.x - 1, 0);
    box.y = std::max(box.y - 1, 0);
    box.width = std::min(box.width + 2, bin.cols - box.x);
    box.height = std::min(box.height + 2, bin.rows - box.y);
    return bin(box);
}

static std::vector<PreprocessResult> PreprocessCellVariants(const cv::Mat& cellBgr) {
    cv::Mat gray;
    bool useCuda = IsCudaAvailable();

#if defined(HAVE_OPENCV_CUDAIMGPROC)
    if (useCuda) {
        try {
            cv::cuda::GpuMat gBgr(cellBgr);
            cv::cuda::GpuMat gGray;
            cv::cuda::cvtColor(gBgr, gGray, cv::COLOR_BGR2GRAY);
            gGray.download(gray);
        }
        catch (...) {
            useCuda = false;
        }
    }
#endif
    if (!useCuda) {
        cv::cvtColor(cellBgr, gray, cv::COLOR_BGR2GRAY);
    }

    cv::Mat blur;
    cv::GaussianBlur(gray, blur, cv::Size(3, 3), 0);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    cv::Mat claheGray;
    clahe->apply(gray, claheGray);

    cv::Mat equalized;
    cv::equalizeHist(gray, equalized);

    struct Variant { cv::Mat base; std::string name; };
    std::vector<Variant> bases = {
        { blur, "blur" },
        { claheGray, "clahe" },
        { equalized, "equalized" }
    };

    std::vector<PreprocessResult> results;
    for (const auto& base : bases) {
        float median = ComputeMedian(base.base);

        // two thresholding strategies: otsu and adaptive mean
        std::vector<cv::Mat> bins;
        cv::Mat otsu;
        cv::threshold(base.base, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        bins.push_back(otsu);

        cv::Mat adap;
        cv::adaptiveThreshold(base.base, adap, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, 5);
        bins.push_back(adap);

        cv::Mat adapGauss;
        cv::adaptiveThreshold(base.base, adapGauss, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 17, 3);
        bins.push_back(adapGauss);

        cv::Mat edges;
        cv::Canny(base.base, edges, 24, 72);
        bins.push_back(edges);

        for (auto& bin : bins) {
            float fgRatio = (float)cv::countNonZero(bin) / (float)(bin.total() + 1e-6f);
            bool invert = (median > 120.0f && fgRatio < 0.5f) || fgRatio > 0.6f;
            if (invert) cv::bitwise_not(bin, bin);

            std::vector<cv::Mat> processed;
            processed.push_back(bin);

            for (int k = 2; k <= 3; k++) {
                cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));

                cv::Mat opened;
                cv::morphologyEx(bin, opened, cv::MORPH_OPEN, kernel);
                processed.push_back(opened);

                cv::Mat closed;
                cv::morphologyEx(bin, closed, cv::MORPH_CLOSE, kernel);
                processed.push_back(closed);

                if (k == 2) {
                    cv::Mat dilated;
                    cv::dilate(bin, dilated, kernel);
                    processed.push_back(dilated);
                }
            }

            for (auto& proc : processed) {
                cv::Mat trimmed = RefineBounding(proc);
                cv::Mat resized;
                cv::resize(trimmed, resized, cv::Size(28, 28), 0, 0, cv::INTER_AREA);

                cv::Mat f;
                cv::Scalar mean, stddev;
                cv::meanStdDev(resized, mean, stddev);
                float stdVal = (float)stddev[0];
                if (stdVal > 1e-3f) {
                    cv::Mat normalized;
                    resized.convertTo(normalized, CV_32F);
                    normalized = (normalized - mean[0]) / (stdVal * 4.0f) + 0.5f;
                    cv::threshold(normalized, normalized, 1.0, 1.0, cv::THRESH_TRUNC);
                    cv::threshold(normalized, normalized, 0.0, 0.0, cv::THRESH_TOZERO);
                    f = normalized;
                }
                else {
                    resized.convertTo(f, CV_32F, 1.0 / 255.0);
                }

                float var = stdVal * stdVal;
                float fg = (float)cv::countNonZero(resized) / (float)(resized.total() + 1e-6f);

                cv::Mat lap;
                cv::Laplacian(resized, lap, CV_32F);
                cv::Scalar lapMean, lapStd;
                cv::meanStdDev(lap, lapMean, lapStd);
                float edgeEnergy = (float)(lapStd[0] * lapStd[0]);

                float quality = var + 0.3f * edgeEnergy;
                if (fg >= 0.06f && fg <= 0.5f) quality += 0.6f;
                else if (fg < 0.02f || fg > 0.65f) quality -= 0.6f;

                PreprocessResult pr{ f, fg, quality };
#if defined(HAVE_OPENCV_CUDAIMGPROC)
                if (useCuda) {
                    try {
                        pr.gpu28f.upload(f);
                        pr.hasGpu = true;
                    }
                    catch (...) {
                        pr.hasGpu = false;
                    }
                }
#endif
                results.push_back(std::move(pr));
            }
        }
    }

    if (results.empty()) {
        PreprocessResult r;
        cellBgr.convertTo(r.mat28f, CV_32F, 1.0 / 255.0);
#if defined(HAVE_OPENCV_CUDAIMGPROC)
        if (useCuda) {
            try {
                r.gpu28f.upload(r.mat28f);
                r.hasGpu = true;
            }
            catch (...) {
                r.hasGpu = false;
            }
        }
#endif
        results.push_back(r);
    }

    return results;
}

static float CorrCoeff(const cv::Mat& a28f, const cv::Mat& b28f) {
    // a,b: 28x28 CV_32F
    cv::Mat ra = a28f.reshape(1, 1);
    cv::Mat rb = b28f.reshape(1, 1);

    // 一个老版本的手写 matchTemplate 等效
    cv::Scalar ma = cv::mean(ra);
    cv::Scalar mb = cv::mean(rb);

    cv::Mat da = ra - ma[0];
    cv::Mat db = rb - mb[0];

    double num = da.dot(db);
    double den = std::sqrt(da.dot(da) * db.dot(db)) + 1e-9;
    return (float)(num / den);
}

static float BestDigitScore(const PreprocessResult& cell, const DigitTemplates& dt, int& bestDigit) {
    float bestScore = -1e9f;
    bestDigit = 0;
    bool useCuda = IsCudaAvailable();

#if defined(HAVE_OPENCV_CUDAIMGPROC)
    cv::cuda::GpuMat cellGpu = cell.hasGpu ? cell.gpu28f : cv::cuda::GpuMat();
    if (useCuda && cellGpu.empty()) useCuda = false;
#endif

    for (int d = 0; d <= 9; d++) {
        if (!dt.loaded[d]) continue;
        const auto& temps = dt.t[d];
        if (temps.empty()) continue;

        float top1 = -1e9f, top2 = -1e9f, top3 = -1e9f;
        
#if defined(HAVE_OPENCV_CUDAIMGPROC)
        if (useCuda && dt.gpu_loaded[d] && !dt.gpu_t[d].empty()) {
            for (const auto& t : dt.gpu_t[d]) {
                cv::cuda::GpuMat result;
                cv::cuda::matchTemplate(cellGpu, t, result, cv::TM_CCOEFF_NORMED);
                cv::Mat host;
                result.download(host);
                float s = host.at<float>(0, 0);
                if (s > top1) {
                    top3 = top2;
                    top2 = top1;
                    top1 = s;
                }
                else if (s > top2) {
                    top3 = top2;
                    top2 = s;
                }
                else if (s > top3) {
                    top3 = s;
                }
            }
        }
        else
#endif
        {
            for (const auto& t : temps) {
                float s = CorrCoeff(cell.mat28f, t);
                if (s > top1) {
                    top3 = top2;
                    top2 = top1;
                    top1 = s;
                }
                else if (s > top2) {
                    top3 = top2;
                    top2 = s;
                }
                else if (s > top3) {
                    top3 = s;
                }
            }
        }

        float score = top1;
        if (temps.size() >= 3) score = top1 * 0.6f + top2 * 0.25f + top3 * 0.15f;
        else if (temps.size() >= 2) score = top1 * 0.7f + top2 * 0.3f;

        if (top2 > -0.5f && (top1 - top2) < 0.08f) {
            score -= 0.04f; // down-weight ambiguous matches
        }

        if (score > bestScore) {
            bestScore = score;
            bestDigit = d;
        }
    }

    return bestScore;
}

static int OcrBoardByTemplates(
    const cv::Mat& warpedBgr,
    int rows,
    int cols,
    const DigitTemplates& dt,
    std::vector<int>& outDigits,
    float& outAvgConf,
    std::vector<float>* outCellConfs
) {
    outDigits.assign(rows * cols, 0);
    if (outCellConfs) outCellConfs->assign(rows * cols, 0.0f);

    int W = warpedBgr.cols;
    int H = warpedBgr.rows;

    int cellW = W / cols;
    int cellH = H / rows;
    if (cellW <= 0 || cellH <= 0) return -10;

    float confSum = 0.0f;
    int confCount = 0;

    const float lowConfThresh = 0.62f;

    // 预留安全边距，避免边线干扰
    int marginX = (int)(cellW * 0.10f);
    int marginY = (int)(cellH * 0.10f);

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

            auto variants = PreprocessCellVariants(cell);
            std::sort(variants.begin(), variants.end(), [](const PreprocessResult& a, const PreprocessResult& b) {
                return a.quality > b.quality;
                });

            int bestD = 0;
            float bestScore = -1e9f;
            float bestQuality = -1e9f;

            for (const auto& var : variants) {
                int digit = 0;
                float score = BestDigitScore(var, dt, digit);

                // prefer higher score; tie-break with preprocess quality
                if (score > bestScore || (std::abs(score - bestScore) < 1e-4f && var.quality > bestQuality)) {
                    bestScore = score;
                    bestD = digit;
                    bestQuality = var.quality;
                }
            }

            // fallback: if still low confidence, retry with the second-best pre-process candidate
            if (bestScore < lowConfThresh && variants.size() > 1) {
                const auto& alt = variants[std::min<size_t>(1, variants.size() - 1)];
                int digit = 0;
                float score = BestDigitScore(alt, dt, digit);
                if (score > bestScore) {
                    bestScore = score;
                    bestD = digit;
                }
            }

            int idx = r * cols + c;
            outDigits[idx] = bestD;
            if (outCellConfs) (*outCellConfs)[idx] = bestScore;
            confSum += bestScore;
            confCount++;
        }
    }

    outAvgConf = (confCount > 0) ? (confSum / confCount) : 0.0f;
    return 0;
}

// ---------- greedy solver (sum==10, full-active rectangle) ----------
struct Move { int r1, c1, r2, c2; };

// ---------- GodBrain V6.2 solver (translated from god_brain_v62.py) ----------
// NOTE:
// - This implementation follows the same search strategy / rules / heuristics as the Python version.
// - RNG streams are independent ("python-random" vs "numpy-random"), but the exact sequences may differ
//   from CPython/NumPy because we use std::mt19937.

struct GBRect { int r1, c1, r2, c2; };
struct GBMove { int r1, c1, r2, c2, count; };

struct GBWeights {
    int w_island{ 0 };
    double w_fragment{ 0.0 };
};

enum class GBMode : int {
    God = 0,
    Classic = 1,
    Omni = 2
};

struct GBPathNode {
    GBRect move;
    std::shared_ptr<GBPathNode> prev;
    int len{ 0 };
};

struct GBState {
    std::vector<uint8_t> map; // 0/1 active
    int score{ 0 };
    double h_score{ 0.0 };
    std::shared_ptr<GBPathNode> path; // persistent list
};

static std::shared_ptr<GBPathNode> GBAppendPath(const std::shared_ptr<GBPathNode>& prev, const GBRect& m) {
    auto node = std::make_shared<GBPathNode>();
    node->move = m;
    node->prev = prev;
    node->len = (prev ? (prev->len + 1) : 1);
    return node;
}

static void GBMaterializePath(const std::shared_ptr<GBPathNode>& tail, std::vector<GBRect>& out) {
    out.clear();
    if (!tail) return;
    out.reserve((size_t)tail->len);
    for (auto p = tail; p; p = p->prev) out.push_back(p->move);
    std::reverse(out.begin(), out.end());
}

static inline int GBRectSum1D(const std::vector<int>& P, int cols, int r1, int c1, int r2, int c2) {
    // prefix P indexed (r,c) on (rows+1)x(cols+1)
    auto idx = [cols](int r, int c) { return r * (cols + 1) + c; };
    int A = P[idx(r2 + 1, c2 + 1)];
    int B = P[idx(r1, c2 + 1)];
    int C = P[idx(r2 + 1, c1)];
    int D = P[idx(r1, c1)];
    return A - B - C + D;
}

static void GBBuildPrefixValCnt(
    const std::vector<uint8_t>& map,
    const std::vector<int>& vals,
    int rows,
    int cols,
    std::vector<int>& P_val,
    std::vector<int>& P_cnt
) {
    P_val.assign((size_t)(rows + 1) * (size_t)(cols + 1), 0);
    P_cnt.assign((size_t)(rows + 1) * (size_t)(cols + 1), 0);

    auto idxP = [cols](int r, int c) { return r * (cols + 1) + c; };
    auto idxC = [cols](int r, int c) { return r * cols + c; };

    for (int r = 1; r <= rows; r++) {
        for (int c = 1; c <= cols; c++) {
            int rr = r - 1, cc = c - 1;
            int ci = idxC(rr, cc);
            int a = map[(size_t)ci] ? 1 : 0;
            int v = map[(size_t)ci] ? vals[(size_t)ci] : 0;

            P_val[(size_t)idxP(r, c)] = P_val[(size_t)idxP(r - 1, c)] + P_val[(size_t)idxP(r, c - 1)] - P_val[(size_t)idxP(r - 1, c - 1)] + v;
            P_cnt[(size_t)idxP(r, c)] = P_cnt[(size_t)idxP(r - 1, c)] + P_cnt[(size_t)idxP(r, c - 1)] - P_cnt[(size_t)idxP(r - 1, c - 1)] + a;
        }
    }
}

#if defined(HAVE_OPENCV_CUDAIMGPROC)
static bool GBBuildPrefixValCntCuda(
    const std::vector<uint8_t>& map,
    const std::vector<int>& vals,
    int rows,
    int cols,
    std::vector<int>& P_val,
    std::vector<int>& P_cnt
) {
    if (!IsCudaAvailable()) return false;

    try {
        cv::Mat mapMat(rows, cols, CV_8UC1);
        cv::Mat valMat(rows, cols, CV_32SC1);
        auto idx = [cols](int r, int c) { return r * cols + c; };
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int i = idx(r, c);
                mapMat.at<uint8_t>(r, c) = map[(size_t)i] ? 1 : 0;
                valMat.at<int>(r, c) = map[(size_t)i] ? vals[(size_t)i] : 0;
            }
        }

        cv::cuda::GpuMat gMap(mapMat);
        cv::cuda::GpuMat gVal(valMat);
        cv::cuda::GpuMat gCntPrefix, gValPrefix;

        cv::cuda::integral(gMap, gCntPrefix, CV_32S);
        cv::cuda::integral(gVal, gValPrefix, CV_32S);

        cv::Mat hCnt, hVal;
        gCntPrefix.download(hCnt);
        gValPrefix.download(hVal);

        const size_t total = (size_t)(rows + 1) * (size_t)(cols + 1);
        P_cnt.assign((const int*)hCnt.ptr<int>(0), (const int*)hCnt.ptr<int>(0) + total);
        P_val.assign((const int*)hVal.ptr<int>(0), (const int*)hVal.ptr<int>(0) + total);
        return true;
    }
    catch (...) {
        return false;
    }
}
#endif

static void GBBuildPrefixValCntAuto(
    const std::vector<uint8_t>& map,
    const std::vector<int>& vals,
    int rows,
    int cols,
    std::vector<int>& P_val,
    std::vector<int>& P_cnt
) {
#if defined(HAVE_OPENCV_CUDAIMGPROC)
    if (GBBuildPrefixValCntCuda(map, vals, rows, cols, P_val, P_cnt)) return;
#endif
    GBBuildPrefixValCnt(map, vals, rows, cols, P_val, P_cnt);
}

static int GBCountIslands(const std::vector<uint8_t>& map, int rows, int cols) {
    int islands = 0;
    auto idx = [cols](int r, int c) { return r * cols + c; };

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int i = idx(r, c);
            if (!map[(size_t)i]) continue;

            // if any 4-neighbor is active -> not an island
            if (r > 0 && map[(size_t)idx(r - 1, c)]) continue;
            if (r < rows - 1 && map[(size_t)idx(r + 1, c)]) continue;
            if (c > 0 && map[(size_t)idx(r, c - 1)]) continue;
            if (c < cols - 1 && map[(size_t)idx(r, c + 1)]) continue;

            islands++;
        }
    }
    return islands;
}

static double GBU01(std::mt19937& rng) {
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

static int GBUniformInt(std::mt19937& rng, int lo, int hi) {
    std::uniform_int_distribution<int> dist(lo, hi);
    return dist(rng);
}

static double GBEvaluateState(
    int score,
    const std::vector<uint8_t>& map,
    int rows,
    int cols,
    const GBWeights& w,
    std::mt19937& rng_np
) {
    // mirrors god_brain_v62.py:_evaluate_state
    double h = (double)score * 2000.0;

    if (w.w_island > 0) {
        int islands = GBCountIslands(map, rows, cols);
        h -= (double)islands * (double)w.w_island;
    }

    if (w.w_fragment > 0.0) {
        int center_r = rows / 2;
        int center_c = cols / 2;
        double center_mass = 0.0;
        auto idx = [cols](int r, int c) { return r * cols + c; };

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (!map[(size_t)idx(r, c)]) continue;
                int dist = std::abs(r - center_r) + std::abs(c - center_c);
                center_mass += (20.0 - (double)dist);
            }
        }

        h -= center_mass * w.w_fragment;
    }

    double noise_level = 50.0;
    if (w.w_island < 20 && w.w_fragment < 1.0) noise_level = 2000.0;
    h += GBU01(rng_np) * noise_level;
    return h;
}

static void GBGetActiveIndices(const std::vector<uint8_t>& map, std::vector<int>& out) {
    out.clear();
    out.reserve(map.size());
    for (int i = 0; i < (int)map.size(); i++) {
        if (map[(size_t)i]) out.push_back(i);
    }
}

static void GBFastScanRectsV6(
    const std::vector<uint8_t>& map,
    const std::vector<int>& vals,
    int rows,
    int cols,
    const std::vector<int>& active_indices,
    std::vector<GBMove>& outMoves
) {
    outMoves.clear();
    const int n_active = (int)active_indices.size();
    if (n_active <= 0) return;

    std::vector<int> P_val, P_cnt;
    GBBuildPrefixValCntAuto(map, vals, rows, cols, P_val, P_cnt);

    outMoves.reserve((size_t)n_active * (size_t)std::min(n_active, 64));
    for (int i = 0; i < n_active; i++) {
        const int idx1 = active_indices[(size_t)i];
        const int r1_raw = idx1 / cols;
        const int c1_raw = idx1 % cols;
        for (int j = i; j < n_active; j++) {
            const int idx2 = active_indices[(size_t)j];
            const int r2_raw = idx2 / cols;
            const int c2_raw = idx2 % cols;
            const int min_r = std::min(r1_raw, r2_raw);
            const int max_r = std::max(r1_raw, r2_raw);
            const int min_c = std::min(c1_raw, c2_raw);
            const int max_c = std::max(c1_raw, c2_raw);

            if (GBRectSum1D(P_val, cols, min_r, min_c, max_r, max_c) != 10) continue;
            int count = GBRectSum1D(P_cnt, cols, min_r, min_c, max_r, max_c);
            outMoves.push_back({ min_r, min_c, max_r, max_c, count });
        }
    }
}

static std::vector<uint8_t> GBApplyMoveFast(const std::vector<uint8_t>& map, const GBRect& rect, int cols) {
    std::vector<uint8_t> out = map;
    for (int r = rect.r1; r <= rect.r2; r++) {
        int base = r * cols;
        for (int c = rect.c1; c <= rect.c2; c++) {
            out[(size_t)(base + c)] = 0;
        }
    }
    return out;
}

static int GBApplyMoveInplaceAndCount(std::vector<uint8_t>& map, const GBRect& rect, int cols) {
    int gained = 0;
    for (int r = rect.r1; r <= rect.r2; r++) {
        int base = r * cols;
        for (int c = rect.c1; c <= rect.c2; c++) {
            size_t idx = (size_t)(base + c);
            if (map[idx]) {
                gained++;
                map[idx] = 0;
            }
            else {
                map[idx] = 0;
            }
        }
    }
    return gained;
}

static GBState GBRunCoreSearchLogic(
    const std::vector<uint8_t>& start_map,
    const std::vector<int>& vals,
    int rows,
    int cols,
    int beam_width,
    GBMode search_mode,
    int start_score,
    const std::shared_ptr<GBPathNode>& start_path,
    const GBWeights& weights,
    std::mt19937& rng_np,
    int max_depth = 160
) {
    GBState init;
    init.map = start_map;
    init.score = start_score;
    init.path = start_path;
    init.h_score = GBEvaluateState(start_score, init.map, rows, cols, weights, rng_np);

    std::vector<GBState> current_beam;
    current_beam.reserve((size_t)std::max(1, beam_width));
    current_beam.push_back(std::move(init));

    GBState best_state_in_run = current_beam[0];

    std::vector<int> active;
    std::vector<GBMove> raw_moves;
    std::vector<GBMove> valid_moves;

    for (int depth = 0; depth < max_depth; depth++) {
        std::vector<GBState> next_candidates;
        next_candidates.reserve((size_t)beam_width * 32);
        bool found_any_move = false;

        for (const auto& state : current_beam) {
            GBGetActiveIndices(state.map, active);
            if ((int)active.size() < 2) {
                if (state.score > best_state_in_run.score) best_state_in_run = state;
                continue;
            }

            GBFastScanRectsV6(state.map, vals, rows, cols, active, raw_moves);
            if (raw_moves.empty()) {
                if (state.score > best_state_in_run.score) best_state_in_run = state;
                continue;
            }

            valid_moves.clear();
            valid_moves.reserve(raw_moves.size());
            for (const auto& m : raw_moves) {
                bool ok = false;
                if (search_mode == GBMode::Classic) {
                    if (m.count == 2) ok = true;
                }
                else {
                    if (m.count >= 2) ok = true;
                }
                if (ok) valid_moves.push_back(m);
            }

            if (valid_moves.empty()) {
                if (state.score > best_state_in_run.score) best_state_in_run = state;
                continue;
            }

            found_any_move = true;

            std::stable_sort(valid_moves.begin(), valid_moves.end(), [](const GBMove& a, const GBMove& b) {
                return a.count > b.count;
            });

            const size_t take = std::min<size_t>(60, valid_moves.size());
            for (size_t i = 0; i < take; i++) {
                const auto& mv = valid_moves[i];
                GBRect rect{ mv.r1, mv.c1, mv.r2, mv.c2 };

                GBState ns;
                ns.map = GBApplyMoveFast(state.map, rect, cols);
                ns.score = state.score + mv.count;
                ns.h_score = GBEvaluateState(ns.score, ns.map, rows, cols, weights, rng_np);
                ns.path = GBAppendPath(state.path, rect);

                next_candidates.push_back(std::move(ns));
            }
        }

        if (!found_any_move) break;
        if (next_candidates.empty()) break;

        std::stable_sort(next_candidates.begin(), next_candidates.end(), [](const GBState& a, const GBState& b) {
            return a.h_score > b.h_score;
        });

        const size_t keep = std::min<size_t>((size_t)std::max(1, beam_width), next_candidates.size());
        current_beam.assign(next_candidates.begin(), next_candidates.begin() + keep);

        if (!current_beam.empty() && current_beam[0].score > best_state_in_run.score) {
            best_state_in_run = current_beam[0];
        }
    }

    return best_state_in_run;
}

struct GBPersonality {
    GBWeights weights;
    const wchar_t* role{ L"" };
};

static GBPersonality GBDispatchPersonality(int workerIndex) {
    // Matches deep_dive.py's default distribution (and typical V6.2 Hydra usage)
    GBPersonality p;
    if (workerIndex < 4) {
        p.weights.w_island = 0;
        p.weights.w_fragment = 0.0;
        p.role = L"Berserker";
    }
    else if (workerIndex < 8) {
        p.weights.w_island = 10;
        p.weights.w_fragment = 0.1;
        p.role = L"Light Walker";
    }
    else if (workerIndex < 12) {
        p.weights.w_island = 5;
        p.weights.w_fragment = 0.01;
        p.role = L"Chaos Gambler";
    }
    else {
        p.weights.w_island = 63;
        p.weights.w_fragment = 1.0;
        p.role = L"Tactician";
    }
    return p;
}

static GBState GBSolveProcessHydra(
    const std::vector<int>& vals,
    int rows,
    int cols,
    int beam_width,
    GBMode mode,
    uint32_t seed,
    double time_limit_sec,
    const GBPersonality& personality
) {
    const uint32_t safe_seed = (uint32_t)(seed % 0xFFFFFFFFu);

    std::mt19937 rng_np(safe_seed);
    std::mt19937 rng_py(safe_seed);

    std::vector<uint8_t> initial_map((size_t)rows * (size_t)cols, 1);

    auto t0 = std::chrono::steady_clock::now();

    // 1) Base run
    GBState base_state;
    if (mode == GBMode::God) {
        GBWeights p1w = personality.weights;
        if (p1w.w_island > 0) {
            // python: p1_weights['w_island'] *= 0.5
            p1w.w_island = (int)std::lround((double)p1w.w_island * 0.5);
        }

        GBState p1 = GBRunCoreSearchLogic(initial_map, vals, rows, cols, beam_width, GBMode::Classic, 0, nullptr, p1w, rng_np);
        base_state = GBRunCoreSearchLogic(p1.map, vals, rows, cols, beam_width, GBMode::Omni, p1.score, p1.path, personality.weights, rng_np);
    }
    else {
        base_state = GBRunCoreSearchLogic(initial_map, vals, rows, cols, beam_width, mode, 0, nullptr, personality.weights, rng_np);
    }

    GBState best_final_state = base_state;

    // 2) Directed destruction loop
    for (;;) {
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - t0).count();
        if (elapsed >= time_limit_sec) break;

        std::vector<GBRect> path;
        GBMaterializePath(best_final_state.path, path);
        if (path.size() < 5) break;

        int L = (int)path.size();

        int cut_start;
        if (GBU01(rng_py) < 0.3) cut_start = GBUniformInt(rng_py, L / 2, L - 3);
        else cut_start = GBUniformInt(rng_py, 0, L - 3);

        int maxCut = std::min(12, L - cut_start);
        int cut_len = GBUniformInt(rng_py, 3, maxCut); // kept for equivalence; intentionally unused
        (void)cut_len;

        std::vector<uint8_t> temp_map((size_t)rows * (size_t)cols, 1);
        int prefix_score = 0;
        std::shared_ptr<GBPathNode> prefix_path = nullptr;

        for (int i = 0; i < cut_start; i++) {
            const auto& rect = path[(size_t)i];
            prefix_score += GBApplyMoveInplaceAndCount(temp_map, rect, cols);
            prefix_path = GBAppendPath(prefix_path, rect);
        }

        GBWeights repair = personality.weights;
        repair.w_island += GBUniformInt(rng_py, -50, 50);

        int widened = (int)std::lround((double)beam_width * 1.2);
        if (widened < 1) widened = 1;

        GBState repaired = GBRunCoreSearchLogic(temp_map, vals, rows, cols, widened, GBMode::Omni, prefix_score, prefix_path, repair, rng_np);
        if (repaired.score > best_final_state.score) {
            best_final_state = std::move(repaired);
        }
    }

    return best_final_state;
}

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
    x = ClampInt(x, 0, sw - 1);
    y = ClampInt(y, 0, sh - 1);

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
extern "C" {

SUM10_API void SUM10_CALL sum10_set_log_callback(sum10_log_callback_t cb) {
    g_log_cb = cb;
    Log(L"[Native] log callback set.");
}

SUM10_API int SUM10_CALL sum10_capture_screen_png(const wchar_t* outPngPath) {
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

SUM10_API int SUM10_CALL sum10_ocr_board(
    const wchar_t* screenPngPath,
    const float* corners8,
    int rows,
    int cols,
    const wchar_t* digitTemplateDir,
    int* outDigits,
    float* outAvgConf,
    float* outCellConf,
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
        if (!HasAnyTemplate(dt)) {
            Log(L"[Native] OCR aborted: no digit templates found.");
            return -4;
        }
        std::vector<int> digits;
        std::vector<float> cellConfs;
        float avgConf = 0.0f;

        int rc = OcrBoardByTemplates(warped, rows, cols, dt, digits, avgConf, outCellConf ? &cellConfs : nullptr);
        if (rc != 0) return rc;

        for (int i = 0; i < rows * cols; i++) {
            outDigits[i] = digits[i];
            if (outCellConf && i < (int)cellConfs.size()) outCellConf[i] = cellConfs[i];
        }
        *outAvgConf = avgConf;

        Log(L"[Native] OCR done.");
        return 0;
    }
    catch (...) {
        return -100;
    }
}

SUM10_API int SUM10_CALL sum10_solve_greedy(
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

SUM10_API int SUM10_CALL sum10_solve_godbrain_v62(
    const int* digits,
    int rows,
    int cols,
    int beamWidth,
    int threads,
    int mode,
    uint32_t baseSeed,
    float timeLimitSec,
    int* outMoves,
    int maxMoves,
    int* outMoveCount,
    int* outScore
) {
    try {
        if (!digits || !outMoves || !outMoveCount || !outScore) return -1;
        if (rows <= 0 || cols <= 0) return -2;
        if (beamWidth <= 0) return -3;
        if (maxMoves <= 0) return -4;
        if (!(mode == 0 || mode == 1 || mode == 2)) return -5;

        const int N = rows * cols;
        std::vector<int> vals((size_t)N);
        for (int i = 0; i < N; i++) vals[(size_t)i] = digits[i];

        int hw = (int)std::thread::hardware_concurrency();
        int workerCount = threads;
        if (workerCount <= 0) workerCount = (hw > 0 ? hw : 1);
        workerCount = std::max(1, std::min(workerCount, 64));

        GBMode gbMode = GBMode::God;
        if (mode == 1) gbMode = GBMode::Classic;
        else if (mode == 2) gbMode = GBMode::Omni;

        double tl = (double)timeLimitSec;
        if (tl <= 0.0) tl = 0.001;

        Log(L"[Native] GodBrain V6.2 solving...");
        {
            std::wstringstream ss;
            ss << L"[Native] cfg beam=" << beamWidth << L", threads=" << workerCount << L", mode=" << mode << L", t=" << timeLimitSec << L"s";
            Log(ss.str().c_str());
        }

        // Run Hydra workers
        std::vector<std::future<GBState>> futs;
        futs.reserve((size_t)workerCount);

        for (int i = 0; i < workerCount; i++) {
            const uint32_t seed = baseSeed + (uint32_t)i;
            const GBPersonality p = GBDispatchPersonality(i);

            futs.push_back(std::async(std::launch::async, [vals, rows, cols, beamWidth, gbMode, seed, tl, p]() {
                return GBSolveProcessHydra(vals, rows, cols, beamWidth, gbMode, seed, tl, p);
            }));
        }

        GBState best;
        bool hasBest = false;
        for (int i = 0; i < (int)futs.size(); i++) {
            GBState st = futs[(size_t)i].get();
            if (!hasBest || st.score > best.score) {
                best = std::move(st);
                hasBest = true;
            }
        }

        std::vector<GBRect> path;
        GBMaterializePath(best.path, path);

        const int n = (int)std::min((size_t)maxMoves, path.size());
        for (int i = 0; i < n; i++) {
            outMoves[i * 4 + 0] = path[(size_t)i].r1;
            outMoves[i * 4 + 1] = path[(size_t)i].c1;
            outMoves[i * 4 + 2] = path[(size_t)i].r2;
            outMoves[i * 4 + 3] = path[(size_t)i].c2;
        }

        *outMoveCount = n;
        *outScore = best.score;

        {
            std::wstringstream ss;
            ss << L"[Native] GodBrain done. score=" << best.score << L", moves=" << n;
            Log(ss.str().c_str());
        }
        return 0;
    }
    catch (...) {
        return -100;
    }
}

SUM10_API int SUM10_CALL sum10_execute_path(
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

} // extern "C"
