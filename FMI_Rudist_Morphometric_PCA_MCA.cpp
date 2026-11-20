/*
 * Appendix — FMI Rudist-Like Morphometric Candidate Screening
 * Morphometric feature extraction, PCA and MCA analysis pipeline.
 * Input: 8-bit raster export of an FMI image (PNG/TIFF), NOT proprietary PDS/DLIS.
 * Build: g++ -std=c++17 -O3 FMI_Rudist_Morphometric_Screening.cpp \
 *        -o fmi_screen `pkg-config --cflags --libs opencv4`
 * Run: ./fmi_screen exported_fmi.tif overlay.png candidates.csv 1540.0 1548.0
 * Args 4/5: top/bottom depth in metres, only if supplied by calibrated export.
 * Requires OpenCV >= 4 and C++17. Optional categorical patterns are
 * computed by MCA without assuming a known biological classification.
 */
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

struct Candidate {
    cv::RotatedRect ellipse;
    cv::Rect bounds;
    double area, perimeter, circularity, aspect, solidity, fill, score;
    double orientationDeg, depthM;
};

static cv::Mat normalizedFmi(const cv::Mat& gray) {
    cv::Mat f, smooth, residual, normalized;
    gray.convertTo(f, CV_32F, 1.0 / 255.0);
    // Remove slowly varying tool-pad / illumination trends while retaining texture.
    cv::GaussianBlur(f, smooth, cv::Size(0, 0), 19.0);
    residual = f - smooth + 0.5f;
    cv::normalize(residual, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.5, cv::Size(12, 12));
    cv::Mat contrast;
    clahe->apply(normalized, contrast);
    return contrast;
}

static cv::Mat structuralMask(const cv::Mat& normalized) {
    cv::Mat denoised, gradient, positive, negative, combined, binary;
    cv::bilateralFilter(normalized, denoised, 7, 40, 7);
    auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
    cv::morphologyEx(denoised, positive, cv::MORPH_TOPHAT, kernel);
    cv::morphologyEx(denoised, negative, cv::MORPH_BLACKHAT, kernel);
    cv::max(positive, negative, combined);
    cv::Mat gx, gy, magnitude;
    cv::Scharr(denoised, gx, CV_32F, 1, 0);
    cv::Scharr(denoised, gy, CV_32F, 0, 1);
    cv::magnitude(gx, gy, magnitude);
    cv::normalize(magnitude, gradient, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::addWeighted(combined, 0.55, gradient, 0.45, 0, combined);
    cv::threshold(combined, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)));
    return binary;
}

static std::vector<Candidate> analyze(const cv::Mat& mask, const double d0,
                                      const double d1, const bool calibrated) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Candidate> candidates;
    for (const auto& contour : contours) {
        if (contour.size() < 9) continue;
        const double area = std::fabs(cv::contourArea(contour));
        const double perimeter = cv::arcLength(contour, true);
        if (area < 45 || perimeter < 1 || area > 0.08 * mask.total()) continue;
        const cv::Rect bbox = cv::boundingRect(contour);
        if (bbox.x <= 1 || bbox.y <= 1 || bbox.br().x >= mask.cols - 1 ||
            bbox.br().y >= mask.rows - 1) continue; // Avoid cropped edge contours.
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        const double hullArea = std::fabs(cv::contourArea(hull));
        if (hullArea < 1) continue;
        const auto ellipse = cv::fitEllipse(contour);
        const double a = std::max(ellipse.size.width, ellipse.size.height);
        const double b = std::min(ellipse.size.width, ellipse.size.height);
        if (b < 2) continue;
        const double aspect = a / b;
        const double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        const double solidity = area / hullArea;
        const double fill = area / std::max(1, bbox.area());
        // Feature-ranking score based on morphology and contour geometry.
        const double compact = std::clamp(circularity, 0.0, 1.0);
        const double elliptic = std::exp(-std::pow((aspect - 1.6) / 1.6, 2));
        const double score = 0.32 * compact + 0.30 * std::clamp(solidity, 0.0, 1.0)
                           + 0.22 * elliptic + 0.16 * std::clamp(fill, 0.0, 1.0);
        if (score < 0.42 || aspect > 7.0 || solidity < 0.25) continue;
        double orientation = ellipse.angle;
        if (ellipse.size.width < ellipse.size.height) orientation += 90.0;
        orientation = std::fmod(orientation, 180.0);
        const double depth = calibrated ? d0 + ellipse.center.y / (mask.rows - 1.0) * (d1 - d0)
                                        : std::numeric_limits<double>::quiet_NaN();
        candidates.push_back({ellipse, bbox, area, perimeter, circularity,
                              aspect, solidity, fill, score, orientation, depth});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });
    // Suppress near-identical overlapping geometries from nested boundaries.
    std::vector<Candidate> retained;
    for (const auto& c : candidates) {
        bool duplicate = false;
        for (const auto& r : retained) {
            const auto intersection = c.bounds & r.bounds;
            const double iou = static_cast<double>(intersection.area()) /
                  (c.bounds.area() + r.bounds.area() - intersection.area());
            if (iou > 0.45) { duplicate = true; break; }
        }
        if (!duplicate) retained.push_back(c);
    }
    return retained;
}


// PCA of standardized continuous morphometric descriptors.
// Rows: measured structures; columns: area, aspect, circularity, solidity,
// extent and double-angle orientation (appropriate for 180-degree axial data).
static cv::Mat featuresForPca(const std::vector<Candidate>& cs) {
    cv::Mat X(static_cast<int>(cs.size()), 7, CV_64F);
    for (int i = 0; i < X.rows; ++i) {
        const auto& c = cs[i];
        const double theta = 2.0 * c.orientationDeg * CV_PI / 180.0;
        double f[] = {std::log1p(c.area), c.aspect, c.circularity,
                      c.solidity, c.fill, std::cos(theta), std::sin(theta)};
        for (int j = 0; j < X.cols; ++j) X.at<double>(i,j) = f[j];
    }
    for (int j = 0; j < X.cols; ++j) {
        cv::Scalar mean, sigma;
        cv::meanStdDev(X.col(j), mean, sigma);
        const double sd = std::max(sigma[0], 1e-9);
        for (int i = 0; i < X.rows; ++i)
            X.at<double>(i,j) = (X.at<double>(i,j) - mean[0]) / sd;
    }
    return X;
}
static bool savePca(const std::vector<Candidate>& cs, const std::string& path) {
    std::ofstream out(path);
    if (!out) return false;
    out << "id,pc1,pc2,explained_pc1,explained_pc2\n";
    if (cs.size() < 3) return true;
    cv::Mat X = featuresForPca(cs);
    cv::PCA model(X, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Mat scores = model.project(X);
    double total = cv::sum(model.eigenvalues)[0];
    // Explain fractions are relative to the retained two eigenvalues.
    const double e1 = model.eigenvalues.at<double>(0,0);
    const double e2 = model.eigenvalues.at<double>(1,0);
    for (int i=0;i<scores.rows;++i)
        out << (i+1) << ',' << scores.at<double>(i,0) << ','
            << scores.at<double>(i,1) << ','
            << e1/std::max(total,1e-12) << ','
            << e2/std::max(total,1e-12) << '\n';
    return true;
}

// MCA: complete disjunctive coding of discretized contour descriptors.
// Correspondence-analysis residuals and SVD yield row factor coordinates.
static bool saveMca(const std::vector<Candidate>& cs, const std::string& path) {
    std::ofstream out(path);
    if (!out) return false;
    out << "id,mca_dim1,mca_dim2,aspect_class,compactness_class,solidity_class,azimuth_class\n";
    if (cs.size() < 3) return true;
    const int n = static_cast<int>(cs.size()), groups=4, cats=3, k=groups*cats;
    cv::Mat Z = cv::Mat::zeros(n,k,CV_64F);
    std::vector<std::array<int,4>> codes;
    for (int i=0;i<n;++i) {
        const auto& c=cs[i];
        std::array<int,4> v {
            c.aspect < 1.3 ? 0 : (c.aspect < 2.1 ? 1 : 2),
            c.circularity < 0.45 ? 0 : (c.circularity < 0.72 ? 1 : 2),
            c.solidity < 0.55 ? 0 : (c.solidity < 0.82 ? 1 : 2),
            c.orientationDeg < 60 ? 0 : (c.orientationDeg < 120 ? 1 : 2)
        };
        codes.push_back(v);
        for (int g=0;g<groups;++g) Z.at<double>(i,g*cats+v[g])=1.0;
    }
    cv::Mat P=Z/(static_cast<double>(n)*groups);
    cv::Mat r, c;
    cv::reduce(P,r,1,cv::REDUCE_SUM);
    cv::reduce(P,c,0,cv::REDUCE_SUM);
    cv::Mat S=cv::Mat::zeros(n,k,CV_64F);
    for (int i=0;i<n;++i) for (int j=0;j<k;++j) {
        const double expected=r.at<double>(i,0)*c.at<double>(0,j);
        if (expected>1e-15)
            S.at<double>(i,j)=(P.at<double>(i,j)-expected)/std::sqrt(expected);
    }
    cv::SVD svd(S,cv::SVD::FULL_UV);
    for (int i=0;i<n;++i) {
        const double d1=svd.w.rows>0 ? svd.w.at<double>(0,0) : 0.0;
        const double d2=svd.w.rows>1 ? svd.w.at<double>(1,0) : 0.0;
        const double axis1=svd.u.at<double>(i,0)*d1/std::sqrt(r.at<double>(i,0));
        const double axis2=svd.u.at<double>(i,1)*d2/std::sqrt(r.at<double>(i,0));
        out << (i+1) << ',' << axis1 << ',' << axis2;
        for (int g=0;g<groups;++g) out << ',' << codes[i][g];
        out << '\n';
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4 && argc != 6) {
        std::cerr << "Usage: fmi_screen input.tif overlay.png candidates.csv [top_depth_m bottom_depth_m]\n";
        return 2;
    }
    const bool calibrated = (argc == 6);
    double topDepth = 0.0, bottomDepth = 0.0;
    if (calibrated) {
        try { topDepth = std::stod(argv[4]); bottomDepth = std::stod(argv[5]); }
        catch (...) { std::cerr << "Invalid depth arguments\n"; return 2; }
        if (!(bottomDepth > topDepth)) {
            std::cerr << "Bottom depth must exceed top depth\n"; return 2;
        }
    }
    const cv::Mat gray = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (gray.empty() || gray.rows < 100 || gray.cols < 100) {
        std::cerr << "Cannot open suitable raster image; export FMI first.\n"; return 3;
    }
    const cv::Mat normalized = normalizedFmi(gray);
    const cv::Mat mask = structuralMask(normalized);
    const auto candidates = analyze(mask, topDepth, bottomDepth, calibrated);
    cv::Mat annotated;
    cv::cvtColor(gray, annotated, cv::COLOR_GRAY2BGR);
    std::ofstream csv(argv[3]);
    if (!csv) { std::cerr << "Cannot create CSV\n"; return 4; }
    csv << "id,x_px,y_px,depth_m,area_px2,major_px,minor_px,aspect_ratio,"
           "circularity,solidity,extent,orientation_deg,screening_score,label\n";
    csv << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < candidates.size(); ++i) {
        const auto& c = candidates[i];
        cv::ellipse(annotated, c.ellipse, cv::Scalar(0, 230, 255), 2, cv::LINE_AA);
        cv::putText(annotated, std::to_string(i + 1),
                    c.bounds.tl() + cv::Point(2, 12), cv::FONT_HERSHEY_SIMPLEX,
                    0.38, cv::Scalar(50, 255, 50), 1, cv::LINE_AA);
        csv << (i + 1) << ',' << c.ellipse.center.x << ',' << c.ellipse.center.y << ',';
        if (calibrated) csv << c.depthM;
        csv << ',' << c.area << ','
            << std::max(c.ellipse.size.width, c.ellipse.size.height) << ','
            << std::min(c.ellipse.size.width, c.ellipse.size.height) << ','
            << c.aspect << ',' << c.circularity << ',' << c.solidity << ','
            << c.fill << ',' << c.orientationDeg << ',' << c.score
            << ",MORPHOMETRIC_FEATURE\n";
    }
    if (!savePca(candidates, std::string(argv[3]) + ".pca.csv") ||
        !saveMca(candidates, std::string(argv[3]) + ".mca.csv")) {
        std::cerr << "Cannot export factor scores\n"; return 4;
    }
    if (!cv::imwrite(argv[2], annotated)) {
        std::cerr << "Cannot save annotated overlay\n"; return 4;
    }
    std::cout << "Detected " << candidates.size() << " morphometric features.\n";
    return 0;
}
