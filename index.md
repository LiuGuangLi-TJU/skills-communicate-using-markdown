# 标题
👌

![key2](https://github.com/user-attachments/assets/927f4d64-334f-456c-ad77-75349cae7c47)
```CPP
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <filesystem>
#include <regex> 
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"
using namespace std;
using namespace cv;

class ImageStitch {
public:

	ImageStitch(const Mat& img_1, const Mat& img_2):
		img1(img_1.clone()), img2(img_2.clone()){}

	// 执行拼接函数，返回是否成功
	bool stitch();

	// 返回拼接图像
	Mat get_stitched_img()const;

	// 返回特征点匹配图像
	Mat get_matching_visualization()const;


private:
	// 查找特征点和描述子
	void find_features();

	// 匹配关键点
	void match_keypoints();
	void match_keypoints_with_gms();

	// 计算单应性矩阵、最终的变换矩阵、平均重投影偏差
	bool calculate_homography_and_mre();

	// 执行图像变换和合成
	void warp_and_compose();

	// 在最终图像上绘制重叠区域
	void draw_overlap();

private:
	
	Mat img1, img2;
	Mat stitched_img;

	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;
	vector<DMatch> good_matches;

	Mat H; // img1到img2的单应性矩阵
	Mat H_trans; // 包含平移的最终变换矩阵
	double mre; // 平均重投影误差

	// 参数：采用极大的特征点检测数、较高的特征点匹配筛选阈值，以获得尽可能精准的结果
	const int num_features_ = 0;
	const float lowe_ratio_ = 0.45f; 
	const double ransac_thresh_ = 1.0;
};

bool ImageStitch::stitch() {
	
	// 寻找特征点并进行匹配
	find_features();
	match_keypoints();
	//match_keypoints_with_gms();
	// 计算单应性矩阵
	if (!calculate_homography_and_mre()) {
		return false; // 如果无法计算H，则拼接失败
	}

	// 变换图像并合成
	warp_and_compose();

	// 绘制重叠区域
	draw_overlap();

	return true;
}

void ImageStitch::find_features() {
	Ptr<SIFT> sift = SIFT::create(num_features_);
	sift->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
	sift->detectAndCompute(img2, noArray(), keypoints2, descriptors2);
}

void ImageStitch::match_keypoints() {
	FlannBasedMatcher matcher;
	vector<vector<DMatch>> knn_matches;
	matcher.knnMatch(descriptors1, descriptors2, knn_matches, 2);

	good_matches.clear();
	for (const auto& m : knn_matches) {
		if (m.size() == 2 && m[0].distance < lowe_ratio_ * m[1].distance) {
			good_matches.push_back(m[0]);
		}
	}


}


bool ImageStitch::calculate_homography_and_mre() {
	if (good_matches.size() < 4) {
		std::cerr << "错误: 匹配点不足，无法计算单应性矩阵。" << std::endl;
		return false;
	}

	std::vector<cv::Point2f> points1, points2;
	for (const auto& match : good_matches) {
		points1.push_back(keypoints1[match.queryIdx].pt);
		points2.push_back(keypoints2[match.trainIdx].pt);
	}
	std::vector<uchar> ransac_mask;
	H = cv::findHomography(points1, points2, cv::USAC_MAGSAC, ransac_thresh_, ransac_mask);

	if (H.empty()) {
		std::cerr << "错误: findHomography 函数未能计算出有效的矩阵。" << std::endl;
		return false;
	}

	// MRE 计算部分
	std::vector<cv::Point2f> points1_projected;
	cv::perspectiveTransform(points1, points1_projected, H);

	double total_error = 0.0;
	int inlier_count = 0;

	for (size_t i = 0; i < points1.size(); ++i) {
		if (ransac_mask[i]) { 
			// 只对RANSAC算法认定的内点计算误差
			total_error += cv::norm(points2[i] - points1_projected[i]);
			inlier_count++;
		}
	}
	mre = (inlier_count > 0) ? (total_error / inlier_count) : -1.0;
	cout <<fixed << setprecision(2) << "平均重投影误差为： "  << mre <<"像素" << endl;
	return true;
}

void ImageStitch::warp_and_compose() {
	
	// 1. 计算最终画布的尺寸 
	std::vector<cv::Point2f> corners1 = {
		{0.0f, 0.0f}, {static_cast<float>(img1.cols), 0.0f},
		{static_cast<float>(img1.cols), static_cast<float>(img1.rows)}, {0.0f, static_cast<float>(img1.rows)}
	};
	std::vector<cv::Point2f> corners1_trans;
	cv::perspectiveTransform(corners1, corners1_trans, H);

	float min_x = 0, min_y = 0, max_x = img2.cols, max_y = img2.rows;
	for (const auto& pt : corners1_trans) {
		min_x = std::min(min_x, pt.x);
		min_y = std::min(min_y, pt.y);
		max_x = std::max(max_x, pt.x);
		max_y = std::max(max_y, pt.y);
	}
	int width = static_cast<int>(ceil(max_x - min_x));
	int height = static_cast<int>(ceil(max_y - min_y));

	cv::Mat translation = (cv::Mat_<double>(3, 3) << 1, 0, -min_x, 0, 1, -min_y, 0, 0, 1);
	H_trans = translation * H;

	// 2. 将原始灰度图转换为BGR图 
	cv::Mat img1_bgr, img2_bgr;
	cv::cvtColor(img1, img1_bgr, cv::COLOR_GRAY2BGR);
	cv::cvtColor(img2, img2_bgr, cv::COLOR_GRAY2BGR);

	// 3. 将 BGR 图变换并放置到32位浮点型图层上
	cv::Mat layer1(height, width, CV_32FC3, cv::Scalar(0, 0, 0));
	cv::Mat layer2(height, width, CV_32FC3, cv::Scalar(0, 0, 0));

	cv::Mat img1_warped;
	cv::warpPerspective(img1_bgr, img1_warped, H_trans, cv::Size(width, height));
	img1_warped.convertTo(layer1, CV_32FC3);

	cv::Mat img2_temp;
	img2_bgr.convertTo(img2_temp, CV_32FC3);
	img2_temp.copyTo(layer2(cv::Rect(static_cast<int>(-min_x), static_cast<int>(-min_y), img2_bgr.cols, img2_bgr.rows)));

	// 4. 创建基础的二值蒙版
	cv::Mat mask1, mask2;
	cv::cvtColor((layer1 > 0), mask1, cv::COLOR_BGR2GRAY);
	cv::cvtColor((layer2 > 0), mask2, cv::COLOR_BGR2GRAY);

	// 平均值融合
	// 直接将两个图层相加
	cv::Mat blended_layer = layer1 + layer2;

	// 找到重叠区域
	cv::Mat overlap = (mask1 & mask2);

	// 在重叠区域，像素值是两图之和，需要除以2来取平均
	// 只对重叠区域进行操作
	blended_layer.forEach<cv::Vec3f>(
		[&](cv::Vec3f& pixel, const int* position) -> void {
			// position[0] 是行(y), position[1] 是列(x)
			if (overlap.at<uchar>(position[0], position[1])) {
				pixel /= 2.0f;
			}
		}
	);

	// 将最终结果转换回8位图像
	blended_layer.convertTo(stitched_img, CV_8UC3);
}

void ImageStitch::draw_overlap() {
	if (H.empty() || H_trans.empty() || stitched_img.empty()) return;

	// 计算img2的四个角点
	vector<Point2f> corners2 = {
		{0.0f, 0.0f}, {static_cast<float>(img2.cols), 0.0f},
		{static_cast<float>(img2.cols), static_cast<float>(img2.rows)}, {0.0f, static_cast<float>(img2.rows)}
	};

	// 将img2的角点反向投影到img1的坐标系中
	vector<Point2f> corners2_on_img1;
	perspectiveTransform(corners2, corners2_on_img1, H.inv());

	// 计算与img1边界的交集
	vector<Point2f> img1_poly = {
		{0.0f, 0.0f}, {static_cast<float>(img1.cols), 0.0f},
		{static_cast<float>(img1.cols), static_cast<float>(img1.rows)}, {0.0f, static_cast<float>(img1.rows)}
	};
	std::vector<Point2f> intersection_poly;
	if (intersectConvexConvex(img1_poly, corners2_on_img1, intersection_poly) && !intersection_poly.empty()) {
		vector<Point2f> common_area_on_stitched;
		perspectiveTransform(intersection_poly, common_area_on_stitched, H_trans);

		vector<vector<Point>> contours;
		vector<Point> points;
		cout << "拼接图像公共区域坐标：" << endl;
		for (const auto& p : common_area_on_stitched) {
			cout << fixed << setprecision(0) << "[" << p.x << ", " << p.y << "]" << endl;
			points.push_back(Point(cvRound(p.x), cvRound(p.y)));
		}
		contours.push_back(points);

		polylines(stitched_img, contours, true, Scalar(0, 255, 0), 3, LINE_AA);
	}
}

Mat ImageStitch::get_stitched_img() const {
	return stitched_img;
}

Mat ImageStitch::get_matching_visualization() const {
	Mat vis_img;
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, vis_img,
	Scalar(0, 0, 255), Scalar(0, 255, 0),
	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	return vis_img;
}

void show_img(const string& winname, Mat img){
	if (img.empty()) {
		cerr << " 图片为空 " << endl;
		return;
	}
	namedWindow(winname, WINDOW_KEEPRATIO);
	imshow(winname, img);
	waitKey(0);
	destroyAllWindows();
}

//此函数的实现需要C++17及更新标准
string get_next_experiment_path(const std::string& base_results_dir) {
	// 确保主结果文件夹存在，如果不存在则创建
	try {
		if (!std::filesystem::exists(base_results_dir)) {
			std::filesystem::create_directories(base_results_dir);
		}
	}
	catch (const std::filesystem::filesystem_error& e) {
		std::cerr << "错误: 无法创建主结果文件夹: " << base_results_dir << " - " << e.what() << std::endl;
		return ""; // 返回空字符串表示失败
	}

	int max_exp_num = 0;
	std::regex exp_regex("exp(\\d+)"); // 正则表达式，用于匹配 "exp" 后跟一个或多个数字

	// 遍历主结果文件夹下的所有条目
	for (const auto& entry : std::filesystem::directory_iterator(base_results_dir)) {
		if (entry.is_directory()) {
			std::string dirname = entry.path().filename().string();
			std::smatch match;
			// 检查文件夹名是否匹配 "exp<数字>" 的格式
			if (std::regex_match(dirname, match, exp_regex)) {
				if (match.size() == 2) {
					// match[0] 是整个字符串 (如 "exp12")
					// match[1] 是第一个捕获组 (即数字 "12")
					int current_num = std::stoi(match[1].str());
					if (current_num > max_exp_num) {
						max_exp_num = current_num;
					}
				}
			}
		}
	}

	// 计算新的实验编号和路径
	int next_exp_num = max_exp_num + 1;
	std::string next_exp_dirname = "exp" + std::to_string(next_exp_num);
	std::filesystem::path next_exp_path = std::filesystem::path(base_results_dir) / next_exp_dirname;

	// 创建新的实验文件夹
	try {
		std::filesystem::create_directory(next_exp_path);
	}
	catch (const std::filesystem::filesystem_error& e) {
		std::cerr << "错误: 无法创建新的实验文件夹: " << next_exp_path << " - " << e.what() << std::endl;
		return "";
	}

	return next_exp_path.string();
}

int main() {
	string img1_path = "E:/FX/alignment/pier01.JPG";
	string img2_path = "E:/FX/alignment/pier02.JPG";

	string result_path = get_next_experiment_path("..\\result");

	cv::Mat img1 = cv::imread(img1_path, cv::IMREAD_GRAYSCALE);
	cv::Mat img2 = cv::imread(img2_path, cv::IMREAD_GRAYSCALE);

	if (img1.empty() || img2.empty()) {
		std::cerr << "无法读取图片，请检查路径" << std::endl;
		return -1;
	}

	// 创建拼接对象
	ImageStitch stitcher(img1, img2);
	// 执行拼接
	if (stitcher.stitch()) {

		std::cout << "拼接完成 " << std::endl;

		// 显示结果
		show_img("特征点匹配", stitcher.get_matching_visualization());
		show_img("拼接结果 ", stitcher.get_stitched_img());

		imwrite(result_path + "/matching.jpg", stitcher.get_matching_visualization());
		imwrite(result_path + "/result.jpg", stitcher.get_stitched_img());
		cout << "保存目录：" << result_path << endl;
	}
	
	else {
		std::cerr << "拼接失败" << std::endl;
		return -1;
	}


	return 0;
}
```
