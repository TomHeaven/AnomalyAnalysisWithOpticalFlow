#pragma once

#include <vector>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu; // for GpuMat and related functions
using namespace std;
//用于将 GpuMat 转化为 vector<Point2f>
static void download(const GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

static void download(const GpuMat& d_mat, vector<float>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC1, (void*)&vec[0]);
	d_mat.download(mat);
}

static void download(const GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

static void upload(const vector<Point2f>& vec, GpuMat& gMat) {
	Mat p(1, vec.size(), CV_32FC2);
	// copy data
	for (int i = 0; i < vec.size(); ++i) {
		Vec2f& t = p.at<Vec2f>(0, i);
		t[0] = vec[i].x;
		t[1] = vec[i].y;
	}
	gMat.upload(p);
}