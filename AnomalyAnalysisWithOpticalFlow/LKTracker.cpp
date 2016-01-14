#include "stdafx.h"
#include "tld_utils.h"
#include "LKTracker.h"
#include "cuda_utils.h"
#include <ctime>

//#define USE_CUDA  // 是否使用CUDA加速
//#define MEASURE_TIME // 是否输出OF计算时间


LKTracker::LKTracker()
{
	term_criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 10, 0.1);
	window_size = Size(15, 15);
	level = 8; //金字塔层数（越大光流搜索的范围越大）
	lambda = 0.005;

#ifdef USE_CUDA
    gFlow = new PyrLKOpticalFlow();
	//设置参数
	gFlow->winSize = window_size;
	gFlow->maxLevel =level;
	gFlow->derivLambda = lambda; //unused
	gFlow->iters = 10;
	gFlow->useInitialFlow = false;
	gFlow->minEigThreshold = 0.1; //unused
#endif
}

LKTracker::~LKTracker(){
	if (gFlow != NULL) {
		gFlow->releaseMemory();
		delete gFlow;
	}
}


bool LKTracker::trackf2f(const GpuMat& gImg1, const GpuMat& gImg2, GpuMat &gPoints1, GpuMat &gPoints2) {
	//计算正反光流
	gFlow->sparse(gImg1, gImg2, gPoints1, gPoints2, gStatus); // compute gPoints2
	gFlow->sparse(gImg2, gImg1, gPoints2, gPointsFB, gFBStatus); //compute gPointsFB

	vector<Point2f> points1, points2;
	download(gPoints1, points1);
	download(gPoints2, points2);

	//Compute the real FB-error
	FB_error.clear();
	for (int i = 0; i < points1.size(); ++i)
	{
		FB_error.push_back(norm(pointsFB[i] - points1[i]));
	}
	//Filter out points with FB_error[i] > mean(FB_error) && points with sim_error[i] > mean(sim_error)
	normCrossCorrelation(gImg1, gImg2, gPoints1, gPoints2, points1, points2);
	bool retVal = filterPts(points1, points2);
	//更新gpu数据
	upload(points1, gPoints1);
	upload(points2, gPoints2);
	return retVal;
}


bool LKTracker::trackf2f(const Mat& img1, const Mat& img2, vector<Point2f> &points1, vector<cv::Point2f> &points2)
{
	//TODO!:implement c function cvCalcOpticalFlowPyrLK() or Faster tracking function
#ifdef  MEASURE_TIME
	clock_t startTime = clock();
#endif
#ifndef USE_CUDA
	//Forward-Backward tracking
	calcOpticalFlowPyrLK(img1, img2, points1, points2, status, similarity, window_size, level, term_criteria, lambda, 0/**/);

	//TomHeaven注释：这里的过滤可能漏掉一些运动速度快的目标，故而去掉过滤
	calcOpticalFlowPyrLK(img2, img1, points2, pointsFB, FB_status, FB_error, window_size, level, term_criteria, lambda, 0);
#else
	/////////// GPU 加速
	/*Mat p1(1, points1.size(), CV_32FC2), p2(1, points2.size(), CV_32FC2);
	// copy data
	for (int i = 0; i < points1.size(); ++i) {
		Vec2f& t1 = p1.at<Vec2f>(0, i);
		t1[0] = points1[i].x;
		t1[1] = points1[i].y;	
	}

	for (int i = 0; i < points2.size(); ++i) {
		Vec2f& t2 = p2.at<Vec2f>(0, i);
		t2[0] = points2[i].x;
		t2[1] = points2[i].y;
	}
	gPoints1.upload(p1);
	gPoints2.upload(p2);*/
	// 上传数据
	gImg1.upload(img1);
	gImg2.upload(img2);
	upload(points1, gPoints1);
	upload(points2, gPoints2);


	//计算正反光流
	gFlow->sparse(gImg1, gImg2, gPoints1, gPoints2, gStatus); // compute gPoints2
	gFlow->sparse(gImg2, gImg1, gPoints2, gPointsFB, gFBStatus); //compute gPointsFB
	
	download(gPoints2, points2);
	download(gStatus, status);
	download(gPointsFB, pointsFB);
#endif

#ifdef  MEASURE_TIME
	clock_t endTime = clock();
	printf("OF time = %.3f\n", double(endTime - startTime) / CLOCKS_PER_SEC);
#endif
	
	//printf("p1: rows = %d, cols = %d, type = %d\n", p1.rows, p1.cols, p1.type());
	//printf("gPoints1: rows = %d, cols = %d, type = %d\n", gPoints1.rows, gPoints1.cols, gPoints1.type());
	//printf("pointsFB: size = %d, point[0] = %f, %f\n", pointsFB.size(), pointsFB[0].x, pointsFB[0].y);
	
	//Compute the real FB-error
	FB_error.clear();
	for (int i = 0; i < points1.size(); ++i)
	{
		FB_error.push_back(norm(pointsFB[i] - points1[i]));
	}
	//Filter out points with FB_error[i] > mean(FB_error) && points with sim_error[i] > mean(sim_error)
	normCrossCorrelation(img1, img2, points1, points2);
	return filterPts(points1, points2);
	//return true;
}


//cuda version
void LKTracker::normCrossCorrelation(const GpuMat& img1, const GpuMat& img2, const GpuMat& gPoints1, const GpuMat& gPoints2, const vector<Point2f>& points1, const vector<Point2f> points2) {
	GpuMat res;
	GpuMat rec0;
	GpuMat rec1;

	similarity.clear();
	for (int i = 0; i < points1.size(); i++) {
		if (status[i] == 1) {
			Rect loc0(points1[i].x, points1[i].y, 10, 10);
			Rect loc1(points2[i].x, points2[i].y, 10, 10);
			rec0 = GpuMat(img1, loc0);
			rec1 = GpuMat(img2, loc1);
			gpu::matchTemplate(rec0, rec1, res, CV_TM_CCOEFF_NORMED);

			similarity.push_back(((float *)(res.data))[0]);
		}
		else {
			similarity.push_back(0.0);
		}
	}
	rec0.release();
	rec1.release();
	res.release();
}

void LKTracker::normCrossCorrelation(const Mat& img1, const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2)
{
	Mat rec0(10, 10, CV_8U);
	Mat rec1(10, 10, CV_8U);
	Mat res(1, 1, CV_32F);

	//double maxDis = 3.0f * sqrt(window_size.width*window_size.width + window_size.height*window_size.height); // 过滤距离过远的光流
	similarity.clear();
	for (int i = 0; i < points1.size(); i++) {
		if (status[i] == 1 /*&& norm(points1[i] - points2[i]) < maxDis*/) {
			getRectSubPix(img1, Size(10, 10), points1[i], rec0);
			getRectSubPix(img2, Size(10, 10), points2[i], rec1);
			matchTemplate(rec0, rec1, res, CV_TM_CCOEFF_NORMED);
			similarity.push_back( ((float *)(res.data))[0] );
		}
		else {
			similarity.push_back(0.0);
		}
	}
	rec0.release();
	rec1.release();
	res.release();
}


bool LKTracker::filterPts(vector<Point2f>& points1, vector<Point2f>& points2)
{
	Scalar val0 = mean(similarity);
	simmed = val0.val[0];
	
	size_t i, k;
	//double maxDis = 3.0f * sqrt(window_size.width*window_size.width + window_size.height*window_size.height); // 过滤距离过远的光流
	for (i = k = 0; i < points2.size(); ++i){
		if (!status[i] /*|| norm(points1[i] - points2[i]) > maxDis*/)
			continue;
		if (similarity[i] > simmed){
			points1[k] = points1[i];
			points2[k] = points2[i];
			FB_error[k] = FB_error[i];
			k++;
		}
	}
	if (k == 0) {
		//printf("k == 0! simmed = %lf, maxDis = %lf\n", simmed, maxDis);
		return false;
	}
	points1.resize(k);
	points2.resize(k);
	FB_error.resize(k);


	Scalar val1 = mean(FB_error);
	fbmed = val1.val[0];
	for (i = k = 0; i < points2.size(); ++i)
	{
		if (!status[i] /* || norm(points1[i] - points2[i]) > maxDis*/)
			continue;

		float xdiff = points2[i].x - points1[i].x;
		float ydiff = points2[i].y - points1[i].y;
		float mag = sqrt(xdiff*xdiff + ydiff*ydiff);

		if (FB_error[i] <= fbmed  && mag > 1)
		{
			points1[k] = points1[i];
			points2[k] = points2[i];
			k++;
		}
	}
	points1.resize(k);
	points2.resize(k);
	if (k > 0)
		return true;
	else
		return false;
}

