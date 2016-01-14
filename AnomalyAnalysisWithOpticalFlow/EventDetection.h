#ifndef EventDetection_H_TOMHEAVEN_20140730
#define EventDetection_H_TOMHEAVEN_20140730

#include <opencv2/opencv.hpp>
#include "LKTracker.h"
#include "cuda_utils.h"
using namespace cv;

using namespace cv::gpu;

/**********************************************************************************************************/
/*   以下为子函数
/**********************************************************************************************************/
//寻找最大值
template <typename Type>
Type FindMaxVal(Type * arr, int N)
{
	Type max = arr[0];
	for(int i=1; i< N; i++)
	{
		if(max < arr[i])
			max = arr[i];
	}
	return max;
}


/** 计算直方图柱值图像
   @param angleMat 光流角度矩阵
   @param flowmagMat 光流速度矩阵
   @param nBins 方向桶的个数
   @param BinsImages 各个方向的直方图矩阵
   @param angleOffset 计算角度时采用的零点
*/
//定义PI角度，有两种取法：一种是角度（180度）；另外一种是弧度（PI，3.1415927...)，这里我们采用角度
const float myPi = 180;
void calcBinsImages(const Mat & angleMat, const Mat & flowmagMat, int nBins, Mat * binsImages, float angleOffSet = 0.0f)
{
	//图像长宽
	int width = angleMat.cols;
	int height =angleMat.rows;

	//直方图每个柱的宽度
	float BinWid = (float)(2*myPi) /(float)nBins;

	//开始
	int bin1, bin2;
	float mag1, mag2;
	float dist,delta;
	for (int y=0; y<height; y++)
	{
		for (int x=0; x<width; x++)
		{
			float angle = angleMat.at<float>(y,x);
			float mag = flowmagMat.at<float>(y,x);

			//如果角度小于angleOffSet，根据函数calcSparseOpticalFlow中角度的计算方式，这说明该点是我们不关注的点，直接跳过即可
			if (angle < angleOffSet || mag < 1e-8)
				continue;

			//如果角度在直方图两边的边界上
			if (angle <= BinWid/2+angleOffSet || angle >= 2*myPi+angleOffSet-BinWid/2)
			{
				bin1 = nBins-1;
				bin2 = 0;
				if (angle < BinWid/2+angleOffSet)
					dist = angle-angleOffSet + BinWid/2;
				else
					dist = angle-angleOffSet - (bin1*BinWid + BinWid/2);
			}
			else
			{
				bin1 = int(floor((angle-angleOffSet-BinWid/2) / BinWid));
				bin2 = bin1+1;
				dist = angle - angleOffSet - (bin1*BinWid + BinWid/2);
			}

			delta = dist/BinWid;
			mag2 = delta * mag;
			mag1 = mag - mag2;

			binsImages[bin1].at<float>(y,x) = mag1;
			binsImages[bin2].at<float>(y,x) = mag2;
		}
	}
}

/**
   计算积分图像 （因OpenCV函数 integral 存在64位浮点数溢出的问题而编写）
   @param src 输入矩阵
   @param dst 积分图像
*/
void myIntegral(const Mat& src, Mat& dst) {
	float rowSum = 0.0f;
	dst = dst.zeros(dst.size(), dst.type());
	for(int i = 0; i < src.rows; ++i, rowSum = 0.0f) {
		for(int j = 0; j < src.cols; ++j) {
			rowSum += src.at<float>(i, j);
			if (i == 0)
				dst.at<float>(i, j) = rowSum;
			else
			    dst.at<float>(i, j) = dst.at<float>(i - 1, j) + rowSum;
		}
	}
}

/**
   计算积分图像 （输入为uchar，输出为float）
   @param src 输入矩阵
   @param dst 积分图像
*/
void myIntegralUchar(const Mat& src, Mat& dst) {
	float rowSum = 0.0f;
	dst = dst.zeros(dst.size(), dst.type());
	for(int i = 0; i < src.rows; ++i, rowSum = 0.0f) {
		for(int j = 0; j < src.cols; ++j) {
			rowSum += (src.at<uchar>(i, j) > 128);
			if (i == 0)
				dst.at<float>(i, j) = rowSum;
			else
				dst.at<float>(i, j) = dst.at<float>(i - 1, j) + rowSum;
		}
	}
}

/** 
  对光流直方图作最大值归一化
  @param binImage 光流直方图
  @param bins 直方图的桶数
*/
void normalize(Mat& binImage, int bins) {
	float maxValue = -1e4f;
	const float eps = 1.0f;
	for (int i = 0; i < bins; ++i) {
		if (binImage.at<float>(i) > maxValue)
			maxValue = binImage.at<float>(i);
	}
	if (maxValue > eps) {
		for (int i = 0; i < bins; ++i)
			binImage.at<float>(i) /= maxValue;
	}
}

/**计算积分直方图
   @param angleMat 光流角度矩阵
   @param flowmagMat 光流速度矩阵
   @param nBins 方向桶的个数
   @param BinsImages 输出的各个方向的直方图矩阵
   @param IntegralImages 输出的光流直方图的积分图像
*/
void calcIntegralHist(const Mat & angleMat, const Mat & flowmagMat, int nBins, Mat * binsImages, Mat * integralImages)
{
	int i;
	for(i=0; i< nBins; i++)
	{
		binsImages[i] = Mat(angleMat.size(), CV_32FC1);
		float * p;
		for(int r =0; r < binsImages[i].rows;r++)
		{
			p = binsImages[i].ptr<float>(r);
			for(int c = 0 ; c < binsImages[i].cols; c++)
				p[c] = 0;
		}
		integralImages[i] = Mat(angleMat.rows+1,angleMat.cols+1,CV_32FC1);
	}
	integralImages[i] = Mat(angleMat.rows+1,angleMat.cols+1,CV_32FC1);
	calcBinsImages(angleMat,flowmagMat,nBins,binsImages);

	for(i=0; i< nBins; i++)
	{
	    // 对 binImages 做归一化
		//normalize(binsImages[i], nBins);
		integral(binsImages[i], integralImages[i], CV_32F);
		//myIntegral(binsImages[i], integralImages[i]);
		//saveImageAsText(binsImages[0], "binsImages.txt");
		//saveImageAsText(integralImages[0], "integral.txt");
		//printf("In calcIntegralHist\n");
	}
}

//画箭头函数 Updated
void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha,
	cv::Scalar& color, int thickness, int lineType)
{

	Point arrow;
	//计算 θ 角（最简单的一种情况在下面图示中已经展示，关键在于 atan2 函数，详情见下面）   
	double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
	line(img, pStart, pEnd, color, thickness, lineType);
	//计算箭角边的另一端的端点位置（上面的还是下面的要看箭头的指向，也就是pStart和pEnd的位置） 
	arrow.x = pEnd.x + len * cos(angle + CV_PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle + CV_PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
	arrow.x = pEnd.x + len * cos(angle - CV_PI * alpha / 180);
	arrow.y = pEnd.y + len * sin(angle - CV_PI * alpha / 180);
	line(img, pEnd, arrow, color, thickness, lineType);
}


/**单密度采样：对当前图像，每间隔一定的间隔（比如5个像素）获取一个采样点，为了剔除一些无关紧要的点，对这些点进行判决，如果特征值过小，则将该点剔除。
   另外，通过引入mask，仅仅对mask覆盖的区域进行计算
   @param grey 输入的灰度图像
   @param points 输出的光流点对
   @param quality 过滤掉的向量特征值占最大特征值的比率
   @param min_distance 输入的空间采样距离
   @param mask 输入的计算时的mask
*/
void DenseSample1(const Mat& grey, std::vector<Point2f>& points, const double quality, const int min_distance, Mat mask)
{
	//确保输入图像grey和mask大小一致
	assert(grey.size()==mask.size());

	//采样点的横向和纵向个数
	int colnum = grey.cols/min_distance;
	int rownum = grey.rows/min_distance;

	//计算每个像素位置的最小的特征值
	Mat eig;
	cornerMinEigenVal(grey, eig, 3, 3);
	

	//找到所有最小的特征值中的最大值，并乘以一个quality系数，作为判决阈值
	double maxVal = 0;
	minMaxLoc(eig, 0, &maxVal);
	const double threshold = maxVal*quality;

	points.clear();
	int offset = min_distance/2;
	//遍历每一个稀疏采样点
	int i = 0, j = 0;
//#pragma omp for private(i, j)
	for(i = 0; i < rownum; i++)
		for(j = 0; j < colnum; j++) 
		{
			//稀疏点在原图像中的坐标
			int x = j*min_distance+offset;
			int y = i*min_distance+offset;

			if(eig.at<float>(y, x) > threshold && mask.at<uchar>(y,x)>0)
				points.push_back(Point2f(float(x), float(y)));

		}
}

/**  计算单密度光流 
  @param img1 输入的前帧图像
  @param img2 输入的后帧图像
  @param xflow 输出的x方向位移
  @param yflow 输出的y方向位移
  @param angle 输出的光流角度
  @param flowmag 输出的光流能量
  @param sparseMask 输出的计算了光流的区域mask
  @param showMat 输出的光流显示图像
  @param min_distance 输入的计算光流时空间采样间隔
  @param mask 输入的计算光流的mask
  @param angleOffset 计算角度时采用的零点
  @return 一个bool值，表示计算是否成功
*/
bool calcSparseOpticalFlow(const Mat & img1, const Mat & img2, Mat & xflow, Mat & yflow, 
						   Mat & angle, Mat & flowmag, Mat & sparseMask, Mat & showMat,const int min_distance, const Mat & mask, const float angleOffset = 0.0f)
{
	size_t i;

	//首先转化成灰度图像
	Mat gray1,gray2;
	cvtColor(img1,gray1,CV_BGR2GRAY);
	cvtColor(img2,gray2,CV_BGR2GRAY);

	//对第一幅图像进行单密度采样
	//std::vector<Point2f> points1(0);
	//DenseSample1(gray1, points1, 0.01, min_distance, mask);
	//DenseSample1(gray1, points1, 0.0000, min_distance, mask);
	//if (points1.size() == 0) 
		//return false;

	//Shi-Tomasi角点
	std::vector<Point2f> points1(0);

	int maxCorners = 10000;
	double qualityLevel = 0.001;

#ifdef  MEASURE_TIME
	clock_t startTime = clock();
#endif
	
#ifdef USE_CUDA 
	GoodFeaturesToTrackDetector_GPU* gfDetector = new GoodFeaturesToTrackDetector_GPU(maxCorners, qualityLevel, min_distance);
	GpuMat gGray1, gCorners, gMask;
	gGray1.upload(gray1);
	gMask.upload(mask);
#ifdef  MEASURE_TIME
	clock_t startTime1 = clock();
#endif
	gfDetector->operator()(gGray1, gCorners, gMask); //void operator ()(const GpuMat& image, GpuMat& corners, const GpuMat& mask = GpuMat());
#ifdef  MEASURE_TIME
	clock_t endTime1 = clock();
	printf("goodFeaturesToTrack pure time = %.3f\n", double(endTime1 - startTime1) / CLOCKS_PER_SEC);
#endif
	download(gCorners, points1);
#else
	goodFeaturesToTrack(gray1, points1, maxCorners, qualityLevel, min_distance, mask);
#endif


#ifdef  MEASURE_TIME
	clock_t endTime = clock();
	printf("goodFeaturesToTrack time = %.3f\n", double(endTime - startTime) / CLOCKS_PER_SEC);
#endif

	if (points1.size() == 0)
		return false;

	//采用TLD中前后向校正跟踪方法计算鲁棒光流
	std::vector<Point2f> points2(0);
	LKTracker track;
	if (!track.trackf2f(gray1, gray2, points1, points2)){
		printf("of_result: 0 matches!\n");
		points1.clear();
		points2.clear();
	}

	//创建XFLOW、YFLOW以及稀疏mask
	xflow = Mat(gray1.size(),CV_32FC1,Scalar(0));
	yflow = Mat(gray1.size(),CV_32FC1,Scalar(0));
	angle = Mat(gray1.size(),CV_32FC1,Scalar(0));
	flowmag = Mat(gray1.size(),CV_32FC1,Scalar(0));
	sparseMask = Mat(gray1.size(),CV_8UC1,Scalar(0));

	//计算
	for(i=0; i< points2.size(); i++)
	{
		if(points2[i].x > gray1.cols-1 || points2[i].x < 0 || points2[i].y > gray1.rows-1 || points2[i].y < 0)
			continue;

		sparseMask.at<uchar>(Point2i(points2[i])) = 255;

		float xdiff = points2[i].x-points1[i].x;
		float ydiff = points2[i].y-points1[i].y;
		float mag = sqrt(xdiff*xdiff+ydiff*ydiff);
		float angle_val = fastAtan2((float)(ydiff), (float)(xdiff)); 

		xflow.at<float>(Point2i(points2[i])) = xdiff;
		yflow.at<float>(Point2i(points2[i])) = ydiff;
		angle.at<float>(Point2i(points2[i])) = angle_val+ angleOffset;
		flowmag.at<float>(Point2i(points2[i])) = mag;
	}

#ifdef SHOW_RES
	//显示光流
	img2.copyTo(showMat);
	for(i=0; i< points2.size(); i++)
	{
		drawArrow(showMat,points2[i],points1[i],5,30,Scalar(0,0,255), 1, 8);
	}
#endif
	return true;
}

#endif