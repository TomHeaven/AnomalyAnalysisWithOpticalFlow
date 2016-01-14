// AnomalyAnalysisWithOpticalFlow.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <Windows.h>
#include "ImageTemplate.h"
#include "ImageUtils.h"
#include "EventDetection.h"
#include "GetBGImage.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <io.h>
#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <vector>
#include <queue>
#include <omp.h>
#include <ctime>

//按16*16分块
#define BLOCK_16
//输出每一帧运行的时间
// #define MEASURE_TIME //在stdafx.h中统一定义
//展示和保存中间结果
//#define SHOW_RES



using namespace std;
using namespace cv;

#define  _CRT_SECURE_NO_WARNINGS


/**  保存特征文件夹 */
char feature_folder[255] = "train_features";
/** 控制是否保存特征的开关 */
bool bSaveFeatures = false;
/** 控制是否执行检测的开关，如果不执行，输出的resVideo是全黑的；将由Matlab的光流来完成detection */
bool bPerformDetection = true;
/** 控制是否保存内容的开关 */
bool bSaveBackground = false;


/** 提取视频中的运动目标并保存到文件 (>= opencv 1.0)
@param Video_Name - 输入视频路径
@param outBackground - 输出视频路径
@param FrmNumforBuildBG - 建立背景模型所需的帧数， 默认 100
@param defaultMinArea - 过滤最小面积的阈值，默认100
@return a boolean value - 是否成功
*/
bool subtractMovingObject(const char* video_Name, const char* outBackground, const char* outForeground, int FrmNumforBuildBG = 100, int defaultMinArea = 100)//
{
	yxFGDetectMPBaseRGB fgdetector;

	int nFrmNum = 0;
	CvScalar meanScalar0, meanScalar;

	vector<CvRect> VecTargetBlob;

	VideoCapture inputVideo(video_Name);        // Open input
	if (!inputVideo.isOpened())
	{
		cout << "Could not open the input video." << video_Name << endl;
		return false;
	}

	Mat frame;
	inputVideo.read(frame);
	VideoWriter backWriter(outBackground, CV_FOURCC('X', 'V', 'I', 'D'), inputVideo.get(CV_CAP_PROP_FPS), cvSize(frame.cols, frame.rows), false);
	VideoWriter foreWriter(outForeground, CV_FOURCC('X', 'V', 'I', 'D'), inputVideo.get(CV_CAP_PROP_FPS), cvSize(frame.cols, frame.rows), false);

	Mat pFrame, pBkImg, pBGImg, pPreImage, pDiffImage, pFinalFGImage, showSrcImage, pFrImg, BianryImage, showSrcImage1, BigImg;

	inputVideo >> pFrame;
	fgdetector.Init(pFrame);

#ifdef SHOW_RES
	namedWindow("background", CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	namedWindow("FGImage", CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	namedWindow("currentFrame", CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	resizeWindow("background", 800, 600);
	resizeWindow("FGImage", 800, 600);
	resizeWindow("currentFrame", 800, 600);
#endif

	while (true)
	{
		inputVideo >> pFrame;
		if (pFrame.empty()) break;

		nFrmNum++;
		/**********************************************************************************************************/
		/*  建立初始背景模型
		/**********************************************************************************************************/
		if (nFrmNum < FrmNumforBuildBG)
		{
			//如果是第一帧，需要申请内存，并初始化
			//	printf("Now: %d\n", nFrmNum);
			//	double timeJianMo=GetTickCount();
			if (fgdetector.ConstructBGModel(pFrame)) break;
			//		timeJianMo=GetTickCount()-timeJianMo;
			//	printf("timeJianMo=%f\n",timeJianMo);
			pBGImg = fgdetector.GetBGImg();

#ifdef SHOW_RES
			//显示背景图像
			imshow("background", pBGImg);
			waitKey(10);
#endif

		}
		if (nFrmNum >= FrmNumforBuildBG)
		{
			if (nFrmNum == FrmNumforBuildBG)
			{
				inputVideo.release();
				inputVideo.open(video_Name);
				inputVideo >> pFrame;

				cvtColor(pBGImg, pBkImg, CV_BGR2GRAY);
				cvtColor(pFrame, pPreImage, CV_BGR2GRAY);
			}
			else
			{
				cvtColor(pFrame, pFrImg, CV_BGR2GRAY);
				imshow("currentFrame", pFrame);
#pragma omp parallel sections
				{
#pragma omp section//差分
					{
						showSrcImage = pFrame.clone();
						showSrcImage1 = pFrame.clone();
						//当前帧转换为灰度图像
						//前一帧灰度图像与当前帧灰度图像相减
						absdiff(pPreImage, pFrImg, pDiffImage);
						//二值化帧差图像
						threshold(pDiffImage, BianryImage, 15, 255.0, CV_THRESH_BINARY);
					}
#pragma omp section//背景减
					{
						//计算当前图像的均值	 
						meanScalar0 = mean(pBkImg);
						meanScalar = mean(pFrImg);

						//更新背景以便适应光照变化
						Scalar diff;
						diff.val[0] = meanScalar.val[0] - meanScalar0.val[0];
						pBkImg = pBkImg + diff;

						// 实时更新背景模型（高斯滑动平均）

						//myRunningAvg(pFrImg,pBkImg, Mat(),0.003f); //并行优化 error in vs 2013

						//	accumulateWeighted(pFrImg,pBkImg,0.003); //此函数有 BUG
						addWeighted(pFrImg, 0.003, pBkImg, 1 - 0.003, 0, pBkImg);  // vs 2013 O2 优化时背景检测算法会出错！！！

						//当前帧跟背景图相减
						absdiff(pBkImg, pFrImg, pFinalFGImage);
						//subtract(pFrImg, pBkImg, pFinalFGImage);
						//abs(pFinalFGImage);
						//二值化前景图
						threshold(pFinalFGImage, pFrImg, 30, 255.0, CV_THRESH_BINARY);
					}
				}
#ifdef SHOW_RES
				//显示背景图像
				imshow("background", pBGImg);
#endif
				BinaryORBinaryImage(BianryImage, pFrImg, pFinalFGImage);

				Mat FGImage;
				//double t2=GetTickCount();
				RemoveSmallErea(pFinalFGImage, FGImage, defaultMinArea);
				//进行形态学滤波，去掉噪音
				//IplConvKernel*Kerne1=cvCreateStructuringElementEx(3,7,0,0,CV_SHAPE_ELLIPSE);
				dilate(FGImage, FGImage, Mat());

#ifdef SHOW_RES
				imshow("FGImage", FGImage);
				waitKey(10);
#endif
				foreWriter.write(FGImage);

				if (bSaveBackground) {
			    	backWriter.write(pBkImg);
			    }

				
			}
			//更新前一帧图像
			cvtColor(pFrame, pPreImage, CV_BGR2GRAY);
			if (cvWaitKey(1) == 27) break;
		}
	}
	destroyAllWindows();
	return true;
}


/**
增加一帧属性到特征
@param feature 输出的六维特征矩阵
@param integralImages 光流积分图像组
@param blockWidth 统计特征分块的宽度
@param blockHeight 统计特征分块的高度
@param blockDeltaX 统计特征分块的宽度间隔
@param blockDeltaY 统计特征分块的高度间隔
*/

void addFrameToFeature(Mat& feature, const Mat* integralImages, const int blockWidth, const int blockHeight, int blockDeltaX, int blockDeltaY) {
	int i, j;
#pragma omp parallel for private (i, j)
	for (i = 0; i < feature.rows; ++i) {
		for (j = 0; j < feature.cols; ++j) {
			Vec7f& vec = feature.at<Vec7f>(i, j); //六通道特征
			int blockX = j * blockDeltaX;
			int blockY = i * blockDeltaY;
			int xLimit = blockX + blockWidth;
			int yLimit = blockY + blockHeight;

			//如果滑窗出界，修正边界
			if (blockX + blockWidth >= integralImages[0].cols) {
				//xLimit = integralImages[0].cols - 1;
				continue;
			}
			if (blockY + blockHeight >= integralImages[0].rows) {
				//yLimit = integralImages[0].rows - 1;
				continue;
			}

			for (int k = 0; k < feature.channels(); ++k) {
				float a = integralImages[k].at<float>(blockY, blockX);
				float b = integralImages[k].at<float>(blockY, xLimit);
				float c = integralImages[k].at<float>(yLimit, blockX);
				float d = integralImages[k].at<float>(yLimit, xLimit);
				vec[k] += (a + d - b - c);
			}
		}
	}
}

/**
从特征矩阵中减掉
@param feature 输出的六维特征矩阵
@param integralImages 光流积分图像组
@param blockWidth 统计特征分块的宽度
@param blockHeight 统计特征分块的高度
@param blockDeltaX 统计特征分块的宽度间隔
@param blockDeltaY 统计特征分块的高度间隔
*/
void removeFrameFromFeature(Mat& feature, const Mat* integralImages, const int blockWidth, const int blockHeight, int blockDeltaX, int blockDeltaY) {
	int i, j;
#pragma omp parallel for private (i, j)
	for (i = 0; i < feature.rows; ++i) {
		for (j = 0; j < feature.cols; ++j) {
			Vec7f& vec = feature.at<Vec7f>(i, j); //六通道特征
			int blockX = j * blockDeltaX;
			int blockY = i * blockDeltaY;
			int xLimit = blockX + blockWidth;
			int yLimit = blockY + blockHeight;

			//如果滑窗出界，修正边界
			if (blockX + blockWidth >= integralImages[0].cols) {
				//xLimit = integralImages[0].cols - 1;
				continue;
			}
			if (blockY + blockHeight >= integralImages[0].rows) {
				//yLimit = integralImages[0].rows - 1;
				continue;
			}

			for (int k = 0; k < feature.channels(); ++k) {
				float a = integralImages[k].at<float>(blockY, blockX);
				float b = integralImages[k].at<float>(blockY, xLimit);
				float c = integralImages[k].at<float>(yLimit, blockX);
				float d = integralImages[k].at<float>(yLimit, xLimit);
				vec[k] -= (a + d - b - c);
			}
		}
	}
}

/** 保存特征到文件
   @param featureBuf 缓存特征的vector
   @param bins 光流特征维数，还有一维静止特征，故实际特征是 bins + 1 维
   */

void saveFeature(vector<Mat>& featureBuf, int bins) {
	if (featureBuf.size() == 0)
		return;

	int rows = featureBuf[0].rows;
	int cols = featureBuf[0].cols;

	printf("rows = %d, cols = %d\n", rows, cols);
	char filename[255];
	printf("开始保存特征数据(OpenCV窗口可能会失去响应，请不要关闭程序)...\n");
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			int fileNum = i * cols + j;
			sprintf_s(filename, "%s/f_%d.txt", feature_folder, fileNum);
			FILE* fout = fopen(filename, "a");
			if (fout == NULL) {
				printf("error open file %s for writing.\n", filename);
				break;
			}

			for (int k = 0; k < featureBuf.size(); ++k) {
				Vec7f& vec = featureBuf[k].at<Vec7f>(i, j);
				//fprintf(fout, "1\t");
				for (int l = 0; l < bins + 1; ++l)
					//	fprintf(fout, "%d:%.0f\t", l, vec[l]);
					fprintf(fout, "%.0f\t", vec[l]);
				fprintf(fout, "\n");
			}
			//fflush(fouts[fileNum]);
			fclose(fout);
		}
	}
	featureBuf.clear();
	printf("完成一次保存。\n");
}


#ifdef  MEASURE_TIME
clock_t startTime;
clock_t endTime;
#endif

/**
计算 busyness 图像或者进行异常检测 (>= opencv 2.0)
@param foregroundVideo - 输入的前景视频
@param originVideo - 输入的原始视频
@param win_size 时间窗口大小
@param bCompBusyness 是否是计算busyness
@param busyness 如果是计算busyness，则返回此值；否则利用此值进行异常检测
@param writer 如果进行异常检测，输出结果到 writer
@param of_time_interval 光流的时间采样间隔
@param of_space_interval 光流的空间采样间隔
@param bins 桶的个数
@param blockWidth 统计特征分块的宽度
@param blockHeight 统计特征分块的高度
@param blockDeltaX 统计特征分块的宽度间隔
@param blockDeltaY 统计特征分块的高度间隔
@param thresh 判别是否异常的阈值参数，在[0, 1]之间，典型值为 0.032
@return 是否成功
*/
bool busynessAndDetectionCommon(const char* foregroundVideo, const char* originVideo, const int& win_size, const bool bCompBusyness, Mat& busyness, VideoWriter& writer, const char* outFlowVideo, const int& of_time_interval, const int& of_space_interval, int bins, int blockWidth, int blockHeight, int blockDeltaX, int blockDeltaY, float thresh)  {
	//参数检查
	assert(win_size >= 1);

	VideoCapture capture(foregroundVideo);
	if (!capture.isOpened())
	{
		printf("Can not open foreground video file %s\n", foregroundVideo);
		return false;
	}

	VideoCapture oCapture(originVideo);
	if (!oCapture.isOpened())
	{
		printf("Can not open original video file %s\n", originVideo);
		return false;
	}

	//定义数据队列 queue  注意：用队列的时候一定要 push(mat.clone())，否则push的是mat的引用！！！

	queue<Mat> ofQueue;       // 光流时间间隔队列
	queue<Mat*> frameQueue; // 前景灰度图像 frame 队列，angle 队列， flowmag 队列

	// rgbForeground 是RGB前景当前帧, foreground 是灰度化的当前帧, curFrame 是原始视频当前帧
	Mat rgbForeground, foreground, curFrame;

	capture.read(rgbForeground);
	oCapture.read(curFrame);
	//检查两个视频的帧大小一致
	assert(rgbForeground.cols == curFrame.cols && rgbForeground.rows == curFrame.rows);
	
#ifdef SHOW_RES
	// 保存光流输出到文件
	VideoWriter flowWriter(outFlowVideo, CV_FOURCC('X', 'V', 'I', 'D'), capture.get(CV_CAP_PROP_FPS), cvSize(curFrame.cols, curFrame.rows), true);
	if (!flowWriter.isOpened())
	{
		printf("Can not open outflow video file %s for writing.\n", outFlowVideo);
		return false;
	}
#endif

	//定义(bins+1)维特征矩阵
	Mat feature(busyness.rows, busyness.cols, CV_32FC(bins + 1), Scalar(0));
	//定义经过块模糊的特征矩阵
	Mat blurredFeature;
	Mat* binsImages = NULL, *integralImages = NULL;

	int nFrame = 0;

#ifdef SHOW_RES
	namedWindow("flow", CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	namedWindow("foreground", CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	namedWindow("flowmag", CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	namedWindow("integral", CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
#endif
	
	//	namedWindow("of_last_frame", CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	if (!bCompBusyness) {
		namedWindow("res", CV_WINDOW_KEEPRATIO | CV_WINDOW_NORMAL);
	}

	//为检测的闭操作做准备
	int dilation_size = 1;
	Mat element = getStructuringElement(MORPH_ELLIPSE,
		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
		Point(dilation_size, dilation_size));
	//////////////////

	//  x光流，y光流，角度，能量，计算出光流的mask，画出箭头的图像
	Mat xflow, yflow, angle, flowmag, sparseMask, showFlowMat;

	//缓存特征后定期保存到文件
	int bufsize = 1000;
	vector<Mat> featureBuf;

	//保存异常块数据
	FILE* fout = NULL;
	if (!bCompBusyness)
		fout = fopen("abnormalBlock.txt", "w");

	while (capture.read(rgbForeground) && oCapture.read(curFrame)) {
#ifdef  MEASURE_TIME
		startTime = clock();
#endif

		++nFrame;
		//转化为灰度 frame
		cvtColor(rgbForeground, foreground, CV_RGB2GRAY);

		//刚开始时，首先读入interval+1帧图像，我们将这些图像称为基准图像组
		if (nFrame == 1) {
			for (int k = 0; k < of_time_interval + 1; k++)
			{
				ofQueue.push(curFrame.clone());
				if (k < of_time_interval)
					oCapture.read(curFrame);
			}
			continue;
		}
		//对当前图像跟基准图像计算单密度光流
		// 为了测试，将mask全部赋值成白色
		//foreground = Mat::ones(foreground.rows, foreground.cols, foreground.type()) * 255;

		if (!calcSparseOpticalFlow(curFrame, ofQueue.front(), xflow, yflow, angle, flowmag, sparseMask, showFlowMat, of_space_interval, foreground))
			continue;

#ifdef  MEASURE_TIME
		endTime = clock();
		printf("calcSparseOpticalFLow time = %.3f\n", double(endTime - startTime) / CLOCKS_PER_SEC);
#endif

		//计算光流积分直方图
		binsImages = new Mat[bins];
		integralImages = new Mat[bins + 1];
		calcIntegralHist(angle, flowmag, bins, binsImages, integralImages); // 注意 calcIntegralHist 会初始化 integraImages[bins] 的内存
		//计算前景积分图
		//binsImages[bins] = Mat(foreground.rows+1,foreground.cols+1,CV_32FC1);
		myIntegralUchar(foreground, integralImages[bins]);
		delete[] binsImages;

		//前景、光流角度、光流能量 入队列
		frameQueue.push(integralImages);

		//增加当前帧到特征
		addFrameToFeature(feature, integralImages, blockWidth, blockHeight, blockDeltaX, blockDeltaY);

#ifdef  MEASURE_TIME
		endTime = clock();
		printf("addFrameToFeature time = %.3f\n", double(endTime - startTime) / CLOCKS_PER_SEC);
#endif

		// 开始计算 busynessf
		if (nFrame >= win_size) {
			//减去已经出窗口的 frame
			Mat* prevIntegralImages = frameQueue.front();
			removeFrameFromFeature(feature, prevIntegralImages, blockWidth, blockHeight, blockDeltaX, blockDeltaY);
			//出队列
			frameQueue.pop();
			delete[] prevIntegralImages;

			//对特征作模糊处理
			int blockKsize = 1;
			int ksize = blockWidth / blockDeltaX * blockKsize;
			blur(feature, blurredFeature, Size(ksize, ksize));

			// 将blurredFeature加入特征队列
			if (bSaveFeatures)
				featureBuf.push_back(blurredFeature.clone());
		
			//保存特征到文件
			// 当 featureBuf 达到内存缓存上限时，将数据写入文件，并清空 featureBuf
			if (bSaveFeatures && featureBuf.size() > bufsize) {
				saveFeature(featureBuf, bins); //保存路径会因训练和检测有所区别。
			}

#ifdef SHOW_RES
			imshow("foreground", foreground);
			imshow("flow", showFlowMat);
			imshow("flowmag", flowmag);
			imshow("integral", integralImages[bins]);
			waitKey(1);
			flowWriter.write(showFlowMat);
#endif

			if (bCompBusyness) {
				//计算最大值以获得 busyness
				max(blurredFeature, busyness, busyness);
			}
			else { //做 detection
				//检测当前观测图像
				//	Mat res(feature.rows * blockHeight, feature.cols * blockHeight, CV_8UC3, Scalar(0));
				Mat res(foreground.rows, foreground.cols, CV_8UC3, Scalar(0));
				int i, j;
				int channel = 0;
				// openmp 加速
				//	double cmpRes = compareHist(feature, busyness, CV_COMP_CORREL);
				//	printf("cmpRes = %d\n", cmpRes);
				if (bPerformDetection) {
					int nAbnormal = 0;
#pragma omp parallel for private (i, j)
					for (i = 0; i < feature.rows; ++i) {
						for (j = 0; j < feature.cols; ++j) {
							//仅仅标记前景中运动的区域
							//for (int k = 0; k < bins + 1; ++k) {
							for (int k = 0; k < bins; ++k) {
								Vec7f& vec2 = blurredFeature.at<Vec7f>(i, j);
								Vec7f& vec1 = busyness.at<Vec7f>(i, j);
								//光流过密检查
								//	float MAX_FLOW =  1.0f * win_size * blockWidth * blockHeight / (of_space_interval * of_space_interval); //数值阈值
								//	if (vec2[k] > MAX_FLOW)
								//	vec2[k] = MAX_FLOW;


								//将当前特征与busyness比较， threshold 是阈值，需要根据 win_size 调整
								float threshold = thresh * win_size * blockWidth * blockHeight / (of_space_interval * of_space_interval); //数值阈值
								if (k == bins) //对于前景特征，阈值不同
									threshold = thresh * win_size * blockWidth * blockHeight;
								float ratio = 0.2f;  //比例阈值，很有必要
								if (float(vec2[k] - vec1[k]) / (vec1[k] + 2) >= ratio &&
									vec2[k] - vec1[k] >= threshold) {
									for (int y = i * blockDeltaY; y < i * blockDeltaY + blockHeight; ++y) {
										//如果滑窗出界，修正边界
										if (y >= res.rows) {
											break;
										}
										for (int x = j * blockDeltaX; x < j * blockDeltaX + blockWidth; ++x) {
											//如果滑窗出界，修正边界
											if (x >= res.cols) {
												break;
											}

											for (int channel = 0; channel < 3; ++channel)
												res.at<Vec3b>(y, x)[channel] = 255;
										}
									} 
									++nAbnormal;
									break;
								} // end if >
							} //end for k
						}  //end for j
					}  //end for i
					//将此检测帧的异常块比例写入文件
					int foregroundSum = 0;
					for (int i = 0; i < foreground.rows; ++i) {
						for (int j = 0; j < foreground.cols; ++j) {
							if (foreground.at<uchar>(i, j) > 128)
								++foregroundSum;
						}
					}
					double ratio = nAbnormal;
					if (foregroundSum > 0)
						ratio /= foregroundSum;
					fprintf(fout, "%d %.3f %d\n", nAbnormal, ratio, foregroundSum);
					//闭操作
					//erode(res, res, element);
					//dilate(res, res, element);
				} // end if (bPerformDetection)

#ifdef SHOW_RES
				imshow("res", res);
				waitKey(1);
#endif
				//保存结果到视频
				writer.write(res);
			}
		}
		//将当前帧加入基准图像组，将用过的帧出队列
		ofQueue.push(curFrame.clone());
		ofQueue.pop();


#ifdef SHOW_RES
		//ESC 可退出
		if (waitKey(8) == 27) {
			break;
		}
#endif
		

#ifdef  MEASURE_TIME
		clock_t endTime = clock();
		printf("one-frame time = %.3f\n", double(endTime - startTime) / CLOCKS_PER_SEC);
#endif
	}

	//收尾工作 + 清理垃圾
	if (bSaveFeatures && featureBuf.size() > 0) {
		saveFeature(featureBuf, bins);

	}

	//列表标出需要 Sparse 计算的块
	if (bSaveFeatures) {
		char filePath[255];
		sprintf(filePath, "%s\\validBlocks.txt", feature_folder);
		FILE* fout = fopen(filePath, "w");
		for (int i = 0; i < busyness.rows; ++i) {
			for (int j = 0; j < busyness.cols; ++j) {
				bool bValid = false;
				//计算 validBlocks 时不考虑静止特征 (只循环到 bins 而不是 bins + 1）
				for (int k = 0; k < bins; ++k) {
					Vec7f& vec = busyness.at<Vec7f>(i, j);
					//将当前特征与busyness比较， threshold 是阈值，需要根据 win_size 调整
					float threshold = thresh * win_size * blockWidth * blockHeight / (of_space_interval * of_space_interval); //数值阈值
					//if (k == bins) //对于前景特征，阈值不同
						//threshold = thresh * win_size * blockWidth * blockHeight;
					if (vec[k] > threshold && threshold > 0) {
						bValid = true;
						break;
					}
				}
				if (bValid) {
					int fileNum = i * busyness.cols + j;
					fprintf(fout, "%d ", fileNum);
				}
			}
		}
		fclose(fout);
	}

	while (!frameQueue.empty()) {
		Mat* prevIntegralImages = frameQueue.front();
		frameQueue.pop();
		delete[] prevIntegralImages;
	}
	destroyAllWindows();
	if (fout != NULL)
		fclose(fout);

	return true;
}


/**
计算 busyness 图像 (>= opencv 2.0)
@param foregroundVideo - 输入的前景视频
@param originVideo - 输入的原始视频
@param win_size - 时间窗口大小
@param of_time_interval - 光流的时间采样间隔
@param of_space_interval - 光流的空间采样间隔
@param blockDeltaX 统计特征分块的宽度间隔
@param blockDeltaY 统计特征分块的高度间隔
@return Mat - busyness 图像(6 通道)：x方向, y方向速度最大值、最小值， 不动 （flowmag < 2*2）, 最大速度范数 max(flowmag)
*/

Mat computeBusyness(const char* foregroundVideo, const char* originVideo, const int& win_size, const char* outFlowVideo, const int& of_time_interval, const int& of_space_interval, int bins, int blockWidth, int blockHeight, int blockDeltaX, int blockDeltaY) {
	VideoCapture capture(foregroundVideo);
	if (!capture.isOpened())
	{
		printf("Can not open foreground video file %s\n", foregroundVideo);
		return Mat::zeros(1, 1, CV_8UC1);
	}
	Mat frame;
	capture.read(frame);
	//声明要计算的busyness
	Mat busyness((frame.rows - blockHeight) / blockDeltaX, (frame.cols - blockWidth) / blockDeltaY, CV_32FC(bins + 1), Scalar(0));
	//Mat busyness(frame.rows / blockDeltaX, frame.cols / blockDeltaY, CV_32FC(bins + 1), Scalar(0)); //无重叠时不需要减去 blockWidth
	//声明一个无用的writer以便调用框架函数
	VideoWriter writer;
	//调用通用框架 (busyness)
	float thresh = 0.02f; //此参数仅检测时有用，此处无用
	busynessAndDetectionCommon(foregroundVideo, originVideo, win_size, true, busyness, writer, outFlowVideo, of_time_interval, of_space_interval, bins, blockWidth, blockHeight, blockDeltaX, blockDeltaY, thresh);
	return busyness;
}


/**
检测输入的前景视频中的异常事件
@param foregroundVideo - 输入的前景视频
@param originVideo - 输入的原始视频
@param outVideo - 输出的异常检测视频
@param win_size - 时间窗口大小
@param of_time_interval - 光流的时间采样间隔
@param of_space_interval - 光流的空间采样间隔
@param blockDeltaX 统计特征分块的宽度间隔
@param blockDeltaY 统计特征分块的高度间隔
@param thresh 判别是否异常的阈值参数，在[0, 1]之间，典型值为 0.02
@return 是否成功
*/

bool detectAbnomal(Mat busyness, const char* foregroundVideo, const char* originVideo, const char* outVideo, const int win_size, const char* outFlowVideo, const int& of_time_interval, int bins, const int& of_space_interval, int blockWidth, int blockHeight, int blockDeltaX, int blockDeltaY, float thresh) {
	VideoCapture capture(foregroundVideo);
	if (!capture.isOpened())
	{
		printf("Can not open foreground video file %s\n", foregroundVideo);
		return false;
	}
	Mat frame;
	capture.read(frame);
	//创建 writer
	VideoWriter writer(outVideo, CV_FOURCC('X', 'V', 'I', 'D'), capture.get(CV_CAP_PROP_FPS), cvSize(frame.cols, frame.rows), true);
	//调用通用框架 (detection)
	return busynessAndDetectionCommon(foregroundVideo, originVideo, win_size, false, busyness, writer, outFlowVideo, of_time_interval, of_space_interval, bins, blockWidth, blockHeight, blockDeltaX, blockDeltaY, thresh);
}


/**
主函数
@param argc 参数个数
@param argv 参数
@return 返回值
*/
int _tmain(int argc, char* argv[])
{
	if (argc < 3) {
		printf("用法：\n");
		printf("AnomalyAnalysisWithOpticalFlow.exe 文件目录 训练文件名 测试文件名 [1-4] thresh [test | train] BLOCK_WIDTH BLOCK_HEIGHT\n");
		printf("例子（运行第二步，共四步）：\n");
		printf("AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi  2\n");
		printf("例子（修改检测参数 thresh为0.2）：\n");
		printf("AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi 3 0.2\n");
		printf("例子（运行第二步测试）：\n");
		printf("AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi  2  0.02 test\n");
		printf("例子（全部运行）：\n");
		printf("AnomalyAnalysisWithOpticalFlow.exe E:\videos train.avi test.avi\n");
		return 0;
	}

	vector<string> files;
	char* folder = argv[1];
	char* trainfile = argv[2];
	char* testfile = argv[3];
	int runPart = 0;
	if (argc >= 4 && argv[4] != NULL) {
		runPart = atoi(argv[4]);
	}
	//检测阈值参数
	float thresh = 0.1f;
	if (argc >= 5 && argv[5] != NULL) {
		thresh = float(atof(argv[5]));
	}

	//参数 bPerformDetection
	if (argc >= 6 && argv[6] != NULL) {
		bPerformDetection = atoi(argv[6]) != 0;
	}

	const int TRAIN = 1;
	const int TEST = 2;
	const int TRAIN_AND_TEST = 3;
	int trainOrTest = TRAIN_AND_TEST;
	if (argc >= 7 && argv[7] != NULL) {
		if (strcmp(argv[7], "train") == 0) {
			trainOrTest = TRAIN;
		}
		else if (strcmp(argv[7], "test") == 0) {
			trainOrTest = TEST;
		}
	}

	//blockSize
#ifdef BLOCK_16
	int blockWidth = 16, blockHeight = 16;
#else
	int blockWidth = 32, blockHeight = 32;
#endif

	if (argc >= 8 && argv[8] != NULL) {
		blockWidth = atoi(argv[8]);
	}
	if (argc >= 9 && argv[9] != NULL) {
		blockHeight = atoi(argv[9]);
	}

	char fileTrain[255];
	sprintf_s(fileTrain, "%s\\%s", folder, trainfile);
	char fileTrainBackground[255];
	sprintf_s(fileTrainBackground, "%s\\%s_background.avi", folder, trainfile);
	char fileTrainForeground[255];
	sprintf_s(fileTrainForeground, "%s\\%s_foreground.avi", folder, trainfile);

	char fileDetection[255];
	sprintf_s(fileDetection, "%s\\%s", folder, testfile);
	char fileDetectionBackground[255];
	sprintf_s(fileDetectionBackground, "%s\\%s_background.avi", folder, testfile);
	char fileDetectionForeground[255];
	sprintf_s(fileDetectionForeground, "%s\\%s_foreground.avi", folder, testfile);


	char fileTrainFlow[255];
	sprintf_s(fileTrainFlow, "%s\\%s_flow.avi", folder, trainfile);
	char fileDetectionFlow[255];
	sprintf_s(fileDetectionFlow, "%s\\%s_flow.avi", folder, testfile);

	char fileDetectionRes[255];
	sprintf_s(fileDetectionRes, "%s\\%s_result.avi", folder, testfile);
	char fileDetectionResCombined[255];
	sprintf_s(fileDetectionResCombined, "%s\\%s_合并结果.avi", folder, testfile);

	//char fileBusyness[255];
	//sprintf_s(fileBusyness, "%s\\%s_busyness.bmp", folder, trainfile);
	char fileBusynessTxt[255];
	sprintf_s(fileBusynessTxt, "%s\\%s_busyness.txt", folder, trainfile);
	//参数
	int win_size = 5;
	//注意：修改bins必须修改 typedef Vec<float, 10> Vec7f
	int of_time_interval = 2, of_space_interval = 1, bins = 9; //blockDeltaX = 16, blockDeltaY = 16;

#ifdef BLOCK_16
	int blockDeltaX = 8, blockDeltaY = 8;
#else
	int blockDeltaX = 16, blockDeltaY = 16;
#endif

	// 是否保存特征
	bSaveFeatures = false;
	// 是否执行检测
	bPerformDetection = true;

	Mat busyness;

	// 1. 提取训练数据和检测视频运动目标并保存
	int frameForBackground = 100;
	int smallAreaLimit = 30;
	if (runPart == 0 || runPart == 1) {
		printf("1. 提取训练数据和检测视频运动目标并保存\n");
		if (trainOrTest == TRAIN || trainOrTest == TRAIN_AND_TEST) //是否提取训练前景
			subtractMovingObject(fileTrain, fileTrainBackground, fileTrainForeground, frameForBackground, smallAreaLimit);
		if (trainOrTest == TEST || trainOrTest == TRAIN_AND_TEST) //是否提取测试前景
			subtractMovingObject(fileDetection, fileDetectionBackground, fileDetectionForeground, frameForBackground, smallAreaLimit);
	}
	// 2. 计算 busyness 图像
	if (runPart == 0 || runPart == 2) {
		if (bSaveFeatures) {
			char cmd[255];
			sprintf(feature_folder, "train_features");
			sprintf(cmd, "mkdir %s", feature_folder);
			system(cmd);
		}

		printf("2. 计算 busyness 图像\n");
		busyness = computeBusyness(fileTrainForeground, fileTrain, win_size, fileTrainFlow, of_time_interval, of_space_interval, bins, blockWidth, blockHeight, blockDeltaX, blockDeltaY);
		saveImageAsText(busyness, fileBusynessTxt);
	}

	// 3. 检测目标视频
	if (runPart == 0 || runPart == 3) {
		printf("3. 检测目标视频\n");
		if (bSaveFeatures) {
			char cmd[255];
			sprintf(feature_folder, "detect_features");
			sprintf(cmd, "mkdir %s", feature_folder);
			system(cmd);
		}
		busyness = loadImageFromText(fileBusynessTxt);
		detectAbnomal(busyness, fileDetectionForeground, fileDetection, fileDetectionRes, win_size, fileDetectionFlow, of_time_interval, bins, of_space_interval, blockWidth, blockHeight, blockDeltaX, blockDeltaY, thresh);
	}

	// 4. 合并视频
	if (runPart == 0 || runPart == 4) {
		printf("4. 合并视频\n");
		combineVideoFiles(fileDetection, fileDetectionRes, fileDetectionResCombined, int(win_size * 0.5 + 0.5), ONE_COL);
	}
	return 0;
}

