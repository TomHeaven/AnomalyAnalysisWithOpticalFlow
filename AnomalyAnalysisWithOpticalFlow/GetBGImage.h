#ifndef GETBGIMAGE_H_TOMHEAVEN_20140801
#define GETBGIMAGE_H_TOMHEAVEN_20140801


//#include "stdafx.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace std;
using namespace cv;
class yxFGDetectMPBaseRGB
{
public:
	yxFGDetectMPBaseRGB();					//构造函数
	~yxFGDetectMPBaseRGB();					//析构函数
	Mat GetFGImg();					//获取前景图像，外部调用
	Mat GetBGImg();					//获取背景图像，外部调用
public:
	void Init(const Mat&img);				//初始化函数
	bool ConstructBGModel(const Mat&img);   //建立背景模型
private:
	//图像大小
	int m_imgHeight;						//图像高度
	int m_imgWidth;							//图像宽度
	//MP算法参数
	double m_defaultMinArea;				//运动物体最小面积，用于剔除噪声
	int m_defaultRadius;;					//球体的半径，也即匹配的阈值，默认为20
	int m_defaultHistBinNum;				//直方图柱数量
	int m_defaultTrainningNum;				//用于背景学习的帧数
	int m_defaultTrainningItv;				//背景学习的帧间隔
	int m_defaultBGLevels;					//背景图像的数量
	//MP算法
	unsigned char ****m_ppppHist;			//每个空间位置沿着时间轴上的直方图
	Mat m_pFGImg;					//前景图像
	Mat m_pBGImg;					//背景图像

	int m_frmNum;						    //记录当前帧的序号，因为前20帧只用来做背景模型，不需要检测

};

#endif