//#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"

//如果采用Vibe算法更新背景
#define	USING_VIBE

class yxFGDetectMPBaseRGB
{
public:
	yxFGDetectMPBaseRGB();					//构造函数
    ~yxFGDetectMPBaseRGB();					//析构函数
	void Process(IplImage *img);			//主处理函数，外部调用
	IplImage* GetFGImg();					//获取前景图像，外部调用
	IplImage* GetBGImg();					//获取背景图像，外部调用
public:
	void Init(IplImage *img);				//初始化函数
	bool ConstructBGModel(IplImage *img);   //建立背景模型
	void FGDetect(IplImage *img);           //前景检测及背景模型更新
	void FilterFGbyHSV(IplImage * srcimg,double sthr=0.1, double vthr=0.4); //根据原始图像的HSV数组对前景进行过滤，主要是为了剔除一些颜色不符合要求的像素点

protected:
	void DeleteSmallArea();                 //删除小面积目标

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
	IplImage * m_pFGImg;					//前景图像
	IplImage * m_pBGImg;					//背景图像

	int m_frmNum;						    //记录当前帧的序号，因为前20帧只用来做背景模型，不需要检测

	//如果采用VIBE算法更新背景
#ifdef USING_VIBE
	//VIBE算法
	int m_defaultLifetime;					//用于标记一个像素最多为前景的次数
	int m_defaultMinMatch;					//当前像素与背景模型匹配的最少个数
	int m_defaultModelNum;					//每个像素点采样数
	int m_defaultBorder;					//领域更新的大小
	int m_defaultUpdateProb;				//若某个像素判断为背景，则更新背景的概率
	//VIBE算法
	unsigned char ****m_pBGModel;			//背景模型
	int **m_ppLifeTime;						//用来记录一个像素被连续判为前景的次数

private:
	//获取随机数
	int GetRandom(int istart,int iend); // 默认istart=0,iend=15
    int GetRandom(int centre);          //返回以centre中心的领域范围里的一个随机数
    int GetRandom();                    //返回一个0到（m_defaultModelNum-1）之间的随机数
#endif
};

