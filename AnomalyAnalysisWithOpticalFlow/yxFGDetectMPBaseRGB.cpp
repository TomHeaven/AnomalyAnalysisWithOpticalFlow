#include "stdafx.h"
#include "yxFGDetectMPBaseRGB.h"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/legacy/legacy.hpp"
//#include "opencv2/core/internal.hpp"
//#include "opencv2/video/tracking.hpp"
//#include "opencv2/video/background_segm.hpp"
//#include "opencv2/legacy/blobtrack.hpp"
//#include "opencv2/legacy/compat.hpp"

#include "math.h"
#include <iostream>
using namespace std;

#define  FG_VIBE_LIFE_SPAN             100        //生命周期
#define  FG_VIBE_MIN_MATCH				2        //最小匹配数
#define  FG_VIBE_RADIUS				   30        //匹配半径
#define  FG_VIBE_MODEL_NUM			   30        //模型中的采样数
#define	 FG_VIBE_UPDATE_PROB		   10        //背景更新的概率
#define  FG_VIBE_BORDER				   10        //边缘宽度，领域更新的半径
#define  FG_MP_HIST_BIN_NUM			   64        //直方图柱数
#define  FG_MP_TRAINNING_NUM		   200        //训练所需帧数
#define	 FG_MP_TRAINNING_INTERVAL       2        //训练间隔，暂时没用
#define  FG_MP_BG_LEVLES				1        //背景层级，暂时没用

//全局函数
void FindMax(unsigned char * arr , int n, unsigned char & maxval, unsigned char & idx)
{
	maxval = arr[0];
	idx = 0;
	for(int i=1; i< n; i++)
	{
		if(arr[i]>maxval)
		{
			maxval = arr[i];
			idx = i;
		}
	}
}
//bubble sort
void BubbleSort(int * arr, int n, int * index)
{
	int i,j, tmp1=0,tmp2=0;
	for(i=0; i< n; i++)
		index[i]=i;

	for (i=0; i< n; i++)
		for(j=n-1; j > i; j--){		
			if(arr[j] > arr[j-1]){
				tmp1 = arr[j-1];
				arr[j-1] = arr[j];
				arr[j] = tmp1;

				tmp2 = index[j-1];
				index[j-1] = index[j];
				index[j] = tmp2;
			}
		}
}

yxFGDetectMPBaseRGB::yxFGDetectMPBaseRGB()
{
	//MP算法参数赋初值
	m_defaultHistBinNum = FG_MP_HIST_BIN_NUM;
	m_defaultTrainningNum = FG_MP_TRAINNING_NUM;       
	m_defaultTrainningItv = FG_MP_TRAINNING_INTERVAL;  
	m_defaultBGLevels = FG_MP_BG_LEVLES;
	m_defaultRadius=FG_VIBE_RADIUS;
	m_pFGImg = NULL;
	m_pBGImg = NULL;

	m_ppppHist = NULL;
	m_frmNum=0;

#ifdef USING_VIBE
	//VIBE算法参数赋初值
	m_defaultLifetime=FG_VIBE_LIFE_SPAN;
	m_defaultMinMatch=FG_VIBE_MIN_MATCH;
	m_defaultModelNum=FG_VIBE_MODEL_NUM;
	m_defaultBorder=FG_VIBE_BORDER;
	m_defaultUpdateProb = FG_VIBE_UPDATE_PROB;
	m_ppLifeTime = NULL;
	m_pBGModel = NULL;
#endif
}

//初始化函数
void yxFGDetectMPBaseRGB::Init(IplImage *img)
{
	if(!img)
	{
		cout<<"the parameter referenced to NULL pointer!"<<endl;
		return;
	}
	//MP算法参数赋初值
	m_imgHeight=img->height;
	m_imgWidth=img->width;
	m_defaultHistBinNum = FG_MP_HIST_BIN_NUM;	
	m_defaultTrainningNum = FG_MP_TRAINNING_NUM;       
	m_defaultTrainningItv = FG_MP_TRAINNING_INTERVAL; 
	m_defaultBGLevels = FG_MP_BG_LEVLES;
	m_defaultRadius=FG_VIBE_RADIUS;
	m_defaultMinArea=200;
	m_frmNum=0;

	//创建前景图像
	if(m_pFGImg==NULL)	
	{
		m_pFGImg=cvCreateImage(cvSize(m_imgWidth,m_imgHeight),IPL_DEPTH_8U,1);
		cvZero(m_pFGImg);
	}
	//创建背景图像
	if(m_pBGImg== NULL)
	{
		m_pBGImg = cvCreateImage(cvSize(m_imgWidth,m_imgHeight),IPL_DEPTH_8U,3);
		cvZero(m_pBGImg);
	}

	//创建直方图容器
	int i,j,k,d;
	if(m_ppppHist == NULL)
	{
		m_ppppHist=new unsigned char ***[m_imgHeight];
		for(i=0;i<m_imgHeight;i++)
		{
			m_ppppHist[i]=new unsigned char **[m_imgWidth];
			for(j=0;j<m_imgWidth;j++)
			{
				m_ppppHist[i][j]=new unsigned char * [3];
				for(d=0; d< 3; d++)
				{
					m_ppppHist[i][j][d]=new unsigned char [m_defaultHistBinNum];
					for(k=0;k<m_defaultHistBinNum;k++)
					{
						m_ppppHist[i][j][d][k]=0;
					}
				}
			}
		}
	}

#ifdef USING_VIBE
	//VIBE算法参数赋初值
	m_defaultLifetime=FG_VIBE_LIFE_SPAN;
	m_defaultMinMatch=FG_VIBE_MIN_MATCH;
	m_defaultModelNum=FG_VIBE_MODEL_NUM;
	m_defaultBorder=FG_VIBE_BORDER;
	m_defaultUpdateProb = FG_VIBE_UPDATE_PROB;

	//创建并初始化前景点掩膜的生命长度
	if(m_ppLifeTime == NULL)
	{
		m_ppLifeTime=new int *[m_imgHeight];
		for(i=0;i<m_imgHeight;i++)
		{
			m_ppLifeTime[i]=new int [m_imgWidth];
			for(j=0;j<m_imgWidth;j++)
			{
				m_ppLifeTime[i][j]=0;
			}
		}
	}
	//创建背景模型容器
	if(m_pBGModel == NULL)
	{
		m_pBGModel=new unsigned char ***[m_imgHeight];
		for(i=0;i<m_imgHeight;i++)
		{
			m_pBGModel[i]=new unsigned char **[m_imgWidth];
			for(j=0;j<m_imgWidth;j++)
			{
				m_pBGModel[i][j]=new unsigned char *[3];
				for(d=0; d< 3; d++)
				{
					m_pBGModel[i][j][d]=new unsigned char [m_defaultModelNum];
					for(k=0;k<m_defaultModelNum;k++)
					{
						m_pBGModel[i][j][d][k]=0;
					}
				}
			}
		}
	}
#endif
}

//创建背景模型
bool yxFGDetectMPBaseRGB::ConstructBGModel(IplImage *img)
{
	if(m_frmNum  >= m_defaultTrainningNum)
		return true;
	//如果帧数小于训练的数量，则继续统计每个像素位置的直方图
	else
	{
		//对图像进行量化，注意量化图像必须是浮点的，否则会出问题
		double qfactor = 256/m_defaultHistBinNum;
		IplImage * qimg = cvCreateImage(cvGetSize(img),IPL_DEPTH_32F ,3);
		cvScale(img,qimg,1/qfactor,0);

		//统计每个像素位置的直方图（每来一帧图像就进行累加）
		int i,j,d,k;
		CvScalar cval;
		for(i=0;i<m_imgHeight;i++)
		{
			for(j=0;j<m_imgWidth;j++)
			{
				cval=cvGet2D(qimg,i,j);
				for(d=0; d< 3; d++)
				{
					int histidx = (int)cval.val[d];
					m_ppppHist[i][j][d][histidx]++;
				}
			}
		}
		cvReleaseImage(&qimg);

		//寻找直方图的前若干个最大值对应的颜色值，这些颜色值构成了背景图像
		CvScalar index;
		unsigned char *tmphist = new unsigned char[m_defaultHistBinNum];
		memset(tmphist,0,m_defaultHistBinNum*sizeof(unsigned char));
		for(i=0;i<m_imgHeight;i++)
		{
			for(j=0;j<m_imgWidth;j++)
			{
				for(d=0; d<3 ;d++)
				{
					for(k=0; k< m_defaultHistBinNum; k++)
					{
						tmphist[k] = m_ppppHist[i][j][d][k];
					}
					//寻找最大值
					unsigned char maxval,idx;
					FindMax(tmphist,m_defaultHistBinNum,maxval,idx);
					index.val[d]=qfactor*idx;
				}
				//将最大值索引设置为背景图像
				cvSet2D(m_pBGImg,i,j,index);
			}
		}
		delete [] tmphist;

		//训练即将结束时，从获取的背景图像中随机抽样，得到背景模型
		if(m_frmNum==(m_defaultTrainningNum-1))
		{
			//首先将直方图释放
			if(m_ppppHist != NULL)
			{
				for(i=0;i<m_imgHeight;i++)
				{
					for(j=0;j<m_imgWidth;j++)
					{
						for(d=0; d< 3; d++)
						{
							delete m_ppppHist[i][j][d];
							m_ppppHist[i][j][d] = NULL;
						}
						delete m_ppppHist[i][j];
						m_ppppHist[i][j] = NULL;
					}
					delete [] m_ppppHist[i];
					m_ppppHist[i] = NULL;
				}
				delete [] m_ppppHist;
				m_ppppHist = NULL;
			}

//如果采用VIBE算法
#ifdef USING_VIBE
			//将背景图像中每个像素的领域进行随机采样，构建背景模型
			CvScalar svalue;
			for(i=m_defaultBorder;i<m_imgHeight-m_defaultBorder;i++)
			{
				for(j=m_defaultBorder;j<m_imgWidth-m_defaultBorder;j++)
				{
					for(k=0; k< m_defaultModelNum; k++)
					{
						int Ri=GetRandom(i);
						int Rj=GetRandom(j);
						svalue=cvGet2D(m_pBGImg,Ri,Rj);
						m_pBGModel[i][j][0][k]=(unsigned char)(svalue.val[0] + 0.5);
						m_pBGModel[i][j][1][k]=(unsigned char)(svalue.val[1] + 0.5);
						m_pBGModel[i][j][2][k]=(unsigned char)(svalue.val[2] + 0.5);
					}
				}
			}
#endif
		}

		//帧数增加1
		m_frmNum++;

		//cvNamedWindow("BG");
		//cvShowImage("BG",m_pBGImg);
		return false;
	}
}
////创建背景模型
//bool yxFGDetectMPBaseRGB::ConstructBGModel(IplImage *img)
//{
//	if(m_frmNum  == 0)
//	{
//		cvCopy(img,m_pBGImg);
//	}
//	//如果帧数小于训练的数量，则继续统计每个像素位置的直方图
//	else
//	{
//			//如果采用VIBE算法
//#ifdef USING_VIBE
//			int i,j,d,k;
//			//将背景图像中每个像素的领域进行随机采样，构建背景模型
//			CvScalar svalue;
//			for(i=m_defaultBorder;i<m_imgHeight-m_defaultBorder;i++)
//			{
//				for(j=m_defaultBorder;j<m_imgWidth-m_defaultBorder;j++)
//				{
//					for(k=0; k< m_defaultModelNum; k++)
//					{
//						int Ri=GetRandom(i);
//						int Rj=GetRandom(j);
//						svalue=cvGet2D(m_pBGImg,Ri,Rj);
//						m_pBGModel[i][j][0][k]=svalue.val[0];
//						m_pBGModel[i][j][1][k]=svalue.val[1];
//						m_pBGModel[i][j][2][k]=svalue.val[2];
//					}
//				}
//			}
//#endif
//		}
//
//		//帧数增加1
//		m_frmNum++;
//
//		//cvNamedWindow("BG");
//		//cvShowImage("BG",m_pBGImg);
//		return false;
//}
//处理的主函数，供外部调用
void yxFGDetectMPBaseRGB::Process(IplImage *img)
{
	//第一帧进行初始化
	if(m_frmNum ==0  || m_pFGImg==NULL)
	{
#ifdef USING_VIBE
		if(m_ppLifeTime==NULL || m_pBGModel==NULL)
#endif
		Init(img);
	}
	//前若干帧仅进行背景建模，不进行检测
	if(m_frmNum<m_defaultTrainningNum)
	{
		ConstructBGModel(img);
	}
	//前景检测
	else
	{
		FGDetect(img);
	}
}

void yxFGDetectMPBaseRGB::FGDetect(IplImage *img)
{

	//对图像进行量化
	//double qfactor = 256/m_defaultHistBinNum;
	//IplImage * qimg = cvCreateImage(cvGetSize(img),8,1);
	//cvScale(grayimg,qimg,1/qfactor,0);
	//cvScale(qimg,qimg,qfactor,0);

	//可以不对图像进行量化，为了保持以下用的图像还是qimg，进行如下操作
	IplImage * qimg = cvCreateImage(cvGetSize(img),8,3);
	cvScale(img,qimg,1,0);

	//初始化前景图像
	cvZero(m_pFGImg);

	//如果仅仅采用MP算法，而不采用VIBE算法
#ifndef USING_VIBE
	//定义临时变量
	double dist0,dist1,dist2,dist;
	int i,j;
	//开始遍历每个像素
	CvScalar bgval,frmval;
	for(i=0;i<m_imgHeight;i++)
	{
		for(j=0;j<m_imgWidth;j++)
		{
			frmval=cvGet2D(qimg,i,j);
			bgval = cvGet2D(m_pBGImg,i,j);
			dist0 = frmval.val[0] - bgval.val[0];
			dist1 = frmval.val[1] - bgval.val[1];
			dist2 = frmval.val[2] - bgval.val[2];
			dist = sqrt(dist0*dist0+dist1*dist1+dist2*dist2);
			if(dist>m_defaultRadius*2)
			{
				//像素值设置为255
				cvSetReal2D(m_pFGImg,i,j,255);
			}
		}
	}
#endif

	//如果采用VIBE算法
#ifdef USING_VIBE
	//定力临时变量
	int i,j,k;
	int matchCnt=0;//距离比较在阈值内的次数
	int iR1,iR2; //产生随机数
	int Ri,Rj;   //产生邻域内X和Y的随机数
	double dist0,dist1,dist2,dist;

	//开始遍历每个像素
	CvScalar svalue;
	for(i=m_defaultBorder;i<m_imgHeight-m_defaultBorder;i++)
	{
		for(j=m_defaultBorder;j<m_imgWidth-m_defaultBorder;j++)
		{
			matchCnt=0;
			svalue=cvGet2D(qimg,i,j);
			for(k=0;k<m_defaultModelNum && matchCnt<m_defaultMinMatch;k++)
			{
				dist0 = svalue.val[0] - m_pBGModel[i][j][0][k];
				dist1 = svalue.val[1] - m_pBGModel[i][j][1][k];
				dist2 = svalue.val[2] - m_pBGModel[i][j][2][k];
				dist = sqrt(dist0*dist0+dist1*dist1+dist2*dist2);
				if(dist<m_defaultRadius)
				{
					matchCnt++;
				}
			}

			//若落在匹配圆内的次数大于m_defaultMinMatch，则认为匹配上，即为背景
			if(matchCnt>=m_defaultMinMatch)
			{
				//背景像素设置为0
				cvSetReal2D(m_pFGImg,i,j,0);
				//生命周期置0
				m_ppLifeTime[i][j]=0;

				//更新背景模型
				iR1=GetRandom(0,m_defaultUpdateProb);
				if(iR1==0)
				{
					iR2=GetRandom();
					m_pBGModel[i][j][0][iR2]=(unsigned char)(svalue.val[0] + 0.5);
					m_pBGModel[i][j][1][iR2]=(unsigned char)(svalue.val[1] + 0.5);
					m_pBGModel[i][j][2][iR2]=(unsigned char)(svalue.val[2] + 0.5);
				}
				//进一步更新邻域模型
				iR1=GetRandom(0,m_defaultUpdateProb);
				if(iR1==0)
				{
					Ri=GetRandom(i);
					Rj=GetRandom(j);
					iR2=GetRandom();
					m_pBGModel[Ri][Rj][0][iR2]=(unsigned char)(svalue.val[0] + 0.5);
					m_pBGModel[Ri][Rj][1][iR2]=(unsigned char)(svalue.val[1] + 0.5);
					m_pBGModel[Ri][Rj][2][iR2]=(unsigned char)(svalue.val[2] + 0.5);
				}
			}
			//否则，设置为前景
			else               
			{
				//像素值设置为255
				cvSetReal2D(m_pFGImg,i,j,255);

				//生命周期加1
				m_ppLifeTime[i][j]=m_ppLifeTime[i][j]+1;

				//如果生命周期的值超过阈值，说明可能存在将背景误认为前景的情况，必须强制更新背景模型若干次
				if(m_ppLifeTime[i][j]>m_defaultLifetime) 
				{
					//生命周期
					m_ppLifeTime[i][j]=0;

					// 并且更新背景模型两次
					iR1=GetRandom();
					m_pBGModel[i][j][0][iR1]=(unsigned char)(svalue.val[0] + 0.5);
					m_pBGModel[i][j][1][iR1]=(unsigned char)(svalue.val[1] + 0.5);
					m_pBGModel[i][j][2][iR1]=(unsigned char)(svalue.val[2] + 0.5);
					iR2=GetRandom();
					m_pBGModel[i][j][0][iR2]=(unsigned char)(svalue.val[0] + 0.5);
					m_pBGModel[i][j][1][iR2]=(unsigned char)(svalue.val[1] + 0.5);
					m_pBGModel[i][j][2][iR2]=(unsigned char)(svalue.val[2] + 0.5);
				}
			}
		}
	}
#endif

	//释放图像
	cvReleaseImage(&qimg);

	//删除小面积目标
	DeleteSmallArea();

	//形态学处理
	//IplImage * tmpimg = cvCreateImage(cvGetSize(img),8,1);
	//IplConvKernel* kernel = cvCreateStructuringElementEx(5,5,2,2,CV_SHAPE_ELLIPSE);
	//cvMorphologyEx(m_pFGImg,m_pFGImg,tmpimg,kernel,CV_MOP_CLOSE ,1);
	//cvReleaseStructuringElement(&kernel);
	//cvReleaseImage(&tmpimg);
	//cvDilate(m_pFGImg,m_pFGImg);
	//cvErode(m_pFGImg,m_pFGImg);
	//cvDilate(m_pFGImg,m_pFGImg);
	//cvErode(m_pFGImg,m_pFGImg);
	//cvDilate(m_pFGImg,m_pFGImg);
	//cvErode(m_pFGImg,m_pFGImg);
}
void yxFGDetectMPBaseRGB::FilterFGbyHSV(IplImage * srcimg,double sthr, double vthr)
{
	assert(srcimg->nChannels==3);
	if(srcimg->width != m_pFGImg->width || srcimg->height != m_pFGImg->height)
		return;
	IplImage * hsv = cvCreateImage(cvGetSize(srcimg),8,3);
	cvCvtColor(srcimg,hsv,CV_BGR2HSV);
	IplImage * simg = cvCreateImage(cvGetSize(srcimg),8,1);
	IplImage * vimg = cvCreateImage(cvGetSize(srcimg),8,1);
	cvSplit(hsv,NULL,simg,vimg,NULL);
	for(int i=0; i< srcimg->height; i++)
		for(int j=0; j < srcimg->width; j++)
		{
			uchar sval = CV_IMAGE_ELEM(simg,uchar,i,j);
			uchar vval = CV_IMAGE_ELEM(vimg,uchar,i,j);
			if(sval < (uchar)(sthr*255) && vval > (uchar)(vthr*255))
				CV_IMAGE_ELEM(m_pFGImg,uchar,i,j)=0;
		}
	cvReleaseImage(&hsv);
	cvReleaseImage(&simg);
	cvReleaseImage(&vimg);
}
void yxFGDetectMPBaseRGB::DeleteSmallArea()
{
	int region_count=0;
	CvSeq *first_seq=NULL, *prev_seq=NULL, *seq=NULL;
	CvMemStorage *storage=cvCreateMemStorage();
	cvClearMemStorage(storage);
    cvFindContours(m_pFGImg,storage,&first_seq,sizeof(CvContour),CV_RETR_EXTERNAL );
	for(seq=first_seq;seq;seq=seq->h_next)
	{
		CvContour *cnt=(CvContour*)seq;
		double area = cvContourArea(cnt);

		if(/*cnt->rect.height*cnt->rect.width*/area<m_defaultMinArea)
		{
			prev_seq=seq->h_prev;
			if(prev_seq)
			{
				prev_seq->h_next=seq->h_next;
				if(seq->h_next) 
					seq->h_next->h_prev=prev_seq;
			}
			else
			{
				first_seq=seq->h_next;
				if(seq->h_next)
					seq->h_next->h_prev=NULL;
			}
		}
		else
		{
			region_count++;
		}
	}

	cvZero(m_pFGImg);
	cvDrawContours(m_pFGImg,first_seq,CV_RGB(0,0,255),CV_RGB(0,0,255),10,-1);
	cvReleaseMemStorage(&storage);
}
#ifdef USING_VIBE
int yxFGDetectMPBaseRGB::GetRandom() //返回一个0到（m_defaultModelNum-1）之间的随机数
{
	int val=int(m_defaultModelNum*1.0*rand()/RAND_MAX + 0.5);
	if(val==m_defaultModelNum)
		return val-1;
	else
		return val;
}

int yxFGDetectMPBaseRGB::GetRandom(int centre)//返回以centre中心的领域范围里的一个随机数
{
	int val=int(centre-m_defaultBorder+rand()%(2*m_defaultBorder) + 0.5);
	if(val<centre-m_defaultBorder)
	{
		val=centre-m_defaultBorder;
	}
	if(val>centre+m_defaultBorder)
	{
		val=centre+m_defaultBorder;
	}
	return val;
}

int yxFGDetectMPBaseRGB::GetRandom(int istart, int iend)
{
	int val=istart+rand()%(iend-istart);
	return val;
}
#endif

IplImage* yxFGDetectMPBaseRGB::GetFGImg()
{
	return m_pFGImg;
}

IplImage* yxFGDetectMPBaseRGB::GetBGImg()
{
	return m_pBGImg;
}

//析构函数
yxFGDetectMPBaseRGB::~yxFGDetectMPBaseRGB()
{
	if(m_pFGImg != NULL)
	{
		cvReleaseImage(&m_pFGImg);
		m_pFGImg = NULL;
	}
	if(m_pBGImg!= NULL)
	{
		cvReleaseImage(&m_pBGImg);
		m_pBGImg= NULL;
	}
	int i,j,d;
	if(m_ppppHist != NULL)
	{
		for(i=0;i<m_imgHeight;i++)
		{
			for(j=0;j<m_imgWidth;j++)
			{
				for(d=0; d< 3; d++)
				{
					delete [] m_ppppHist[i][j][d];
					m_ppppHist[i][j][d] = NULL;
				}
				delete [] m_ppppHist[i][j];
				m_ppppHist[i][j] = NULL;
			}
			delete []m_ppppHist[i];
			m_ppppHist[i] = NULL;
		}
		delete [] m_ppppHist;
		m_ppppHist = NULL;
	}

#ifdef  USING_VIBE
	if (m_ppLifeTime!= NULL)
	{
		for(i=0;i<m_imgHeight;i++)
		{
			delete []m_ppLifeTime[i];
			m_ppLifeTime[i] = NULL;
		}
		delete []m_ppLifeTime;
		m_ppLifeTime = NULL;
	}
	if(m_pBGModel != NULL)
	{
		for(i=0;i<m_imgHeight;i++)
		{
			for(j=0;j<m_imgWidth;j++)
			{
				for(d=0; d< 3; d++)
				{
					delete [] m_pBGModel[i][j][d];
					m_pBGModel[i][j][d] = NULL;
				}
				delete [] m_pBGModel[i][j];
				m_pBGModel[i][j] = NULL;
			}
			delete []m_pBGModel[i];
			m_pBGModel[i] = NULL;
		}
		delete [] m_pBGModel;
		m_pBGModel = NULL;
	}
#endif
}
