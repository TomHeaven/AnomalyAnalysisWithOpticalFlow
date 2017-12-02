
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/gpu/gpu.hpp>

using namespace cv;
using namespace cv::gpu; // for GpuMat and related functions
using namespace std;


class LKTracker
{
private:
  std::vector<cv::Point2f> pointsFB;
  cv::Size window_size;
  int level;
  std::vector<uchar> status;
  std::vector<uchar> FB_status;
  std::vector<float> similarity;
  std::vector<float> FB_error;
  float simmed;
  float fbmed;
  cv::TermCriteria term_criteria;
  float lambda;
  void normCrossCorrelation(const cv::Mat& img1,const cv::Mat& img2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
  bool filterPts(std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);

  //Cuda加速版本
#ifdef USE_CUDA
  GpuMat gImg1, gImg2, gPoints1, gPoints2, gStatus, gPointsFB, gFBStatus;
  PyrLKOpticalFlow* gFlow;
#endif

  void LKTracker::normCrossCorrelation(const GpuMat& img1, const GpuMat& img2, const GpuMat& gPoints1, const GpuMat& gPoints2, const vector<Point2f>& points1, const vector<Point2f> points2);
 // bool filterPts(GpuMat points1, GpuMat points2); // not implemented
public:
  LKTracker();
  ~LKTracker();
  bool trackf2f(const cv::Mat& img1, const cv::Mat& img2,
                std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
  float getFB(){return fbmed;}

  //Cuda加速版本
#ifdef USE_CUDA
  bool LKTracker::trackf2f(const GpuMat& img1, const GpuMat& img2, GpuMat &points1, GpuMat &points2);
#endif
};

