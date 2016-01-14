#include "stdafx.h"
#include "tld_utils.h"
#include "opencv2/core/core.hpp"
using namespace cv;
using namespace std;

void drawBox(Mat& image, Rect box, Scalar color, int thick){
  rectangle( image, Point2d(box.x, box.y), Point2d(box.x+box.width,box.y+box.height),color, thick);
} 

void drawPoints(Mat& image, vector<Point2f> points,Scalar color){
  for( vector<Point2f>::const_iterator i = points.begin(), ie = points.end(); i != ie; ++i )
      {
      Point center( cvRound(i->x ), cvRound(i->y));
      circle(image,*i,2,color,1);
      }
}

Mat createMask(const Mat& image, Rect box){
  Mat mask = Mat::zeros(image.rows,image.cols,CV_8U);
  drawBox(mask,box,Scalar::all(255), CV_FILLED);
  return mask;
}

float median(vector<float> v)
{
    int n = int(floor(double(v.size() / 2)) + 0.5);
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

vector<int> index_shuffle(int begin,int end){
  vector<int> indexes(end-begin);
  for (int i=begin;i<end;i++){
    indexes[i]=i;
  }
  random_shuffle(indexes.begin(),indexes.end());
  return indexes;
}

