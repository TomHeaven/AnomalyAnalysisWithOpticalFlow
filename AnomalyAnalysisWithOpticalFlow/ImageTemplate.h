#ifndef IMAGETEMPLATE_H_
#define IMAGETEMPLATE_H_
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core_c.h>

template<class T> class Image
{
private:
	IplImage *imgp;
public:
	Image(IplImage *img=0){imgp=img;}
	~Image(){imgp=0;}
	void operator=(IplImage*img){imgp=img;}
	inline T*operator[](const int rowIndx){
		return((T*)(imgp->imageData+rowIndx*imgp->widthStep));
	}
};
typedef struct{
	unsigned char b,g,r;
}RgbPixel;
typedef struct{
	float b,g,r;
}RgbPixelFloat;

typedef struct{
	float Y,Cb,Cr;
}YCbCrPixelFloat;
typedef Image<RgbPixel>RgbImage;
typedef Image<RgbPixelFloat> RgbImageFloat;
typedef Image<unsigned char> BwImage;
typedef Image<float>BwImageFloat;
typedef Image<YCbCrPixelFloat>YCbCrImageFloat;
#endif /*IMAGETEMPLATE_H_*/