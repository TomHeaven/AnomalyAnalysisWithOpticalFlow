#ifndef MATRIX_H_TOMHEAVEN_20140730
#define MATRIX_H_TOMHEAVEN_20140730

#define  _CRT_SECURE_NO_WARNINGS

/************************************************************************/
/** 定义结构 Matrix 用于计算                                            */
/************************************************************************/

typedef struct Matrix {
	int ** data;
	int width;
	int height;

	/** 构造函数  */
	Matrix() {
		width = height = 0;
		data = NULL;
	}
	/** 构造函数1 */
	Matrix(int w, int h) {
		width = w;
		height = h;

		data = new int*[h];
	    for(int i = 0; i < h; ++i) {
			data[i] = new int[w];
			for(int j = 0; j <w; ++j)
				data[i][j] = 0;
		}
	}

	/** 析构函数 */
	~Matrix() {
		release();
	}

	/** 释放内存 */
	void release() {
		for(int i = 0; i < height; ++i) {
			if (data != NULL && data[i] != NULL)
				delete(data[i]);
		}
		if (data != NULL)
			delete(data); 
	}

	/** 构造函数2 */
	Matrix(const Matrix& m) {
		width = m.width;
		height = m.height;

		data = new int*[height];
		for(int i = 0; i < height; ++i) {
			data[i] = new int[width];
			for(int j = 0; j <width; ++j)
				data[i][j] = m.data[i][j];
		}
	}

	/** 打印 */
	void printMatrix() {
		for(int i = 0; i < height; ++i) {
			for(int j = 0; j <width; ++j)
				printf("%d ", data[i][j]);
			printf("\n");

		}
	}

	/** 保存矩阵到文件
	   @param filepath 保存的文件路径
	   @return 一个bool值，表示是否成功。
	*/
	bool saveMatrix(const char* filepath) {
		FILE* fout = NULL;
		fopen_s(&fout, filepath, "w");
		if (fout == NULL)
			return false;
		fprintf(fout, "%d %d\n", height, width);
		for(int i = 0; i < height; ++i) {
			for(int j = 0; j <width; ++j)
				fprintf(fout, "%d ", data[i][j]);
			fprintf(fout, "\n");

		}
		fclose(fout);
		return true;
	}

	/** 从文件加载矩阵
	  @param filepath 加载的文件路径
	  @return 一个bool值，表示是否成功。
	*/
	bool loadMatrix(const char* filepath) {
		//释放已有内存
		for(int i = 0; i < height; ++i) {
			if (data != NULL && data[i] != NULL)
			    delete(data[i]);
		}
		if (data != NULL)
		    delete(data);
		

		FILE* fin = NULL;
		fopen_s(&fin, filepath, "r");
		if (fin == NULL)
			return false;
		fscanf_s(fin, "%d%d", &height, &width);

		//申请新内存并读取数据
		data = new int*[height];
		for(int i = 0; i < height; ++i) {
			data[i] = new int[width];
			for(int j = 0; j < width; ++j)
				fscanf_s(fin, "%d", &data[i][j]);
		}

		fclose(fin);
		return true;
	}

	

	/** 求两个同形矩阵对应元素之和，并赋值到自身 
	    @param m 运算符的第二个矩阵
	*/
	void add(const Matrix& m) {
		CV_Assert(height == m.height && width == m.width);
		for(int i = 0; i < height; ++i) {
			for(int j = 0; j <width; ++j) {
                 data[i][j] += m.data[i][j];
			}
		}
	}

	/** 求两个同形矩阵对应元素之差，并赋值到自身 
	    @param m 运算符的第二个矩阵
	*/
	void subtract(const Matrix& m) {
		CV_Assert(height != m.height || width != m.width);
		for(int i = 0; i < height; ++i) {
			for(int j = 0; j <width; ++j) {
                 data[i][j] -= m.data[i][j];
			}
		}
	}


	/** 求所有元素最大值
	 @return 最大值 int
	*/
	int max() {
		int maxValue = 0;
		for(int i = 0; i < height; ++i) {
			for(int j = 0; j <width; ++j) {
				if (data[i][j] > maxValue) {
					maxValue = data[i][j];
				}
			}
		}
		return maxValue;
	}

	/** 求两个同形矩阵对应元素最大值，并赋值到自身 */
	void max(const Matrix& m) {
		CV_Assert(height == m.height && width == m.width);
		for(int i = 0; i < width; ++i) {
			for(int j = 0; j < height; ++j) {
				data[j][i] = std::max(data[j][i], m.data[j][i]);
			}
		}
	}

	/** 归一化矩阵值域到 [0, 255] */
	void normalize() {
		int maxValue = max();
		for(int i = 0; i < height; ++i) {
			for(int j = 0; j <width; ++j) {
			    data[i][j] = (int)(float(data[i][j]) / maxValue * 255 + 0.5);
			}
		}
	}
	
	/** 转化为 IplImage 图像，注意用完之后要 cvRleaseImage(IlpImage**)，否则会造成内存泄漏。
	    @return 单通道IplImage*
	*/
	IplImage* toIplImage() {
		Matrix m(*this);
		m.normalize();
		IplImage* img = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
		for(int i = 0; i < height; ++i) {
			for(int j = 0; j <width; ++j) {
				setPixel(img, j, i, m.data[i][j]);
			//	if (m.data[i][j] == 0)
				//    printf("m.data[i][j] = %d\n", m.data[i][j]);
			}
		}
		return img;
	}
}  Matrix;

#endif