// stdafx.h : 标准系统包含文件的包含文件，
// 或是经常使用但不常更改的
// 特定于项目的包含文件
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>



// TODO: 在此处引用程序需要的其他头文件

#define USE_CUDA  // 是否使用CUDA加速
#define MEASURE_TIME //测量时间
#define SHOW_RES  //展示中间结果

#ifdef MEASURE_TIME
#include <ctime>
#endif