#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

#pragma region OPENCV3_LIBRARY_LINKER
#ifdef _DEBUG
#define CV_EXT "d.lib"
#else
#define CV_EXT ".lib"
#endif
#define CV_VER  CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#pragma comment(lib, "opencv_world" CV_VER CV_EXT)
#pragma endregion

#pragma region MACRO

//cv::Matのピクセル値拾得用マクロ
#define matB(IMG,X,Y)		((IMG).data[((IMG).step*(Y) + (IMG).channels()*(X)) + 0])
#define matG(IMG,X,Y)		((IMG).data[((IMG).step*(Y) + (IMG).channels()*(X)) + 1])
#define matR(IMG,X,Y)		((IMG).data[((IMG).step*(Y) + (IMG).channels()*(X)) + 2])
#define matGRAY(IMG,X,Y)	matB(IMG,X,Y)
#define matBf(IMG,X,Y)		(((Point3f*)((IMG).data + (IMG).step.p[0] * Y))[X].x)
#define matGf(IMG,X,Y)		(((Point3f*)((IMG).data + (IMG).step.p[0] * Y))[X].y)
#define matRf(IMG,X,Y)		(((Point3f*)((IMG).data + (IMG).step.p[0] * Y))[X].z)
#define matBd(IMG,X,Y)		(((Point3d*)((IMG).data + (IMG).step.p[0] * Y))[X].x)
#define matGd(IMG,X,Y)		(((Point3d*)((IMG).data + (IMG).step.p[0] * Y))[X].y)
#define matRd(IMG,X,Y)		(((Point3d*)((IMG).data + (IMG).step.p[0] * Y))[X].z)

#pragma endregion