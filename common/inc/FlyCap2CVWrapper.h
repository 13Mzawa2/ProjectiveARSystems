#pragma once

#ifdef _DEBUG
#define FC2_EXT ".lib"
#define CV_EXT "d.lib"
#else
#define FC2_EXT ".lib"
#define CV_EXT ".lib"
#endif
#pragma comment(lib, "FlyCapture2" FC2_EXT)
#pragma comment(lib, "FlyCapture2GUI" FC2_EXT)
#pragma comment(lib, "opencv_world300" CV_EXT)

#include <opencv2\opencv.hpp>
#include <FlyCapture2.h>
#include <FlyCapture2GUI.h>

class FlyCap2CVWrapper
{
protected:
	FlyCapture2::Camera flycam;
	FlyCapture2::CameraInfo flycamInfo;
	FlyCapture2::Error flycamError;
	FlyCapture2::Image flyImg, bgrImg;
	cv::Mat cvImg;

public:
	FlyCap2CVWrapper();
	~FlyCap2CVWrapper();
	cv::Mat readImage();
	//	Settings
	void autoExposure(bool flag, float absValue);
	void autoWhiteBalance(bool flag, int red, int blue);
	void autoSaturation(bool flag, float absValue);
	void autoShutter(bool flag, float ms);
	void autoGain(bool flag, float dB);
	void autoFrameRate(bool flag, float fps);
	bool checkError();
};

