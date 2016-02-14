/*************************************************************
FlyCapture2 to OpenCV Wrapper Class
FlyCapture2SDK�ŃL���v�`�������摜��OpenCV�ɓn�����߂̃N���X
�Q�l�F
����������߂��@http://13mzawa2.hateblo.jp/entry/2015/11/13/030939

*************************************************************/

#include "FlyCap2CVWrapper.h"

using namespace FlyCapture2;

FlyCap2CVWrapper::FlyCap2CVWrapper()
{
	// Connect the camera
	flycamError = flycam.Connect(0);
	if (flycamError != PGRERROR_OK)
	{
		std::cout << "Failed to connect to camera" << std::endl;
		return;
	}

	// Get the camera info and print it out
	flycamError = flycam.GetCameraInfo(&flycamInfo);
	if (flycamError != PGRERROR_OK)
	{
		std::cout << "Failed to get camera info from camera" << std::endl;
		return;
	}
	std::cout << flycamInfo.vendorName << " "
		<< flycamInfo.modelName << " "
		<< flycamInfo.serialNumber << std::endl;

	//	Set Video Property
	//	Video Mode: Custom(Format 7)
	//	Frame Rate: 120fps
	flycamError = flycam.SetVideoModeAndFrameRate(VIDEOMODE_FORMAT7, FRAMERATE_FORMAT7);
	Format7ImageSettings imgSettings;
	imgSettings.offsetX = 208;
	imgSettings.offsetY = 218;
	imgSettings.width = 800;
	imgSettings.height = 600;
	imgSettings.pixelFormat = PIXEL_FORMAT_422YUV8;
	flycamError = flycam.SetFormat7Configuration(&imgSettings, 100.0f);
	if (flycamError != PGRERROR_OK)
	{
		std::cout << "Failed to set video mode and frame rate" << std::endl;
		return;
	}
	//	Disable Auto changes
	autoFrameRate(false, 90.0f);
	autoWhiteBalance(false, 640, 640);
	autoExposure(false, -2.0f);
	autoSaturation(false, 100.0f);
	autoShutter(false, 6.8f);
	autoGain(false, 12.0f);

	flycamError = flycam.StartCapture();
	if (flycamError == PGRERROR_ISOCH_BANDWIDTH_EXCEEDED)
	{
		std::cout << "Bandwidth exceeded" << std::endl;
		return;
	}
	else if (flycamError != PGRERROR_OK)
	{
		std::cout << "Failed to start image capture" << std::endl;
		return;
	}
}

FlyCap2CVWrapper::~FlyCap2CVWrapper()
{
	flycamError = flycam.StopCapture();
	if (flycamError != PGRERROR_OK)
	{
		// This may fail when the camera was removed, so don't show 
		// an error message
	}
	flycam.Disconnect();
}

//	�����I�o�ݒ�
//	true -> auto, false -> manual
void FlyCap2CVWrapper::autoExposure(bool flag, float absValue = 1.585f)
{
	Property prop;
	prop.type = AUTO_EXPOSURE;
	prop.onOff = true;
	prop.autoManualMode = flag;
	prop.absControl = true;
	prop.absValue = absValue;
	flycamError = flycam.SetProperty(&prop);
	if (flycamError != PGRERROR_OK)
	{
		std::cout << "Failed to change Auto Exposure Settings" << std::endl;
	}
	return;
}

//	�����z���C�g�o�����X�ݒ�
void FlyCap2CVWrapper::autoWhiteBalance(bool flag, int red = 640, int blue = 640)
{
	Property prop;
	prop.type = WHITE_BALANCE;
	prop.onOff = true;
	prop.autoManualMode = flag;
	prop.valueA = red;
	prop.valueB = blue;
	flycamError = flycam.SetProperty(&prop);
	if (flycamError != PGRERROR_OK)
	{
		std::cout << "Failed to change Auto White Balance Settings" << std::endl;
	}
	return;
}

//	����Satulation�ݒ�
void FlyCap2CVWrapper::autoSaturation(bool flag, float percent = 50.0f)
{
	Property prop;
	prop.type = SATURATION;
	prop.onOff = true;
	prop.autoManualMode = flag;
	prop.absControl = true;
	prop.absValue = percent;
	flycamError = flycam.SetProperty(&prop);
	if (flycamError != PGRERROR_OK)
	{
		std::cout << "Failed to change Auto Satulation Settings" << std::endl;
	}
	return;
}

//	�����V���b�^�[���x�ݒ�
void FlyCap2CVWrapper::autoShutter(bool flag, float ms = 7.5f)
{
	Property prop;
	prop.type = SHUTTER;
	prop.autoManualMode = flag;
	prop.absControl = true;
	prop.absValue = ms;
	flycamError = flycam.SetProperty(&prop);
	if (flycamError != PGRERROR_OK)
	{
		std::cout << "Failed to change Auto Shutter Settings" << std::endl;
	}
	return;
}

//	�����Q�C���ݒ�
void FlyCap2CVWrapper::autoGain(bool flag, float gain = 0.0f)
{
	Property prop;
	prop.type = GAIN;
	prop.autoManualMode = flag;
	prop.absControl = true;
	prop.absValue = gain;
	flycamError = flycam.SetProperty(&prop);
	if (flycamError != PGRERROR_OK)
	{
		std::cout << "Failed to change Auto Gain Settings" << std::endl;
	}
	return;
}

//	�t���[�����[�g�ݒ�
void FlyCap2CVWrapper::autoFrameRate(bool flag, float fps = 85.0f)
{
	Property prop;
	prop.type = FRAME_RATE;
	prop.autoManualMode = flag;
	prop.absControl = true;
	prop.absValue = fps;
	flycamError = flycam.SetProperty(&prop);
	if (flycamError != PGRERROR_OK)
	{
		std::cout << "Failed to change Frame Rate Settings" << std::endl;
	}
	return;
}

//	cv::Mat�ւ̓]��
cv::Mat FlyCap2CVWrapper::readImage()
{
	// Get the image
	flycamError = flycam.RetrieveBuffer(&flyImg);
	if (flycamError != PGRERROR_OK)
	{
		std::cout << "capture error" << std::endl;
		return cvImg;
	}
	// convert to bgr
	flyImg.Convert(FlyCapture2::PIXEL_FORMAT_BGR, &bgrImg);
	// convert to OpenCV Mat
	unsigned int rowBytes = (unsigned int)((double)bgrImg.GetReceivedDataSize() / (double)bgrImg.GetRows());
	cvImg = cv::Mat(bgrImg.GetRows(), bgrImg.GetCols(), CV_8UC3, bgrImg.GetData(), rowBytes);

	return cvImg.clone();
}

bool FlyCap2CVWrapper::checkError()
{
	return flycamError != PGRERROR_OK;
}