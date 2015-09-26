#pragma once

#include <opencv2\opencv.hpp>
#include <Kinect.h>
#pragma comment(lib, "kinect20.lib")

template<class Interface>
inline void safeRelease(Interface *&interfacePointer)
{
	if (interfacePointer != NULL)
	{
		interfacePointer->Release();
		interfacePointer = NULL;
	}
}
inline void safeRelease(IKinectSensor *&kinectSensor)
{
	if (kinectSensor) kinectSensor->Close();
	if (kinectSensor != NULL)
	{
		kinectSensor->Release();
		kinectSensor = NULL;
	}
}

class Kinect2WithOpenCVWrapper
{
protected:
	//	OpenCV Buffer Size
	cv::Size colorBufferSize;
	cv::Size depthBufferSize;
	UINT colorBufferBlockSize;
	UINT depthBufferBlockSize;

public:
	//	KinectSDK2 Interface
	IKinectSensor *kinect;
	//	for color buffer
	IColorFrameSource *colorSource;
	IColorFrameReader *colorReader;
	IFrameDescription *colorDescription;
	IColorFrame *colorFrame;
	//	for depth buffer
	IDepthFrameSource *depthSource;
	IDepthFrameReader *depthReader;
	IFrameDescription *depthDescription;
	IDepthFrame *depthFrame;
	//	mapping engine
	ICoordinateMapper *mapper;
	//	OpenCV Image Buffer
	cv::Mat colorBuffer;		//	1920 x 1080, BGRA, 8UC4
	cv::Mat depthBuffer;		//	512 x 424, Gray, 16UC1
	cv::Mat xyzBuffer;			//	512 x 424, XYZ, 32FC3
	cv::Mat coordColorBuffer;	//	512 x 424, BGRA, 8UC4

	Kinect2WithOpenCVWrapper()
	{
		//	DefaultのKinectを取得
		if (FAILED(GetDefaultKinectSensor(&kinect)))
		{
			std::cerr << "Error: GetDefaultKinectSensor" << std::endl;
			exit(-1);
		}
		//	Kinectを開く
		if (FAILED(kinect->Open()))
		{
			std::cerr << "Error: IKinectSensor::Open" << std::endl;
			exit(-1);
		}
	}

	//------------------------
	//	Color Frame Open
	//------------------------
	inline void enableColorFrame()
	{
		//	KinectのColorフレームを読みだすための準備
		if (FAILED(kinect->get_ColorFrameSource(&colorSource)))
		{
			std::cerr << "Error: IKinectSensor::get_ColorFrameSource" << std::endl;
			exit(-1);
		}
		if (FAILED(colorSource->OpenReader(&colorReader)))
		{
			std::cerr << "Error: IColorFrameSource::OpenReader" << std::endl;
			exit(-1);
		}
		//	OpenCVのColorフレームの準備
		if (FAILED(colorSource->get_FrameDescription(&colorDescription)))
		{
			std::cerr << "Error: IColorFrameSource::get_FrameDescription" << std::endl;
			exit(-1);
		}
		//	OpenCV側のバッファ作成
		//	this.colorBufferを用意
		colorDescription->get_Width(&colorBufferSize.width);
		colorDescription->get_Height(&colorBufferSize.height);
		colorBuffer = cv::Mat(colorBufferSize, CV_8UC4);
		colorBufferBlockSize = colorBuffer.total() * colorBuffer.channels() * sizeof(uchar);
	}

	//--------------------------
	//	Depth Frame Open
	//--------------------------
	inline void enableDepthFrame()
	{
		//	KinectのDepthフレームを読みだすための準備
		if (FAILED(kinect->get_DepthFrameSource(&depthSource)))
		{
			std::cerr << "Error: IKinectSensor::get_DepthFrameSource" << std::endl;
			exit(-1);
		}
		if (FAILED(depthSource->OpenReader(&depthReader)))
		{
			std::cerr << "Error: IDepthFrameSource::OpenReader" << std::endl;
			exit(-1);
		}
		//	OpenCVのDepthフレームの準備
		if (FAILED(depthSource->get_FrameDescription(&depthDescription)))
		{
			std::cerr << "Error: IDepthFrameSource::get_FrameDescription" << std::endl;
			exit(-1);
		}
		//	OpenCV側のバッファ作成
		depthDescription->get_Width(&depthBufferSize.width);
		depthDescription->get_Height(&depthBufferSize.height);
		depthBuffer = cv::Mat(depthBufferSize, CV_16UC1);			//	Depthバッファは16bit
		depthBufferBlockSize = depthBuffer.total() * depthBuffer.channels() * sizeof(ushort);
	}

	//------------------------------
	//	Coordinate Mapper
	//------------------------------
	inline void enableCoordinateMapper()
	{
		if (FAILED(kinect->get_CoordinateMapper(&mapper)))
		{
			std::cerr << "Error: IKinectSensor::get_CoordinateMapper" << std::endl;
			exit(-1);
		}
	}

	//------------------------------------------------
	//	colorBufferに保存し，ミラーリングを直して渡す
	//------------------------------------------------
	inline void getColorFrame(cv::Mat &colorImg)
	{
		colorImg = cv::Mat(colorBufferSize, CV_8UC3);	//	渡す時に 3 channels に変更
		getColorFrame();
		cv::cvtColor(colorBuffer, colorImg, CV_BGRA2BGR);
		flip(colorImg, colorImg, 1);		//	左右反転
	}
	inline void getColorFrame(void)
	{
		colorFrame = nullptr;
		cv::Mat buffer(colorBufferSize, CV_8UC4);
		//	Colorフレーム取得
		if (SUCCEEDED(colorReader->AcquireLatestFrame(&colorFrame)))
		{	//	Colorフレームからデータを取得できたらOpenCV側バッファにコピー
			if (SUCCEEDED(colorFrame->CopyConvertedFrameDataToArray(
				colorBufferBlockSize,
				reinterpret_cast<BYTE*>(buffer.data),
				ColorImageFormat_Bgra)))
			{
				buffer.copyTo(colorBuffer);
			}
		}
	}

	//------------------------------------------------
	//	depthBufferに保存し，ミラーリングを直して渡す
	//------------------------------------------------
	inline void getDepthFrame(cv::Mat &depthImg)
	{
		depthImg = cv::Mat(depthBufferSize, CV_16UC1);
		getDepthFrame();
		flip(depthBuffer, depthImg, 1);		//	左右反転
	}
	inline void getDepthFrame(void)
	{
		depthFrame = nullptr;
		cv::Mat buffer(depthBufferSize, CV_16UC1);
		//	Depthフレーム取得
		if (SUCCEEDED(depthReader->AcquireLatestFrame(&depthFrame)))
		{	//	DepthフレームからデータをOpenCV側バッファにコピー
			if (SUCCEEDED(depthFrame->AccessUnderlyingBuffer(
				&depthBufferBlockSize,
				reinterpret_cast<UINT16**>(&buffer.data))))
			{
				buffer.copyTo(depthBuffer);
			}
		}
	}

	//------------------------------------------------
	//	mapperによってずれを修正したcolorImgを渡す
	//------------------------------------------------
	inline void getCoordinatedColorFrame(cv::Mat &coordColorImg)
	{
		coordColorImg = cv::Mat(depthBufferSize, CV_8UC3);
		getCoordinatedColorFrame();
		cv::cvtColor(coordColorBuffer, coordColorImg, CV_BGRA2BGR);
	}
	inline void getCoordinatedColorFrame(void)
	{
		cv::Mat buffer = cv::Mat(depthBufferSize, CV_8UC4);
		if (!colorBuffer.empty() && !depthBuffer.empty())
		{
			std::vector<ColorSpacePoint> colorSpacePoints(depthBuffer.total());
			if (SUCCEEDED(mapper->MapDepthFrameToColorSpace(
				depthBuffer.total(), reinterpret_cast<UINT16*>(depthBuffer.data),		//	取得したデプスバッファ
				depthBuffer.total(), &colorSpacePoints[0])))							//	depth -> color変換LUT
			{
				buffer = cv::Scalar(0);
#pragma omp parallel for
				for (int y = 0; y < depthBuffer.rows; y++)
				{
#pragma omp parallel for
					for (int x = 0; x < depthBuffer.cols; x++)
					{
						unsigned int idx = y * depthBuffer.cols + x;
						cv::Point colorLoc(
							static_cast<int>(floor(colorSpacePoints[idx].X + 0.5)),
							static_cast<int>(floor(colorSpacePoints[idx].Y + 0.5)));		//	四捨五入のため0.5を足す
						if (colorLoc.x >= 0 && colorLoc.x < colorBuffer.cols
							&& colorLoc.y >= 0 && colorLoc.y < colorBuffer.rows)
							buffer.at<cv::Vec4b>(idx) = colorBuffer.at<cv::Vec4b>(colorLoc);	//	LUTに従いバッファに保存
					}
				}
				
				buffer.copyTo(coordColorBuffer);
			}
		}
	}

	//------------------------------------------------
	//	DepthカメラからXYZ座標を復元
	//------------------------------------------------
	inline void getXYZFrame(cv::Mat &xyzMat)
	{
		xyzMat = cv::Mat(depthBufferSize, CV_32FC3);		//	(x, y, z) float精度
		getXYZFrame();
		xyzBuffer.copyTo(xyzMat);
	}
	inline void getXYZFrame(void)
	{
		cv::Mat buffer = cv::Mat(depthBuffer.size(), CV_32FC3);
		if (!depthBuffer.empty())
		{
			std::vector<CameraSpacePoint> xyzPoints(depthBuffer.total());
			if (SUCCEEDED(mapper->MapDepthFrameToCameraSpace(
				depthBuffer.total(), reinterpret_cast<UINT16*>(depthBuffer.data),
				depthBuffer.total(), &xyzPoints[0])))
			{
#pragma omp parallel for
				for (int y = 0; y < depthBuffer.rows; y++)
				{
					for (int x = 0; x < depthBuffer.cols; x++)
					{
						int idx = y * depthBuffer.cols + x;
						buffer.at<cv::Vec3f>(idx)[0] = xyzPoints[idx].X;
						buffer.at<cv::Vec3f>(idx)[1] = xyzPoints[idx].Y;
						buffer.at<cv::Vec3f>(idx)[2] = xyzPoints[idx].Z;
					}
				}
				buffer.copyTo(xyzBuffer);
			}
		}
	}

	virtual ~Kinect2WithOpenCVWrapper()
	{
		safeRelease(kinect);
	}

	//	フレームの解放（次フレームの待機）
	inline void releaseFrames()
	{
		safeRelease(depthFrame);
		safeRelease(colorFrame);
	}

	inline void releaseAllInterface()
	{
		//	全インターフェースの解放
		safeRelease(depthSource);
		safeRelease(depthReader);
		safeRelease(depthDescription);
		safeRelease(colorSource);
		safeRelease(colorReader);
		safeRelease(colorDescription);
		safeRelease(mapper);
		safeRelease(kinect);
	}
};

