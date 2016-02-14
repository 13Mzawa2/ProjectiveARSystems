﻿
#pragma region Disable Warning C4996
//
// Disable Warning C4996
//
#ifndef _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES
#define _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES 1
#endif
#ifndef _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES
#define _CRT_SECURE_CPP_OVERLOAD_SECURE_NAMES 1
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#pragma endregion
//===========================================
//	Includes
//===========================================
//	Library Linker Scripts
#include <iostream>
#include "OpenGLHeader.h"

#pragma region OPENCV3_LIBRARY_LINKER
#include <opencv2/opencv.hpp>
#ifdef _DEBUG
#define CV_EXT "d.lib"
#else
#define CV_EXT ".lib"
#endif
#define CV_VER  CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#pragma comment(lib, "opencv_world" CV_VER CV_EXT)
#pragma endregion

#pragma region ARTOOLKIT5_LIBRARY_LINKER
#include <AR/ar.h>
#include <AR/arMulti.h>
#ifdef _DEBUG
#define AR_EXT "d.lib"
#else
#define AR_EXT ".lib"
#endif
#pragma comment(lib, "AR" AR_EXT)
#pragma comment(lib, "ARICP" AR_EXT)
#pragma comment(lib, "ARMulti" AR_EXT)
#pragma endregion

//	Original Libraries
#include "FlyCap2CVWrapper.h"
#include "OBJRenderingEngine.h"
#include "GLImage.h"

using namespace cv;
using namespace std;

//===========================================
//	import file path
//===========================================
//	Shader file
const char vertexDir[] = "./shader/vertex.glsl";
const char fragmentDir[] = "./shader/fragment.glsl";
//	.obj Wavefront Object
const char objDir[] = "../common/data/model/drop/drop_modified_x004.obj";
const char textureDir[] = "../common/data/model/drop/textures/txt_001_diff_cmyk_gamma.png";
const char cubeMarkerObjDir[] = "../common/data/model/MultiMarker/MultiMarker.obj";
//	Look-Up Table File Path
const char *lutDir[5] = {
	"../common/data/lut/LUT_dichromat_typeP.png",
	"../common/data/lut/LUT_dichromat_typeD.png",
	"../common/data/lut/LUT_dichromat_typeT.png",
	"../common/data/lut/LUT_elder_70.png",
	"../common/data/lut/LUT_elder_80.png"
};
const char *textureSetDir[9] = {
	"../common/data/model/drop/textures/txt_001_diff.bmp",
	"../common/data/model/drop/textures/劣化/txt_001_diff_劣化15.bmp",
	"../common/data/model/drop/textures/劣化/txt_001_diff_劣化30.bmp",
	"../common/data/model/drop/textures/劣化/txt_001_diff_劣化45.bmp",
	"../common/data/model/drop/textures/劣化/txt_001_diff_劣化60.bmp",
	"../common/data/model/drop/textures/劣化/txt_001_diff_劣化75.bmp",
	"../common/data/model/drop/textures/劣化/txt_001_diff_劣化90.bmp",
	"../common/data/model/drop/textures/劣化/txt_001_diff_劣化105.bmp",
	"../common/data/model/drop/textures/劣化/txt_001_diff_劣化120.bmp"
};
//	Camera Calibration File
//	OpenCVのキャリブレーションデータを用いる
const char calibDir[] = "./data/calibdata.xml";
const char dummyCalibDir[] = "./data/LogitechPro4000.dat";
//	MultiMarker Setting File
const char markerConfigDir[] = "./data/CubeMarker/cubemarker_artk5.dat";

//===========================================
//	Camera Variables
//===========================================
//	Video Module
FlyCap2CVWrapper flycap;				//	FlyCapture2のラッパー
Mat colorImg;							//	flycapから渡される画像
//	Param
Mat cameraMatrix, distCoeffs;			//	カメラ内部パラメータ
Mat cameraMatrixProj, distCoeffsProj;	//	プロジェクタ内部パラメータ
Mat RProCam, TProCam;					//	外部パラメータ
Size cameraSize, projSize;				//	カメラ・プロジェクタの解像度
//	Dest
glm::mat4 glmProjMat, glmProjMatProj;	//	フラスタム行列
glm::mat4 glmTransProCam;				//	プロジェクタ位置姿勢
Mat mapC1, mapC2;						//	カメラの歪み補正マップ
Mat mapP1, mapP2;						//	プロジェクタの歪み補正マップ

//===========================================
//	ARToolKit Variables
//===========================================
ARParamLT *cparamLT;
ARHandle *arhandle;
AR3DHandle *ar3dhandle;
ARPattHandle *pattHandle;
ARMultiMarkerInfoT *multiConfig;
static double borderSize = 0.125;

//-----------------------------------------------------
//	GLFW User Interface
//-----------------------------------------------------
#define PROJ_WIN_ID 2

GLFWwindow	*mainWindow, *subWindow;		//	マルチウィンドウ
int subWinW, subWinH;
static float objTx = 0, objTy = 0, objTz = 0;
static glm::vec3 projT(0.0, 0.0, 0.0);	//	手動位置補正：並進
static glm::vec3 projR(0.0, 0.0, 0.0);	//	手動位置補正：回転(Rx, Ry, Rz)
static glm::quat current = glm::quat(1.0, 0.0, 0.0, 0.0);
static float objAngle = 0.0f;
//	履歴データ 0が最も新しい 過去2フレームを使用
glm::mat4 prePose1, prePose2;	//	過去2フレーム
static float weightV = 0.7, weightA = 0.2;			//	mean = mix(p0, mix(p1, p2, weightA), weightV) 
static double threshR = 5.0e-4, threshT = 0.6;		//	物体が止まっていると認識する閾値
//static float weightV = 0.0, weightA = 0.0;			//	mean = mix(p0, mix(p1, p2, weightA), weightV) 
//static double threshR = 0, threshT = 0;		//	物体が止まっていると認識する閾値

double xBegin, yBegin;
int pressedMouseButton = 0;
bool visible = false;

//-----------------------------------------------------
//	OpenGL / GLSL Rendering Engine
//-----------------------------------------------------
OBJRenderingEngine mainRenderer, subRenderer;
OBJRenderingEngine markerLight;
cv::Mat visionLUT;
glm::mat4 Model, View, Projection;
glm::vec3 colorWhite(1.0, 1.0, 1.0);	//	現在の白色点
glm::vec3 backupWhite(1.0, 1.0, 1.0);	//	LUT変換時の白色点

//-----------------------------------------------------
//	Prototypes
//-----------------------------------------------------
int initWindow(void);
void initMainWindow(void);
void initSubWindow(void);
void initCamera(void);
void initARTK(void);
void mouseEvent(GLFWwindow *window, int button, int state, int optionkey);
void cursorPosEvent(GLFWwindow *window, double x, double y);
void scrollEvent(GLFWwindow *window, double xofset, double yofset);
void safeTerminate(void);
void showMatrix(glm::mat4 &m);
void lookup(Mat &src, Mat &dst, Mat &LUT);
glm::vec3 lookupColor(glm::vec3 &src, Mat LUT);

int initWindow(void)
{
	//	GLFWの初期化
	if (glfwInit() != GL_TRUE)
	{
		cerr << "GLFWの初期化に失敗しました．\n";
		return EXIT_FAILURE;
	}
	//	Window設定
	glfwWindowHint(GLFW_SAMPLES, 4);								//	4x アンチエイリアス
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);						//	リサイズ不可
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);					//	OpenGLバージョン3.3を利用
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);					//	
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);	//	古いOpenGLを使わない

	// Main Windowの用意(カメラ画像サイズ)
	mainWindow = glfwCreateWindow(cameraSize.width, cameraSize.height, "Main Window", NULL, NULL);
	if (mainWindow == NULL){
		cerr << "GLFWウィンドウの生成に失敗しました. Intel GPUを使用している場合は, OpenGL 3.3と相性が良くないため，2.1を試してください．\n";
		glfwTerminate();
		return EXIT_FAILURE;
	}

	// Sub Windowの用意(全画面)
	int monitorCount;
	GLFWmonitor **monitors = glfwGetMonitors(&monitorCount);
	subWindow = glfwCreateWindow(projSize.width, projSize.height, "Sub Window", monitors[PROJ_WIN_ID], NULL);
	if (subWindow == NULL){
		cerr << "GLFWウィンドウの生成に失敗しました. Intel GPUを使用している場合は, OpenGL 3.3と相性が良くないため，2.1を試してください．\n";
		glfwTerminate();
		return EXIT_FAILURE;
	}
	glfwGetWindowSize(subWindow ,&subWinW, &subWinH);
	glfwMakeContextCurrent(mainWindow);

	// Initialize GLEW
	glewExperimental = true;	// Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "GLEWの初期化に失敗しました．\n");
		return EXIT_FAILURE;
	}
	return 0;
}

void initMainWindow(void)
{
	//	Main Window Setting
	glfwMakeContextCurrent(mainWindow);				//	main windowをカレントにする
	glfwSwapInterval(0);				//	SwapBufferのインターバル
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LESS);				//	カメラに近い面だけレンダリングする

	//-----------------------------
	//	Launch Rendering Engine
	//-----------------------------
	mainRenderer.setVisionLUT(visionLUT);					//	LUTを読み込む
	loadOBJ(objDir, mainRenderer.obj);						//	.objファイルを読み込み
	mainRenderer.shader.initGLSL(vertexDir, fragmentDir);	//	プログラマブルシェーダをロード
	mainRenderer.texImg = imread(textureDir);				//	テクスチャ画像を読み込む
	flip(mainRenderer.texImg, mainRenderer.texImg, 1);
	mainRenderer.init();
}

void initSubWindow(void)
{
	//	Sub Window Setting
	glfwMakeContextCurrent(subWindow);				//	sub windowをカレントにする
	glfwSwapInterval(0);				//	SwapBufferのインターバル
	glClearColor(colorWhite[0],colorWhite[1],colorWhite[2], 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LESS);				//	カメラに近い面だけレンダリングする

	//-----------------------------
	//	Launch Rendering Engine
	//-----------------------------
	subRenderer.setVisionLUT(visionLUT);		//	LUTを読み込む
	loadOBJ(objDir, subRenderer.obj);						//	.objファイルを読み込み
	subRenderer.shader.initGLSL(vertexDir, fragmentDir);	//	プログラマブルシェーダをロード
	subRenderer.texImg = mainRenderer.texImg.clone();				//	テクスチャ画像を読み込む
	subRenderer.init();

	//	立方体マーカーに当てる白い光源
	markerLight.setVisionLUT(visionLUT);
	loadOBJ(cubeMarkerObjDir, markerLight.obj);
	markerLight.shader.initGLSL(vertexDir, fragmentDir);
	markerLight.init();
	markerLight.objectColor = colorWhite;
	markerLight.useLight = false;

	//	読み込んでいるLUTの白色点を記憶
	backupWhite = lookupColor(glm::vec3(1.0, 1.0, 1.0), visionLUT);

	//	キー入力を受け付けるようにする
	glfwSetInputMode(subWindow, GLFW_STICKY_KEYS, GL_TRUE);
	//	マウス操作を可能にする
	glfwSetMouseButtonCallback(subWindow, mouseEvent);
	glfwSetCursorPosCallback(subWindow, cursorPosEvent);
	glfwSetScrollCallback(subWindow, scrollEvent);
}

void initCamera(void)
{
	//	カメラパラメータのロード
	FileStorage fs(calibDir, FileStorage::READ);
	FileNode Camera = fs["Camera"];
	Camera["size"] >> cameraSize;
	Camera["CameraMatrix"] >> cameraMatrix;
	Camera["DistCoeffs"] >> distCoeffs;

	FileNode Projector = fs["Projector"];
	Projector["size"] >> projSize;
	Projector["CameraMatrix"] >> cameraMatrixProj;
	Projector["DistCoeffs"] >> distCoeffsProj;

	FileNode ProCam = fs["ProCam"];
	ProCam["R"] >> RProCam;
	ProCam["T"] >> TProCam;

	//cameraMatrixProj.at<double>(0, 2) = projSize.width - cameraMatrixProj.at<double>(0,2);		//	1-2x/w = -1+2c_x/w

	//	カメラパラメータから行列作成
	glmProjMat = cvtCVCameraParam2GLProjection(cameraMatrix, cameraSize, 0.1, 5000.0);
	glmProjMatProj = cvtCVCameraParam2GLProjection(cameraMatrixProj, projSize, 0.1, 5000.0);
	glmTransProCam = composeRT(RProCam, TProCam);
	//	Undistort Map (Camera)
	initUndistortRectifyMap(
		cameraMatrix, distCoeffs,
		Mat(), cameraMatrix, cameraSize, CV_32FC1,
		mapC1, mapC2);
	//	Undistort Map (Projector)
	initUndistortRectifyMap(
		cameraMatrixProj, distCoeffsProj,
		Mat(), cameraMatrixProj, projSize, CV_32FC1,
		mapP1, mapP2);

}

void initARTK(void)
{
	//	Set camera parameters and make ARToolKit handles
	ARParam cparam;
	arParamLoad(dummyCalibDir, 1, &cparam);
	arParamChangeSize(&cparam, cameraSize.width, cameraSize.height, &cparam);
	cparam.xsize = cameraSize.width;
	cparam.ysize = cameraSize.height;
	//	OpenCV cameraMatrix を代入
	cparam.mat[0][0] = cameraMatrix.at<double>(0, 0);
	cparam.mat[1][1] = cameraMatrix.at<double>(1, 1);
	cparam.mat[0][2] = cameraMatrix.at<double>(0, 2);
	cparam.mat[1][2] = cameraMatrix.at<double>(1, 2);
	cparam.mat[2][2] = 1.0;
	//	ARToolKitでは歪み0のレンズとする（OpenCVに任せる）
	//	詳細はARToolKit5のarParamObserv2Ideal()関数の実装を参照
	cparam.dist_factor[0] = 0.0;
	cparam.dist_factor[1] = 0.0;
	cparam.dist_factor[2] = 0.0;
	cparam.dist_factor[3] = 0.0;
	//cparam.dist_factor[0] = distCoeffs.at<double>(0);	//	k1
	//cparam.dist_factor[1] = distCoeffs.at<double>(1);	//	k2
	//cparam.dist_factor[2] = distCoeffs.at<double>(2);	//	p1
	//cparam.dist_factor[3] = distCoeffs.at<double>(3);	//	p2
	cparam.dist_factor[4] = cameraMatrix.at<double>(0,0);	//	fx
	cparam.dist_factor[5] = cameraMatrix.at<double>(1,1);	//	fy
	cparam.dist_factor[6] = cameraMatrix.at<double>(0,2);	//	x0
	cparam.dist_factor[7] = cameraMatrix.at<double>(1,2);	//	y0
	cparam.dist_factor[8] = 1.0;	//	s
	cparam.dist_function_version = 4;		//	OpenCV lens model

	cparamLT = arParamLTCreate(&cparam, AR_PARAM_LT_DEFAULT_OFFSET);
	arhandle = arCreateHandle(cparamLT);
	arSetPixelFormat(arhandle, AR_PIXEL_FORMAT_BGR);
	arSetDebugMode(arhandle, AR_DEBUG_DISABLE);
	arSetLabelingThreshMode(arhandle, AR_LABELING_THRESH_MODE_AUTO_OTSU);
	ar3dhandle = ar3DCreateHandle(&cparam);
	//	Setup Cube Marker
	pattHandle = arPattCreateHandle();
	multiConfig = arMultiReadConfigFile(markerConfigDir, pattHandle);
	if (multiConfig->patt_type == AR_MULTI_PATTERN_DETECTION_MODE_TEMPLATE) {
		arSetPatternDetectionMode(arhandle, AR_TEMPLATE_MATCHING_COLOR);
	}
	else if (multiConfig->patt_type == AR_MULTI_PATTERN_DETECTION_MODE_MATRIX) {
		arSetPatternDetectionMode(arhandle, AR_MATRIX_CODE_DETECTION);
	}
	else { // AR_MULTI_PATTERN_DETECTION_MODE_TEMPLATE_AND_MATRIX
		arSetPatternDetectionMode(arhandle, AR_TEMPLATE_MATCHING_COLOR_AND_MATRIX);
	}
	arPattAttach(arhandle, pattHandle);
	arSetBorderSize(arhandle, borderSize);
}

int main(void)
{
	initCamera();

	visionLUT = imread(lutDir[0]);
	if (visionLUT.empty())
	{
		cerr << "LUTの読み込みに失敗しました．ディレクトリを確認してください．\n"
			<< "ディレクトリの場所：" << lutDir << endl;
		glfwTerminate();
		return EXIT_FAILURE;
	}

	if(initWindow() == EXIT_FAILURE) exit(-1);

	cout << "\nMain Windowの設定" << endl;
	initMainWindow();
	cout << "\nSub Windowの設定" << endl;
	initSubWindow();
	cout << "\n画像描画の設定" << endl;
	GLImage glImg(mainWindow);
	//GLImage glImgProj(subWindow);

	initARTK();


	//--------------------
	//	タイマー設定
	//--------------------
	double currentTime = 0.0, processTime = 0.0;
	glfwSetTime(0.0);

	//	メインループ
	while (1)							
	{

		//------------------------------
		//	タイマー計測開始
		//------------------------------
		currentTime = glfwGetTime();

		//------------------------------
		//	Detect Cube Marker
		//------------------------------
		colorImg = flycap.readImage();
		flip(colorImg, colorImg, 1);
		Mat temp = colorImg.clone();
		remap(colorImg, temp, mapC1, mapC2, INTER_LINEAR);

		//	カメラからマーカーまで
		static glm::mat4 markerTransMat = glm::mat4(1.0f);
		if (arDetectMarker(arhandle, (ARUint8*)colorImg.data)<0) break;
		arGetTransMatMultiSquare(ar3dhandle, arGetMarker(arhandle), arGetMarkerNum(arhandle), multiConfig);
		if (multiConfig->prevF != 0)
		{
			visible = true;
			double m_modelview[16];
			for (int k = 0; k < 3; k++) {
				for (int j = 0; j < 4; j++) {
					m_modelview[k + j * 4] = multiConfig->trans[k][j];
				}
				m_modelview[3 + k * 4] = 0.0;
			}
			m_modelview[3 + 3 * 4] = 1.0;
			markerTransMat = glm::make_mat4(m_modelview);
		}
		else
		{
			visible = false;
		}
		//-------------------------------
		//	検出位置補正
		//-------------------------------
		//	現フレーム＆過去フレームから分解
		vector<glm::vec3> transv(3);
		vector<glm::quat> quatv(3);
		transv[0] = glm::vec3(markerTransMat[3]);
		transv[1] = glm::vec3(prePose1[3]);
		transv[2] = glm::vec3(prePose2[3]);
		quatv[0] = glm::toQuat(markerTransMat);
		quatv[1] = glm::toQuat(prePose1);
		quatv[2] = glm::toQuat(prePose2);
		//	移動平均
		glm::vec3 t_mean = glm::mix(transv[0], glm::mix(transv[1], transv[2], weightA), weightV);
		glm::quat q_mean1;
		if (glm::dot(quatv[1], quatv[2]) > 0.0) q_mean1 = glm::mix(quatv[1], quatv[2], weightA);
		else q_mean1 = glm::mix(quatv[1], -quatv[2], weightA);
		glm::quat q_mean;
		if (glm::dot(quatv[0], q_mean1) > 0.0) q_mean = glm::mix(quatv[0], q_mean1, weightV);
		else q_mean = glm::mix(quatv[0], -q_mean1, weightV);
		//	閾値処理
		//	並進
		glm::vec3 t_current = transv[1];
		if (glm::distance(t_current, t_mean) > threshT) t_current = t_mean;
		//	回転
		glm::quat q_current = quatv[1];
		double dist_q = glm::length(q_current) + glm::length(q_mean) - glm::dot(q_current, q_mean);		//	||q_current - q_mean||
		if (dist_q > threshR) q_current = q_mean;
		//	行列の再構成
		glm::mat4 r_current = glm::mat4_cast(q_current);
		prePose2 = prePose1;								//	i-1 -> i-2 の更新
		prePose1 = glm::translate(t_current) * r_current;	//	i -> i-1 の更新
		markerTransMat = prePose1;
		


		//------------------------------
		//	Main Winodw
		//------------------------------
		glfwMakeContextCurrent(mainWindow);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		glImg.draw(temp);
		glClear(GL_DEPTH_BUFFER_BIT);

		//	プロジェクション行列
		Projection = glmProjMat;

		// カメラ行列
		View = glm::lookAt(
			glm::vec3(0, 0, 0), // カメラの原点
			glm::vec3(0, 0, 1), // 見ている点
			glm::vec3(0, -1, 0)  // カメラの上方向
			)
			;

		//	モデル行列
		//	マーカーからモデルまで
		glm::mat4 marker2model = glm::mat4(1.0)
			* glm::translate(glm::vec3(0.0f, 0.0f, -70.0f))
			//* glm::rotate(glm::mat4(1.0), (float)(180.0f*CV_PI/180.0f), glm::vec3(0.0, 1.0, 0.0))
			* glm::scale(glm::vec3(1.04, 1.04, 1.04))
			;
		Model = glm::mat4(1.0)
			* markerTransMat
			* marker2model;
		
		//	Render Object
		//	Our ModelViewProjection : multiplication of our 3 matrices
		mainRenderer.shader.enable();
		mainRenderer.MV = View * Model;
		mainRenderer.MVP = Projection * mainRenderer.MV;
		mainRenderer.lightDirection = glm::vec3(mainRenderer.MV[3]);
		mainRenderer.lightColor = colorWhite;
		
		//	Execute Rendering
		static bool showModel = true;
		if (showModel && visible)	mainRenderer.render();


		//	描画結果を反映
		glfwSwapBuffers(mainWindow);

		//--------------------------------
		//	Sub Window
		//--------------------------------
		glfwMakeContextCurrent(subWindow);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		Projection = glmProjMatProj;
		//	カメラを原点としたプロジェクタ位置姿勢
		View = glm::mat4(1.0)
			* glm::lookAt(
			glm::vec3(0, 0, 0), // カメラの原点
			glm::vec3(0, 0, 1), // 見ている点
			glm::vec3(0, -1, 0)  // カメラの上方向
			)
			* glm::rotate(projR[0], glm::vec3(1, 0, 0))
			* glm::rotate(projR[1], glm::vec3(0, 1, 0))
			* glm::rotate(projR[2], glm::vec3(0, 0, 1))
			* glm::translate(projT)		//	キャリブレーションの手動修正
			* glmTransProCam
			;
		

		//	カメラを原点としたワールド座標系
		Model = glm::mat4(1.0)
			* markerTransMat
			* marker2model;

		//	Render Object
		// Our ModelViewProjection : multiplication of our 3 matrices
		subRenderer.shader.enable();
		subRenderer.MV = View * Model;
		subRenderer.MVP = Projection * View * Model;
		subRenderer.lightDirection = glm::vec3(subRenderer.MV[3]);
		subRenderer.lightColor = colorWhite;
		
		if (visible) subRenderer.render();

		glDisable(GL_DEPTH_TEST);

		//	Render Marker Light
		Model = markerTransMat
			//* glm::rotate(glm::mat4(1.0), (float)(180.0f*CV_PI / 180.0f), glm::vec3(0.0, 1.0, 0.0))
			;

		markerLight.shader.enable();
		markerLight.MV = View * Model;
		markerLight.MVP = Projection * View * Model;
		markerLight.objectColor = colorWhite;
		markerLight.render();
		glEnable(GL_DEPTH_TEST);


		//Mat projImg(projSize, CV_8UC3);
		//glReadBuffer(GL_BACK);
		//glReadPixels(0, 0, projSize.width, projSize.height, GL_BGR, GL_UNSIGNED_BYTE, projImg.data);
		//flip(projImg, projImg, 0);
		//imwrite("test1.png", projImg);
		//remap(projImg, projImg, mapP1, mapP2, INTER_LINEAR);

		//imwrite("test.png", projImg);
		//glImg.draw(projImg);
		
		// Swap buffers
		glfwSwapBuffers(subWindow);

		//------------------------------
		//	Key Events
		//------------------------------
		//	Save MainWindow Front Image
		if (glfwGetKey(subWindow, GLFW_KEY_S) == GLFW_PRESS)
		{
			Mat capture = colorImg.clone();
			glfwMakeContextCurrent(mainWindow);
			glReadBuffer(GL_FRONT);
			glReadPixels(0, 0, cameraSize.width, cameraSize.height, GL_BGR, GL_UNSIGNED_BYTE, capture.data);
			flip(capture, capture, 0);
			imwrite("capture.png", capture);
		}

		//if (glfwGetKey(subWindow, GLFW_KEY_D) == GLFW_PRESS)
		//{
		//	cout << "T = [" << objTx << "," << objTy << "," << objTz << "]\n";
		//	cout << "Q = [" << current.w << "," << current.x << "," << current.y << "," << current.z << "]\n";
		//}

		//	Show Current Pro-Cam Translation Matrix
		if (glfwGetKey(subWindow, GLFW_KEY_P) == GLFW_PRESS)
		{
			showMatrix(glm::mat4(1.0)
				* glm::rotate(projR[0], glm::vec3(1, 0, 0))
				* glm::rotate(projR[1], glm::vec3(0, 1, 0))
				* glm::rotate(projR[2], glm::vec3(0, 0, 1))
				* glm::translate(projT)
				*glmTransProCam);
		}

		//	Rotation
		if (glfwGetKey(subWindow, GLFW_KEY_I) == GLFW_PRESS)
		{
			if (glfwGetKey(subWindow, GLFW_KEY_X) == GLFW_PRESS)
				projR[0] += 0.0005;
			if (glfwGetKey(subWindow, GLFW_KEY_Y) == GLFW_PRESS)
				projR[1] += 0.0005;
			if (glfwGetKey(subWindow, GLFW_KEY_Z) == GLFW_PRESS)
				projR[2] += 0.0005;
		}
		else
		{
			if (glfwGetKey(subWindow, GLFW_KEY_X) == GLFW_PRESS)
				projR[0] -= 0.0005;
			if (glfwGetKey(subWindow, GLFW_KEY_Y) == GLFW_PRESS)
				projR[1] -= 0.0005;
			if (glfwGetKey(subWindow, GLFW_KEY_Z) == GLFW_PRESS)
				projR[2] -= 0.0005;
		}
		//	Translation
		if (glfwGetKey(subWindow, GLFW_KEY_SLASH) == GLFW_PRESS)			//	L	+ x
			projT[0] += 0.1;
		if (glfwGetKey(subWindow, GLFW_KEY_COMMA) == GLFW_PRESS)		//	.	- x
			projT[0] -= 0.1;
		if (glfwGetKey(subWindow, GLFW_KEY_L) == GLFW_PRESS)		//	,	+ y
			projT[1] += 0.1;
		if (glfwGetKey(subWindow, GLFW_KEY_PERIOD) == GLFW_PRESS)		//	/	- y
			projT[1] -= 0.1;
		if (glfwGetKey(subWindow, GLFW_KEY_K) == GLFW_PRESS)			//	K	+ z
			projT[2] += 0.1;
		if (glfwGetKey(subWindow, GLFW_KEY_SEMICOLON) == GLFW_PRESS)	//	;	- z
			projT[2] -= 0.1;
		//	Set Default
		if (glfwGetKey(subWindow, GLFW_KEY_HOME) == GLFW_PRESS)
		{
			projR = glm::vec3(0.0); 
			projT = glm::vec3(0.0);
		}
		//	Show/Hide Model
		static bool keyHoldingM = false;
		int cM = glfwGetKey(subWindow, GLFW_KEY_M);
		if (keyHoldingM || cM == GLFW_PRESS)
		{
			if (keyHoldingM && cM == GLFW_RELEASE)
			{
				showModel = !showModel;
				keyHoldingM = false;
			}
			else
				keyHoldingM = true;
		}

		//	Change Texture
		static int texNum = 0;
		static bool keyHolding7 = false;
		int c7 = glfwGetKey(subWindow, GLFW_KEY_7);
		if (keyHolding7 || c7 == GLFW_PRESS)
		{
			if (keyHolding7 && c7 == GLFW_RELEASE)
			{
				if (texNum == 0) texNum = 0;
				else texNum--;
				mainRenderer.texImg = imread(textureSetDir[texNum]);
				flip(mainRenderer.texImg, mainRenderer.texImg, 1);
				mainRenderer.setObjectTexture();
				subRenderer.texImg = imread(textureSetDir[texNum]);
				flip(subRenderer.texImg, subRenderer.texImg, 1);
				subRenderer.setObjectTexture();
				cout << "テクスチャ変更：劣化度" << texNum * 15 << endl;
				keyHolding7 = false;
			}
			else
				keyHolding7 = true;
		}
		static bool keyHolding8 = false;
		int c8 = glfwGetKey(subWindow, GLFW_KEY_8);
		if (keyHolding8 || c8 == GLFW_PRESS)
		{
			if (keyHolding8 && c8 == GLFW_RELEASE)
			{
				if (texNum == 8) texNum = 8;
				else texNum++;
				mainRenderer.texImg = imread(textureSetDir[texNum]);
				flip(mainRenderer.texImg, mainRenderer.texImg, 1);
				mainRenderer.setObjectTexture();
				subRenderer.texImg = imread(textureSetDir[texNum]);
				flip(subRenderer.texImg, subRenderer.texImg, 1);
				subRenderer.setObjectTexture();
				if (texNum >= 7)
					cout << "テクスチャ変更：劣化度" << 110 + (texNum - 6) * 5 << endl;
				else
					cout << "テクスチャ変更：劣化度" << texNum * 15 << endl;
				keyHolding8 = false;
			}
			else
				keyHolding8 = true;
		}


		//	Change LUT
		static bool keyHoldingSpace = false;
		int cSpace = glfwGetKey(subWindow, GLFW_KEY_SPACE);
		if (keyHoldingSpace || cSpace == GLFW_PRESS)
		{
			if (keyHoldingSpace && cSpace == GLFW_RELEASE)
			{
				mainRenderer.useLUT = !mainRenderer.useLUT;
				subRenderer.useLUT = !subRenderer.useLUT;
				if (subRenderer.useLUT)
					colorWhite = backupWhite;
				else
					colorWhite = glm::vec3(1.0, 1.0, 1.0);

				glClearColor(colorWhite[0], colorWhite[1], colorWhite[2], 1.0);
				keyHoldingSpace = false;
			}
			else
			{
				keyHoldingSpace = true;
			}
		}
		if (glfwGetKey(subWindow, GLFW_KEY_1) == GLFW_PRESS)
		{
			visionLUT = imread(lutDir[0]);
			mainRenderer.updateLUT(visionLUT);
			subRenderer.updateLUT(visionLUT);
			backupWhite = lookupColor(glm::vec3(1.0, 1.0, 1.0), visionLUT);
			if (subRenderer.useLUT) colorWhite = backupWhite;
			cout << "Loaded: " << lutDir[0] << endl;
		}
		if (glfwGetKey(subWindow, GLFW_KEY_2) == GLFW_PRESS)
		{
			visionLUT = imread(lutDir[1]);
			mainRenderer.updateLUT(visionLUT);
			subRenderer.updateLUT(visionLUT);
			backupWhite = lookupColor(glm::vec3(1.0, 1.0, 1.0), visionLUT);
			if (subRenderer.useLUT) colorWhite = backupWhite;
			cout << "Loaded: " << lutDir[1] << endl;
		}
		if (glfwGetKey(subWindow, GLFW_KEY_3) == GLFW_PRESS)
		{
			visionLUT = imread(lutDir[2]);
			mainRenderer.updateLUT(visionLUT);
			subRenderer.updateLUT(visionLUT);
			backupWhite = lookupColor(glm::vec3(1.0, 1.0, 1.0), visionLUT);
			if (subRenderer.useLUT) colorWhite = backupWhite;
			cout << "Loaded: " << lutDir[2] << endl;
		}
		if (glfwGetKey(subWindow, GLFW_KEY_4) == GLFW_PRESS)
		{
			visionLUT = imread(lutDir[3]);
			mainRenderer.updateLUT(visionLUT);
			subRenderer.updateLUT(visionLUT);
			backupWhite = lookupColor(glm::vec3(1.0, 1.0, 1.0), visionLUT);
			if (subRenderer.useLUT) colorWhite = backupWhite;
			cout << "Loaded: " << lutDir[3] << endl;
		}
		if (glfwGetKey(subWindow, GLFW_KEY_5) == GLFW_PRESS)
		{
			visionLUT = imread(lutDir[4]);
			mainRenderer.updateLUT(visionLUT);
			subRenderer.updateLUT(visionLUT);
			backupWhite = lookupColor(glm::vec3(1.0, 1.0, 1.0), visionLUT);
			if (subRenderer.useLUT) colorWhite = backupWhite;
			cout << "Loaded: " << lutDir[4] << endl;
		}
		//	Quit Program
		if (glfwGetKey(mainWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS		//	Escキー
			|| glfwGetKey(subWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS
			|| glfwWindowShouldClose(mainWindow))			//	ウィンドウの閉じるボタン
		{
			safeTerminate();
			break;
		}

		//-------------------------------
		//	タイマー計測終了
		//-------------------------------
		processTime = glfwGetTime() - currentTime;
		cout << "FPS : " << 1.0 / processTime << "\r";

		glfwPollEvents();
	}

	return EXIT_SUCCESS;
}


void mouseEvent(GLFWwindow *window, int button, int state, int optionkey)
{
	if (state == GLFW_PRESS)
	{
		switch (button)
		{
		case GLFW_MOUSE_BUTTON_RIGHT:
		case GLFW_MOUSE_BUTTON_MIDDLE:
		case GLFW_MOUSE_BUTTON_LEFT:
			pressedMouseButton = button;
			break;
		}
		glfwGetCursorPos(subWindow, &xBegin, &yBegin);
	}
}

void cursorPosEvent(GLFWwindow *window, double x, double y)
{
	double xDisp = x - xBegin;
	double yDisp = y - yBegin;
	double dx = -xDisp / subWinW / 5.0;
	double dy = yDisp / subWinH / 5.0;
	double length = ::sqrt(dx*dx + dy*dy);	//	クォータニオンの長さ
	double rad;
	glm::quat after;

	switch (pressedMouseButton)
	{
	case GLFW_MOUSE_BUTTON_LEFT:
		if (length > 0)
		{
			rad = length * glm::pi<float>();
			after = glm::quat(cos(rad), -sin(rad) * dy / length, sin(rad) * dx / length, 0.0);
			current = after * current;
			objAngle += dx;
		}
		break;
	case GLFW_MOUSE_BUTTON_MIDDLE:
		objTx += dx * 300;
		objTy += dy * 300;
		break;
	case GLFW_MOUSE_BUTTON_RIGHT:
		break;
	}
	xBegin = x;
	yBegin = y;

}

void scrollEvent(GLFWwindow *window, double xofset, double yofset)
{
	objTz += yofset;
}

void safeTerminate()
{
	arPattDetach(arhandle);
	arPattDeleteHandle(pattHandle);
	ar3DDeleteHandle(&ar3dhandle);
	arDeleteHandle(arhandle);
	arParamLTFree(&cparamLT);
	glfwTerminate();
}


void showMatrix(glm::mat4 &m)
{
	cout << "[";
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 4; j++){
			cout << m[j][i] << "\t";
		}
		cout << "\n ";
	}
	for (int j = 0; j < 4; j++){
		cout << m[j][3] << "\t";
	}
	cout << "]" << endl;
}


//	LUTに従って画素値の入れ替え
void lookup(Mat &src, Mat &dst, Mat &LUT)
{
	//	LUTのアクセス用マクロ
#define BIT(B, G, R) ((B) << 16 | (G) << 8 | (R))
	dst = Mat(src.rows, src.cols, CV_8UC3);
	for (int y = 0; y < dst.rows; y++)
	{
		for (int x = 0; x < dst.cols; x++)
		{
			matB(dst, x, y) = LUT.at<Vec3b>(BIT(matB(src, x, y), matG(src, x, y), matR(src, x, y)), 0)[0];
			matG(dst, x, y) = LUT.at<Vec3b>(BIT(matB(src, x, y), matG(src, x, y), matR(src, x, y)), 0)[1];
			matR(dst, x, y) = LUT.at<Vec3b>(BIT(matB(src, x, y), matG(src, x, y), matR(src, x, y)), 0)[2];
		}
	}
}

glm::vec3 lookupColor(glm::vec3 &src, Mat LUT)
{
	Vec3b temp((uchar)(src[2] * 255.0), (uchar)(src[1] * 255.0), (uchar)(src[0] * 255.0));
	temp = LUT.at<Vec3b>(BIT(temp[0], temp[1], temp[2]));
	return glm::vec3((double)temp[2]/255.0, (double)temp[1]/255.0, (double)temp[0]/255.0);
}