
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

#include <iostream>
#include "objloader.hpp"
#include "FlyCap2CVWrapper.h"
#include "OpenGLHeader.h"
#include "Shader.h"
#include "GLImage.h"
#include "OpenCVCamera.h"

#include <ARToolKitPlus\TrackerSingleMarkerImpl.h>
#pragma comment(lib, "ARToolKitPlus.lib")

using namespace cv;
using namespace std;

//-----------------------------------------------------
//	import file path
//-----------------------------------------------------
const char vertexDir[] = "./shader/vertex.glsl";
const char fragmentDir[] = "./shader/fragment.glsl";
const char objDir[] = "../common/data/model/CalibBox/CalibBox.obj";
const char textureDir[] = "../common/data/model/CalibBox/textures/txt_001_diff.bmp";
//const char lutDir[] = "../common/data/lut/LUT_dichromat_typeP.png";
const char *lutDir[5] = {
	"../common/data/lut/LUT_dichromat_typeP.png",
	"../common/data/lut/LUT_dichromat_typeD.png",
	"../common/data/lut/LUT_dichromat_typeT.png",
	"../common/data/lut/LUT_elder_70.png",
	"../common/data/lut/LUT_elder_80.png"
};
const char calibDir[] = "./data/calibdata.xml";
//const char calibARDir[] = "./data/calib_artkp.dat";

//-----------------------------------------------------
//	for Calibration Data
//-----------------------------------------------------
#define MARKER_SIZE 48.0f
FlyCap2CVWrapper flycap;
Mat colorImg;
Mat cameraMatrix, distCoeffs, cameraMatrixProj, distCoeffsProj, RProCam, TProCam;
Size cameraSize, projSize;
glm::mat4 glmProjMat, glmTransProCam;
Mat mapC1, mapC2;
ARToolKitPlus::TrackerSingleMarker *tracker;

//-----------------------------------------------------
//	GLFW User Interface
//-----------------------------------------------------
#define PROJ_WIN_ID 2

GLFWwindow	*mainWindow, *subWindow;		//	マルチウィンドウ
int subWinW, subWinH;
//static float objTx = 85.9375, objTy = 588.54609, objTz = -40.4250069;
static float objTx = 0, objTy = 0, objTz = 0;
static glm::vec3 projT(0.0, 0.0, 0.0);
//static glm::quat current = glm::quat(-0.3691, 0.00095, 0.00852, -0.9293);
static glm::quat current = glm::quat(1.0, 0.0, 0.0, 0.0);
static float objAngle = 0.0f;
//	履歴データ 0が最も新しい 過去2フレームを使用
std::vector<glm::mat4> prePose = { glm::mat4(1.0), glm::mat4(1.0), glm::mat4(1.0) };
static float weightV = 0.9, weightA = 0.3;			//	mean = mix(p0, mix(p1, p2, weightA), weightV) 
static double threshR = 1.0e-6, threshT = 0.5;		//	物体が止まっていると認識する閾値
//static float weightV = 0.0, weightA = 0.0;			//	mean = mix(p0, mix(p1, p2, weightA), weightV) 
//static double threshR = 0, threshT = 0;		//	物体が止まっていると認識する閾値

double xBegin, yBegin;
int pressedMouseButton = 0;
bool visible = false;

//-----------------------------------------------------
//	OpenGL / GLSL Rendering Engine
//-----------------------------------------------------

#define GLSL_LOCATION_VERTEX	0
#define GLSL_LOCATION_UV		1
#define GLSL_LOCATION_NORMAL	2

typedef struct Renderer
{
	Object obj;
	Shader shader;
	////	uniform IDs
	//	in vertex.glsl
	GLuint mvpID;			//	uniform mat4 MVP;
	GLuint mvID;			//	uniform mat4 MV;
	//	in fragment.glsl
	GLuint textureSamplerID;		//	uniform sampler2D myTextureSampler;
	GLuint lutSamplerID;
	GLuint lightDirectionID;		//	uniform vec3 LightDirection;
	GLuint lightColorID;			//	uniform vec3 LightColor;
	GLuint lutSwitchID;
	////	object buffers
	GLuint vertexArray;		//	頂点情報を保持する配列
	GLuint vertexBuffer;	//	location = 0
	GLuint uvBuffer;		//	location = 1
	GLuint normalBuffer;	//	location = 2
	GLuint textureObject;	//	テクスチャにアクセスするためのオブジェクト
	GLuint lutBuffer;		//	LookUpTable
	////	uniform variables
	glm::mat4 MVP;
	glm::mat4 MV;
	glm::vec3 lightDirection;
	glm::vec3 lightColor;
	bool useLUT = false;
};


Renderer mainRenderer, subRenderer;
Mat	texImg;
Mat lutMat;

void getUniformID(Renderer &r);
void setObjectTexture(Renderer &r, Mat &texture);
void setLUT(Renderer &r, Mat &lut);
void updateLUT(Renderer &r, Mat &lut);
void setObjectVertices(Renderer &r);
void renderObject(Renderer &r);

//-----------------------------------------------------
//	Prototypes
//-----------------------------------------------------
int initWindow(void);
void initMainWindow(void);
void initSubWindow(void);
void initCamera(void);
void mouseEvent(GLFWwindow *window, int button, int state, int optionkey);
void cursorPosEvent(GLFWwindow *window, double x, double y);
void scrollEvent(GLFWwindow *window, double xofset, double yofset);
void safeTerminate();
void cameraFrustumRH(Mat cameraMatrix, Size cameraSize, glm::mat4 &projMatrix, double znear, double zfar);
void composeRT(Mat R, Mat T, glm::mat4 &RT);
void showMatrix(glm::mat4 &m);


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

	// Main Windowの用意
	mainWindow = glfwCreateWindow(cameraSize.width, cameraSize.height, "Main Window", NULL, NULL);
	if (mainWindow == NULL){
		cerr << "GLFWウィンドウの生成に失敗しました. Intel GPUを使用している場合は, OpenGL 3.3と相性が良くないため，2.1を試してください．\n";
		glfwTerminate();
		return EXIT_FAILURE;
	}

	// Sub Windowの用意
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

	//	プログラマブルシェーダをロード
	mainRenderer.shader.initGLSL(vertexDir, fragmentDir);
	//	Uniform変数へのハンドルを取得
	getUniformID(mainRenderer);

	//	.objファイルを読み込み
	loadOBJ(objDir, mainRenderer.obj);
	////	OBJファイルの中身をバッファに転送
	setObjectVertices(mainRenderer);

	//	テクスチャ画像を読み込む
	setObjectTexture(mainRenderer, texImg);

	//	LUTを読み込む
	setLUT(mainRenderer, lutMat);
}

void initSubWindow(void)
{
	//	Sub Window Setting
	glfwMakeContextCurrent(subWindow);				//	sub windowをカレントにする
	glfwSwapInterval(0);				//	SwapBufferのインターバル

	glClearColor(1.0, 1.0, 1.0, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LESS);				//	カメラに近い面だけレンダリングする

	//	プログラマブルシェーダをロード
	//subRenderer.shader.initGLSL(vertexDir, fragmentDir);
	subRenderer.shader = mainRenderer.shader;
	//	Uniform変数へのハンドルを取得
	getUniformID(subRenderer);

	//	.objファイルを読み込み
	loadOBJ(objDir, subRenderer.obj);
	////	OBJファイルの中身をバッファに転送
	setObjectVertices(subRenderer);

	//	テクスチャ画像を読み込む
	setObjectTexture(subRenderer, texImg);

	//	LUTを読み込む
	setLUT(subRenderer, lutMat);

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

	cameraFrustumRH(cameraMatrix, cameraSize, glmProjMat, 0.1, 5000);
	//TProCam.at<double>(0) = 21.4989;
	//TProCam.at<double>(1) = 1.98882; 
	//TProCam.at<double>(2) = 76.1548;
	composeRT(RProCam, TProCam, glmTransProCam);
	//	Undistort Map
	initUndistortRectifyMap(
		cameraMatrix, distCoeffs,
		Mat(), cameraMatrix, cameraSize, CV_32FC1,
		mapC1, mapC2);


}

int main(void)
{
	initCamera();

	if(initWindow() == EXIT_FAILURE) exit(-1);
	//	カメラの準備
	//namedWindow("camera");

	texImg = imread(textureDir);
	if (texImg.empty())
	{
		cerr << "テクスチャの読み込みに失敗しました．ディレクトリを確認してください．\n"
			<< "ディレクトリの場所：" << textureDir << endl;
		glfwTerminate();
		return EXIT_FAILURE;
	}
	lutMat = imread(lutDir[0]);
	if (lutMat.empty())
	{
		cerr << "LUTの読み込みに失敗しました．ディレクトリを確認してください．\n"
			<< "ディレクトリの場所：" << lutDir << endl;
		glfwTerminate();
		return EXIT_FAILURE;
	}

	cout << "\nMain Windowの設定" << endl;
	initMainWindow();
	cout << "\nSub Windowの設定" << endl;
	initSubWindow();
	cout << "\n画像描画の設定" << endl;
	GLImage glImg;
	glImg.init(mainWindow);


	//	ARToolKitPlusの初期化
	ARToolKitPlus::Camera *param = OpenCVCamera::fromOpenCV(cameraMatrix, distCoeffs, cameraSize);

	ARToolKitPlus::Logger *logger = nullptr;
	tracker = new ARToolKitPlus::TrackerSingleMarkerImpl<6, 6, 6, 1, 10>(cameraSize.width, cameraSize.height);
	//tracker->init("data/LogitechPro4000.dat", 0.1f, 5000.0f);	
	tracker->init(NULL, 0.1f, 5000.0f);	//	ファイルは使用しない
	tracker->setCamera(param);
	//tracker->changeCameraSize(cameraSize.width, cameraSize.height);
	tracker->activateAutoThreshold(true);
	tracker->setNumAutoThresholdRetries(5);
	tracker->setBorderWidth(0.125f);			//	BCH boader width = 12.5%
	tracker->setPatternWidth(29.5f);			//	marker physical width = 60.0mm
	tracker->setPixelFormat(ARToolKitPlus::PIXEL_FORMAT_BGR);		//	With OpenCV
	tracker->setUndistortionMode(ARToolKitPlus::UNDIST_NONE);		//	UndistortionはOpenCV側で行う
	tracker->setMarkerMode(ARToolKitPlus::MARKER_ID_BCH);
	tracker->setPoseEstimator(ARToolKitPlus::POSE_ESTIMATOR_RPP);

	//ARToolKitPlus::Camera *pm = tracker->getCamera();
	////tracker->setCamera(pm);

	//for (int i = 0; i < 3; i++){
	//	for (int j = 0; j < 4; j++)
	//		std::cout << pm->mat[i][j] << " ";
	//	std::cout << "\n";
	//}
	//cout << endl;
	//for (int i = 0; i < 4; i++)
	//	std::cout << pm->dist_factor[i] << " ";
	//std::cout << std::endl;
	//cout << "(" << pm->xsize << ", " << pm->ysize << ")" << endl;

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
		//	画像処理
		//------------------------------
		colorImg = flycap.readImage();
		flip(colorImg, colorImg, 1);
		Mat temp;
		remap(colorImg, temp, mapC1, mapC2, INTER_LINEAR);
		//Mat temp_gaussian;
		//cvtColor(temp, temp_gaussian, CV_BGR2GRAY);
		//adaptiveThreshold(temp_gaussian, temp_gaussian, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 281, 10);
		//blur(temp_gaussian, temp_gaussian, Size(3, 3));
		//threshold(temp_gaussian, temp_gaussian, 128, 255, THRESH_BINARY);
		//cvtColor(temp_gaussian, temp_gaussian, CV_GRAY2BGR);
		//imshow("gaussian", temp_gaussian);

		//	カメラからマーカーまで
		static glm::mat4 markerTransMat = glm::mat4(1.0f); 
		ARToolKitPlus::ARMarkerInfo *markers;
		int markerID = tracker->calc(temp.data, -1, true, &markers);
		float conf = (float)tracker->getConfidence();		//	信頼度
		if (markerID == 4)
		{
			visible = true;
			//	コーナー点を描画
			Point center(markers->pos[0], markers->pos[1]); 
			circle(temp, center, 6, Scalar(0, 0, 255));
			for (int j = 0; j < 4; j++) {
				Point p(markers->vertex[j][0], markers->vertex[j][1]);
				circle(temp, p, 6, Scalar(255, 0, 255));
			}
			//cout << markers->vertex[0][0] << ", " << markers->vertex[0][1] << "\n";
			markerTransMat = glm::make_mat4(tracker->getModelViewMatrix())
				//* glm::mat4_cast(current)
				* glm::rotate(glm::mat4(1.0), objAngle, glm::vec3(0, 0, 1))
				* glm::translate(glm::vec3(objTx, objTy, objTz))
				;

			//cout << "confidence: " << conf << " ";
			//cout << "detected!";
			//showMatrix(markerTransMat);
		}
		else
		{
			visible = false;
		}
		//------------------------------
		//	Get AR Marker Transform
		//------------------------------
		////	カメラ画像取得
		//colorImg = flycap.readImage();
		//flip(colorImg, colorImg, 1);
		////	ARToolKitに2値化画像を転送
		//Mat threshImg, cameraBGRA, temp;
		////cv::cvtColor(colorImg, threshImg, CV_BGR2GRAY);
		////cv::adaptiveThreshold(threshImg, threshImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 55, 10);
		////cv::GaussianBlur(threshImg, threshImg, cv::Size(9, 9), 3.0);
		////cv::cvtColor(threshImg, cameraBGRA, CV_GRAY2BGRA);
		//cvtColor(colorImg, cameraBGRA, CV_BGR2BGRA);
		//ARUint8 *imgData = (ARUint8*)cameraBGRA.data;
		//cvtColor(colorImg, threshImg, CV_BGR2GRAY);
		//threshold(threshImg, threshImg, 128, 255, CV_THRESH_BINARY);
		//cv::imshow("camera", threshImg);
		//
		////	ARマーカーを認識
		//ARMarkerInfo *markerInfo;
		//int markerNum, thresh = 128;
		//if (arDetectMarker(imgData, thresh, &markerInfo, &markerNum) < 0)
		//{
		//	cerr << "Error: at arDetectMarker() function." << endl;
		//	safeTerminate();
		//	exit(-1);
		//}
		//int k = -1;
		//for (int j = 0; j < markerNum; j++)
		//{
		//	if (marker.patt_id == markerInfo[j].id)
		//	{	//	markerと最も一致度の高いIDを抽出
		//		if (k == -1) k = j;
		//		else if (markerInfo[k].cf < markerInfo[j].cf) k = j;
		//	}
		//}
		////	マーカー位置姿勢を取得
		//glm::mat4 markerTransMat;		//	最初は単位行列
		//if (k == -1) marker.visible = 0;
		//else
		//{	//	過去情報を利用してブレを抑える
		//	if (marker.visible == 0)
		//		arGetTransMat(&markerInfo[k], marker.patt_center, marker.patt_width, marker.patt_trans);
		//	else
		//		arGetTransMatCont(&markerInfo[k], marker.patt_trans, marker.patt_center, marker.patt_width, marker.patt_trans);
		//	marker.visible = 1;
		//	for (int i = 0; i < 3; i++){
		//		for (int j = 0; j < 4; j++){
		//			markerTransMat[j][i] = marker.patt_trans[i][j];
		//		}
		//	}
		//	////	マーカー位置姿勢の加重平均をとる
		//	prePose.insert(prePose.begin(), markerTransMat);	//	prePose[0]に挿入
		//	prePose.pop_back();		//	-3フレーム目を削除
		//	//	回転(expマップ上で加重平均)
		//	//	クォータニオンの生成
		//	glm::quat q[3] = {
		//		glm::toQuat(prePose[0]),
		//		glm::toQuat(prePose[1]),
		//		glm::toQuat(prePose[2])
		//	};
		//	//	球面線形補間
		//	glm::quat q_mean = glm::slerp(q[0], glm::slerp(q[1], q[2], 0.9f), 0.9f);		//	x(1-a)+ya
		//	//	クォータニオンを回転行列に変換
		//	glm::mat4 r_mean = glm::mat4_cast(q_mean);
		//	//	閾値処理
		//	static glm::quat q_mean_temp = q_mean;
		//	//cout << abs(glm::dot(q_mean, q[1]) - glm::length(q_mean)) << endl;
		//	if (abs(glm::dot(q_mean, q[1]) - glm::length(q_mean)) < threshR)
		//		r_mean = glm::mat4_cast(q_mean_temp);
		//	else
		//		q_mean_temp = q[0];
		//	//	平行移動ベクトルは単純に加重平均
		//	glm::vec4 t_mean = glm::mix((prePose[0])[3], glm::mix((prePose[1])[3], (prePose[2])[3], weightA), weightV);
		//	//	閾値処理
		//	static glm::vec4 t_temp = t_mean;
		//	//cout << glm::distance(t, prePose[1][3]) << endl;
		//	if (glm::distance(t_mean, prePose[1][3]) < threshT)
		//		t_mean = t_temp;
		//	else
		//		t_temp = t_mean;
		//	//	並進・回転の合成
		//	markerTransMat = glm::translate(glm::vec3(t_mean)) * r_mean;
		//	//for (int i = 0; i < 4; i++){
		//	//	for (int j = 0; j < 4; j++)
		//	//		cout << markerTransMat[i][j] << ", ";
		//	//	cout << ";\n";
		//	//}
		//}
		
		//------------------------------
		//	Main Winodw
		//------------------------------
		glfwMakeContextCurrent(mainWindow);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		glImg.draw(temp);
		glClear(GL_DEPTH_BUFFER_BIT);

		//	プロジェクション行列
		glm::mat4 Projection = glm::mat4(1.0);
		//Projection = glm::make_mat4(tracker->getProjectionMatrix());
		cameraFrustumRH(cameraMatrix, cameraSize, glmProjMat, 0.1, 5000);
		Projection = glmProjMat;
		// カメラ行列
		glm::mat4 View = glm::mat4(1.0)
			* glm::lookAt(
			glm::vec3(0, 0, 0), // カメラの原点
			glm::vec3(0, 0, 1), // 見ている点
			glm::vec3(0, 1, 0)  // カメラの上方向
			)
			;

		glm::mat4 Model;  // 各モデルを変える！
		//	マーカーからモデルまで
		glm::mat4 marker2model = glm::mat4(1.0)
			* glm::rotate(glm::mat4(1.0), (float)(180.0f*CV_PI/180.0f), glm::vec3(0.0, 1.0, 0.0))
			//* glm::translate(glm::vec3(0.0f, 0.0, 216.0))
			//* glm::scale(glm::vec3(1.0, 1.0, 1.0))
			;
		Model = glm::mat4(1.0)
			* markerTransMat
			* marker2model;
		
		//	Render Object
		//	Our ModelViewProjection : multiplication of our 3 matrices
		mainRenderer.shader.enable();
		mainRenderer.MV = View * Model;
		mainRenderer.MVP = Projection * mainRenderer.MV;
		mainRenderer.lightDirection = glm::vec3(markerTransMat[3]) - glm::vec3(glmTransProCam[3]);
		mainRenderer.lightColor = glm::vec3(1.0, 1.0, 1.0);
		
		//	Execute Rendering
		static bool showModel = true;
		if (showModel && visible)	renderObject(mainRenderer);

		//	描画結果を反映
		glfwSwapBuffers(mainWindow);

		//--------------------------------
		//	Sub Window
		//--------------------------------
		glfwMakeContextCurrent(subWindow);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//	Render Object
		// Our ModelViewProjection : multiplication of our 3 matrices
		subRenderer.shader.enable();
		
		cameraFrustumRH(cameraMatrixProj, projSize, glmProjMat, 0.1, 5000);
		Projection = glmProjMat;
		//	カメラを原点としたプロジェクタ位置姿勢
		View = glm::mat4(1.0)
			* glm::lookAt(
			glm::vec3(0, 0, 0), // カメラの原点
			glm::vec3(0, 0, 1), // 見ている点
			glm::vec3(0, 1, 0)  // カメラの上方向
			)
			* glmTransProCam
			* glm::translate(projT)		//	キャリブレーションの手動修正
			;
			//* projectorPose;

		//	カメラを原点としたワールド座標系
		Model = glm::mat4(1.0)
			* markerTransMat
			* marker2model;

		subRenderer.MV = View * Model;
		subRenderer.MVP = Projection * View * Model;
		subRenderer.lightDirection = glm::vec3(markerTransMat[3]) - glm::vec3(glmTransProCam[3]);
		subRenderer.lightColor = glm::vec3(1.0, 1.0, 1.0);
		
		if (visible)
		renderObject(subRenderer);

		// Swap buffers
		glfwSwapBuffers(subWindow);

		//------------------------------
		//	Key Events
		//------------------------------

		if (glfwGetKey(subWindow, GLFW_KEY_D) == GLFW_PRESS)
		{
			cout << "T = [" << objTx << "," << objTy << "," << objTz << "]\n";
			cout << "Q = [" << current.w << "," << current.x << "," << current.y << "," << current.z << "]\n";
		}

		if (glfwGetKey(subWindow, GLFW_KEY_P) == GLFW_PRESS)
		{
			showMatrix(glmTransProCam
				* glm::translate(projT));
		}
		if (glfwGetKey(subWindow, GLFW_KEY_I) == GLFW_PRESS)
		{
			if (glfwGetKey(subWindow, GLFW_KEY_X) == GLFW_PRESS)
				projT[0] += 0.1;
			if (glfwGetKey(subWindow, GLFW_KEY_Y) == GLFW_PRESS)
				projT[1] += 0.1;
			if (glfwGetKey(subWindow, GLFW_KEY_Z) == GLFW_PRESS)
				projT[2] += 0.1;
		}
		else
		{
			if (glfwGetKey(subWindow, GLFW_KEY_X) == GLFW_PRESS)
				projT[0] -= 0.1;
			if (glfwGetKey(subWindow, GLFW_KEY_Y) == GLFW_PRESS)
				projT[1] -= 0.1;
			if (glfwGetKey(subWindow, GLFW_KEY_Z) == GLFW_PRESS)
				projT[2] -= 0.1;
		}
		if (glfwGetKey(subWindow, GLFW_KEY_M) == GLFW_PRESS)
		{
			showModel = !showModel;
		}

		//	Change LUT
		static bool keyHolding = false;
		int c = glfwGetKey(subWindow, GLFW_KEY_SPACE);
		if (keyHolding || c == GLFW_PRESS)
		{
			if (keyHolding && c == GLFW_RELEASE)
			{
				mainRenderer.useLUT = !mainRenderer.useLUT;
				subRenderer.useLUT = !subRenderer.useLUT;
				keyHolding = false;
			}
			else
				keyHolding = true;
		}
		if (glfwGetKey(subWindow, GLFW_KEY_1) == GLFW_PRESS)
		{
			lutMat = imread(lutDir[0]);
			updateLUT(mainRenderer, lutMat);
			updateLUT(subRenderer, lutMat);
			cout << "Loaded: " << lutDir[0] << endl;
		}
		if (glfwGetKey(subWindow, GLFW_KEY_2) == GLFW_PRESS)
		{
			lutMat = imread(lutDir[1]);
			updateLUT(mainRenderer, lutMat);
			updateLUT(subRenderer, lutMat);
			cout << "Loaded: " << lutDir[1] << endl;
		}
		if (glfwGetKey(subWindow, GLFW_KEY_3) == GLFW_PRESS)
		{
			lutMat = imread(lutDir[2]);
			updateLUT(mainRenderer, lutMat);
			updateLUT(subRenderer, lutMat);
			cout << "Loaded: " << lutDir[2] << endl;
		}
		if (glfwGetKey(subWindow, GLFW_KEY_4) == GLFW_PRESS)
		{
			lutMat = imread(lutDir[3]);
			updateLUT(mainRenderer, lutMat);
			updateLUT(subRenderer, lutMat);
			cout << "Loaded: " << lutDir[3] << endl;
		}
		if (glfwGetKey(subWindow, GLFW_KEY_5) == GLFW_PRESS)
		{
			lutMat = imread(lutDir[4]);
			updateLUT(mainRenderer, lutMat);
			updateLUT(subRenderer, lutMat);
			cout << "Loaded: " << lutDir[4] << endl;
		}

		if (glfwGetKey(mainWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS		//	Escキー
			|| glfwGetKey(subWindow, GLFW_KEY_ESCAPE) == GLFW_PRESS
			|| glfwWindowShouldClose(mainWindow))			//	ウィンドウの閉じるボタン
		{
			if (tracker)
				delete tracker;
			tracker = NULL;
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

void getUniformID(Renderer &r)
{
	r.mvpID = glGetUniformLocation(r.shader.program, "MVP");
	r.mvID = glGetUniformLocation(r.shader.program, "MV");
	r.textureSamplerID = glGetUniformLocation(r.shader.program, "myTextureSampler");
	r.lutSamplerID = glGetUniformLocation(r.shader.program, "lutSampler");
	r.lightDirectionID = glGetUniformLocation(r.shader.program, "LightDirection");
	r.lightColorID = glGetUniformLocation(r.shader.program, "LightColor");
	r.lutSwitchID = glGetUniformLocation(r.shader.program, "lutSwitch");
}

void setObjectTexture(Renderer &r, Mat &texture)
{
	//	テクスチャ画像を読み込む
	glGenTextures(1, &r.textureObject);
	glBindTexture(GL_TEXTURE_2D, r.textureObject);
	//	OpenGLに画像を渡す
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
		texture.cols, texture.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, texture.data);
	//	テクスチャの繰り返し設定
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	//	画像を拡大(MAGnifying)するときは線形(LINEAR)フィルタリングを使用
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//	画像を縮小(MINifying)するとき、線形(LINEAR)フィルタした、二つのミップマップを線形(LINEARYLY)に混ぜたものを使用
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	//	ミップマップを作成
	glGenerateMipmap(GL_TEXTURE_2D);
}

void setLUT(Renderer &r, Mat &lut)
{
	//	テクスチャ画像を読み込む
	glGenTextures(1, &r.lutBuffer);
	glBindTexture(GL_TEXTURE_3D, r.lutBuffer);
	//	OpenGLに画像を渡す
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB,
		256, 256, 256, 0, GL_BGR, GL_UNSIGNED_BYTE, lut.data);
	//	テクスチャの拡大縮小に線形補間を使用
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
}

void updateLUT(Renderer &r, Mat &lut)
{
	glBindTexture(GL_TEXTURE_3D, r.lutBuffer);
	//	OpenGLに画像を渡す
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB,
		256, 256, 256, 0, GL_BGR, GL_UNSIGNED_BYTE, lut.data);
}

void setObjectVertices(Renderer &r)
{
	//	頂点配列オブジェクトを設定
	glGenVertexArrays(1, &r.vertexArray);
	glBindVertexArray(r.vertexArray);

	//	頂点バッファをOpenGLに渡す
	glGenBuffers(1, &r.vertexBuffer);							//	バッファを1つ作成
	glBindBuffer(GL_ARRAY_BUFFER, r.vertexBuffer);			//	以降のコマンドをvertexbufferバッファに指定
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * r.obj.vertices.size(), &(r.obj).vertices[0], GL_STATIC_DRAW);		//	頂点をOpenGLのvertexbuferに渡す

	//	UV座標バッファ
	glGenBuffers(1, &r.uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, r.uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * r.obj.uvs.size(), &(r.obj).uvs[0], GL_STATIC_DRAW);

	//	法線バッファ
	glGenBuffers(1, &r.normalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, r.normalBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * r.obj.normals.size(), &(r.obj).normals[0], GL_STATIC_DRAW);
}

void renderObject(Renderer &r)
{
	////	Execute Rendering
	// 現在バインドしているシェーダのuniform変数に変換を送る
	// レンダリングするモデルごとに実行
	glUniformMatrix4fv(r.mvpID, 1, GL_FALSE, &r.MVP[0][0]);
	glUniformMatrix4fv(r.mvID, 1, GL_FALSE, &r.MV[0][0]);
	glUniform3fv(r.lightDirectionID, 1, &r.lightDirection[0]);
	glUniform3fv(r.lightColorID, 1, &r.lightColor[0]);
	glUniform1i(r.lutSwitchID, r.useLUT);

	//	テクスチャユニット0にtextureBufferをバインド
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, r.textureObject);
	//	0番目のテクスチャユニットを"myTextureSampler"にセット
	glUniform1i(r.textureSamplerID, 0);

	//	テクスチャユニット1にlutBufferをバインド
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_3D, r.lutBuffer);
	glUniform1i(r.lutSamplerID, 1);

	//	最初の属性バッファ：頂点
	glEnableVertexAttribArray(GLSL_LOCATION_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, r.vertexBuffer);
	glVertexAttribPointer(
		GLSL_LOCATION_VERTEX,	// shader内のlocation
		3,						// 要素サイズ
		GL_FLOAT,				// 要素の型
		GL_FALSE,				// 正規化？
		0,						// ストライド
		(void*)0				// 配列バッファオフセット
		);
	//	2番目の属性バッファ : UV
	glEnableVertexAttribArray(GLSL_LOCATION_UV);
	glBindBuffer(GL_ARRAY_BUFFER, r.uvBuffer);
	glVertexAttribPointer(GLSL_LOCATION_UV, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	//	3番目の属性バッファ : 法線
	glEnableVertexAttribArray(GLSL_LOCATION_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, r.normalBuffer);
	glVertexAttribPointer(GLSL_LOCATION_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	//	三角形ポリゴンを描画
	glDrawArrays(GL_TRIANGLES, 0, r.obj.vertices.size());
	//	描画後にバッファをクリア
	glDisableVertexAttribArray(GLSL_LOCATION_VERTEX);
	glDisableVertexAttribArray(GLSL_LOCATION_UV);
	glDisableVertexAttribArray(GLSL_LOCATION_NORMAL);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindTexture(GL_TEXTURE_3D, 0);
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
	glfwTerminate();
}

//	OpenCVカメラパラメータからOpenGL(GLM)プロジェクション行列を得る関数
void cameraFrustumRH(Mat camMat, Size camSz, glm::mat4 &projMat, double znear, double zfar)
{
	//	Load camera parameters
	double fx = camMat.at<double>(0, 0);
	double fy = camMat.at<double>(1, 1);
	double s = camMat.at<double>(0, 1);
	double cx = camMat.at<double>(0, 2);
	double cy = camMat.at<double>(1, 2);
	double w = camSz.width, h = camSz.height;

	//	参考:https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
	//	With window_coords=="y_down", we have:
	//	[2 * K00 / width,	-2 * K01 / width,	(width - 2 * K02 + 2 * x0) / width,		0]
	//	[0,					2 * K11 / height,	(-height + 2 * K12 + 2 * y0) / height,	0]
	//	[0,					0,					(-zfar - znear) / (zfar - znear),		-2 * zfar*znear / (zfar - znear)]
	//	[0,					0,					-1,										0]

	glm::mat4 projection(
		-2.0 * fx / w,		0,						0,										0,
		0,					-2.0 * fy / h,			0,										0,
		1.0 - 2.0 * cx / w,	- 1.0 + 2.0 * cy / h,	-(zfar + znear) / (zfar - znear),		-1.0,
		0,					0,						-2.0 * zfar * znear / (zfar - znear),	0);


	projMat = projection;
}

void composeRT(Mat R, Mat T, glm::mat4 &RT)
{
	glm::mat4 trans(1.0);
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			trans[j][i] = R.at<double>(i, j);
		}
		trans[3][i] = T.at<double>(i);
	}
	RT = trans;
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
