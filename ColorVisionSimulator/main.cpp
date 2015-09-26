
#include <iostream>
#include "Shader.h"
#include "objloader.hpp"
#include "Kinect2WithOpenCVWrapper.h"
#include "OpenGLHeader.h"
#include "OpenCV3Linker.h"
#include "ARTKLinker.h"

using namespace cv;
using namespace std;

//-----------------------------------------------------
//	import file path
//-----------------------------------------------------
const char vertexDir[] = "./shader/vertex.glsl";
const char fragmentDir[] = "./shader/fragment.glsl";
const char objDir[] = "../common/data/model/ARbox/ARbox.obj";
const char textureDir[] = "../common/data/model/ARbox/textures/txt_001_diff.bmp";
//const char lutDir[] = "../common/data/lut/LUT_dichromat_typeP.png";
const char *lutDir[5] = {
	"../common/data/lut/LUT_dichromat_typeP.png",
	"../common/data/lut/LUT_dichromat_typeD.png",
	"../common/data/lut/LUT_dichromat_typeT.png",
	"../common/data/lut/LUT_elder_70.png",
	"../common/data/lut/LUT_elder_80.png"
};

//-----------------------------------------------------
//	Constants
//-----------------------------------------------------
//	RoomAliveToolkitでの測定結果
//	KinectのRGBカメラパラメータ
glm::mat3 colorCameraMatrix(
	1088.5942262014253, 0, 0,
	0, 1088.4801711642506, 0,
	987.9108474275381, 527.64605393047646, 1);
double colorLensDistortion[4] = {
	0.04229, -0.05348, -0.00024, 0.00335 };
glm::mat3 projectorCameraMatrix(
	2898.8350799438763, 0, 0,
	0, 2898.8350799438763, 0,
	768.59447467126, 273.6576771850149, 1);
glm::mat4 projectorPose(
	0.988940417766571, 0.066547796130180359, -0.13254536688327789, 0,
	-0.011403728276491165, 0.92515689134597778, 0.37941363453865051, 0,
	0.14787440001964569, -0.37370595335960388, 0.915683925151825, 0,
	-70.988007578053944, 606.12622324461007, -999.45766623241938, 1);


//-----------------------------------------------------
//	for ARToolkit
//-----------------------------------------------------
#define MARKER_SIZE 48.0f
ARParam cparam;
ARTKMarker marker = {
	"data/markerB.pat",
	-1,
	0,
	0,
	MARKER_SIZE,
	{0.0, 0.0},
	{0.0}
};
Kinect2WithOpenCVWrapper kinect;
Mat colorImg;

//-----------------------------------------------------
//	GLFW User Interface
//-----------------------------------------------------
#define PROJ_WIN_ID 2

GLFWwindow	*mainWindow, *subWindow;		//	マルチウィンドウ
int subWinW, subWinH;
//static float objTx = 85.9375, objTy = 588.54609, objTz = -40.4250069;
static float objTx = 0, objTy = 0, objTz = 0;
//static glm::quat current = glm::quat(-0.3691, 0.00095, 0.00852, -0.9293);
static glm::quat current = glm::quat(0.0, 0.0, 0.0, 1.0);
double xBegin, yBegin;
int pressedMouseButton = 0;

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

glm::mat4 projectionMatfromCameraMatrix(glm::mat3 cameraMat, int winW, int winH, double znear, double zfar);
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
void initARTK(void);
void mouseEvent(GLFWwindow *window, int button, int state, int optionkey);
void cursorPosEvent(GLFWwindow *window, double x, double y);
void scrollEvent(GLFWwindow *window, double xofset, double yofset);
void safeTerminate();
void arglCameraFrustumRH(ARParam *cparam, const double focalmin, const double focalmax, GLdouble m_projection[16]);

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
	mainWindow = glfwCreateWindow(1920/2, 1080/2, "Main Window", NULL, NULL);
	if (mainWindow == NULL){
		cerr << "GLFWウィンドウの生成に失敗しました. Intel GPUを使用している場合は, OpenGL 3.3と相性が良くないため，2.1を試してください．\n";
		glfwTerminate();
		return EXIT_FAILURE;
	}

	// Sub Windowの用意
	int monitorCount;
	GLFWmonitor **monitors = glfwGetMonitors(&monitorCount);
	subWindow = glfwCreateWindow(1024, 768, "Sub Window", monitors[PROJ_WIN_ID], NULL);
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

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LESS);				//	カメラに近い面だけレンダリングする

	//	プログラマブルシェーダをロード
	subRenderer.shader.initGLSL(vertexDir, fragmentDir);
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

void initARTK(void)
{

	//	パターンファイルのロード
	if ((marker.patt_id = arLoadPatt(marker.patt_name)) < 0)
	{
		cout << "パターンファイルが読み込めませんでした．ディレクトリを確認してください．\n"
			<< "場所：" << marker.patt_name << endl;
		safeTerminate();
		exit(-1);
	}
	//	カメラパラメータのロード
	ARParam wparam;
	arParamLoad("data/kinect_param.dat", 1, &wparam);
	arParamChangeSize(&wparam, colorImg.cols, colorImg.rows, &cparam);
	arParamDisp(&cparam);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			colorCameraMatrix[i][j] = cparam.mat[i][j];
		}
	}
	//cparam.xsize = colorImg.cols; cparam.ysize = colorImg.rows;
	//for (int i = 0; i < 4; i++)
	//{
	//	for (int j = 0; j < 3; j++)
	//	{
	//		cparam.mat[i][j] = colorCameraMatrix[i][j];
	//	}
	//}
	//for (int i = 0; i < 4; i++)
	//{
	//	cparam.dist_factor[i] = colorLensDistortion[i];
	//}
	//arParamDisp(&cparam);

	arInitCparam(&cparam);
}

int main(void)
{
	//	カメラの準備
	kinect.enableColorFrame();
	namedWindow("camera");
	while (1)
	{
		kinect.getColorFrame(colorImg);
		safeRelease(kinect.colorFrame);
		Mat temp;
		cv::resize(colorImg, temp, colorImg.size() / 2);
		cv::imshow("camera", temp);
		if (waitKey(15) == ' ') break;
	}
	initWindow();

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


	initMainWindow();
	initSubWindow();

	initARTK();

	//	メインループ
	while (glfwGetKey(mainWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS		//	Escキー
		&& glfwGetKey(subWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS
		&& !glfwWindowShouldClose(mainWindow))							//	ウィンドウの閉じるボタン
	{
		//------------------------------
		//	Key Events
		//------------------------------
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
		if (glfwGetKey(subWindow, GLFW_KEY_D) == GLFW_PRESS)
		{
			cout << "T = [" << objTx << "," << objTy << "," << objTz << "]\n";
			cout << "Q = [" << current.w << "," << current.x << "," << current.y << "," << current.z << "]\n";
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

		//------------------------------
		//	Get AR Marker
		//------------------------------
		//	カメラ画像取得
		kinect.getColorFrame(colorImg);
		safeRelease(kinect.colorFrame);
		//	ARToolKitに2値化画像を転送
		Mat threshImg, cameraBGRA, temp;
		cv::cvtColor(colorImg, threshImg, CV_BGR2GRAY);
		cv::adaptiveThreshold(threshImg, threshImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 55, 10);
		cv::GaussianBlur(threshImg, threshImg, cv::Size(9, 9), 2.0);
		cv::cvtColor(threshImg, cameraBGRA, CV_GRAY2BGRA);
		ARUint8 *imgData = (ARUint8*)cameraBGRA.data;
		temp = Mat(cameraBGRA.size(), CV_8UC4);
		::memcpy(temp.data, imgData, temp.total()*temp.channels());
		cv::resize(temp, temp, colorImg.size()/2);
		cv::imshow("camera", temp);
		
		//	ARマーカーを認識
		ARMarkerInfo *markerInfo;
		int markerNum, thresh = 128;
		if (arDetectMarker(imgData, thresh, &markerInfo, &markerNum) < 0)
		{
			cerr << "Error: at arDetectMarker() function." << endl;
			safeTerminate();
			exit(-1);
		}
		int k = -1;
		for (int j = 0; j < markerNum; j++)
		{
			if (marker.patt_id == markerInfo[j].id)
			{	//	markerと最も一致度の高いIDを抽出
				if (k == -1) k = j;
				else if (markerInfo[k].cf < markerInfo[j].cf) k = j;
			}
		}
		//	マーカー位置姿勢を取得
		glm::mat4 markerTransMat(1.0f);		//	最初は単位行列
		if (k == -1) marker.visible = 0;
		else
		{	//	過去情報を利用してブレを抑える
			if (marker.visible == 0)
				arGetTransMat(&markerInfo[k], marker.patt_center, marker.patt_width, marker.patt_trans);
			else
				arGetTransMatCont(&markerInfo[k], marker.patt_trans, marker.patt_center, marker.patt_width, marker.patt_trans);
			marker.visible = 1;
			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 4; j++)
					markerTransMat[j][i] = marker.patt_trans[i][j];
		}
		
		//------------------------------
		//	Main Winodw
		//------------------------------
		glfwMakeContextCurrent(mainWindow);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// 射影行列：45°の視界、アスペクト比4:3、表示範囲：0.1単位  100単位
		//glm::mat4 Projection = glm::perspective(45.0f, 1920.0f / 1080.0f, 0.1f, 10000.0f);
		//glm::mat4 pj = projectionMatfromCameraMatrix(colorCameraMatrix, subWinW, subWinH, 0.001, 10000.0);
		//for (int i = 0; i < 4; i++){
		//	for (int j = 0; j < 4; j++)
		//		cout << pj[i][j] << ",";
		//	cout << ";\n";
		//}
		glm::mat4 glmProjMat(1.0); double p[16];
		arglCameraFrustumRH(&cparam, 0.1f, 10000.0f, p);
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				glmProjMat[i][j] = p[i * 4 + j];
				//cout << glmProjMat[i][j] << ",";
			}
			//cout << ";\n";
		}
		glm::mat4 Projection = glmProjMat;
		// カメラ行列
		glm::mat4 View = glm::mat4(1.0)
			* glm::lookAt(
			glm::vec3(0, 0, 0), // カメラの原点
			glm::vec3(0, 0, 1), // 見ている点
			glm::vec3(0, -1, 0)  // カメラの上方向
			);
		// モデル行列：単位行列(モデルは原点にあります。)
		glm::mat4 marker2model = glm::mat4(1.0)
			* glm::scale(glm::vec3(1.0, 1.0, 1.0))
			* glm::translate(-glm::vec3(MARKER_SIZE/2 + 11.0f, -171.6f + MARKER_SIZE/2 + 11.0f, 0));
			//* glm::rotate(glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0))
			//* glm::translate(glm::vec3(MARKER_SIZE / 2 + 11.0f + 150.0f, MARKER_SIZE / 2 + 11.0f, 105.6f));
		glm::mat4 Model;  // 各モデルを変える！
		Model = glm::mat4(1.0f)
			* markerTransMat
			* marker2model;
			//* glm::translate(glm::vec3(0.0, 10.0, 0.0))
			//* glm::rotate(angle, glm::vec3(1.0, 1.0, 0.0));
		
		//	Render Object
		//	Our ModelViewProjection : multiplication of our 3 matrices
		mainRenderer.shader.enable();
		mainRenderer.MV = View * Model;
		mainRenderer.MVP = Projection * mainRenderer.MV;
		mainRenderer.lightDirection = glm::vec3(markerTransMat[3]) - glm::vec3(projectorPose[3]);
		mainRenderer.lightColor = glm::vec3(1.0, 1.0, 1.0);
		
		//	Execute Rendering
		renderObject(mainRenderer);

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
		
		//ARParam projParam;
		//projParam.xsize = 1024; projParam.ysize = 768;
		//for (int i = 0; i < 4; i++)
		//	projParam.dist_factor[i] = 0.0;
		//for (int i = 0; i < 4; i++){
		//	for (int j = 0; j < 3; j++)
		//		projParam.mat[i][j] = projectorCameraMatrix[i][j];
		//}
		//double projmat[16];
		//arglCameraFrustumRH(&projParam, 0.001, 10000.0, projmat);
		//glm::mat4 glmProjMat;
		//for (int i = 0; i < 4; i++){
		//	for (int j = 0; j < 4; j++)
		//		glmProjMat[i][j] = projmat[i + j * 4];
		//}
		ARParam subparam;
		//arParamChangeSize(&cparam, colorImg.cols, colorImg.rows, &subparam);
		subparam.xsize = subWinW; subparam.ysize = subWinH;
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 3; j++){
				if (i == 3) subparam.mat[j][i] = 0.0;
				else subparam.mat[j][i] = projectorCameraMatrix[i][j];
			}
			subparam.dist_factor[i] = 0.0;
		}
		//arParamDisp(&subparam);
		arglCameraFrustumRH(&subparam, 0.1f, 10000.0f, p);
		for (int i = 0; i < 4; i++){
			for (int j = 0; j < 4; j++){
				glmProjMat[i][j] = p[i * 4 + j];
				//cout << glmProjMat[i][j] << ",";
			}
			//cout << ";\n";
		}
		Projection = glmProjMat;
		//Projection = projectionMatfromCameraMatrix(projectorCameraMatrix, subWinW, subWinH, 0.1, 10000.0);
		//	カメラを原点としたプロジェクタ位置姿勢
		View = glm::mat4(1.0)
			* glm::lookAt(
			glm::vec3(0, 0, 0), // カメラの原点
			glm::vec3(0, 0, 1), // 見ている点
			glm::vec3(0, -1, 0)  // カメラの上方向
			)
			* glm::inverse(projectorPose)
			//* glm::translate(glm::vec3(objTx, objTy, objTz))
			//* glm::mat4_cast(current)
			* glm::translate(glm::vec3(116.699, 549.479, -105.775))
			* glm::mat4_cast(glm::quat(-0.940652, -0.338357, 0.0235807, 0.0112587))
			;
			//* projectorPose;

		//	カメラを原点としたワールド座標系
		Model = glm::mat4(1.0)
			* markerTransMat
			* marker2model;
			//* glm::mat4_cast(current)
			//* glm::translate(glm::vec3(objTx, objTy, objTz));

		subRenderer.MV = View * Model;
		subRenderer.MVP = Projection * View * Model;
		subRenderer.lightDirection = glm::vec3(markerTransMat[3]) - glm::vec3(projectorPose[3]);
		subRenderer.lightColor = glm::vec3(1.0, 1.0, 1.0);
		
		renderObject(subRenderer);

		// Swap buffers
		glfwSwapBuffers(subWindow);
		glfwPollEvents();

	}
	safeTerminate();

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

//	Camera MatrixからOpenGLの透視投影行列を作成
glm::mat4 projectionMatfromCameraMatrix(glm::mat3 cameraMat, int winW, int winH, double znear, double zfar)
{
	//	Load camera parameters
	float fx = cameraMat[0][0];
	float fy = cameraMat[1][1];
	float cx = cameraMat[2][0];
	float cy = cameraMat[2][1];
	glm::mat4 projection(
		-2.0 * fx / winW, 0, 0, 0,
		0, -2.0 * fy / winH, 0, 0,
		2.0 * cx / winW - 1.0, 2.0 * cy / winH - 1.0, -(zfar + znear) / (zfar - znear), -1.0,
		0, 0, -2.0 * zfar * znear / (zfar - znear), 0);
	return projection;
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
	double dx = -xDisp / subWinW / 2.0;
	double dy = yDisp / subWinH / 2.0;
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
		}
		break;
	case GLFW_MOUSE_BUTTON_MIDDLE:
		objTx += dx * 1000;
		objTy += dy * 1000;
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
	safeRelease(kinect.colorDescription);
	safeRelease(kinect.colorReader);
	safeRelease(kinect.colorSource);
	safeRelease(kinect.kinect);
	glfwTerminate();
}

void arglCameraFrustumRH(ARParam *cparam, const double focalmin, const double focalmax, GLdouble m_projection[16])
{
	double   icpara[3][4];
	double   trans[3][4];
	double   p[3][3], q[4][4];
	int      width, height;
	int      i, j;

	width = cparam->xsize;
	height = cparam->ysize;

	if (arParamDecompMat(cparam->mat, icpara, trans) < 0) {
		printf("arglCameraFrustum(): arParamDecompMat() indicated parameter error.\n"); // Windows bug: when running multi-threaded, can't write to stderr!
		return;
	}
	for (i = 0; i < 4; i++) {
		icpara[1][i] = (height - 1)*(icpara[2][i]) - icpara[1][i];
	}

	for (i = 0; i < 3; i++) {
		for (j = 0; j < 3; j++) {
			p[i][j] = icpara[i][j] / icpara[2][2];
		}
	}
	q[0][0] = (2.0 * p[0][0] / (width - 1));
	q[0][1] = (2.0 * p[0][1] / (width - 1));
	q[0][2] = -((2.0 * p[0][2] / (width - 1)) - 1.0);
	q[0][3] = 0.0;

	q[1][0] = 0.0;
	q[1][1] = -(2.0 * p[1][1] / (height - 1));
	q[1][2] = -((2.0 * p[1][2] / (height - 1)) - 1.0);
	q[1][3] = 0.0;

	q[2][0] = 0.0;
	q[2][1] = 0.0;
	q[2][2] = (focalmax + focalmin) / (focalmin - focalmax);
	q[2][3] = 2.0 * focalmax * focalmin / (focalmin - focalmax);

	q[3][0] = 0.0;
	q[3][1] = 0.0;
	q[3][2] = -1.0;
	q[3][3] = 0.0;

	for (i = 0; i < 4; i++) { // Row.
		// First 3 columns of the current row.
		for (j = 0; j < 3; j++) { // Column.
			m_projection[i + j * 4] = q[i][0] * trans[0][j] +
				q[i][1] * trans[1][j] +
				q[i][2] * trans[2][j];
		}
		// Fourth column of the current row.
		m_projection[i + 3 * 4] = q[i][0] * trans[0][3] +
			q[i][1] * trans[1][3] +
			q[i][2] * trans[2][3] +
			q[i][3];
	}
}
