
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
//	RoomAliveToolkit�ł̑��茋��
//	Kinect��RGB�J�����p�����[�^
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

GLFWwindow	*mainWindow, *subWindow;		//	�}���`�E�B���h�E
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
	GLuint vertexArray;		//	���_����ێ�����z��
	GLuint vertexBuffer;	//	location = 0
	GLuint uvBuffer;		//	location = 1
	GLuint normalBuffer;	//	location = 2
	GLuint textureObject;	//	�e�N�X�`���ɃA�N�Z�X���邽�߂̃I�u�W�F�N�g
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
	//	GLFW�̏�����
	if (glfwInit() != GL_TRUE)
	{
		cerr << "GLFW�̏������Ɏ��s���܂����D\n";
		return EXIT_FAILURE;
	}
	//	Window�ݒ�
	glfwWindowHint(GLFW_SAMPLES, 4);								//	4x �A���`�G�C���A�X
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);						//	���T�C�Y�s��
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);					//	OpenGL�o�[�W����3.3�𗘗p
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);					//	
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);	//	�Â�OpenGL���g��Ȃ�

	// Main Window�̗p��
	mainWindow = glfwCreateWindow(1920/2, 1080/2, "Main Window", NULL, NULL);
	if (mainWindow == NULL){
		cerr << "GLFW�E�B���h�E�̐����Ɏ��s���܂���. Intel GPU���g�p���Ă���ꍇ��, OpenGL 3.3�Ƒ������ǂ��Ȃ����߁C2.1�������Ă��������D\n";
		glfwTerminate();
		return EXIT_FAILURE;
	}

	// Sub Window�̗p��
	int monitorCount;
	GLFWmonitor **monitors = glfwGetMonitors(&monitorCount);
	subWindow = glfwCreateWindow(1024, 768, "Sub Window", monitors[PROJ_WIN_ID], NULL);
	if (subWindow == NULL){
		cerr << "GLFW�E�B���h�E�̐����Ɏ��s���܂���. Intel GPU���g�p���Ă���ꍇ��, OpenGL 3.3�Ƒ������ǂ��Ȃ����߁C2.1�������Ă��������D\n";
		glfwTerminate();
		return EXIT_FAILURE;
	}
	glfwGetWindowSize(subWindow ,&subWinW, &subWinH);
	glfwMakeContextCurrent(mainWindow);

	// Initialize GLEW
	glewExperimental = true;	// Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "GLEW�̏������Ɏ��s���܂����D\n");
		return EXIT_FAILURE;
	}
	return 0;
}

void initMainWindow(void)
{
	//	Main Window Setting
	glfwMakeContextCurrent(mainWindow);				//	main window���J�����g�ɂ���

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LESS);				//	�J�����ɋ߂��ʂ��������_�����O����

	//	�v���O���}�u���V�F�[�_�����[�h
	mainRenderer.shader.initGLSL(vertexDir, fragmentDir);
	//	Uniform�ϐ��ւ̃n���h�����擾
	getUniformID(mainRenderer);

	//	.obj�t�@�C����ǂݍ���
	loadOBJ(objDir, mainRenderer.obj);
	////	OBJ�t�@�C���̒��g���o�b�t�@�ɓ]��
	setObjectVertices(mainRenderer);

	//	�e�N�X�`���摜��ǂݍ���
	setObjectTexture(mainRenderer, texImg);

	//	LUT��ǂݍ���
	setLUT(mainRenderer, lutMat);
}

void initSubWindow(void)
{
	//	Sub Window Setting
	glfwMakeContextCurrent(subWindow);				//	sub window���J�����g�ɂ���

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LESS);				//	�J�����ɋ߂��ʂ��������_�����O����

	//	�v���O���}�u���V�F�[�_�����[�h
	subRenderer.shader.initGLSL(vertexDir, fragmentDir);
	//	Uniform�ϐ��ւ̃n���h�����擾
	getUniformID(subRenderer);

	//	.obj�t�@�C����ǂݍ���
	loadOBJ(objDir, subRenderer.obj);
	////	OBJ�t�@�C���̒��g���o�b�t�@�ɓ]��
	setObjectVertices(subRenderer);

	//	�e�N�X�`���摜��ǂݍ���
	setObjectTexture(subRenderer, texImg);

	//	LUT��ǂݍ���
	setLUT(subRenderer, lutMat);

	//	�L�[���͂��󂯕t����悤�ɂ���
	glfwSetInputMode(subWindow, GLFW_STICKY_KEYS, GL_TRUE);
	//	�}�E�X������\�ɂ���
	glfwSetMouseButtonCallback(subWindow, mouseEvent);
	glfwSetCursorPosCallback(subWindow, cursorPosEvent);
	glfwSetScrollCallback(subWindow, scrollEvent);
}

void initARTK(void)
{

	//	�p�^�[���t�@�C���̃��[�h
	if ((marker.patt_id = arLoadPatt(marker.patt_name)) < 0)
	{
		cout << "�p�^�[���t�@�C�����ǂݍ��߂܂���ł����D�f�B���N�g�����m�F���Ă��������D\n"
			<< "�ꏊ�F" << marker.patt_name << endl;
		safeTerminate();
		exit(-1);
	}
	//	�J�����p�����[�^�̃��[�h
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
	//	�J�����̏���
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
		cerr << "�e�N�X�`���̓ǂݍ��݂Ɏ��s���܂����D�f�B���N�g�����m�F���Ă��������D\n"
			<< "�f�B���N�g���̏ꏊ�F" << textureDir << endl;
		glfwTerminate();
		return EXIT_FAILURE;
	}
	lutMat = imread(lutDir[0]);
	if (lutMat.empty())
	{
		cerr << "LUT�̓ǂݍ��݂Ɏ��s���܂����D�f�B���N�g�����m�F���Ă��������D\n"
			<< "�f�B���N�g���̏ꏊ�F" << lutDir << endl;
		glfwTerminate();
		return EXIT_FAILURE;
	}


	initMainWindow();
	initSubWindow();

	initARTK();

	//	���C�����[�v
	while (glfwGetKey(mainWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS		//	Esc�L�[
		&& glfwGetKey(subWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS
		&& !glfwWindowShouldClose(mainWindow))							//	�E�B���h�E�̕���{�^��
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
		//	�J�����摜�擾
		kinect.getColorFrame(colorImg);
		safeRelease(kinect.colorFrame);
		//	ARToolKit��2�l���摜��]��
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
		
		//	AR�}�[�J�[��F��
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
			{	//	marker�ƍł���v�x�̍���ID�𒊏o
				if (k == -1) k = j;
				else if (markerInfo[k].cf < markerInfo[j].cf) k = j;
			}
		}
		//	�}�[�J�[�ʒu�p�����擾
		glm::mat4 markerTransMat(1.0f);		//	�ŏ��͒P�ʍs��
		if (k == -1) marker.visible = 0;
		else
		{	//	�ߋ����𗘗p���ău����}����
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

		// �ˉe�s��F45���̎��E�A�A�X�y�N�g��4:3�A�\���͈́F0.1�P��  100�P��
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
		// �J�����s��
		glm::mat4 View = glm::mat4(1.0)
			* glm::lookAt(
			glm::vec3(0, 0, 0), // �J�����̌��_
			glm::vec3(0, 0, 1), // ���Ă���_
			glm::vec3(0, -1, 0)  // �J�����̏����
			);
		// ���f���s��F�P�ʍs��(���f���͌��_�ɂ���܂��B)
		glm::mat4 marker2model = glm::mat4(1.0)
			* glm::scale(glm::vec3(1.0, 1.0, 1.0))
			* glm::translate(-glm::vec3(MARKER_SIZE/2 + 11.0f, -171.6f + MARKER_SIZE/2 + 11.0f, 0));
			//* glm::rotate(glm::radians(180.0f), glm::vec3(0.0, 0.0, 1.0))
			//* glm::translate(glm::vec3(MARKER_SIZE / 2 + 11.0f + 150.0f, MARKER_SIZE / 2 + 11.0f, 105.6f));
		glm::mat4 Model;  // �e���f����ς���I
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

		//	�`�挋�ʂ𔽉f
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
		//	�J���������_�Ƃ����v���W�F�N�^�ʒu�p��
		View = glm::mat4(1.0)
			* glm::lookAt(
			glm::vec3(0, 0, 0), // �J�����̌��_
			glm::vec3(0, 0, 1), // ���Ă���_
			glm::vec3(0, -1, 0)  // �J�����̏����
			)
			* glm::inverse(projectorPose)
			//* glm::translate(glm::vec3(objTx, objTy, objTz))
			//* glm::mat4_cast(current)
			* glm::translate(glm::vec3(116.699, 549.479, -105.775))
			* glm::mat4_cast(glm::quat(-0.940652, -0.338357, 0.0235807, 0.0112587))
			;
			//* projectorPose;

		//	�J���������_�Ƃ������[���h���W�n
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
	//	�e�N�X�`���摜��ǂݍ���
	glGenTextures(1, &r.textureObject);
	glBindTexture(GL_TEXTURE_2D, r.textureObject);
	//	OpenGL�ɉ摜��n��
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
		texture.cols, texture.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, texture.data);
	//	�e�N�X�`���̌J��Ԃ��ݒ�
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	//	�摜���g��(MAGnifying)����Ƃ��͐��`(LINEAR)�t�B���^�����O���g�p
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//	�摜���k��(MINifying)����Ƃ��A���`(LINEAR)�t�B���^�����A��̃~�b�v�}�b�v����`(LINEARYLY)�ɍ��������̂��g�p
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	//	�~�b�v�}�b�v���쐬
	glGenerateMipmap(GL_TEXTURE_2D);
}

void setLUT(Renderer &r, Mat &lut)
{
	//	�e�N�X�`���摜��ǂݍ���
	glGenTextures(1, &r.lutBuffer);
	glBindTexture(GL_TEXTURE_3D, r.lutBuffer);
	//	OpenGL�ɉ摜��n��
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB,
		256, 256, 256, 0, GL_BGR, GL_UNSIGNED_BYTE, lut.data);
	//	�e�N�X�`���̊g��k���ɐ��`��Ԃ��g�p
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
}

void updateLUT(Renderer &r, Mat &lut)
{
	glBindTexture(GL_TEXTURE_3D, r.lutBuffer);
	//	OpenGL�ɉ摜��n��
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB,
		256, 256, 256, 0, GL_BGR, GL_UNSIGNED_BYTE, lut.data);
}

void setObjectVertices(Renderer &r)
{
	//	���_�z��I�u�W�F�N�g��ݒ�
	glGenVertexArrays(1, &r.vertexArray);
	glBindVertexArray(r.vertexArray);

	//	���_�o�b�t�@��OpenGL�ɓn��
	glGenBuffers(1, &r.vertexBuffer);							//	�o�b�t�@��1�쐬
	glBindBuffer(GL_ARRAY_BUFFER, r.vertexBuffer);			//	�ȍ~�̃R�}���h��vertexbuffer�o�b�t�@�Ɏw��
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * r.obj.vertices.size(), &(r.obj).vertices[0], GL_STATIC_DRAW);		//	���_��OpenGL��vertexbufer�ɓn��

	//	UV���W�o�b�t�@
	glGenBuffers(1, &r.uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, r.uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * r.obj.uvs.size(), &(r.obj).uvs[0], GL_STATIC_DRAW);

	//	�@���o�b�t�@
	glGenBuffers(1, &r.normalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, r.normalBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * r.obj.normals.size(), &(r.obj).normals[0], GL_STATIC_DRAW);
}

void renderObject(Renderer &r)
{
	////	Execute Rendering
	// ���݃o�C���h���Ă���V�F�[�_��uniform�ϐ��ɕϊ��𑗂�
	// �����_�����O���郂�f�����ƂɎ��s
	glUniformMatrix4fv(r.mvpID, 1, GL_FALSE, &r.MVP[0][0]);
	glUniformMatrix4fv(r.mvID, 1, GL_FALSE, &r.MV[0][0]);
	glUniform3fv(r.lightDirectionID, 1, &r.lightDirection[0]);
	glUniform3fv(r.lightColorID, 1, &r.lightColor[0]);
	glUniform1i(r.lutSwitchID, r.useLUT);

	//	�e�N�X�`�����j�b�g0��textureBuffer���o�C���h
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, r.textureObject);
	//	0�Ԗڂ̃e�N�X�`�����j�b�g��"myTextureSampler"�ɃZ�b�g
	glUniform1i(r.textureSamplerID, 0);

	//	�e�N�X�`�����j�b�g1��lutBuffer���o�C���h
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_3D, r.lutBuffer);
	glUniform1i(r.lutSamplerID, 1);

	//	�ŏ��̑����o�b�t�@�F���_
	glEnableVertexAttribArray(GLSL_LOCATION_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, r.vertexBuffer);
	glVertexAttribPointer(
		GLSL_LOCATION_VERTEX,	// shader����location
		3,						// �v�f�T�C�Y
		GL_FLOAT,				// �v�f�̌^
		GL_FALSE,				// ���K���H
		0,						// �X�g���C�h
		(void*)0				// �z��o�b�t�@�I�t�Z�b�g
		);
	//	2�Ԗڂ̑����o�b�t�@ : UV
	glEnableVertexAttribArray(GLSL_LOCATION_UV);
	glBindBuffer(GL_ARRAY_BUFFER, r.uvBuffer);
	glVertexAttribPointer(GLSL_LOCATION_UV, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	//	3�Ԗڂ̑����o�b�t�@ : �@��
	glEnableVertexAttribArray(GLSL_LOCATION_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, r.normalBuffer);
	glVertexAttribPointer(GLSL_LOCATION_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	//	�O�p�`�|���S����`��
	glDrawArrays(GL_TRIANGLES, 0, r.obj.vertices.size());
	//	�`���Ƀo�b�t�@���N���A
	glDisableVertexAttribArray(GLSL_LOCATION_VERTEX);
	glDisableVertexAttribArray(GLSL_LOCATION_UV);
	glDisableVertexAttribArray(GLSL_LOCATION_NORMAL);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindTexture(GL_TEXTURE_3D, 0);
}

//	Camera Matrix����OpenGL�̓������e�s����쐬
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
	double length = ::sqrt(dx*dx + dy*dy);	//	�N�H�[�^�j�I���̒���
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
