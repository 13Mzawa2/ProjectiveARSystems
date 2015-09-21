#include <iostream>
#include "OpenCV3Linker.h"
#include "OpenGLHeader.h"
#include "Shader.h"
#include "objloader.hpp"
#include "OBJRenderer.h"

using namespace cv;
using namespace std;

const char vertexDir[] = "shader/Vertex.glsl";
const char fragmentDir[] = "shader/Fragment.glsl";
const char objDir[] = "../common/data/model/drop_modified_x004.obj";
const char textureDir[] = "../common/data/model/textures/txt_001_diff.bmp";
const char lutDir[] = "../common/data/lut/LUT_dichromat_typeP.png";

GLFWwindow	*mainWindow, *subWindow;		//	マルチウィンドウ
OBJRenderer renderer, subRenderer;		//	objファイルのレンダリングに必要なものまとめ
Mat			texImg;
int subWinW, subWinH;
static float objTx = 0, objTy = 0, objTz = 0;
static glm::quat current = quat(0.292, -0.80, 0.523, 0.161);
double xBegin, yBegin;
int pressedMouseButton = 0;

int initWindow(void);
void initMainWindow(void);
void initSubWindow(void);
void mouseEvent(GLFWwindow *window, int button, int state, int optionkey);
void cursorPosEvent(GLFWwindow *window, double x, double y);
void scrollEvent(GLFWwindow *window, double xofset, double yofset);
mat4x4 projectionMatfromCameraMatrix()

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
	mainWindow = glfwCreateWindow(1024, 768, "Main Window", NULL, NULL);
	if (mainWindow == NULL){
		cerr << "GLFWウィンドウの生成に失敗しました. Intel GPUを使用している場合は, OpenGL 3.3と相性が良くないため，2.1を試してください．\n";
		glfwTerminate();
		return EXIT_FAILURE;
	}

	// Sub Windowの用意
	int monitorCount;
	GLFWmonitor **monitors = glfwGetMonitors(&monitorCount);
	subWindow = glfwCreateWindow(1024, 768, "Sub Window", monitors[1], NULL);
	if (subWindow == NULL){
		cerr << "GLFWウィンドウの生成に失敗しました. Intel GPUを使用している場合は, OpenGL 3.3と相性が良くないため，2.1を試してください．\n";
		glfwTerminate();
		return EXIT_FAILURE;
	}
	glfwGetWindowSize(subWindow ,&subWinW, &subWinH);

	//	キー入力を受け付けるようにする
	glfwSetInputMode(mainWindow, GLFW_STICKY_KEYS, GL_TRUE);

	//	マウス操作を可能にする
	glfwSetMouseButtonCallback(subWindow, mouseEvent);
	glfwSetCursorPosCallback(subWindow, cursorPosEvent);
	glfwSetScrollCallback(subWindow, scrollEvent);

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
	renderer.loadShader(vertexDir, fragmentDir);
	renderer.getUniformID();

	//	.objファイルを読み込みます。
	renderer.loadObject(objDir);
	renderer.setObjData();

	//	テクスチャ画像を読み込む
	renderer.loadTexture(texImg);
	renderer.setupTexture();
	
}

void initSubWindow(void)
{
	//	Sub Window Setting
	glfwMakeContextCurrent(subWindow);				//	sub windowをカレントにする

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LESS);				//	カメラに近い面だけレンダリングする

	//subRenderer.clone();			//	設定の使い回ししようと思ったけどうまくいかない

	//	プログラマブルシェーダをロード
	subRenderer.loadShader(vertexDir, fragmentDir);
	subRenderer.getUniformID();

	//	.objファイルを読み込みます。
	subRenderer.loadObject(objDir);
	subRenderer.setObjData();

	//	テクスチャ画像を読み込む
	subRenderer.loadTexture(texImg);
	subRenderer.setupTexture();
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
	double dx = xDisp / subWinW / 2.0;
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
			after = quat(cos(rad), -sin(rad) * dy / length, sin(rad) * dx / length, 0.0);
			current = after * current;
		}
		break;
	case GLFW_MOUSE_BUTTON_MIDDLE:
		objTx += dx * 100;
		objTy += dy * 100;
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

int main(void)
{
	initWindow();

	texImg = imread(textureDir);
	if (texImg.empty())
	{
		cerr << "テクスチャの読み込みに失敗しました．ディレクトリを確認してください．\n"
			<< "ディレクトリの場所：" << textureDir << endl;
		glfwTerminate();
		return EXIT_FAILURE;
	}
	flip(texImg, texImg, 1);
	initMainWindow();
	initSubWindow();

	//	メインループ
	while (glfwGetKey(mainWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS		//	Escキー
		&& !glfwWindowShouldClose(mainWindow))							//	ウィンドウの閉じるボタン
	{
		//------------------------------
		//	Main Winodw
		//------------------------------
		glfwMakeContextCurrent(mainWindow);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		

		// 射影行列：45°の視界、アスペクト比4:3、表示範囲：0.1単位  100単位
		glm::mat4 Projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 10000.0f);
		// カメラ行列
		glm::mat4 View = glm::lookAt(
			glm::vec3(40, 500, 30), // ワールド空間でカメラは(4,3,3)にあります。
			glm::vec3(0, 0, 0), // 原点を見ています。
			glm::vec3(0, 0, 1)  // 頭が上方向(0,-1,0にセットすると上下逆転します。)
			);
		// モデル行列：単位行列(モデルは原点にあります。)
		glm::mat4 Model;  // 各モデルを変える！
		static float angle = 0.0f;
		angle += 0.1f;
		if (angle >= 360.0) angle -= 360.0;
		Model = glm::translate(glm::vec3(0.0, 0.0, 0.0))
			* glm::rotate(angle, glm::vec3(0.0, 0.0, 1.0))
			* glm::mat4(1.0f);
		
		//	Render Object
		//	Our ModelViewProjection : multiplication of our 3 matrices
		renderer.shader.enable();
		renderer.MV = View * Model;
		renderer.MVP = Projection * renderer.MV; // 行列の掛け算は逆になることを思い出してください。
		renderer.lightDirection = glm::vec3(200.0, 500.0, 100.0);
		renderer.lightColor = glm::vec3(1.0, 1.0, 1.0);
		renderer.render();

		glfwSwapBuffers(mainWindow);

		//--------------------------------
		//	Sub Window
		//--------------------------------
		glfwMakeContextCurrent(subWindow);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//	画像処理(Hueを時間で更新)
		Mat temp;
		cvtColor(texImg, temp, CV_BGR2HSV);
		for (int i = 0; i < temp.rows; i++)
		{
			for (int j = 0; j < temp.cols; j++)
			{
				float hue = matB(temp, j, i) + angle;
				matB(temp, j, i) = (int)hue;
				if (matB(temp, j, i) > 180)
				{
					matB(temp, j, i) = 0;
					hue = 0.0;
				}
			}
		}
		cvtColor(temp, temp, CV_HSV2BGR);
		subRenderer.updateTexture(temp);

		//	Render Object
		// Our ModelViewProjection : multiplication of our 3 matrices
		subRenderer.shader.enable();
		//	RoomAliveToolkitでの測定結果
		mat3x3 cameraMatrix(
			2631.8585662663163, 0, 0,
			0, 2631.8585662663163, 0,
			537.97898525592848, 81.662059249010937, 1);
		mat4x4 projectorPose(
			0.99881625175476074, -0.013360070064663887, 0.046772114932537079, 0,
			-0.0026347176171839237, 0.94527167081832886, 0.32627373933792114, 0,
			-0.0485713928937912, -0.32601076364517212, 0.94411748647689819, 0,
			0.064024697539966549, 0.32449587982305783, -0.83255910737444971, 1);
		
		//Projection = glm::perspective(24.0f, (float)subWinW / (float)subWinH, 0.1f, 10000.0f);
		View = glm::lookAt(
			glm::vec3(0, 0, 0),				//	カメラ座標を原点とする
			glm::vec3(0, 0, 1),				//	画面の奥はZ軸
			glm::vec3(0, 1, 0)				//	画面の上はY軸
			);

		Model = 
			 glm::translate(glm::vec3(objTx, objTy, objTz))
			* glm::translate(glm::vec3(-21.5, 119.0, 630.0))
			* glm::mat4_cast(current)
			* glm::mat4(1.0);
		subRenderer.MV = View * Model;
		subRenderer.MVP = Projection * subRenderer.MV; // 行列の掛け算は逆になることを思い出してください。
		subRenderer.lightDirection = glm::vec3(10.0, 0.0, 10.0);
		subRenderer.lightColor = glm::vec3(1.0, 1.0, 1.0);
		subRenderer.render();

		// Swap buffers
		glfwSwapBuffers(subWindow);
		glfwPollEvents();

	}

	glfwTerminate();


	return EXIT_SUCCESS;
}