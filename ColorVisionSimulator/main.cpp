#include <iostream>
#include "OpenCV3Linker.h"
#include "OpenGLHeader.h"
#include "Shader.h"
#include "objloader.hpp"

using namespace cv;
using namespace std;

#define PROJ_WIN_ID 2

const char vertexDir[] = "./shader/vertex.glsl";
const char fragmentDir[] = "./shader/fragment.glsl";
const char objDir[] = "../common/data/model/ARbox/ARbox.obj";
const char textureDir[] = "../common/data/model/ARbox/textures/txt_001_diff.bmp";
const char lutDir[] = "../common/data/lut/LUT_dichromat_typeP.png";

//	RoomAliveToolkitでの測定結果
mat3x3 projectorCameraMatrix(
	2163.9840607833007, 0, 0,
	0, 2163.9840607833007, 0,
	384.14708782496359, 24.7008330562985, 1);
mat4x4 projectorPose(
	0.99780523777008057, -0.011467255651950836, 0.06521625816822052, 0,
	-0.020374707877635956, 0.883938729763031, 0.467158704996109, 0,
	-0.063004210591316223, -0.46746215224266052, 0.88176506757736206, 0,
	-0.0015859748836387856, 0.39699462236674066, -0.5658578787135, 1);

GLFWwindow	*mainWindow, *subWindow;		//	マルチウィンドウ
Mat	texImg;
Mat lutMat;

int subWinW, subWinH;
static float objTx = -45.117, objTy = -450.715, objTz = -1185.972;
static glm::quat current = quat(0.098108, 0.071614, -0.434337, 0.892501);
double xBegin, yBegin;
int pressedMouseButton = 0;

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
	mat4 MVP;
	mat4 MV;
	vec3 lightDirection;
	vec3 lightColor;
	bool useLUT = false;
};

Renderer mainRenderer, subRenderer;

int initWindow(void);
void initMainWindow(void);
void initSubWindow(void);
void mouseEvent(GLFWwindow *window, int button, int state, int optionkey);
void cursorPosEvent(GLFWwindow *window, double x, double y);
void scrollEvent(GLFWwindow *window, double xofset, double yofset);
mat4 projectionMatfromCameraMatrix(mat3 cameraMat, int winW, int winH, double near, double far);
void getUniformID(Renderer &r);
void setObjectTexture(Renderer &r, Mat &texture);
void setLUT(Renderer &r, Mat &lut);
void setObjectVertices(Renderer &r);
void renderObject(Renderer &r);


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
	lutMat = imread(lutDir);
	if (lutMat.empty())
	{
		cerr << "LUTの読み込みに失敗しました．ディレクトリを確認してください．\n"
			<< "ディレクトリの場所：" << lutDir << endl;
		glfwTerminate();
		return EXIT_FAILURE;
	}


	initMainWindow();
	initSubWindow();

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
		angle += 0.001f;
		if (angle >= 360.0) angle -= 360.0;
		Model = glm::translate(glm::vec3(0.0, 10.0, 0.0))
			* glm::rotate(angle, glm::vec3(1.0, 1.0, 0.0))
			* glm::mat4(1.0f);
		
		//	Render Object
		//	Our ModelViewProjection : multiplication of our 3 matrices
		mainRenderer.shader.enable();
		mainRenderer.MV = View * Model;
		mainRenderer.MVP = Projection * mainRenderer.MV;
		mainRenderer.lightDirection = glm::vec3(200.0, 500.0, 100.0);
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
		
		Projection = projectionMatfromCameraMatrix(projectorCameraMatrix, subWinW, subWinH, 0.001, 10000.0);
		//	カメラを原点としたプロジェクタ位置姿勢
		View = projectorPose;
		//	カメラを原点としたワールド座標系
		Model = 
			glm::translate(glm::vec3(objTx, objTy, objTz))
			* glm::mat4_cast(current)
			* glm::mat4(1.0);
		subRenderer.MV = View * Model;
		subRenderer.MVP = Projection * subRenderer.MV;
		subRenderer.lightDirection = glm::vec3(0.0, 0.0, 10.0);
		subRenderer.lightColor = glm::vec3(1.0, 1.0, 1.0);
		
		renderObject(subRenderer);

		// Swap buffers
		glfwSwapBuffers(subWindow);
		glfwPollEvents();

	}

	glfwTerminate();


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
mat4 projectionMatfromCameraMatrix(mat3 cameraMat, int winW, int winH, double near, double far)
{
	//	Load camera parameters
	float fx = cameraMat[0][0];
	float fy = cameraMat[1][1];
	float cx = cameraMat[2][0];
	float cy = cameraMat[2][1];
	mat4 projection(
		-2.0 * fx / winW, 0, 0, 0,
		0, -2.0 * fy / winH, 0, 0,
		2.0 * cx / winW - 1.0, 2.0 * cy / winH - 1.0, -(far + near) / (far - near), -1.0,
		0, 0, -2.0 * far * near / (far - near), 0);
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
			after = quat(cos(rad), -sin(rad) * dy / length, sin(rad) * dx / length, 0.0);
			current = after * current;
		}
		break;
	case GLFW_MOUSE_BUTTON_MIDDLE:
		objTx += dx * 100;
		objTy -= dy * 100;
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