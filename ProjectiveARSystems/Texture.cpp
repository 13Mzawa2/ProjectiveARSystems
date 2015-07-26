#include <iostream>
#include <OpenCVAdapter.hpp>
#include "OpenGLHeader.h"
#include "Shader.h"
#include "objloader.hpp"
#include "OBJRenderer.h"

using namespace cv;

GLFWwindow	*mainWindow, *subWindow;		//	マルチウィンドウ
Shader		shader;			//	シェーダークラス
OBJRenderer renderer;		//	objファイルのレンダリングに必要な関数まとめ

int initWindow(void);


int initWindow(void)
{
	//	GLFWの初期化
	if (glfwInit() != GL_TRUE)
	{
		fprintf(stderr, "Failed to initialize GLFW\n");
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
		fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
		glfwTerminate();
		return EXIT_FAILURE;
	}
	//// Sub Windowの用意
	//subWindow = glfwCreateWindow(1024, 768, "Sub Window", NULL, NULL);
	//if (subWindow == NULL){
	//	fprintf(stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
	//	glfwTerminate();
	//	return EXIT_FAILURE;
	//}
	glfwMakeContextCurrent(mainWindow);				//	main windowをカレントにする

	//	キー入力を受け付けるようにする
	glfwSetInputMode(mainWindow, GLFW_STICKY_KEYS, GL_TRUE);

	// Initialize GLEW
	glewExperimental = true;	// Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return EXIT_FAILURE;
	}
	return 0;
}

int main(void)
{
	initWindow();

	glClearColor(0.8, 0.8, 0.6, 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_LESS);				//	カメラに近い面だけレンダリングする

	//	プログラマブルシェーダをロード
	shader.initGLSL("shader/vertex.glsl", "shader/fragment.glsl");
	//	.objファイルを読み込みます。
	Object obj;
	bool res = loadOBJ("model/drop_modified_x004.obj", obj);
	//	テクスチャ画像を読み込む
	Mat texImg = imread("model/textures/txt_001_diff.bmp");
	flip(texImg, texImg, 1);
	//	新たに作られたテクスチャを"バインド"します。つまりここから後のテクスチャ関数はこのテクスチャを変更します。
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);

	//	OpenGLに画像を渡します。
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texImg.cols, texImg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, texImg.data);
	//	テクスチャの繰り返し設定
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	//	画像を拡大(MAGnifying)するときは線形(LINEAR)フィルタリングを使用
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//	画像を縮小(MINifying)するとき、線形(LINEAR)フィルタした、二つのミップマップを線形(LINEARYLY)に混ぜたものを使用
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	//	ミップマップを作成
	glGenerateMipmap(GL_TEXTURE_2D);

	//	Vertex Array IDを設定
	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	//	uniformへのハンドルを取得
	//	初期化時だけ
	GLuint MatrixID = glGetUniformLocation(shader.program, "MVP");		//	vertex.shader内の uniform mat4 MVP; へのID
	GLuint NormalMatID = glGetUniformLocation(shader.program, "MV");
	GLuint textureID = glGetUniformLocation(shader.program, "myTextureSampler");	// ひとつのOpenGLテクスチャを作ります。
	GLuint LightDirectionID = glGetUniformLocation(shader.program, "LightDirection");
	GLuint LightColorID = glGetUniformLocation(shader.program, "LightColor");

	//	頂点バッファをOpenGLに渡す
	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);							//	バッファを1つ作成
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);			//	以降のコマンドをvertexbufferバッファに指定
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * obj.vertices.size() , &obj.vertices[0], GL_STATIC_DRAW);		//	頂点をOpenGLのvertexbuferに渡す

	//	UV座標バッファ
	GLuint uvbuffer;
	glGenBuffers(1, &uvbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * obj.uvs.size(), &obj.uvs[0], GL_STATIC_DRAW);

	//	法線バッファ
	GLuint normalbuffer;
	glGenBuffers(1, &normalbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * obj.normals.size(), &obj.normals[0], GL_STATIC_DRAW);

	//	メインループ
	while (glfwGetKey(mainWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS		//	Escキー
		&& !glfwWindowShouldClose(mainWindow))							//	ウィンドウの閉じるボタン
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(shader.program);			//	シェーダを利用
		glEnable(GL_DEPTH_TEST);
		//glDepthFunc(GL_LESS);			// 前のものよりもカメラに近ければ、フラグメントを受け入れる

		// 射影行列：45°の視界、アスペクト比4:3、表示範囲：0.1単位  100単位
		glm::mat4 Projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 10000.0f);
		//glm::mat4x4 Projection = glm::ortho(-5.0f, 5.0f, -5.0f, 5.0f, 0.1f, 100.0f);
		// カメラ行列
		glm::mat4 View = glm::lookAt(
			glm::vec3(40, 500, 30), // ワールド空間でカメラは(4,3,3)にあります。
			glm::vec3(0, 0, 0), // 原点を見ています。
			glm::vec3(0, 1, 0)  // 頭が上方向(0,-1,0にセットすると上下逆転します。)
			);
		// モデル行列：単位行列(モデルは原点にあります。)
		glm::mat4 Model = glm::mat4(1.0f);  // 各モデルを変える！
		static float angle = 0.0f;
		angle += 0.1f;
		if (angle >= 360.0) angle -= 360.0;
		Model = glm::translate(glm::vec3(0.0, 0.0, 0.0))
			* glm::rotate(angle, glm::vec3(0.0, 0.0, 1.0))
			//* glm::rotate(90.0f, glm::vec3(0.0, 1.0, 0.0))
			* Model;
		// Our ModelViewProjection : multiplication of our 3 matrices
		glm::mat4 MV = View * Model;
		glm::mat4 MVP = Projection * MV; // 行列の掛け算は逆になることを思い出してください。
		glm::vec3 lightDir = vec3(500.0, 200.0, 100.0);
		glm::vec3 lightColor = vec3(1.0, 1.0, 1.0);

		// 現在バインドしているシェーダの"MVP" uniformに変換を送る
		// レンダリングする各モデルごとに実行
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);
		glUniformMatrix4fv(NormalMatID, 1, GL_FALSE, &MV[0][0]);
		glUniform3fv(LightDirectionID, 1, &lightDir[0]);
		glUniform3fv(LightColorID, 1, &lightColor[0]);

		// Bind our texture in Texture Unit 0
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		// Set our "myTextureSampler" sampler to user Texture Unit 0
		glUniform1i(textureID, 0);

		// 最初の属性バッファ：頂点
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
		glVertexAttribPointer(
			0,                  // 属性0：0に特に理由はありません。しかし、シェーダ内のlayoutとあわせないといけません。
			3,                  // サイズ
			GL_FLOAT,           // タイプ
			GL_FALSE,           // 正規化？
			0,                  // ストライド
			(void*)0            // 配列バッファオフセット
			);
		// 2番目の属性バッファ : UV
		glEnableVertexAttribArray(1);
		glBindBuffer(GL_ARRAY_BUFFER, uvbuffer);
		glVertexAttribPointer(
			1,                                // 属性。1という数字に意味はありません。ただしシェーダのlayoutとあわせる必要があります。
			2,                                // サイズ
			GL_FLOAT,                         // タイプ
			GL_FALSE,                         // 正規化？
			0,                                // ストライド
			(void*)0                          // 配列バッファオフセット
			);
		// 3番目の属性バッファ : 法線
		glEnableVertexAttribArray(2);
		glBindBuffer(GL_ARRAY_BUFFER, normalbuffer);
		glVertexAttribPointer(
			2,                                // 属性。1という数字に意味はありません。ただしシェーダのlayoutとあわせる必要があります。
			3,                                // サイズ
			GL_FLOAT,                         // タイプ
			GL_FALSE,                         // 正規化？
			0,                                // ストライド
			(void*)0                          // 配列バッファオフセット
			);


		// 三角形を描きます！
		glDrawArrays(GL_TRIANGLES, 0, obj.vertices.size()); // 頂点0から始まります。合計3つの頂点です。→1つの三角形です。

		glDisableVertexAttribArray(0);
		glDisableVertexAttribArray(1);
		glDisableVertexAttribArray(2);

		// Swap buffers
		glfwSwapBuffers(mainWindow);
		glfwPollEvents();

	}

	glfwTerminate();


	return EXIT_SUCCESS;
}