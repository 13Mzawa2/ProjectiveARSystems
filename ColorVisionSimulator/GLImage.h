//	OpenGL/GLSLを利用した画像描画


#include "OpenGLHeader.h"
#include "Shader.h"
#include <opencv2\opencv.hpp>

class GLImage
{
private:
	GLFWwindow *imgWindow;
	GLuint vao;		//	頂点配列オブジェクト
	GLuint vbo;		//	頂点バッファオブジェクト
	GLuint image;	//	テクスチャオブジェクト
	GLuint imageLoc;
	Shader s;
	int vertices;
public:
	GLImage()
	{
	}
	void init(GLFWwindow *window)
	{
		int w, h;
		glfwMakeContextCurrent(window);
		glfwGetWindowSize(window, &w, &h);
		imgWindow = window;

		// 頂点配列オブジェクト
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		// 頂点バッファオブジェクト
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		// [-1, 1] の正方形
		static const GLfloat position[][2] =
		{
			{ -1.0f, -1.0f },
			{ 1.0f, -1.0f },
			{ 1.0f, 1.0f },
			{ -1.0f, 1.0f }
		};
		vertices = sizeof(position) / sizeof (position[0]);
		glBufferData(GL_ARRAY_BUFFER, sizeof(position), position, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
		glEnableVertexAttribArray(0);

		//	テクスチャ
		glGenTextures(1, &image);
		glBindTexture(GL_TEXTURE_RECTANGLE, image);
		glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

		//	シェーダのロード
		s.initGLSL("./shader/vertex_drawpix.glsl", "./shader/fragment_drawpix.glsl");
		imageLoc = glGetUniformLocation(s.program, "image");
	}
	void draw(cv::Mat frame)
	{
		glfwMakeContextCurrent(imgWindow);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// 切り出した画像をテクスチャに転送する
		cv::flip(frame, frame, 0);
		glBindTexture(GL_TEXTURE_RECTANGLE, image);
		glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, frame.cols, frame.rows, GL_BGR, GL_UNSIGNED_BYTE, frame.data);

		// シェーダプログラムの使用開始
		s.enable();

		// uniform サンプラの指定
		glUniform1i(imageLoc, 0);

		// テクスチャユニットとテクスチャの指定
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_RECTANGLE, image);

		// 描画に使う頂点配列オブジェクトの指定
		glBindVertexArray(vao);

		// 図形の描画
		glDrawArrays(GL_TRIANGLE_FAN, 0, vertices);

		// 頂点配列オブジェクトの指定解除
		glBindVertexArray(0);

		// シェーダプログラムの使用終了
		s.disable();
	}
};