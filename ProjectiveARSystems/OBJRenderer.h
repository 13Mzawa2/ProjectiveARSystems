#pragma once

#include <opencv2/opencv.hpp>
#include "OpenGLHeader.h"
#include "Shader.h"
#include "objloader.hpp"

#define GLSL_VERTEX_LOCATION		0
#define GLSL_UV_LOCATION			1
#define GLSL_NORMAL_LOCATION		2

class OBJRenderer
{
protected:
	//	レンダリングソース
	Object obj;			//	頂点属性群
	cv::Mat texImg;		//	テクスチャ

	//	Shaderのuniform変数ハンドラ
	GLuint mvpID;
	GLuint normalMatID;
	GLuint samplerID;			//	テクスチャサンプラー
	GLuint lightDirectionID;
	GLuint lightColorID;

	//	各種オブジェクトバッファ
	GLuint vertexArray;		//	各種頂点属性を保持する頂点配列
	GLuint textureBuffer;	//	テクスチャにアクセスするためのオブジェクト
	GLuint vertexBuffer;
	GLuint uvBuffer;
	GLuint normalBuffer;

public:
	//	Shader変数
	Shader shader;		//	シェーダー
	glm::mat4 MVP;				//	= Projection * View * Model
	glm::mat4 MV;				//	= View * Model
	glm::vec3 lightDirection;	//	光線ベクトル
	glm::vec3 lightColor;		//	光源色

	OBJRenderer();
	~OBJRenderer();

	//	ファイルローダ
	void loadShader(Shader &_shader);		//	上手く働くか不明
	void loadShader(const char *VertexShaderName, const char *FragmentShaderName);
	void loadObject(Object &_obj);			//	上手く働くか不明
	void loadObject(const char *fileName);
	void loadTexture(cv::Mat &texture);

	//	OpenGL，シェーダの初期設定
	void setupTexture(void);
	void setObjData(void);
	void getUniformID(void);
	OBJRenderer &clone(void);			//	同じレンダリング情報を使いまわしたいときに使用

	//	描画処理
	void updateTexture(cv::Mat &texture);		//	オンラインでテクスチャを変更するときの関数
	void render(void);
	
};

