#pragma once

#include "OpenGLHeader.h"

class Shader
{
private:
	GLuint vertexShader, fragmentShader;		//	シェーダオブジェクト
	const char *vertexFileName, *fragmentFileName;		//	シェーダファイル名
public:
	GLuint program;			//	シェーダプログラム
	Shader();
	~Shader();
	Shader &operator=(Shader &_s)
	{
		initGLSL(_s.vertexFileName, _s.fragmentFileName);
		return *this;
	}
	void readShaderCompile(GLuint shader, const char *file);	//	.shaderのコンパイル
	void link(GLuint prog);		//	コンパイルしたshaderをリンクする
	//	初期化
	//	フラグメントシェーダーの有無で分ける
	void initGLSL(const char *vertexFile);
	void initGLSL(const char *vertexFile, const char *fragmentFile);
	//	有効化
	void enable(){ glUseProgram(program); }
	void disable(){ glUseProgram(0); }

};

