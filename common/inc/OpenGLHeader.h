//	GLEW, GLFW, GLMのインクルードヘッダ
//	GLEW : OpenGL1.1以降の機能を使えるようにする
//	GLFW : GLUTに代わるOpenGL表示レイヤー
//	GLM  : 座標変換等のクラスを提供

#pragma once

#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "glfw3.lib")
#include <stdio.h>
#include <stdlib.h>

#include <GLEW\glew.h>

#include <GLFW/glfw3.h>	

#include <glm/glm.hpp>						//	基本的な演算
#include <glm/gtc/matrix_transform.hpp>		//	glm::perspectiveなど
#include <glm/gtx/transform.hpp>			//	glm::translateなど
using namespace glm;