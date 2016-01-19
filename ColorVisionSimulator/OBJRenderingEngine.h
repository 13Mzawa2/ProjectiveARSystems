//===========================================
//	OpenGL/GLSL Simple Object Rendering Engine
//===========================================
#pragma once

//===========================================
//	Macros
//===========================================
#define GLSL_LOCATION_VERTEX	0
#define GLSL_LOCATION_UV		1
#define GLSL_LOCATION_NORMAL	2

//===========================================
//	Includes
//===========================================
//	Library Linker Scripts
#include "OpenCV3Linker.h"
#include "OpenGLHeader.h"
//	Original Libraries
#include "Shader.h"			//	Shader Class
#include "objloader.hpp"	//	Load OBJ file

class OBJRenderingEngine
{
protected:
	////	uniform IDs
	//	in vertex.glsl
	GLuint mvpID;			//	uniform mat4 MVP;
	GLuint mvID;			//	uniform mat4 MV;
	//	in fragment.glsl
	GLuint textureSamplerID;		//	uniform sampler2D myTextureSampler;
	GLuint lightDirectionID;		//	uniform vec3 LightDirection;
	GLuint lightColorID;			//	uniform vec3 LightColor;
	GLuint lutSamplerID;			//	uniform sampler3D lutSampler;
	GLuint lutSwitchID;				//	uniform bool lutSwitch;
	////	object buffers
	GLuint vertexArray;		//	���_����ێ�����z��
	GLuint vertexBuffer;	//	location = 0
	GLuint uvBuffer;		//	location = 1
	GLuint normalBuffer;	//	location = 2
	GLuint textureObject;	//	�e�N�X�`���ɃA�N�Z�X���邽�߂̃I�u�W�F�N�g
	GLuint visionLUTBuffer;	//	���o�E�F�oLUT

public:
	//	public variables
	Object obj;
	cv::Mat texImg;
	Shader shader;
	//	uniform variables	LUT�z�񎩑͎̂����Ȃ�
	glm::mat4 MVP;
	glm::mat4 MV;
	glm::vec3 lightDirection;
	glm::vec3 lightColor;
	bool useLUT = false;

	OBJRenderingEngine();
	~OBJRenderingEngine();
	OBJRenderingEngine &operator=(OBJRenderingEngine &r);		//	public�̕ϐ����R�s�[�CShader�͍ăR���p�C���Cinit()�͂��Ȃ�
	void getUniformID();
	void setObjectTexture();
	void setVisionLUT(cv::Mat LUT);
	void updateLUT(cv::Mat newLUT);
	void setObjectVertices();
	void init();
	void render();

};

//	�⏕�@�\�FOpenCV�����p�����[�^��OpenGL�s��
glm::mat4 cvtCVCameraParam2GLProjection(cv::Mat camMat, cv::Size camSz, double znear, double zfar);
glm::mat4 composeRT(cv::Mat R, cv::Mat T);