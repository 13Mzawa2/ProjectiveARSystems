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
	//	�����_�����O�\�[�X
	Object obj;			//	���_�����Q
	cv::Mat texImg;		//	�e�N�X�`��

	//	Shader��uniform�ϐ��n���h��
	GLuint mvpID;
	GLuint normalMatID;
	GLuint samplerID;			//	�e�N�X�`���T���v���[
	GLuint lightDirectionID;
	GLuint lightColorID;

	//	�e��I�u�W�F�N�g�o�b�t�@
	GLuint vertexArray;		//	�e�풸�_������ێ����钸�_�z��
	GLuint textureBuffer;	//	�e�N�X�`���ɃA�N�Z�X���邽�߂̃I�u�W�F�N�g
	GLuint vertexBuffer;
	GLuint uvBuffer;
	GLuint normalBuffer;

public:
	//	Shader�ϐ�
	Shader shader;		//	�V�F�[�_�[
	glm::mat4 MVP;				//	= Projection * View * Model
	glm::mat4 MV;				//	= View * Model
	glm::vec3 lightDirection;	//	�����x�N�g��
	glm::vec3 lightColor;		//	�����F

	OBJRenderer();
	~OBJRenderer();

	//	�t�@�C�����[�_
	void loadShader(const char *VertexShaderName, const char *FragmentShaderName);
	void loadObject(const char *fileName);
	void loadTexture(cv::Mat &texture);

	//	OpenGL�C�V�F�[�_�̏����ݒ�
	void setupTexture(void);
	void setObjData(void);
	void getUniformID(void);

	//	�`�揈��
	void updateTexture(cv::Mat &texture);		//	�I�����C���Ńe�N�X�`����ύX����Ƃ��̊֐�
	void render(void);
	
};

