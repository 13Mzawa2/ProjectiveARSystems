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
const char lutDir[] = "../common/data/lut/LUT_dichromat_typeP.bmp";

//	RoomAliveToolkit�ł̑��茋��
mat3x3 projectorCameraMatrix(
	2163.9840607833007, 0, 0,
	0, 2163.9840607833007, 0,
	384.14708782496359, 24.7008330562985, 1);
mat4x4 projectorPose(
	0.99780523777008057, -0.011467255651950836, 0.06521625816822052, 0,
	-0.020374707877635956, 0.883938729763031, 0.467158704996109, 0,
	-0.063004210591316223, -0.46746215224266052, 0.88176506757736206, 0,
	-0.0015859748836387856, 0.39699462236674066, -0.5658578787135, 1);

GLFWwindow	*mainWindow, *subWindow;		//	�}���`�E�B���h�E
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
	////	object buffers
	GLuint vertexArray;		//	���_����ێ�����z��
	GLuint vertexBuffer;	//	location = 0
	GLuint uvBuffer;		//	location = 1
	GLuint normalBuffer;	//	location = 2
	GLuint textureObject;	//	�e�N�X�`���ɃA�N�Z�X���邽�߂̃I�u�W�F�N�g
	GLuint lutBuffer;		//	LookUpTable
	////	uniform variables
	mat4 MVP;
	mat4 MV;
	vec3 lightDirection;
	vec3 lightColor;
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
	mainWindow = glfwCreateWindow(1024, 768, "Main Window", NULL, NULL);
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


int main(void)
{
	initWindow();

	texImg = imread(textureDir);
	if (texImg.empty())
	{
		cerr << "�e�N�X�`���̓ǂݍ��݂Ɏ��s���܂����D�f�B���N�g�����m�F���Ă��������D\n"
			<< "�f�B���N�g���̏ꏊ�F" << textureDir << endl;
		glfwTerminate();
		return EXIT_FAILURE;
	}
	lutMat = imread(lutDir);
	if (lutMat.empty())
	{
		cerr << "LUT�̓ǂݍ��݂Ɏ��s���܂����D�f�B���N�g�����m�F���Ă��������D\n"
			<< "�f�B���N�g���̏ꏊ�F" << lutDir << endl;
		glfwTerminate();
		return EXIT_FAILURE;
	}


	initMainWindow();
	initSubWindow();

	//	���C�����[�v
	while (glfwGetKey(mainWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS		//	Esc�L�[
		&& glfwGetKey(subWindow, GLFW_KEY_ESCAPE) != GLFW_PRESS
		&& !glfwWindowShouldClose(mainWindow))							//	�E�B���h�E�̕���{�^��
	{
		//------------------------------
		//	Main Winodw
		//------------------------------
		glfwMakeContextCurrent(mainWindow);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		

		// �ˉe�s��F45���̎��E�A�A�X�y�N�g��4:3�A�\���͈́F0.1�P��  100�P��
		glm::mat4 Projection = glm::perspective(45.0f, 4.0f / 3.0f, 0.1f, 10000.0f);
		// �J�����s��
		glm::mat4 View = glm::lookAt(
			glm::vec3(40, 500, 30), // ���[���h��ԂŃJ������(4,3,3)�ɂ���܂��B
			glm::vec3(0, 0, 0), // ���_�����Ă��܂��B
			glm::vec3(0, 0, 1)  // ���������(0,-1,0�ɃZ�b�g����Ə㉺�t�]���܂��B)
			);
		// ���f���s��F�P�ʍs��(���f���͌��_�ɂ���܂��B)
		glm::mat4 Model;  // �e���f����ς���I
		static float angle = 0.0f;
		angle += 0.001f;
		if (angle >= 360.0) angle -= 360.0;
		Model = glm::translate(glm::vec3(0.0, 0.0, 0.0))
			* glm::rotate(angle, glm::vec3(0.0, 0.0, 1.0))
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
		
		Projection = projectionMatfromCameraMatrix(projectorCameraMatrix, subWinW, subWinH, 0.001, 10000.0);
		//	�J���������_�Ƃ����v���W�F�N�^�ʒu�p��
		View = projectorPose;
		//	�J���������_�Ƃ������[���h���W�n
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
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
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
	// �����_�����O����e���f�����ƂɎ��s
	glUniformMatrix4fv(r.mvpID, 1, GL_FALSE, &r.MVP[0][0]);
	glUniformMatrix4fv(r.mvID, 1, GL_FALSE, &r.MV[0][0]);
	glUniform3fv(r.lightDirectionID, 1, &r.lightDirection[0]);
	glUniform3fv(r.lightColorID, 1, &r.lightColor[0]);

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
	double length = ::sqrt(dx*dx + dy*dy);	//	�N�H�[�^�j�I���̒���
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