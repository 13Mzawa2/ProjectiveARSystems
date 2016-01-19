/********************************************************
OpenGL Image with OpenCV
GLFW��OpenCV��cv::Mat��w�i�`�悷�邽�߂̃N���X

How to Use:
1. ���C�����[�v�ɓ���O��GLImage�𐶐�
2. �`�悵����GLFWwindow��^����GLImage��������
3. ���C�����[�v���Ŏ��̗l�ɏ���(ex. mainWindow�̔w�i��frameImg��`��)

//	Change Current Window
glfwMakeContextCurrent(mainWindow);
//	Clear Buffer Bits
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//	Draw Image
glImg.draw(frameImg);		//	<- NO glfwSwapBuffers()
//	Clear Depth Bits (so you can overwride CG on frameImg)
glClear(GL_DEPTH_BUFFER_BIT);
//	Draw your CG
//	End Draw
glfwSwapBuffers(mainWindow);

Change 20160119:
�E�R���X�g���N�^�ŏ������ł���悤�ɂ���
�E�R�����g�啝�ǉ�
�EGLSL���C�����C�������ĊO���t�@�C����s�v�ɂ���

*********************************************************/

#pragma once

#include "OpenGLHeader.h"
#include "Shader.h"
#include <opencv2\opencv.hpp>

class GLImage
{
private:
	GLFWwindow *imgWindow;
	GLuint vao;		//	���_�z��I�u�W�F�N�g
	GLuint vbo;		//	���_�o�b�t�@�I�u�W�F�N�g
	GLuint image;	//	�e�N�X�`���I�u�W�F�N�g
	GLuint imageLoc;//	�I�u�W�F�N�g�̏ꏊ
	Shader s;		//	�V�F�[�_
	//	�o�[�e�b�N�X�V�F�[�_
	const char *vertexSource = 
		"#version 330 core \n" 
		"layout(location = 0) in vec4 pv;\n" 
		"void main(void)\n" 
		"{\n" 
		"	gl_Position = pv;\n" 
		"}\n";
	//	�t���O�����g�V�F�[�_
	const char *fragmentSource =
		"#version 330 core \n"
		"uniform sampler2DRect image;\n"
		"layout(location = 0) out vec4 fc;\n"
		"void main(void)\n"
		"{\n"
		"	fc = texture(image, gl_FragCoord.xy);\n"
		"}\n";
	int vertices;

public:
	GLImage()
	{
	}
	GLImage(GLFWwindow *window)
	{
		init(window);
	}
	void init(GLFWwindow *window)
	{
		int w, h;
		glfwMakeContextCurrent(window);
		glfwGetWindowSize(window, &w, &h);
		imgWindow = window;

		// ���_�z��I�u�W�F�N�g
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		// ���_�o�b�t�@�I�u�W�F�N�g
		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);

		// [-1, 1] �̐����`
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

		//	�e�N�X�`��
		glGenTextures(1, &image);
		glBindTexture(GL_TEXTURE_RECTANGLE, image);
		glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGB, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

		//	�V�F�[�_�̃��[�h
		s.initInlineGLSL(vertexSource, fragmentSource);
		imageLoc = glGetUniformLocation(s.program, "image");
	}
	void draw(cv::Mat frame)
	{
		glfwMakeContextCurrent(imgWindow);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// �؂�o�����摜���e�N�X�`���ɓ]������
		cv::flip(frame, frame, 0);
		glBindTexture(GL_TEXTURE_RECTANGLE, image);
		glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0, 0, frame.cols, frame.rows, GL_BGR, GL_UNSIGNED_BYTE, frame.data);

		// �V�F�[�_�v���O�����̎g�p�J�n
		s.enable();

		// uniform �T���v���̎w��
		glUniform1i(imageLoc, 0);

		// �e�N�X�`�����j�b�g�ƃe�N�X�`���̎w��
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_RECTANGLE, image);

		// �`��Ɏg�����_�z��I�u�W�F�N�g�̎w��
		glBindVertexArray(vao);

		// �}�`�̕`��
		glDrawArrays(GL_TRIANGLE_FAN, 0, vertices);

		// ���_�z��I�u�W�F�N�g�̎w�����
		glBindVertexArray(0);

		// �V�F�[�_�v���O�����̎g�p�I��
		s.disable();
	}
};