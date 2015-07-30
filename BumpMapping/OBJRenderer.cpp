#include "OBJRenderer.h"


OBJRenderer::OBJRenderer()
{
}


OBJRenderer::~OBJRenderer()
{
}

void OBJRenderer::loadShader(const char *vertex, const char *fragment)
{
	shader.initGLSL(vertex, fragment);
}

void OBJRenderer::loadObject(const char *filename)
{
	loadOBJ(filename, obj);
}

void OBJRenderer::loadTexture(cv::Mat &texture)
{
	texImg = texture.clone();
}

//	�e�N�X�`���I�u�W�F�N�g���쐬����
void OBJRenderer::setupTexture(void)
{
	glGenTextures(1, &textureBuffer);
	glBindTexture(GL_TEXTURE_2D, textureBuffer);
	//	OpenGL�ɉ摜��n���܂��B
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
		texImg.cols, texImg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, texImg.data);
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

//	uniform�ϐ��ւ̃n���h�����擾
//	GLSL�̕ϐ���ύX�����炱�����ύX����
void OBJRenderer::getUniformID(void)
{
	mvpID = glGetUniformLocation(shader.program, "MVP");		//	vertex.shader���� uniform mat4 MVP; �ւ�ID
	normalMatID = glGetUniformLocation(shader.program, "MV");
	samplerID = glGetUniformLocation(shader.program, "myTextureSampler");	// �ЂƂ�OpenGL�e�N�X�`�������܂��B
	lightDirectionID = glGetUniformLocation(shader.program, "LightDirection");
	lightColorID = glGetUniformLocation(shader.program, "LightColor");

}

//	OBJ�t�@�C���̒��g���o�b�t�@�ɓn��
void OBJRenderer::setObjData(void)
{
	//	���_�z��I�u�W�F�N�g��ݒ�
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	//	���_�o�b�t�@��OpenGL�ɓn��
	glGenBuffers(1, &vertexBuffer);							//	�o�b�t�@��1�쐬
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);			//	�ȍ~�̃R�}���h��vertexbuffer�o�b�t�@�Ɏw��
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * obj.vertices.size(), &obj.vertices[0], GL_STATIC_DRAW);		//	���_��OpenGL��vertexbufer�ɓn��

	//	UV���W�o�b�t�@
	glGenBuffers(1, &uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * obj.uvs.size(), &obj.uvs[0], GL_STATIC_DRAW);

	//	�@���o�b�t�@
	glGenBuffers(1, &normalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * obj.normals.size(), &obj.normals[0], GL_STATIC_DRAW);

}

void OBJRenderer::updateTexture(cv::Mat &texture)
{
	texImg = texture.clone();
	glBindTexture(GL_TEXTURE_2D, textureBuffer);
	//	OpenGL�ɉ摜��n���܂��B
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texImg.cols, texImg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, texImg.data);
	//	�~�b�v�}�b�v���쐬
	glGenerateMipmap(GL_TEXTURE_2D);
}

void OBJRenderer::render(void)
{
	// ���݃o�C���h���Ă���V�F�[�_��"MVP" uniform�ɕϊ��𑗂�
	// �����_�����O����e���f�����ƂɎ��s
	glUniformMatrix4fv(mvpID, 1, GL_FALSE, &MVP[0][0]);
	glUniformMatrix4fv(normalMatID, 1, GL_FALSE, &MV[0][0]);
	glUniform3fv(lightDirectionID, 1, &lightDirection[0]);
	glUniform3fv(lightColorID, 1, &lightColor[0]);

	//	�e�N�X�`�����j�b�g0��textureBuffer���o�C���h
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureBuffer);
	//	0�Ԗڂ̃e�N�X�`�����j�b�g��"myTextureSampler"�ɃZ�b�g
	glUniform1i(samplerID, 0);

	// �ŏ��̑����o�b�t�@�F���_
	glEnableVertexAttribArray(GLSL_VERTEX_LOCATION);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glVertexAttribPointer(
		GLSL_VERTEX_LOCATION,	// ����0�F0�ɓ��ɗ��R�͂���܂���B�������A�V�F�[�_����layout�Ƃ��킹�Ȃ��Ƃ����܂���B
		3,						// �T�C�Y
		GL_FLOAT,				// �^�C�v
		GL_FALSE,				// ���K���H
		0,						// �X�g���C�h
		(void*)0				// �z��o�b�t�@�I�t�Z�b�g
		);
	// 2�Ԗڂ̑����o�b�t�@ : UV
	glEnableVertexAttribArray(GLSL_UV_LOCATION);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glVertexAttribPointer(GLSL_UV_LOCATION, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	// 3�Ԗڂ̑����o�b�t�@ : �@��
	glEnableVertexAttribArray(GLSL_NORMAL_LOCATION);
	glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
	glVertexAttribPointer(GLSL_NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	// �O�p�`�|���S����`��
	glDrawArrays(GL_TRIANGLES, 0, obj.vertices.size());

	glDisableVertexAttribArray(GLSL_VERTEX_LOCATION);
	glDisableVertexAttribArray(GLSL_UV_LOCATION);
	glDisableVertexAttribArray(GLSL_NORMAL_LOCATION);

}