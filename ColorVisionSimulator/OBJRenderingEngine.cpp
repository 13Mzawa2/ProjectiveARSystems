#include "OBJRenderingEngine.h"

OBJRenderingEngine::OBJRenderingEngine()
{

}

OBJRenderingEngine::~OBJRenderingEngine()
{

}

//	�R�s�[�R���X�g���N�^
//	obj, shader, texImg�����ɓǂݍ��܂�Ă��邱�Ƃ��O��
OBJRenderingEngine &OBJRenderingEngine::operator=(OBJRenderingEngine &r)
{
	obj = r.obj;
	shader = r.shader;		//	���̒���initGLSL()���Ăяo���Ă���
	texImg = r.texImg.clone();

	MVP = r.MVP;
	MV = r.MV;
	lightDirection = r.lightDirection;
	lightColor = r.lightColor;

	return *this;
}

//	Get GLSL Uniform variable IDs
//	�V�F�[�_�v���O������Uniform�ϐ���ǉ�����ꍇ�͂����œo�^
void OBJRenderingEngine::getUniformID()
{
	mvpID = glGetUniformLocation(shader.program, "MVP");
	mvID = glGetUniformLocation(shader.program, "MV");
	textureSamplerID = glGetUniformLocation(shader.program, "myTextureSampler");
	lightDirectionID = glGetUniformLocation(shader.program, "LightDirection");
	lightColorID = glGetUniformLocation(shader.program, "LightColor");
	lightSwitchID = glGetUniformLocation(shader.program, "LightSwitch");
	objectColorID = glGetUniformLocation(shader.program, "ObjectColor");
	lutSamplerID = glGetUniformLocation(shader.program, "lutSampler");
	lutSwitchID = glGetUniformLocation(shader.program, "lutSwitch");
}

//	OBJ�t�@�C���̃e�N�X�`����OpenGL�ɓo�^
void OBJRenderingEngine::setObjectTexture()
{
	//	�e�N�X�`���摜��ǂݍ���
	glGenTextures(1, &textureObject);
	glBindTexture(GL_TEXTURE_2D, textureObject);
	//	OpenGL�ɉ摜��n��
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

//	���o�E�F�oLUT��3����LUT�Ƃ���OpenGL�ɓo�^
void OBJRenderingEngine::setVisionLUT(cv::Mat lut)
{
	//	�e�N�X�`���摜��ǂݍ���
	glGenTextures(1, &visionLUTBuffer);
	glBindTexture(GL_TEXTURE_3D, visionLUTBuffer);
	//	OpenGL�ɉ摜��n��
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB,
		256, 256, 256, 0, GL_BGR, GL_UNSIGNED_BYTE, lut.data);
	//	�e�N�X�`���̊g��k���ɐ��`��Ԃ��g�p
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
}

//	���o�E�F�oLUT�̐؂�ւ�
void OBJRenderingEngine::updateLUT(cv::Mat lut)
{
	glBindTexture(GL_TEXTURE_3D, visionLUTBuffer);
	//	OpenGL�ɉ摜��n��
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB,
		256, 256, 256, 0, GL_BGR, GL_UNSIGNED_BYTE, lut.data);
}

//	OBJ�t�@�C���̒��_����OpenGL�ɓo�^
void OBJRenderingEngine::setObjectVertices()
{
	//	���_�z��I�u�W�F�N�g��ݒ�
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	//	���_�o�b�t�@��OpenGL�ɓn��
	glGenBuffers(1, &vertexBuffer);							//	�o�b�t�@��1�쐬
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);			//	�ȍ~�̃R�}���h��vertexbuffer�o�b�t�@�Ɏw��
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * obj.vertices.size(), &(obj).vertices[0], GL_STATIC_DRAW);		//	���_��OpenGL��vertexbufer�ɓn��

	//	UV���W�o�b�t�@
	glGenBuffers(1, &uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * obj.uvs.size(), &(obj).uvs[0], GL_STATIC_DRAW);

	//	�@���o�b�t�@
	glGenBuffers(1, &normalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * obj.normals.size(), &(obj).normals[0], GL_STATIC_DRAW);
}

//	���ɓǂݍ��܂�Ă���obj, shader, texImg���g���ď�����
void OBJRenderingEngine::init()
{
	getUniformID();
	setObjectVertices();
	setObjectTexture();
}

//	�`�施��
void OBJRenderingEngine::render()
{
	////	Execute Rendering
	// ���݃o�C���h���Ă���V�F�[�_��uniform�ϐ��ɕϊ��𑗂�
	// �����_�����O���郂�f�����ƂɎ��s
	glUniformMatrix4fv(mvpID, 1, GL_FALSE, &MVP[0][0]);
	glUniformMatrix4fv(mvID, 1, GL_FALSE, &MV[0][0]);
	glUniform3fv(lightDirectionID, 1, &lightDirection[0]);
	glUniform3fv(lightColorID, 1, &lightColor[0]);
	glUniform3fv(objectColorID, 1, &objectColor[0]);
	glUniform1i(lutSwitchID, useLUT);
	glUniform1i(lightSwitchID, useLight);

	//	�e�N�X�`�����j�b�g0��textureBuffer���o�C���h
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureObject);
	//	0�Ԗڂ̃e�N�X�`�����j�b�g��"myTextureSampler"�ɃZ�b�g
	glUniform1i(textureSamplerID, 0);

	//	�e�N�X�`�����j�b�g1��visionLUTBuffer���o�C���h
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_3D, visionLUTBuffer);
	glUniform1i(lutSamplerID, 1);

	//	�ŏ��̑����o�b�t�@�F���_
	glEnableVertexAttribArray(GLSL_LOCATION_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
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
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glVertexAttribPointer(GLSL_LOCATION_UV, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	//	3�Ԗڂ̑����o�b�t�@ : �@��
	glEnableVertexAttribArray(GLSL_LOCATION_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
	glVertexAttribPointer(GLSL_LOCATION_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	//	�O�p�`�|���S����`��
	glDrawArrays(GL_TRIANGLES, 0, obj.vertices.size());
	//	�`���Ƀo�b�t�@���N���A
	glDisableVertexAttribArray(GLSL_LOCATION_VERTEX);
	glDisableVertexAttribArray(GLSL_LOCATION_UV);
	glDisableVertexAttribArray(GLSL_LOCATION_NORMAL);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindTexture(GL_TEXTURE_3D, 0);
}

//	Convert OpenCV Camera Paramator to OpenGL Projection Matrix
//	@Param
//		camMat: OpenCV camera matrix, from OpenCV camera calibration
//		camSz:  camera window size
//		znear:  near z point of frustum clipping
//		zfar:   far z point of frustum clipping
//	@Return
//		projection: OpenGL Projection Matrix
glm::mat4 cvtCVCameraParam2GLProjection(cv::Mat camMat, cv::Size camSz, double znear, double zfar)
{
	//	Load camera parameters
	double fx = camMat.at<double>(0, 0);
	double fy = camMat.at<double>(1, 1);
	double s = camMat.at<double>(0, 1);
	double cx = camMat.at<double>(0, 2);
	double cy = camMat.at<double>(1, 2);
	double w = camSz.width, h = camSz.height;

	//	�Q�l:https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
	//	With window_coords=="y_down", we have:
	//	[2 * K00 / width,	-2 * K01 / width,	(width - 2 * K02 + 2 * x0) / width,		0]
	//	[0,					2 * K11 / height,	(-height + 2 * K12 + 2 * y0) / height,	0]
	//	[0,					0,					(-zfar - znear) / (zfar - znear),		-2 * zfar*znear / (zfar - znear)]
	//	[0,					0,					-1,										0]

	glm::mat4 projection(
		-2.0 * fx / w, 0, 0, 0,
		0, -2.0 * fy / h, 0, 0,
		1.0 - 2.0 * cx / w, -1.0 + 2.0 * cy / h, -(zfar + znear) / (zfar - znear), -1.0,
		0, 0, -2.0 * zfar * znear / (zfar - znear), 0);

	return projection;
}

//	Compose OpenCV R matrix and T matrix
glm::mat4 composeRT(cv::Mat R, cv::Mat T)
{
	glm::mat4 RT;
	glm::mat4 trans(1.0);
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			trans[j][i] = R.at<double>(i, j);
		}
		trans[3][i] = T.at<double>(i);
	}
	RT = trans;
	return RT;
}
