#version 330 core
 
// ���_�V�F�[�_����̒l���������݂܂�
in vec4 vertexPosition_cameraspace;
in vec2 UV;
in vec3 Normal;
 
// �A�E�g�v�b�g�f�[�^
out vec3 color;
 
// ���ׂẴ��b�V���ň��̒l
uniform sampler2D myTextureSampler;		//	�e�N�X�`���T���v���[
uniform sampler3D lutSampler;			//	LUT�擾�p
uniform vec3 LightDirection;			//	��������
uniform vec3 LightColor;				//	�����F
uniform bool lutSwitch;					//	true => LUT��K�p
 
void main()
{
	vec3 fnormal = normalize(Normal);					//	fragment�ɓn���ꂽNormal�͐��K������Ă��Ȃ�
	vec3 halfway = normalize(LightDirection - vec3(vertexPosition_cameraspace));
	float cosine = max(dot(fnormal, halfway), 0);
	vec3 diffuse = LightColor * cosine;
	vec3 ambient = diffuse * vec3(0.1, 0.1, 0.1);

    // �A�E�g�v�b�g�J���[ = �w�肵��UV���W�̃e�N�X�`���̐F�ɃV�F�[�f�B���O�����s
    vec3 tempcolor = texture( myTextureSampler, UV ).rgb * diffuse + ambient;
	vec3 temp = clamp(tempcolor, 0.0, 0.9999);		//	��E��1.0����LUT�̐F������

	if(lutSwitch)
		color = texture(lutSampler, temp).rgb;
	else
		color = temp;
}