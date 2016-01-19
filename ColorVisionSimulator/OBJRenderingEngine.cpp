#include "OBJRenderingEngine.h"

OBJRenderingEngine::OBJRenderingEngine()
{

}

OBJRenderingEngine::~OBJRenderingEngine()
{

}

//	コピーコンストラクタ
//	obj, shader, texImgが既に読み込まれていることが前提
OBJRenderingEngine &OBJRenderingEngine::operator=(OBJRenderingEngine &r)
{
	obj = r.obj;
	shader = r.shader;		//	この中でinitGLSL()を呼び出している
	texImg = r.texImg.clone();

	MVP = r.MVP;
	MV = r.MV;
	lightDirection = r.lightDirection;
	lightColor = r.lightColor;

	return *this;
}

//	Get GLSL Uniform variable IDs
//	シェーダプログラムにUniform変数を追加する場合はここで登録
void OBJRenderingEngine::getUniformID()
{
	mvpID = glGetUniformLocation(shader.program, "MVP");
	mvID = glGetUniformLocation(shader.program, "MV");
	textureSamplerID = glGetUniformLocation(shader.program, "myTextureSampler");
	lightDirectionID = glGetUniformLocation(shader.program, "LightDirection");
	lightColorID = glGetUniformLocation(shader.program, "LightColor");
	lutSamplerID = glGetUniformLocation(shader.program, "lutSampler");
	lutSwitchID = glGetUniformLocation(shader.program, "lutSwitch");
}

//	OBJファイルのテクスチャをOpenGLに登録
void OBJRenderingEngine::setObjectTexture()
{
	//	テクスチャ画像を読み込む
	glGenTextures(1, &textureObject);
	glBindTexture(GL_TEXTURE_2D, textureObject);
	//	OpenGLに画像を渡す
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
		texImg.cols, texImg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, texImg.data);
	//	テクスチャの繰り返し設定
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	//	画像を拡大(MAGnifying)するときは線形(LINEAR)フィルタリングを使用
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	//	画像を縮小(MINifying)するとき、線形(LINEAR)フィルタした、二つのミップマップを線形(LINEARYLY)に混ぜたものを使用
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	//	ミップマップを作成
	glGenerateMipmap(GL_TEXTURE_2D);
}

//	視覚・色覚LUTを3次元LUTとしてOpenGLに登録
void OBJRenderingEngine::setVisionLUT(cv::Mat lut)
{
	//	テクスチャ画像を読み込む
	glGenTextures(1, &visionLUTBuffer);
	glBindTexture(GL_TEXTURE_3D, visionLUTBuffer);
	//	OpenGLに画像を渡す
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB,
		256, 256, 256, 0, GL_BGR, GL_UNSIGNED_BYTE, lut.data);
	//	テクスチャの拡大縮小に線形補間を使用
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
}

//	視覚・色覚LUTの切り替え
void OBJRenderingEngine::updateLUT(cv::Mat lut)
{
	glBindTexture(GL_TEXTURE_3D, visionLUTBuffer);
	//	OpenGLに画像を渡す
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB,
		256, 256, 256, 0, GL_BGR, GL_UNSIGNED_BYTE, lut.data);
}

//	OBJファイルの頂点情報をOpenGLに登録
void OBJRenderingEngine::setObjectVertices()
{
	//	頂点配列オブジェクトを設定
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	//	頂点バッファをOpenGLに渡す
	glGenBuffers(1, &vertexBuffer);							//	バッファを1つ作成
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);			//	以降のコマンドをvertexbufferバッファに指定
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * obj.vertices.size(), &(obj).vertices[0], GL_STATIC_DRAW);		//	頂点をOpenGLのvertexbuferに渡す

	//	UV座標バッファ
	glGenBuffers(1, &uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * obj.uvs.size(), &(obj).uvs[0], GL_STATIC_DRAW);

	//	法線バッファ
	glGenBuffers(1, &normalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * obj.normals.size(), &(obj).normals[0], GL_STATIC_DRAW);
}

//	既に読み込まれているobj, shader, texImgを使って初期化
void OBJRenderingEngine::init()
{
	getUniformID();
	setObjectVertices();
	setObjectTexture();
}

//	描画命令
void OBJRenderingEngine::render()
{
	////	Execute Rendering
	// 現在バインドしているシェーダのuniform変数に変換を送る
	// レンダリングするモデルごとに実行
	glUniformMatrix4fv(mvpID, 1, GL_FALSE, &MVP[0][0]);
	glUniformMatrix4fv(mvID, 1, GL_FALSE, &MV[0][0]);
	glUniform3fv(lightDirectionID, 1, &lightDirection[0]);
	glUniform3fv(lightColorID, 1, &lightColor[0]);
	glUniform1i(lutSwitchID, useLUT);

	//	テクスチャユニット0にtextureBufferをバインド
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureObject);
	//	0番目のテクスチャユニットを"myTextureSampler"にセット
	glUniform1i(textureSamplerID, 0);

	//	テクスチャユニット1にvisionLUTBufferをバインド
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_3D, visionLUTBuffer);
	glUniform1i(lutSamplerID, 1);

	//	最初の属性バッファ：頂点
	glEnableVertexAttribArray(GLSL_LOCATION_VERTEX);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glVertexAttribPointer(
		GLSL_LOCATION_VERTEX,	// shader内のlocation
		3,						// 要素サイズ
		GL_FLOAT,				// 要素の型
		GL_FALSE,				// 正規化？
		0,						// ストライド
		(void*)0				// 配列バッファオフセット
		);
	//	2番目の属性バッファ : UV
	glEnableVertexAttribArray(GLSL_LOCATION_UV);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glVertexAttribPointer(GLSL_LOCATION_UV, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	//	3番目の属性バッファ : 法線
	glEnableVertexAttribArray(GLSL_LOCATION_NORMAL);
	glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
	glVertexAttribPointer(GLSL_LOCATION_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	//	三角形ポリゴンを描画
	glDrawArrays(GL_TRIANGLES, 0, obj.vertices.size());
	//	描画後にバッファをクリア
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

	//	参考:https://strawlab.org/2011/11/05/augmented-reality-with-OpenGL
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
