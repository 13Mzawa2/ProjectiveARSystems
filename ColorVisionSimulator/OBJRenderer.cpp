#include "OBJRenderer.h"


OBJRenderer::OBJRenderer()
{
}


OBJRenderer::~OBJRenderer()
{
}

void OBJRenderer::loadShader(Shader &_shader)
{
	shader = _shader;
}

void OBJRenderer::loadShader(const char *vertex, const char *fragment)
{
	shader.initGLSL(vertex, fragment);
}

void OBJRenderer::loadObject(Object &_obj)
{
	obj = _obj;
}

void OBJRenderer::loadObject(const char *filename)
{
	loadOBJ(filename, obj);
}

void OBJRenderer::loadTexture(cv::Mat &texture)
{
	texImg = texture.clone();
}

//	テクスチャオブジェクトを作成する
void OBJRenderer::setupTexture(void)
{
	glGenTextures(1, &textureBuffer);
	glBindTexture(GL_TEXTURE_2D, textureBuffer);
	//	OpenGLに画像を渡します。
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

//	uniform変数へのハンドルを取得
//	GLSLの変数を変更したらここも変更する
void OBJRenderer::getUniformID(void)
{
	mvpID = glGetUniformLocation(shader.program, "MVP");		//	vertex.shader内の uniform mat4 MVP; へのID
	normalMatID = glGetUniformLocation(shader.program, "MV");
	samplerID = glGetUniformLocation(shader.program, "myTextureSampler");	// ひとつのOpenGLテクスチャを作ります。
	lightDirectionID = glGetUniformLocation(shader.program, "LightDirection");
	lightColorID = glGetUniformLocation(shader.program, "LightColor");

}

//	OBJファイルの中身をバッファに渡す
void OBJRenderer::setObjData(void)
{
	//	頂点配列オブジェクトを設定
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	//	頂点バッファをOpenGLに渡す
	glGenBuffers(1, &vertexBuffer);							//	バッファを1つ作成
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);			//	以降のコマンドをvertexbufferバッファに指定
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * obj.vertices.size(), &obj.vertices[0], GL_STATIC_DRAW);		//	頂点をOpenGLのvertexbuferに渡す

	//	UV座標バッファ
	glGenBuffers(1, &uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec2) * obj.uvs.size(), &obj.uvs[0], GL_STATIC_DRAW);

	//	法線バッファ
	glGenBuffers(1, &normalBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * obj.normals.size(), &obj.normals[0], GL_STATIC_DRAW);

}

OBJRenderer &OBJRenderer::clone(void)
{
	OBJRenderer r;
	r.obj = obj;
	r.shader = shader;
	r.loadTexture(texImg);
	r.MVP = MVP;
	r.MV = MV;
	r.lightDirection = lightDirection;
	r.lightColor = lightColor;

	setupTexture();
	setObjData();
	getUniformID();
	
	return r;
}

void OBJRenderer::updateTexture(cv::Mat &texture)
{
	texImg = texture.clone();
	glBindTexture(GL_TEXTURE_2D, textureBuffer);
	//	OpenGLに画像を渡します。
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texImg.cols, texImg.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, texImg.data);
	//	ミップマップを作成
	glGenerateMipmap(GL_TEXTURE_2D);
}

void OBJRenderer::render(void)
{
	// 現在バインドしているシェーダの"MVP" uniformに変換を送る
	// レンダリングする各モデルごとに実行
	glUniformMatrix4fv(mvpID, 1, GL_FALSE, &MVP[0][0]);
	glUniformMatrix4fv(normalMatID, 1, GL_FALSE, &MV[0][0]);
	glUniform3fv(lightDirectionID, 1, &lightDirection[0]);
	glUniform3fv(lightColorID, 1, &lightColor[0]);

	//	テクスチャユニット0にtextureBufferをバインド
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureBuffer);
	//	0番目のテクスチャユニットを"myTextureSampler"にセット
	glUniform1i(samplerID, 0);

	// 最初の属性バッファ：頂点
	glEnableVertexAttribArray(GLSL_VERTEX_LOCATION);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glVertexAttribPointer(
		GLSL_VERTEX_LOCATION,	// 属性0：0に特に理由はありません。しかし、シェーダ内のlayoutとあわせないといけません。
		3,						// サイズ
		GL_FLOAT,				// タイプ
		GL_FALSE,				// 正規化？
		0,						// ストライド
		(void*)0				// 配列バッファオフセット
		);
	// 2番目の属性バッファ : UV
	glEnableVertexAttribArray(GLSL_UV_LOCATION);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glVertexAttribPointer(GLSL_UV_LOCATION, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
	// 3番目の属性バッファ : 法線
	glEnableVertexAttribArray(GLSL_NORMAL_LOCATION);
	glBindBuffer(GL_ARRAY_BUFFER, normalBuffer);
	glVertexAttribPointer(GLSL_NORMAL_LOCATION, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

	// 三角形ポリゴンを描画
	glDrawArrays(GL_TRIANGLES, 0, obj.vertices.size());

	glDisableVertexAttribArray(GLSL_VERTEX_LOCATION);
	glDisableVertexAttribArray(GLSL_UV_LOCATION);
	glDisableVertexAttribArray(GLSL_NORMAL_LOCATION);

}