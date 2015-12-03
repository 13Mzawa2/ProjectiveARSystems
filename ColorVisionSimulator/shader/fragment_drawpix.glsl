#version 330 core

uniform sampler2DRect image;

layout (location = 0) out vec4 fc;

void main(void)
{
  fc = texture(image, gl_FragCoord.xy);
}