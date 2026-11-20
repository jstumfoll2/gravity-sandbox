#version 310 es
precision mediump float;
uniform float uKeep;
out vec4 oColor;
void main() { oColor = vec4(0.0, 0.0, 0.0, uKeep); }
