#version 310 es
precision mediump float;
in vec3 vColor;
uniform float uBrightness;
out vec4 oColor;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    float a = exp(-3.0 * r2);
    oColor = vec4(vColor * (a * uBrightness), 1.0);
}
