#version 310 es
precision mediump float;
in vec2 vUv;
uniform sampler2D uTex;
uniform float uExposure;
out vec4 oColor;
void main() {
    vec3 c = texture(uTex, vUv).rgb * uExposure;
    c = vec3(1.0) - exp(-c);           // soft tone map, so dense cores don't clip flat
    c = pow(c, vec3(1.0 / 1.1));
    oColor = vec4(c + vec3(0.004, 0.005, 0.012), 1.0);
}
