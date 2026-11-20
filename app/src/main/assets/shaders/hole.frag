#version 310 es
precision mediump float;
in float vFlash;
in float vFed;
out vec4 oColor;
void main() {
    vec2 c = gl_PointCoord * 2.0 - 1.0;
    float r = length(c) * 3.0;          // in event-horizon radii
    if (r > 3.0) discard;
    vec3 ring = vec3(1.0, 0.62, 0.25) * (0.6 + min(vFed * 2.0, 1.5)) + vec3(0.6, 0.7, 1.0) * vFlash;
    float ringA = exp(-((r - 1.15) * 5.0) * ((r - 1.15) * 5.0));
    float glow = exp(-(r - 1.0) * 2.2) * 0.25;
    if (r < 1.0) {
        oColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        float a = clamp(ringA + glow, 0.0, 1.0);
        oColor = vec4(ring * (ringA + glow), a);
    }
}
