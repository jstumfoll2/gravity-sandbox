#version 310 es
precision highp float;
layout(location = 0) in vec4 aPos;   // x, y, mass, alive
layout(location = 1) in vec4 aVel;   // vx, vy, heat, density
uniform vec2 uScale;      // world -> clip
uniform vec2 uCenter;
uniform float uPointPx;
uniform float uPxPerUnit;
uniform float uDensGlow;
uniform float uHeatScale;
out vec3 vColor;

vec3 palette(float t) {
    // cold dust -> blue-white -> yellow -> orange-white star
    vec3 c0 = vec3(0.35, 0.45, 0.85);
    vec3 c1 = vec3(0.85, 0.90, 1.00);
    vec3 c2 = vec3(1.00, 0.80, 0.40);
    vec3 c3 = vec3(1.00, 0.45, 0.15);
    if (t < 0.33) return mix(c0, c1, t / 0.33);
    if (t < 0.66) return mix(c1, c2, (t - 0.33) / 0.33);
    return mix(c2, c3, clamp((t - 0.66) / 0.34, 0.0, 1.0));
}

void main() {
    if (aPos.w < 0.5) {
        gl_Position = vec4(4.0, 4.0, 4.0, 1.0);
        gl_PointSize = 1.0;
        vColor = vec3(0.0);
        return;
    }
    gl_Position = vec4((aPos.xy - uCenter) * uScale, 0.0, 1.0);
    float dens = aVel.w;
    float heat = aVel.z * uHeatScale;
    float t = clamp(0.15 * dens + heat, 0.0, 1.0);
    vColor = palette(t) * (1.0 + uDensGlow * min(dens, 20.0));
    gl_PointSize = max(1.5, uPointPx * (1.0 + 0.08 * min(dens, 20.0)));
}
