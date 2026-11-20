#version 310 es
precision highp float;
layout(location = 0) in vec4 aPos;   // x, y, mass, radius
layout(location = 1) in vec4 aVel;   // vx, vy, swallowed, flash
uniform vec2 uScale;
uniform vec2 uCenter;
uniform float uPxPerUnit;
out float vFlash;
out float vFed;
void main() {
    if (aPos.z <= 0.0) {
        gl_Position = vec4(4.0, 4.0, 4.0, 1.0);
        gl_PointSize = 1.0;
        vFlash = 0.0; vFed = 0.0;
        return;
    }
    gl_Position = vec4((aPos.xy - uCenter) * uScale, 0.0, 1.0);
    // sprite covers 3 event-horizon radii so the ring and glow fit
    gl_PointSize = max(12.0, aPos.w * uPxPerUnit * 6.0);
    vFlash = aVel.w;
    vFed = aVel.z;
}
