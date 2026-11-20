package io.github.jstumfoll2.gravity

import android.content.Context
import android.opengl.GLES31.*
import android.opengl.GLSurfaceView
import android.os.SystemClock
import java.io.File
import java.util.Random
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.math.sqrt

/** What the renderer tells the UI (always called on the GL thread; the UI posts to itself). */
interface SimListener {
    fun onStats(text: String)
    fun onMessage(text: String)
    fun onFatal(text: String)
    fun onRecordingChanged(recording: Boolean, replaying: Boolean)
    fun onGpuInfo(text: String)
}

/**
 * Owns all GL state and runs the simulation. Everything here runs on the GL thread:
 * the UI talks to it through GLSurfaceView.queueEvent.
 */
class SimRenderer(private val ctx: Context, private val listener: SimListener) : GLSurfaceView.Renderer {

    companion object {
        const val MAX_HOLES = 16
        const val MAX_TOUCH = 10
        const val STRIDE = 32 // bytes per particle and per hole: two vec4
    }

    // ---- GL objects
    private lateinit var stepProg: Program
    private lateinit var holesProg: Program
    private lateinit var particleProg: Program
    private lateinit var fadeProg: Program
    private lateinit var presentProg: Program
    private lateinit var holeProg: Program
    private val partBuf = IntArray(2)
    private var holeBuf = 0
    private var absorbBuf = 0
    private var vao = 0
    private var emptyVao = 0
    private var fbo = 0
    private var fboTex = 0
    private var tile = 256
    private var ready = false

    // ---- simulation state
    private var capacity = 0
    private var cur = 0               // index of the buffer holding the current state
    private var cursor = 0            // ring cursor for spawned particles
    private var holesUsed = 0
    private var holeNext = 0
    private var preset = Preset.DISK
    private var seed = 1L
    private var rnd = Random(1)
    private var frame = 0L
    private var t0Ms = 0L

    // ---- view
    private var width = 1
    private var height = 1
    private var halfW = 100f
    private var halfH = 100f

    // ---- input
    var mode = Mode.SPAWN
    private class Pointer(var x: Float, var y: Float, var t: Long, val mode: Mode) {
        var px = x; var py = y          // position at the previous frame (for spawn streaks)
        var vx = 0f; var vy = 0f
    }
    private val pointers = HashMap<Int, Pointer>()
    private val pending = ArrayList<InputEvent>()

    // ---- record / replay
    private var recording: Recording? = null
    private var replay: Recording? = null
    private var replayIdx = 0
    private val recDir get() = File(ctx.filesDir, "recordings")

    // ---- timing
    private var lastStats = 0L
    private var framesSinceStats = 0
    private var gpuMsAcc = 0.0
    private var gpuMsN = 0
    private var benchLeft = 0
    private val benchMs = ArrayList<Double>()

    private val spawnBuf = Gl.floats(400 * 8 * 2)
    private val touchArr = FloatArray(MAX_TOUCH * 4)

    // ------------------------------------------------------------------ lifecycle

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        ready = false
        try {
            val ver = glGetString(GL_VERSION) ?: ""
            val m = Regex("OpenGL ES (\\d+)\\.(\\d+)").find(ver)
            val major = m?.groupValues?.get(1)?.toInt() ?: 0
            val minor = m?.groupValues?.get(2)?.toInt() ?: 0
            if (major < 3 || (major == 3 && minor < 1)) {
                listener.onFatal("This GPU reports \"$ver\". Gravity Sandbox needs OpenGL ES 3.1 compute shaders.")
                return
            }
            val maxInv = IntArray(1)
            glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, maxInv, 0)
            tile = if (maxInv[0] >= 256) 256 else 128
            val defs = mapOf("TILE" to "$tile", "T" to "$tile")
            stepProg = Gl.compute(ctx, "step.comp", defs)
            holesProg = Gl.compute(ctx, "holes.comp", defs)
            particleProg = Gl.graphics(ctx, "particle.vert", "particle.frag")
            fadeProg = Gl.graphics(ctx, "fullscreen.vert", "fade.frag")
            presentProg = Gl.graphics(ctx, "fullscreen.vert", "present.frag")
            holeProg = Gl.graphics(ctx, "hole.vert", "hole.frag")

            glGenBuffers(2, partBuf, 0)
            holeBuf = Gl.genBuffer()
            absorbBuf = Gl.genBuffer()
            val v = IntArray(2)
            glGenVertexArrays(2, v, 0)
            vao = v[0]; emptyVao = v[1]

            listener.onGpuInfo("${glGetString(GL_RENDERER)} · ${ver.take(40)} · workgroup $tile")
            ready = true
            reset(preset, seed)
        } catch (e: Throwable) {
            listener.onFatal(e.message ?: e.toString())
        }
    }

    override fun onSurfaceChanged(gl: GL10?, w: Int, h: Int) {
        width = maxOf(1, w); height = maxOf(1, h)
        glViewport(0, 0, width, height)
        if (ready) makeFbo()
    }

    private fun makeFbo() {
        if (fbo != 0) {
            glDeleteFramebuffers(1, intArrayOf(fbo), 0)
            glDeleteTextures(1, intArrayOf(fboTex), 0)
        }
        val ids = IntArray(1)
        glGenFramebuffers(1, ids, 0); fbo = ids[0]
        glGenTextures(1, ids, 0); fboTex = ids[0]
        glBindTexture(GL_TEXTURE_2D, fboTex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        val ext = glGetString(GL_EXTENSIONS) ?: ""
        val half = ext.contains("GL_EXT_color_buffer_half_float") || ext.contains("GL_EXT_color_buffer_float")
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        var ok = false
        if (half) {
            // 16-bit float target: thousands of faint additive sprites accumulate without banding
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_HALF_FLOAT, null)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTex, 0)
            ok = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
        }
        if (!ok) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, null)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fboTex, 0)
        }
        glClearColor(0f, 0f, 0f, 1f)
        glClear(GL_COLOR_BUFFER_BIT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
    }

    // ------------------------------------------------------------------ commands (GL thread)

    fun reset(p: Preset, s: Long) {
        if (!ready) { preset = p; seed = s; return }
        preset = p; seed = s
        rnd = Random(s xor 0x5DEECE66DL)
        capacity = (Params.count.value.toInt() / tile).coerceAtLeast(1) * tile
        val scene = Presets.build(p, capacity, s, Params.g.value)
        val fb = Gl.floats(scene.particles.size).put(scene.particles)
        for (b in partBuf) {
            fb.position(0)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, b)
            glBufferData(GL_SHADER_STORAGE_BUFFER, capacity * STRIDE, fb, GL_DYNAMIC_COPY)
        }
        cur = 0
        cursor = scene.used % capacity
        // holes: all dead, then the preset's
        val hb = Gl.floats(MAX_HOLES * 8)
        for (i in 0 until MAX_HOLES) hb.put(floatArrayOf(1e6f, 1e6f, 0f, 0f, 0f, 0f, 0f, 0f))
        hb.position(0)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, holeBuf)
        glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_HOLES * STRIDE, hb, GL_DYNAMIC_COPY)
        holesUsed = 0; holeNext = 0
        for (h in scene.holes) writeHole(h)
        val ab = Gl.ints(MAX_HOLES * 3)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, absorbBuf)
        glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_HOLES * 3 * 4, ab, GL_DYNAMIC_COPY)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        pointers.clear()
        pending.clear()
        frame = 0
        t0Ms = SystemClock.uptimeMillis()
        if (fbo != 0) {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glClear(GL_COLOR_BUFFER_BIT)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
        }
        Gl.check("reset")
    }

    fun newScene(p: Preset) {
        stopReplay(false)
        if (recording != null) stopRecording()
        reset(p, System.nanoTime())
    }

    fun restart() = newScene(preset)

    /** Raw touch from the UI thread, already in pixels; converted to world units here. */
    fun touch(kind: Int, id: Int, xPx: Float, yPx: Float, eventTime: Long) {
        if (replay != null) {
            if (kind == 0) { stopReplay(true) } else return
        }
        val wx = (xPx / width * 2f - 1f) * halfW
        val wy = (1f - yPx / height * 2f) * halfH
        pending += InputEvent(0, eventTime - t0Ms, kind, id, wx, wy, mode)
    }

    fun paramChanged(key: String, value: Float) {
        recording?.events?.add(InputEvent(frame, 0, 3, 0, value, 0f, Mode.SPAWN, key))
    }

    fun startRecording() {
        stopReplay(false)
        val s = System.nanoTime()
        reset(preset, s)
        recording = Recording(preset, s, Params.snapshot())
        listener.onRecordingChanged(true, false)
        listener.onMessage("Recording from a fresh ${preset.label}")
    }

    fun stopRecording() {
        val r = recording ?: return
        recording = null
        r.frames = frame
        val name = "rec_" + java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US).format(java.util.Date()) + ".txt"
        r.write(File(recDir, name))
        r.write(File(recDir, "last.txt"))
        listener.onRecordingChanged(false, false)
        listener.onMessage("Saved ${r.events.size} events, $frame frames")
    }

    fun startReplay() {
        if (recording != null) stopRecording()
        val r = Recording.read(File(recDir, "last.txt"))
        if (r == null) { listener.onMessage("Nothing recorded yet"); return }
        Params.apply(r.params)
        reset(r.preset, r.seed)
        replay = r
        replayIdx = 0
        listener.onRecordingChanged(false, true)
        listener.onMessage("Replaying ${r.frames} frames (touch to take over)")
    }

    private fun stopReplay(byUser: Boolean) {
        if (replay == null) return
        replay = null
        pointers.clear()
        listener.onRecordingChanged(false, false)
        if (byUser) listener.onMessage("Replay stopped, you have control")
    }

    fun benchmark() {
        benchLeft = 150
        benchMs.clear()
        listener.onMessage("Benchmarking ${capacity} particles…")
    }

    // ------------------------------------------------------------------ per frame

    override fun onDrawFrame(gl: GL10?) {
        if (!ready) {
            glClearColor(0.02f, 0.02f, 0.05f, 1f); glClear(GL_COLOR_BUFFER_BIT)
            return
        }
        updateCamera()
        processInput()
        applyThumbs()

        val timing = Params.gpuTiming.value > 0.5f || benchLeft > 0
        val tStart = if (timing) { glFinish(); System.nanoTime() } else 0L
        val steps = Params.substeps.value.toInt().coerceIn(1, 8)
        for (s in 0 until steps) step()
        if (timing) {
            glFinish()
            val ms = (System.nanoTime() - tStart) / 1e6 / steps
            gpuMsAcc += ms; gpuMsN++
            if (benchLeft > 0) {
                if (benchLeft <= 120) benchMs += ms   // skip warm-up frames
                benchLeft--
                if (benchLeft == 0) reportBench()
            }
        }
        render()
        frame++
        replay?.let { if (frame >= it.frames && replayIdx >= it.events.size) { stopReplay(false); listener.onMessage("Replay finished") } }
        stats()
    }

    private fun updateCamera() {
        val half = 100f / Params.zoom.value
        if (width <= height) { halfW = half; halfH = half * height / width }
        else { halfH = half; halfW = half * width / height }
    }

    private fun processInput() {
        val events: List<InputEvent>
        val rp = replay
        if (rp != null) {
            val list = ArrayList<InputEvent>()
            while (replayIdx < rp.events.size && rp.events[replayIdx].frame <= frame) list += rp.events[replayIdx++]
            events = list
        } else {
            events = pending.map { it.copy(frame = frame) }
            pending.clear()
            recording?.events?.addAll(events)
        }
        for (p in pointers.values) { p.px = p.x; p.py = p.y }
        for (e in events) {
            when (e.kind) {
                0 -> pointers[e.id] = Pointer(e.x, e.y, e.tMs, e.mode)
                1 -> pointers[e.id]?.let { p ->
                    val dt = (e.tMs - p.t) / 1000f
                    if (dt > 0.001f) {
                        val ivx = (e.x - p.x) / dt; val ivy = (e.y - p.y) / dt
                        p.vx += (ivx - p.vx) * 0.5f; p.vy += (ivy - p.vy) * 0.5f
                    }
                    p.x = e.x; p.y = e.y; p.t = e.tMs
                }
                2 -> pointers.remove(e.id)?.let { p ->
                    if (e.tMs - p.t > 80) { p.vx = 0f; p.vy = 0f }   // finger had stopped before lifting
                    if (p.mode == Mode.HOLE) {
                        val f = Params.fling.value
                        writeHole(Presets.hole(e.x, e.y, p.vx * f, p.vy * f, Params.holeMass.value))
                    }
                }
                3 -> Params.byKey[e.key]?.let { if (!it.restart) it.value = e.x.coerceIn(it.min, it.max) }
            }
        }
    }

    private fun applyThumbs() {
        val m = Presets.particleMass(capacity)
        val rate = Params.spawnRate.value.toInt().coerceIn(1, 400)
        val spread = Params.spawnSpread.value
        val fling = Params.fling.value
        val spin = Params.spin.value
        val g = Params.g.value
        for (p in pointers.values) {
            if (p.mode != Mode.SPAWN) continue
            spawnBuf.clear()
            for (k in 0 until rate) {
                val t = rnd.nextFloat()
                val x = p.px + (p.x - p.px) * t + rnd.nextGaussian().toFloat() * spread
                val y = p.py + (p.y - p.py) * t + rnd.nextGaussian().toFloat() * spread
                var vx = p.vx * fling + rnd.nextGaussian().toFloat() * 1.5f
                var vy = p.vy * fling + rnd.nextGaussian().toFloat() * 1.5f
                if (spin > 0f) {
                    val r = sqrt(x * x + y * y) + 1f
                    val vc = sqrt(g / r) * spin
                    vx += -y / r * vc; vy += x / r * vc
                }
                spawnBuf.put(x).put(y).put(m).put(1f).put(vx).put(vy).put(0f).put(0f)
            }
            writeParticles(rate)
        }
    }

    /** Writes [n] particles from spawnBuf into the ring, wrapping around the end. */
    private fun writeParticles(n: Int) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, partBuf[cur])
        var done = 0
        while (done < n) {
            val chunk = minOf(n - done, capacity - cursor)
            spawnBuf.position(done * 8)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, cursor * STRIDE, chunk * STRIDE, spawnBuf)
            cursor = (cursor + chunk) % capacity
            done += chunk
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    }

    private fun writeHole(h: FloatArray) {
        val slot = holeNext
        holeNext = (holeNext + 1) % MAX_HOLES
        holesUsed = maxOf(holesUsed, slot + 1)
        val b = Gl.floats(8).put(h)
        b.position(0)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, holeBuf)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, slot * STRIDE, STRIDE, b)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        // clear its swallow counters
        val z = Gl.ints(3)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, absorbBuf)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, slot * 12, 12, z)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    }

    private fun step() {
        val dt = Params.dt.value
        val g = Params.g.value
        val soft = Params.soft.value
        val bound = Params.bound.value

        // thumbs that pull or push
        var nt = 0
        for (p in pointers.values) {
            if (nt >= MAX_TOUCH) break
            val sign = when (p.mode) { Mode.PULL -> 1f; Mode.PUSH -> -1f; else -> continue }
            touchArr[nt * 4] = p.x; touchArr[nt * 4 + 1] = p.y
            touchArr[nt * 4 + 2] = sign * Params.pull.value; touchArr[nt * 4 + 3] = Params.pullRadius.value
            nt++
        }

        stepProg.use()
        stepProg.i("uCount", capacity)
        stepProg.i("uHoleCount", holesUsed)
        stepProg.f("uDt", dt)
        stepProg.f("uG", g)
        stepProg.f("uSoft2", soft * soft)
        stepProg.f("uH", Params.h.value)
        stepProg.f("uK", Params.k.value)
        stepProg.f("uVisc", Params.visc.value)
        stepProg.f("uDrag", Params.drag.value)
        stepProg.f("uHoleSoft2", 0.25f)
        stepProg.f("uBound", bound)
        stepProg.f("uHeatGain", Params.heatGain.value)
        stepProg.f("uHeatDecay", Params.heatDecay.value)
        stepProg.i("uTouchCount", nt)
        if (nt > 0) glUniform4fv(stepProg.loc("uTouch"), nt, touchArr, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, partBuf[cur])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, partBuf[1 - cur])
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, holeBuf)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, absorbBuf)
        glDispatchCompute(capacity / tile, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        if (holesUsed > 0) {
            holesProg.use()
            holesProg.i("uCount", capacity)
            holesProg.i("uHoleCount", holesUsed)
            holesProg.f("uDt", dt)
            holesProg.f("uG", g)
            holesProg.f("uHoleSoft2", 0.25f)
            holesProg.f("uRadiusPerMass", Params.holeRadius.value)
            holesProg.f("uMinRadius", Params.holeMinRadius.value)
            holesProg.f("uBound", bound)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, partBuf[1 - cur])
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, holeBuf)
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, absorbBuf)
            glDispatchCompute(1, 1, 1)
        }
        // covers the next step's reads, vertex fetch for drawing, and CPU buffer updates
        glMemoryBarrier(GL_ALL_BARRIER_BITS)
        cur = 1 - cur
    }

    private fun render() {
        val sx = 1f / halfW; val sy = 1f / halfH
        val cx = 0f; val cy = 0f
        val pxPerUnit = width / (2f * halfW)

        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glViewport(0, 0, width, height)
        glEnable(GL_BLEND)

        // fade the previous frame (trails): dst *= keep
        glBlendFunc(GL_ZERO, GL_SRC_ALPHA)
        fadeProg.use()
        fadeProg.f("uKeep", Params.trails.value)
        glBindVertexArray(emptyVao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        // particles, additive
        glBlendFunc(GL_ONE, GL_ONE)
        particleProg.use()
        particleProg.v2("uScale", sx, sy)
        particleProg.v2("uCenter", cx, cy)
        particleProg.f("uPointPx", Params.pointPx.value * minOf(width, height) / 720f)
        particleProg.f("uPxPerUnit", pxPerUnit)
        particleProg.f("uDensGlow", Params.densGlow.value)
        particleProg.f("uHeatScale", 0.05f)
        particleProg.f("uBrightness", Params.brightness.value)
        bindAttribs(partBuf[cur])
        glDrawArrays(GL_POINTS, 0, capacity)

        // to screen with tone mapping
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDisable(GL_BLEND)
        presentProg.use()
        presentProg.f("uExposure", Params.exposure.value)
        presentProg.i("uTex", 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, fboTex)
        glBindVertexArray(emptyVao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        // black holes on top, drawn crisp (no trails)
        if (holesUsed > 0) {
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            holeProg.use()
            holeProg.v2("uScale", sx, sy)
            holeProg.v2("uCenter", cx, cy)
            holeProg.f("uPxPerUnit", pxPerUnit)
            bindAttribs(holeBuf)
            glDrawArrays(GL_POINTS, 0, holesUsed)
            glDisable(GL_BLEND)
        }
        glBindVertexArray(0)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
    }

    private fun bindAttribs(buffer: Int) {
        glBindVertexArray(vao)
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, false, STRIDE, 0)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 4, GL_FLOAT, false, STRIDE, 16)
    }

    private fun stats() {
        framesSinceStats++
        val now = SystemClock.uptimeMillis()
        if (lastStats == 0L) lastStats = now
        val el = now - lastStats
        if (el < 500) return
        val fps = framesSinceStats * 1000.0 / el
        val sb = StringBuilder()
        sb.append("%.0f fps · %,d particles".format(fps, capacity))
        if (gpuMsN > 0) sb.append(" · physics %.1f ms/step".format(gpuMsAcc / gpuMsN))
        if (holesUsed > 0) sb.append(" · ${holesUsed} hole slot${if (holesUsed > 1) "s" else ""}")
        if (recording != null) sb.append(" · ● REC")
        if (replay != null) sb.append(" · ▶ replay ${frame}/${replay?.frames}")
        listener.onStats(sb.toString())
        lastStats = now; framesSinceStats = 0; gpuMsAcc = 0.0; gpuMsN = 0
    }

    private fun reportBench() {
        if (benchMs.isEmpty()) return
        val sorted = benchMs.sorted()
        val med = sorted[sorted.size / 2]
        val p90 = sorted[(sorted.size * 9) / 10]
        val pairs = capacity.toDouble() * capacity / (med / 1000.0) / 1e9
        listener.onMessage("N=%,d: median %.2f ms/step, p90 %.2f ms, %.1f G pairs/s".format(capacity, med, p90, pairs))
    }
}
