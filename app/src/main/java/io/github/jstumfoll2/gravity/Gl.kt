package io.github.jstumfoll2.gravity

import android.content.Context
import android.opengl.GLES31.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer

/** A linked GL program with a uniform-location cache. */
class Program(val id: Int) {
    private val locs = HashMap<String, Int>()
    fun loc(name: String): Int = locs.getOrPut(name) { glGetUniformLocation(id, name) }
    fun use() = glUseProgram(id)
    fun f(name: String, v: Float) = glUniform1f(loc(name), v)
    fun i(name: String, v: Int) = glUniform1i(loc(name), v)
    fun v2(name: String, x: Float, y: Float) = glUniform2f(loc(name), x, y)
}

object Gl {
    fun asset(ctx: Context, name: String, defines: Map<String, String> = emptyMap()): String {
        var src = ctx.assets.open("shaders/$name").bufferedReader().use { it.readText() }
        for ((k, v) in defines) src = src.replace(Regex("#define $k \\S+"), "#define $k $v")
        return src
    }

    private fun shader(type: Int, src: String, name: String): Int {
        val s = glCreateShader(type)
        glShaderSource(s, src)
        glCompileShader(s)
        val ok = IntArray(1)
        glGetShaderiv(s, GL_COMPILE_STATUS, ok, 0)
        if (ok[0] == 0) {
            val log = glGetShaderInfoLog(s)
            glDeleteShader(s)
            throw RuntimeException("Shader $name failed to compile:\n$log")
        }
        return s
    }

    private fun link(name: String, vararg shaders: Int): Program {
        val p = glCreateProgram()
        for (s in shaders) glAttachShader(p, s)
        glLinkProgram(p)
        val ok = IntArray(1)
        glGetProgramiv(p, GL_LINK_STATUS, ok, 0)
        for (s in shaders) glDeleteShader(s)
        if (ok[0] == 0) throw RuntimeException("Program $name failed to link:\n${glGetProgramInfoLog(p)}")
        return Program(p)
    }

    fun compute(ctx: Context, name: String, defines: Map<String, String> = emptyMap()): Program =
        link(name, shader(GL_COMPUTE_SHADER, asset(ctx, name, defines), name))

    fun graphics(ctx: Context, vert: String, frag: String): Program =
        link("$vert+$frag", shader(GL_VERTEX_SHADER, asset(ctx, vert), vert), shader(GL_FRAGMENT_SHADER, asset(ctx, frag), frag))

    fun floats(n: Int): FloatBuffer =
        ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).asFloatBuffer()

    fun ints(n: Int): IntBuffer =
        ByteBuffer.allocateDirect(n * 4).order(ByteOrder.nativeOrder()).asIntBuffer()

    fun genBuffer(): Int { val a = IntArray(1); glGenBuffers(1, a, 0); return a[0] }

    fun check(where: String) {
        val e = glGetError()
        if (e != GL_NO_ERROR) android.util.Log.w("gravity", "GL error 0x${Integer.toHexString(e)} at $where")
    }
}
