package io.github.jstumfoll2.gravity

import android.annotation.SuppressLint
import android.content.Context
import android.opengl.GLSurfaceView
import android.view.MotionEvent

/** GLSurfaceView that forwards every finger to the renderer on the GL thread. */
@SuppressLint("ViewConstructor")
class SimView(context: Context, val renderer: SimRenderer) : GLSurfaceView(context) {
    init {
        setEGLContextClientVersion(3)
        preserveEGLContextOnPause = true
        setRenderer(renderer)
        renderMode = RENDERMODE_CONTINUOUSLY
    }

    @SuppressLint("ClickableViewAccessibility")
    override fun onTouchEvent(e: MotionEvent): Boolean {
        val t = e.eventTime
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_POINTER_DOWN -> send(e, e.actionIndex, 0, t)
            MotionEvent.ACTION_MOVE -> for (i in 0 until e.pointerCount) send(e, i, 1, t)
            MotionEvent.ACTION_UP, MotionEvent.ACTION_POINTER_UP -> send(e, e.actionIndex, 2, t)
            MotionEvent.ACTION_CANCEL -> for (i in 0 until e.pointerCount) send(e, i, 2, t)
        }
        return true
    }

    private fun send(e: MotionEvent, index: Int, kind: Int, t: Long) {
        val id = e.getPointerId(index)
        val x = e.getX(index)
        val y = e.getY(index)
        queueEvent { renderer.touch(kind, id, x, y, t) }
    }
}
