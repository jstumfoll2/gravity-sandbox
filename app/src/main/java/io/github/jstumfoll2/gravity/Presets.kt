package io.github.jstumfoll2.gravity

import java.util.Random
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** A starting state: particles (8 floats each) plus black holes. Total particle mass is 1. */
class Scene(val particles: FloatArray, val used: Int, val holes: List<FloatArray>)

enum class Preset(val label: String) {
    DISK("Disk + hole"),
    CLOUD("Collapsing cloud"),
    GALAXIES("Galaxy collision"),
    RINGS("Rings"),
    EMPTY("Empty");

    companion object {
        /** Presets fill this share of the buffer, leaving room for thumbs to add more before recycling. */
        const val FILL = 0.7f
    }
}

object Presets {
    fun build(preset: Preset, capacity: Int, seed: Long, g: Float): Scene {
        val rnd = Random(seed)
        val n = if (preset == Preset.EMPTY) 0 else (capacity * Preset.FILL).toInt()
        val out = FloatArray(capacity * 8)
        // everything starts dead
        for (i in 0 until capacity) {
            out[i * 8] = 1e6f; out[i * 8 + 1] = 1e6f
        }
        val m = particleMass(capacity)
        val holes = ArrayList<FloatArray>()
        when (preset) {
            Preset.DISK -> {
                val mBh = 1.0f
                holes += hole(0f, 0f, 0f, 0f, mBh)
                disk(out, 0, n, rnd, 0f, 0f, 0f, 0f, 12f, 85f, mBh, m, g, 1f)
            }
            Preset.CLOUD -> {
                for (i in 0 until n) {
                    val r = 70f * sqrt(rnd.nextFloat())
                    val a = rnd.nextFloat() * 2f * PI.toFloat()
                    val x = r * cos(a); val y = r * sin(a)
                    // slow solid-body spin: ~30% of what would hold it up
                    val menc = (r / 70f) * (r / 70f)
                    val vc = sqrt(g * menc / (r + 1f)) * 0.3f
                    put(out, i, x, y, -sin(a) * vc, cos(a) * vc, m)
                }
            }
            Preset.GALAXIES -> {
                val half = n / 2
                val mBh = 0.5f
                holes += hole(-60f, -25f, 9f, 3f, mBh)
                holes += hole(60f, 25f, -9f, -3f, mBh)
                disk(out, 0, half, rnd, -60f, -25f, 9f, 3f, 5f, 40f, mBh, m, g, 1f, diskMass = 0.5f)
                disk(out, half, n - half, rnd, 60f, 25f, -9f, -3f, 5f, 40f, mBh, m, g, -1f, diskMass = 0.5f)
            }
            Preset.RINGS -> {
                val mBh = 1.5f
                holes += hole(0f, 0f, 0f, 0f, mBh)
                val rings = floatArrayOf(25f, 45f, 70f)
                for (i in 0 until n) {
                    val ring = rings[i % rings.size]
                    val r = ring + rnd.nextGaussian().toFloat() * 1.5f
                    val a = rnd.nextFloat() * 2f * PI.toFloat()
                    val vc = sqrt(g * (mBh + 0.5f) / r)
                    put(out, i, r * cos(a), r * sin(a), -sin(a) * vc, cos(a) * vc, m)
                }
            }
            Preset.EMPTY -> {}
        }
        return Scene(out, n, holes)
    }

    /** Every particle has this mass, so a preset that fills 70% of the buffer weighs 1. */
    fun particleMass(capacity: Int) = 1f / (capacity * Preset.FILL)

    fun hole(x: Float, y: Float, vx: Float, vy: Float, mass: Float) =
        floatArrayOf(x, y, mass, maxOf(Params.holeMinRadius.value, Params.holeRadius.value * mass), vx, vy, 0f, 0f)

    private fun put(out: FloatArray, i: Int, x: Float, y: Float, vx: Float, vy: Float, m: Float) {
        val o = i * 8
        out[o] = x; out[o + 1] = y; out[o + 2] = m; out[o + 3] = 1f
        out[o + 4] = vx; out[o + 5] = vy; out[o + 6] = 0f; out[o + 7] = 0f
    }

    /** Keplerian disk around a central mass, including the disk's own enclosed mass. */
    private fun disk(
        out: FloatArray, start: Int, count: Int, rnd: Random,
        cx: Float, cy: Float, cvx: Float, cvy: Float,
        rIn: Float, rOut: Float, centralMass: Float, m: Float, g: Float, dir: Float,
        diskMass: Float = 1f,
    ) {
        for (k in 0 until count) {
            val u = rnd.nextFloat()
            // surface density ~ 1/r: radius uniform in [rIn, rOut]
            val r = rIn + (rOut - rIn) * u
            val a = rnd.nextFloat() * 2f * PI.toFloat()
            val menc = diskMass * u
            val vc = sqrt(g * (centralMass + menc) / r) * dir
            // a little velocity dispersion so it is not perfectly cold
            val jx = rnd.nextGaussian().toFloat() * 0.03f * vc
            val jy = rnd.nextGaussian().toFloat() * 0.03f * vc
            put(out, start + k, cx + r * cos(a), cy + r * sin(a), cvx - sin(a) * vc + jx, cvy + cos(a) * vc + jy, m)
        }
    }
}
