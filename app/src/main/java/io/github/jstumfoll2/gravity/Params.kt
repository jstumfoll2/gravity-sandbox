package io.github.jstumfoll2.gravity

import android.content.SharedPreferences
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.roundToInt

/** One tunable value. Sliders map 0..1000 onto [min, max], linearly or on a log scale. */
class Param(
    val key: String,
    val label: String,
    val min: Float,
    val max: Float,
    val default: Float,
    val log: Boolean = false,
    val integer: Boolean = false,
    val step: Float = 0f,
    val restart: Boolean = false,
    val section: String = "",
    val help: String = "",
) {
    @Volatile var value: Float = default

    fun toSlider(v: Float = value): Int {
        val t = if (log) (ln(v / min) / ln(max / min)) else (v - min) / (max - min)
        return (t.coerceIn(0f, 1f) * SLIDER_STEPS).roundToInt()
    }

    fun fromSlider(s: Int): Float {
        val t = s / SLIDER_STEPS.toFloat()
        var v = if (log) min * exp(t * ln(max / min)) else min + t * (max - min)
        if (step > 0f) v = (v / step).roundToInt() * step
        if (integer) v = v.roundToInt().toFloat()
        return v.coerceIn(min, max)
    }

    fun format(v: Float = value): String = when {
        integer -> v.roundToInt().toString()
        v == 0f -> "0"
        kotlin.math.abs(v) >= 100f -> "%.0f".format(v)
        kotlin.math.abs(v) >= 10f -> "%.1f".format(v)
        kotlin.math.abs(v) >= 0.1f -> "%.2f".format(v)
        else -> "%.4f".format(v)
    }

    companion object { const val SLIDER_STEPS = 1000 }
}

object Params {
    // Simulation
    val count = Param("count", "Particles", 1024f, 65536f, 8192f, log = true, integer = true, step = 1024f, restart = true,
        section = "Simulation", help = "All-pairs gravity costs N². Applies on reset.")
    val g = Param("g", "Gravity G", 0f, 150000f, 40000f, section = "Simulation", help = "G × total mass of a preset")
    val soft = Param("soft", "Softening", 0.1f, 6f, 1f, log = true, section = "Simulation", help = "Gravity is smoothed inside this distance")
    val dt = Param("dt", "Time step", 0.001f, 0.025f, 1f / 120f, log = true, section = "Simulation")
    val substeps = Param("substeps", "Steps per frame", 1f, 8f, 2f, integer = true, section = "Simulation")
    val drag = Param("drag", "Drag", 0f, 2f, 0f, section = "Simulation")
    val bound = Param("bound", "World radius", 100f, 2000f, 600f, log = true, section = "Simulation",
        help = "Particles that fly further are removed")

    // Matter
    val h = Param("h", "Contact radius", 0.2f, 5f, 1.5f, log = true, section = "Matter",
        help = "Particles closer than this push apart and share momentum")
    val k = Param("k", "Pressure", 0f, 200f, 30f, section = "Matter", help = "0 = pure collisionless gravity")
    val visc = Param("visc", "Viscosity", 0f, 12f, 2f, section = "Matter", help = "Sticky collisions; lets clumps settle")
    val heatGain = Param("heatGain", "Heating", 0f, 0.2f, 0.02f, section = "Matter", help = "Collisions heat matter (redder glow)")
    val heatDecay = Param("heatDecay", "Cooling", 0f, 5f, 0.6f, section = "Matter")

    // Black holes
    val holeMass = Param("holeMass", "New hole mass", 0.02f, 5f, 0.4f, log = true, section = "Black holes",
        help = "In units of a preset's total mass")
    val holeRadius = Param("holeRadius", "Horizon per mass", 0.5f, 30f, 6f, log = true, section = "Black holes",
        help = "Event horizon radius = this × mass")
    val holeMinRadius = Param("holeMinRadius", "Min horizon", 0.3f, 10f, 1.5f, log = true, section = "Black holes")

    // Thumbs
    val spawnRate = Param("spawnRate", "Spawn per frame", 1f, 400f, 40f, log = true, integer = true, section = "Thumbs")
    val spawnSpread = Param("spawnSpread", "Spawn spread", 0f, 15f, 2.5f, section = "Thumbs")
    val fling = Param("fling", "Fling strength", 0f, 3f, 1f, section = "Thumbs", help = "Spawned velocity = finger velocity × this")
    val spin = Param("spin", "Spawn orbit", 0f, 1.5f, 0f, section = "Thumbs",
        help = "Adds circular-orbit speed around the centre to new particles")
    val pull = Param("pull", "Pull / push force", 0f, 6000f, 1200f, section = "Thumbs")
    val pullRadius = Param("pullRadius", "Pull radius", 3f, 120f, 30f, log = true, section = "Thumbs")

    // Look
    val zoom = Param("zoom", "Zoom", 0.2f, 4f, 1f, log = true, section = "Look")
    val pointPx = Param("pointPx", "Particle size", 1f, 14f, 3f, help = "At 720 px screen width; scales with the screen", section = "Look")
    val brightness = Param("brightness", "Brightness", 0.02f, 2f, 0.35f, log = true, section = "Look")
    val exposure = Param("exposure", "Exposure", 0.2f, 6f, 1.5f, log = true, section = "Look")
    val trails = Param("trails", "Trails", 0f, 0.97f, 0.55f, section = "Look")
    val densGlow = Param("densGlow", "Clump glow", 0f, 0.5f, 0.08f, section = "Look")
    val gpuTiming = Param("gpuTiming", "Measure GPU time", 0f, 1f, 0f, integer = true, section = "Look",
        help = "1 = wait for the GPU each frame to time the physics (costs a little fps)")

    val all: List<Param> = listOf(
        count, g, soft, dt, substeps, drag, bound,
        h, k, visc, heatGain, heatDecay,
        holeMass, holeRadius, holeMinRadius,
        spawnRate, spawnSpread, fling, spin, pull, pullRadius,
        zoom, pointPx, brightness, exposure, trails, densGlow, gpuTiming,
    )
    val byKey: Map<String, Param> = all.associateBy { it.key }

    fun load(sp: SharedPreferences) {
        for (p in all) p.value = sp.getFloat(p.key, p.default).coerceIn(p.min, p.max)
    }

    fun save(sp: SharedPreferences) {
        val e = sp.edit()
        for (p in all) e.putFloat(p.key, p.value)
        e.apply()
    }

    fun resetDefaults() { for (p in all) p.value = p.default }

    fun snapshot(): Map<String, Float> = all.associate { it.key to it.value }

    fun apply(values: Map<String, Float>) {
        for ((key, v) in values) byKey[key]?.let { it.value = v.coerceIn(it.min, it.max) }
    }
}
