package io.github.jstumfoll2.gravity

import java.io.File

/** Thumb mode chosen on the toolbar when a finger touched down. */
enum class Mode { SPAWN, PULL, PUSH, HOLE }

/**
 * One input event in world coordinates. [frame] is the simulation frame it applies to,
 * [tMs] the touch event time relative to the recording start (for finger velocity).
 * kind: 0 down, 1 move, 2 up, 3 param change (key in [key], value in [x]).
 */
data class InputEvent(
    val frame: Long,
    val tMs: Long,
    val kind: Int,
    val id: Int,
    val x: Float,
    val y: Float,
    val mode: Mode = Mode.SPAWN,
    val key: String = "",
)

/**
 * A recording: the starting preset, seed and every slider value, then the input stream.
 * The simulation runs a fixed number of steps per frame and spawns from a seeded RNG,
 * so replaying on the same phone reproduces the run (GPU float maths is deterministic
 * per device; swallowed mass uses integer atomics so its order does not matter).
 */
class Recording(
    val preset: Preset,
    val seed: Long,
    val params: Map<String, Float>,
    val events: MutableList<InputEvent> = ArrayList(),
) {
    var frames: Long = 0

    fun write(f: File) {
        f.parentFile?.mkdirs()
        f.bufferedWriter().use { w ->
            w.write("gravity-sandbox-recording 1\n")
            w.write("preset ${preset.name}\nseed $seed\nframes $frames\n")
            for ((k, v) in params) w.write("param $k $v\n")
            for (e in events) {
                w.write("e ${e.frame} ${e.tMs} ${e.kind} ${e.id} ${e.x} ${e.y} ${e.mode.name} ${e.key.ifEmpty { "-" }}\n")
            }
        }
    }

    companion object {
        fun read(f: File): Recording? {
            if (!f.exists()) return null
            var preset = Preset.DISK
            var seed = 1L
            var frames = 0L
            val params = LinkedHashMap<String, Float>()
            val events = ArrayList<InputEvent>()
            f.forEachLine { line ->
                val t = line.split(' ')
                when (t[0]) {
                    "preset" -> preset = runCatching { Preset.valueOf(t[1]) }.getOrDefault(Preset.DISK)
                    "seed" -> seed = t[1].toLong()
                    "frames" -> frames = t[1].toLong()
                    "param" -> params[t[1]] = t[2].toFloat()
                    "e" -> events += InputEvent(
                        t[1].toLong(), t[2].toLong(), t[3].toInt(), t[4].toInt(), t[5].toFloat(), t[6].toFloat(),
                        Mode.valueOf(t[7]), if (t[8] == "-") "" else t[8],
                    )
                }
            }
            return Recording(preset, seed, params, events).also { it.frames = frames }
        }
    }
}
