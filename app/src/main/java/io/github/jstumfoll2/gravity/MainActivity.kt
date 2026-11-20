package io.github.jstumfoll2.gravity

import android.app.Activity
import android.content.SharedPreferences
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.widget.Button
import android.widget.FrameLayout
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.PopupMenu
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView

class MainActivity : Activity(), SimListener {

    private lateinit var prefs: SharedPreferences
    private lateinit var renderer: SimRenderer
    private lateinit var simView: SimView
    private lateinit var statsView: TextView
    private lateinit var messageView: TextView
    private lateinit var gpuInfoView: TextView
    private lateinit var tunePanel: ScrollView
    private lateinit var recButton: Button
    private lateinit var replayButton: Button
    private val modeButtons = HashMap<Mode, Button>()
    private val sliderRefreshers = ArrayList<() -> Unit>()
    private val main = Handler(Looper.getMainLooper())
    private val hideMessage = Runnable { messageView.animate().alpha(0f).setDuration(400).start() }

    private val accent = Color.parseColor("#FFB347")
    private val panelBg = Color.parseColor("#E00B0D18")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        prefs = getSharedPreferences("params", MODE_PRIVATE)
        Params.load(prefs)

        renderer = SimRenderer(this, this)
        simView = SimView(this, renderer)

        val root = FrameLayout(this)
        root.setBackgroundColor(Color.BLACK)
        root.addView(simView, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT))

        statsView = TextView(this).apply {
            setTextColor(Color.parseColor("#B0C4DE"))
            textSize = 11f
            typeface = Typeface.MONOSPACE
            setPadding(dp(10), dp(8), dp(10), dp(4))
            setShadowLayer(3f, 0f, 0f, Color.BLACK)
        }
        root.addView(statsView, FrameLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT, Gravity.TOP or Gravity.START))

        messageView = TextView(this).apply {
            setTextColor(Color.WHITE)
            textSize = 13f
            gravity = Gravity.CENTER
            setPadding(dp(14), dp(8), dp(14), dp(8))
            background = rounded(Color.parseColor("#C0101830"), 16)
            alpha = 0f
        }
        root.addView(messageView, FrameLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT, Gravity.TOP or Gravity.CENTER_HORIZONTAL).apply {
            topMargin = dp(34); leftMargin = dp(16); rightMargin = dp(16)
        })

        root.addView(buildToolbar(), FrameLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT, Gravity.BOTTOM))

        tunePanel = buildTunePanel()
        tunePanel.visibility = View.GONE
        root.addView(tunePanel, FrameLayout.LayoutParams(MATCH_PARENT, MATCH_PARENT).apply {
            bottomMargin = dp(56)
        })

        setContentView(root)
        immersive()
        selectMode(Mode.SPAWN)
    }

    @Suppress("DEPRECATION")
    private fun immersive() {
        window.decorView.systemUiVisibility = (View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY
            or View.SYSTEM_UI_FLAG_FULLSCREEN
            or View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
            or View.SYSTEM_UI_FLAG_LAYOUT_STABLE
            or View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION
            or View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN)
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        if (hasFocus) immersive()
    }

    override fun onResume() { super.onResume(); simView.onResume() }

    override fun onPause() {
        simView.queueEvent { renderer.stopRecording() }
        simView.onPause()
        Params.save(prefs)
        super.onPause()
    }

    // ---------------------------------------------------------------- toolbar

    private fun buildToolbar(): View {
        val row = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(6), dp(6), dp(6), dp(6))
        }
        for ((m, label) in listOf(Mode.SPAWN to "Spawn", Mode.PULL to "Pull", Mode.PUSH to "Push", Mode.HOLE to "Hole")) {
            val b = button(label) { selectMode(m) }
            modeButtons[m] = b
            row.addView(b)
        }
        row.addView(spacer())
        row.addView(button("Scene") { v -> scenePicker(v) })
        row.addView(button("Reset") { simView.queueEvent { renderer.restart() } })
        row.addView(button("Tune") { toggleTune() })
        recButton = button("● Rec") {
            simView.queueEvent { if (recButtonIsStop) renderer.stopRecording() else renderer.startRecording() }
        }
        row.addView(recButton)
        replayButton = button("▶ Replay") { simView.queueEvent { renderer.startReplay() } }
        row.addView(replayButton)
        val scroll = HorizontalScrollView(this).apply {
            isHorizontalScrollBarEnabled = false
            setBackgroundColor(Color.parseColor("#A0000000"))
            addView(row)
        }
        return scroll
    }

    private var recButtonIsStop = false

    private fun selectMode(m: Mode) {
        simView.queueEvent { renderer.mode = m }
        for ((k, b) in modeButtons) {
            b.background = if (k == m) rounded(accent, 10) else rounded(Color.parseColor("#33FFFFFF"), 10)
            b.setTextColor(if (k == m) Color.BLACK else Color.WHITE)
        }
        val hint = when (m) {
            Mode.SPAWN -> "Drag to spray matter. Flick to throw it."
            Mode.PULL -> "Hold to pull matter toward your thumb"
            Mode.PUSH -> "Hold to push matter away"
            Mode.HOLE -> "Tap to drop a black hole, flick to throw one"
        }
        onMessage(hint)
    }

    private fun scenePicker(anchor: View) {
        val menu = PopupMenu(this, anchor)
        Preset.values().forEachIndexed { i, p -> menu.menu.add(0, i, i, p.label) }
        menu.setOnMenuItemClickListener { item ->
            val p = Preset.values()[item.itemId]
            simView.queueEvent { renderer.newScene(p) }
            true
        }
        menu.show()
    }

    // ---------------------------------------------------------------- tuning panel

    private fun toggleTune() {
        if (tunePanel.visibility == View.VISIBLE) {
            tunePanel.visibility = View.GONE
            Params.save(prefs)
        } else {
            sliderRefreshers.forEach { it() }
            tunePanel.visibility = View.VISIBLE
        }
    }

    private fun buildTunePanel(): ScrollView {
        val col = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(24))
        }
        col.addView(TextView(this).apply {
            text = "Tuning"
            setTextColor(Color.WHITE); textSize = 20f; typeface = Typeface.DEFAULT_BOLD
        })
        gpuInfoView = TextView(this).apply {
            setTextColor(Color.parseColor("#8090A0")); textSize = 11f
        }
        col.addView(gpuInfoView)

        val actions = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, dp(8), 0, dp(4)) }
        actions.addView(button("Benchmark") { simView.queueEvent { renderer.benchmark() } })
        actions.addView(button("Defaults") {
            Params.resetDefaults()
            Params.save(prefs)
            sliderRefreshers.forEach { it() }
            simView.queueEvent { renderer.restart() }
        })
        actions.addView(button("Close") { toggleTune() })
        col.addView(actions)

        var section = ""
        for (p in Params.all) {
            if (p.section != section) {
                section = p.section
                col.addView(TextView(this).apply {
                    text = section.uppercase()
                    setTextColor(accent); textSize = 12f; typeface = Typeface.DEFAULT_BOLD
                    letterSpacing = 0.1f
                    setPadding(0, dp(16), 0, dp(2))
                })
            }
            col.addView(sliderRow(p))
        }

        return ScrollView(this).apply {
            setBackgroundColor(panelBg)
            addView(col)
            // keep touches from falling through to the simulation
            isClickable = true
        }
    }

    private fun sliderRow(p: Param): View {
        val box = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(6), 0, 0) }
        val head = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val label = TextView(this).apply {
            text = p.label + if (p.restart) "  (applies on reset)" else ""
            setTextColor(Color.WHITE); textSize = 14f
        }
        val value = TextView(this).apply {
            setTextColor(accent); textSize = 14f; typeface = Typeface.MONOSPACE
            gravity = Gravity.END
        }
        head.addView(label, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        head.addView(value)
        box.addView(head)
        val bar = SeekBar(this).apply { max = Param.SLIDER_STEPS }
        box.addView(bar, LinearLayout.LayoutParams(MATCH_PARENT, dp(32)))
        if (p.help.isNotEmpty()) {
            box.addView(TextView(this).apply {
                text = p.help; setTextColor(Color.parseColor("#7A8899")); textSize = 11f
            })
        }
        val refresh = {
            bar.progress = p.toSlider()
            value.text = p.format()
        }
        refresh()
        sliderRefreshers += refresh
        bar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(sb: SeekBar, progress: Int, fromUser: Boolean) {
                if (!fromUser) return
                val v = p.fromSlider(progress)
                value.text = p.format(v)
                if (!p.restart) {
                    p.value = v
                    simView.queueEvent { renderer.paramChanged(p.key, v) }
                }
            }
            override fun onStartTrackingTouch(sb: SeekBar) {}
            override fun onStopTrackingTouch(sb: SeekBar) {
                if (p.restart) {
                    p.value = p.fromSlider(sb.progress)
                    simView.queueEvent { renderer.restart() }
                }
                Params.save(prefs)
            }
        })
        // long-press the name to restore this one slider
        label.setOnLongClickListener {
            p.value = p.default
            refresh()
            if (p.restart) simView.queueEvent { renderer.restart() }
            else simView.queueEvent { renderer.paramChanged(p.key, p.default) }
            true
        }
        return box
    }

    // ---------------------------------------------------------------- SimListener (GL thread)

    override fun onStats(text: String) { main.post { statsView.text = text } }

    override fun onMessage(text: String) {
        main.post {
            messageView.text = text
            messageView.animate().cancel()
            messageView.alpha = 1f
            main.removeCallbacks(hideMessage)
            main.postDelayed(hideMessage, if (text.length > 50) 5000 else 2500)
        }
    }

    override fun onFatal(text: String) {
        main.post {
            messageView.animate().cancel()
            main.removeCallbacks(hideMessage)
            messageView.text = text
            messageView.alpha = 1f
        }
    }

    override fun onRecordingChanged(recording: Boolean, replaying: Boolean) {
        main.post {
            recButtonIsStop = recording
            recButton.text = if (recording) "■ Stop" else "● Rec"
            recButton.setTextColor(if (recording) Color.parseColor("#FF6060") else Color.WHITE)
            replayButton.setTextColor(if (replaying) accent else Color.WHITE)
        }
    }

    override fun onGpuInfo(text: String) { main.post { gpuInfoView.text = text } }

    // ---------------------------------------------------------------- helpers

    private fun dp(v: Int): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v.toFloat(), resources.displayMetrics).toInt()

    private fun rounded(color: Int, radiusDp: Int) = GradientDrawable().apply {
        setColor(color); cornerRadius = dp(radiusDp).toFloat()
    }

    private fun spacer() = View(this).apply { layoutParams = LinearLayout.LayoutParams(dp(10), 1) }

    private fun button(label: String, onClick: (View) -> Unit): Button = Button(this).apply {
        text = label
        isAllCaps = false
        textSize = 13f
        setTextColor(Color.WHITE)
        background = rounded(Color.parseColor("#33FFFFFF"), 10)
        minWidth = 0; minimumWidth = 0
        minHeight = 0; minimumHeight = 0
        setPadding(dp(12), dp(8), dp(12), dp(8))
        layoutParams = LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT).apply { marginEnd = dp(6) }
        setOnClickListener { onClick(it) }
    }
}
