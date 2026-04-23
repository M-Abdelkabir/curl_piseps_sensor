package com.example.curl_piseps

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.os.Build
import android.media.MediaRecorder
import android.media.MediaPlayer
import android.media.ToneGenerator
import android.hardware.camera2.CameraManager
import android.util.Log
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.util.*
import java.util.concurrent.Executors
import java.util.concurrent.ExecutorService

class MainActivity : AppCompatActivity(), SensorEventListener {

    private lateinit var sensorManager: SensorManager
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null

    private val windowSize = 100
    private val accelBuffer = Collections.synchronizedList(mutableListOf<FloatArray>())
    private val gyroBuffer = Collections.synchronizedList(mutableListOf<FloatArray>())

    private var module: Module? = null
    private lateinit var resultTextView: TextView
    private lateinit var statusTextView: TextView
    private lateinit var startButton: Button
    private lateinit var stopButton: Button
    private lateinit var recordButton: Button
    private lateinit var perfectButton: Button
    private lateinit var imperfectButton: Button
    private lateinit var recordPerfectVoiceButton: Button
    private lateinit var recordImperfectVoiceButton: Button

    private var cameraManager: CameraManager? = null
    private var cameraId: String? = null
    private var vibrator: Vibrator? = null
    private var mediaRecorder: MediaRecorder? = null
    private var mediaPlayer: MediaPlayer? = null

    private var isRecordingVoice = false
    private lateinit var perfectVoicePath: String
    private lateinit var imperfectVoicePath: String

    private var isSensing = false
    private var isRecording = false
    private val recordedAccel = mutableListOf<FloatArray>()
    private val recordedGyro = mutableListOf<FloatArray>()
    
    private val inferenceExecutor: ExecutorService = Executors.newSingleThreadExecutor()
    private var isProcessing = false
    
    // Normalization stats (updated after training)
    private var means = floatArrayOf(-0.9451f, 0.6241f, 3.0560f, 0.0056f, 0.0406f, 0.0248f)
    private var stds = floatArrayOf(3.4365f, 1.5086f, 6.9630f, 1.6976f, 1.9673f, 1.1526f)

    // Butterworth filter coefficients (4th order, 10Hz cutoff @ 50Hz)
    private val b = floatArrayOf(0.0048f, 0.0193f, 0.0289f, 0.0193f, 0.0048f)
    private val a = floatArrayOf(1.0000f, -2.3695f, 2.3140f, -1.0547f, 0.1874f)
    
    // Filter state for 6 channels (3 accel + 3 gyro)
    private val filterStatesX = Array(6) { FloatArray(4) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            setContentView(R.layout.activity_main)

            resultTextView = findViewById(R.id.resultTextView)
            statusTextView = findViewById(R.id.statusTextView)
            startButton = findViewById(R.id.startButton)
            stopButton = findViewById(R.id.stopButton)

            sensorManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
            accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
            gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

            if (accelerometer == null || gyroscope == null) {
                Toast.makeText(this, "Required sensors not available!", Toast.LENGTH_LONG).show()
                statusTextView.text = "Error: Sensors Missing"
            }

            try {
                module = LiteModuleLoader.load(assetFilePath(this, "curl_classifier.pt"))
                statusTextView.text = "Model Loaded Successfully"
            } catch (e: Exception) {
                Log.e("PytorchMobile", "Error loading model", e)
                statusTextView.text = "Error Loading Model: ${e.message}"
                Toast.makeText(this, "Model load failed!", Toast.LENGTH_LONG).show()
            }

            startButton.setOnClickListener {
                startSensing()
            }

            stopButton.setOnClickListener {
                stopSensing()
            }

            recordButton = findViewById(R.id.recordButton)
            perfectButton = findViewById(R.id.perfectButton)
            imperfectButton = findViewById(R.id.imperfectButton)

            recordButton.setOnClickListener {
                if (isRecording) {
                    stopRecording()
                } else {
                    startRecording()
                }
            }

            perfectButton.setOnClickListener {
                saveRecording("perfect")
            }

            imperfectButton.setOnClickListener {
                saveRecording("imperfect")
            }

            // Request Permissions
            val permissions = arrayOf(
                android.Manifest.permission.CAMERA,
                android.Manifest.permission.RECORD_AUDIO
            )
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                requestPermissions(permissions, 100)
            }

            // Initialize Feedback Managers
            cameraManager = getSystemService(Context.CAMERA_SERVICE) as CameraManager
            cameraId = cameraManager?.cameraIdList?.firstOrNull { id ->
                val characteristics = cameraManager?.getCameraCharacteristics(id)
                characteristics?.get(android.hardware.camera2.CameraCharacteristics.FLASH_INFO_AVAILABLE) == true
            }

            vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                val vibratorManager = getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
                vibratorManager.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                getSystemService(Context.VIBRATOR_SERVICE) as Vibrator
            }

            perfectVoicePath = "${getExternalFilesDir(null)}/perfect_voice.3gp"
            imperfectVoicePath = "${getExternalFilesDir(null)}/imperfect_voice.3gp"

            recordPerfectVoiceButton = findViewById(R.id.recordPerfectVoiceButton)
            recordImperfectVoiceButton = findViewById(R.id.recordImperfectVoiceButton)

            recordPerfectVoiceButton.setOnClickListener {
                handleVoiceRecording(perfectVoicePath, recordPerfectVoiceButton)
            }

            recordImperfectVoiceButton.setOnClickListener {
                handleVoiceRecording(imperfectVoicePath, recordImperfectVoiceButton)
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error in onCreate", e)
            // Even if layout fails, this might help see what's wrong in Logcat
        }
    }

    private fun startSensing() {
        if (!isSensing) {
            accelBuffer.clear()
            gyroBuffer.clear()
            for (i in 0 until 6) {
                filterStatesX[i].fill(0f)
            }
            
            val accelRegistered = accelerometer?.let { 
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) 
            } ?: false
            
            val gyroRegistered = gyroscope?.let { 
                sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) 
            } ?: false

            if (accelRegistered || gyroRegistered) {
                isSensing = true
                startButton.isEnabled = false
                stopButton.isEnabled = true
                statusTextView.text = "Status: Monitoring..."
                resultTextView.text = "Waiting for data..."
            } else {
                Toast.makeText(this, "Could not start sensors", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun stopSensing() {
        if (isSensing) {
            sensorManager.unregisterListener(this)
            isSensing = false
            startButton.isEnabled = true
            stopButton.isEnabled = false
            statusTextView.text = "Status: Analyzing..."
            
            // Run final prediction on the collected buffer
            runFinalInference()
        }
    }

    override fun onPause() {
        super.onPause()
        if (isSensing) {
            stopSensing()
        }
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (!isSensing) return

        if (event.sensor.type == Sensor.TYPE_ACCELEROMETER) {
            accelBuffer.add(event.values.clone())
            if (isRecording) recordedAccel.add(event.values.clone())
        } else if (event.sensor.type == Sensor.TYPE_GYROSCOPE) {
            gyroBuffer.add(event.values.clone())
            if (isRecording) recordedGyro.add(event.values.clone())
        }

        if (accelBuffer.size > 2000) accelBuffer.removeAt(0) // Prevent memory leak if left running
        if (gyroBuffer.size > 2000) gyroBuffer.removeAt(0)
/*
        if (accelBuffer.size >= windowSize && gyroBuffer.size >= windowSize && !isProcessing) {
            isProcessing = true
            inferenceExecutor.execute {
                try {
                    processAndInference()
                } finally {
                    isProcessing = false
                }
            }
        }
*/
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    private fun processAndInference() {
        val accelData: List<FloatArray>
        val gyroData: List<FloatArray>
        
        synchronized(accelBuffer) {
            if (accelBuffer.size < windowSize) return
            accelData = accelBuffer.take(windowSize).toList()
            repeat(50) { if (accelBuffer.isNotEmpty()) accelBuffer.removeAt(0) }
        }
        
        synchronized(gyroBuffer) {
            if (gyroBuffer.size < windowSize) return
            gyroData = gyroBuffer.take(windowSize).toList()
            repeat(50) { if (gyroBuffer.isNotEmpty()) gyroBuffer.removeAt(0) }
        }

        try {
            val inputData = FloatArray(1 * 6 * windowSize)
            for (i in 0 until windowSize) {
                inputData[0 * windowSize + i] = (applyFilter(accelData[i][0], 0) - means[0]) / (stds[0] + 1e-7f)
                inputData[1 * windowSize + i] = (applyFilter(accelData[i][1], 1) - means[1]) / (stds[1] + 1e-7f)
                inputData[2 * windowSize + i] = (applyFilter(accelData[i][2], 2) - means[2]) / (stds[2] + 1e-7f)
                inputData[3 * windowSize + i] = (applyFilter(gyroData[i][0], 3) - means[3]) / (stds[3] + 1e-7f)
                inputData[4 * windowSize + i] = (applyFilter(gyroData[i][1], 4) - means[4]) / (stds[4] + 1e-7f)
                inputData[5 * windowSize + i] = (applyFilter(gyroData[i][2], 5) - means[5]) / (stds[5] + 1e-7f)
            }

            val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 6, windowSize.toLong()))
            val outputTensor = module?.forward(IValue.from(inputTensor))?.toTensor()
            val scores = outputTensor?.dataAsFloatArray

            if (scores != null && scores.size >= 2) {
                val maxIdx = if (scores[1] > scores[0]) 1 else 0
                val label = if (maxIdx == 1) "Perfect" else "Imperfect"
                runOnUiThread {
                    resultTextView.text = "Result: $label\n(P: ${String.format("%.2f", scores[1])}, I: ${String.format("%.2f", scores[0])})"
                }
            }
        } catch (e: Exception) {
            Log.e("Inference", "Error during inference", e)
        }
    }

    private fun applyFilter(value: Float, channel: Int): Float {
        val y = b[0] * value + filterStatesX[channel][0]
        filterStatesX[channel][0] = b[1] * value - a[1] * y + filterStatesX[channel][1]
        filterStatesX[channel][1] = b[2] * value - a[2] * y + filterStatesX[channel][2]
        filterStatesX[channel][2] = b[3] * value - a[3] * y + filterStatesX[channel][3]
        filterStatesX[channel][3] = b[4] * value - a[4] * y
        return y
    }

    private fun runFinalInference() {
        val accelData = synchronized(accelBuffer) { accelBuffer.toList() }
        val gyroData = synchronized(gyroBuffer) { gyroBuffer.toList() }

        if (accelData.size < windowSize || gyroData.size < windowSize) {
            runOnUiThread { 
                resultTextView.text = "Result: Too short sequence"
                statusTextView.text = "Status: Stopped"
            }
            return
        }

        inferenceExecutor.execute {
            try {
                val totalLen = minOf(accelData.size, gyroData.size)
                
                // 1. Filter the entire sequence first (to match Python implementation)
                val filteredData = Array(6) { FloatArray(totalLen) }
                for (i in 0 until 6) filterStatesX[i].fill(0f)
                
                for (t in 0 until totalLen) {
                    filteredData[0][t] = applyFilter(accelData[t][0], 0)
                    filteredData[1][t] = applyFilter(accelData[t][1], 1)
                    filteredData[2][t] = applyFilter(accelData[t][2], 2)
                    filteredData[3][t] = applyFilter(gyroData[t][0], 3)
                    filteredData[4][t] = applyFilter(gyroData[t][1], 4)
                    filteredData[5][t] = applyFilter(gyroData[t][2], 5)
                }

                // 2. Run sliding windows on the filtered data
                var totalPerfect = 0f
                var totalImperfect = 0f
                var windowCount = 0

                val maxWindowsStart = totalLen - windowSize
                val step = 25 
                
                for (start in 0..maxWindowsStart step step) {
                    val inputData = FloatArray(1 * 6 * windowSize)
                    for (c in 0 until 6) {
                        for (i in 0 until windowSize) {
                            val rawValue = filteredData[c][start + i]
                            inputData[c * windowSize + i] = (rawValue - means[c]) / (stds[c] + 1e-7f)
                        }
                    }

                    val inputTensor = Tensor.fromBlob(inputData, longArrayOf(1, 6, windowSize.toLong()))
                    val outputTensor = module?.forward(IValue.from(inputTensor))?.toTensor()
                    val logits = outputTensor?.dataAsFloatArray

                    if (logits != null && logits.size >= 2) {
                        // Apply Softmax to get probabilities 0-1
                        val exp0 = Math.exp(logits[0].toDouble())
                        val exp1 = Math.exp(logits[1].toDouble())
                        val sum = exp0 + exp1
                        totalImperfect += (exp0 / sum).toFloat()
                        totalPerfect += (exp1 / sum).toFloat()
                        windowCount++
                    }
                }

                if (windowCount > 0) {
                    val avgPerfect = totalPerfect / windowCount
                    val avgImperfect = totalImperfect / windowCount
                    val label = if (avgPerfect > avgImperfect) "Perfect" else "Imperfect"
                    runOnUiThread {
                        resultTextView.text = "Final Result: $label\n(Avg P: ${String.format("%.2f", avgPerfect)}, I: ${String.format("%.2f", avgImperfect)})"
                        statusTextView.text = "Status: Analysis Complete"
                        triggerFeedback(label == "Perfect")
                    }
                }
            } catch (e: Exception) {
                Log.e("FinalInference", "Error during inference", e)
                runOnUiThread { statusTextView.text = "Error in Analysis" }
            }
        }
    }

    private fun startRecording() {
        recordedAccel.clear()
        recordedGyro.clear()
        isRecording = true
        recordButton.text = "Stop Record"
        perfectButton.isEnabled = false
        imperfectButton.isEnabled = false
        statusTextView.text = "Status: Recording..."
        if (!isSensing) startSensing()
    }

    private fun stopRecording() {
        isRecording = false
        recordButton.text = "Start Record"
        perfectButton.isEnabled = true
        imperfectButton.isEnabled = true
        statusTextView.text = "Status: Record Finished"
    }

    private fun saveRecording(label: String) {
        val timestamp = java.text.SimpleDateFormat("yyyy-MM-dd_HH-mm-ss", Locale.getDefault()).format(Date())
        val dir = File(getExternalFilesDir(null), "data/$label/$timestamp")
        if (!dir.exists()) dir.mkdirs()

        val accelFile = File(dir, "Accelerometer.csv")
        val gyroFile = File(dir, "Gyroscope.csv")

        try {
            accelFile.printWriter().use { out ->
                out.println("seconds_elapsed,x,y,z")
                recordedAccel.forEachIndexed { i, values ->
                    out.println("${i * 0.02},${values[0]},${values[1]},${values[2]}")
                }
            }
            gyroFile.printWriter().use { out ->
                out.println("seconds_elapsed,x,y,z")
                recordedGyro.forEachIndexed { i, values ->
                    out.println("${i * 0.02},${values[0]},${values[1]},${values[2]}")
                }
            }
            Toast.makeText(this, "Saved to $label", Toast.LENGTH_SHORT).show()
            perfectButton.isEnabled = false
            imperfectButton.isEnabled = false
        } catch (e: Exception) {
            Log.e("SaveData", "Error saving data", e)
            Toast.makeText(this, "Save failed: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    @Throws(IOException::class)
    fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { `is` ->
            FileOutputStream(file).use { os ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (`is`.read(buffer).also { read = it } != -1) {
                    os.write(buffer, 0, read)
                }
                os.flush()
            }
            return file.absolutePath
        }
    }

    private fun handleVoiceRecording(path: String, button: Button) {
        if (isRecordingVoice) {
            stopVoiceRecording()
            button.text = if (path == perfectVoicePath) "Rec P Voice" else "Rec I Voice"
        } else {
            startVoiceRecording(path)
            button.text = "Stop Rec"
        }
    }

    private fun startVoiceRecording(path: String) {
        try {
            mediaRecorder = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                MediaRecorder(this)
            } else {
                @Suppress("DEPRECATION")
                MediaRecorder()
            }
            mediaRecorder?.apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.THREE_GPP)
                setOutputFile(path)
                setAudioEncoder(MediaRecorder.AudioEncoder.AMR_NB)
                prepare()
                start()
            }
            isRecordingVoice = true
            Toast.makeText(this, "Recording...", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e("VoiceRecord", "Error starting recording", e)
            Toast.makeText(this, "Record Error: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun stopVoiceRecording() {
        try {
            mediaRecorder?.apply {
                stop()
                release()
            }
            mediaRecorder = null
            isRecordingVoice = false
            Toast.makeText(this, "Recording Saved", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e("VoiceRecord", "Error stopping recording", e)
        }
    }

    private fun playVoice(path: String) {
        val file = File(path)
        if (!file.exists()) return
        
        try {
            mediaPlayer?.release()
            mediaPlayer = MediaPlayer().apply {
                setDataSource(path)
                prepare()
                start()
            }
        } catch (e: Exception) {
            Log.e("VoicePlay", "Error playing voice", e)
        }
    }

    private fun triggerFeedback(isPerfect: Boolean) {
        if (isPerfect) {
            // Flash for Perfect
            cameraId?.let { id ->
                try {
                    cameraManager?.setTorchMode(id, true)
                    Timer().schedule(object : TimerTask() {
                        override fun run() {
                            cameraManager?.setTorchMode(id, false)
                        }
                    }, 1000)
                } catch (e: Exception) {
                    Log.e("Flash", "Error toggling flash", e)
                }
            }
            playVoice(perfectVoicePath)
        } else {
            // Vibration for Imperfect
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                vibrator?.vibrate(VibrationEffect.createOneShot(1000, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                @Suppress("DEPRECATION")
                vibrator?.vibrate(1000)
            }
            playVoice(imperfectVoicePath)
        }
    }
}
