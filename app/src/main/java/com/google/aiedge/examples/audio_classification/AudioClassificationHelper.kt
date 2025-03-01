/*
 * Copyright 2024 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.aiedge.examples.audio_classification

import android.annotation.SuppressLint
import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Performs classification on sound.
 *
 * <p>The API supports models which accept sound input via {@code AudioRecord} and one classification output tensor.
 * The output of the recognition is emitted as LiveData of Map.
 *
 */
class AudioClassificationHelper(private val context: Context, val options: Options = Options()) {
    class Options(
        /** Overlap factor of recognition period */
        var overlapFactor: Float = DEFAULT_OVERLAP,
        /** Probability value above which a class is labeled as active (i.e., detected) the display.  */
        var probabilityThreshold: Float = DEFAULT_PROBABILITY_THRESHOLD,
        /** The enum contains the model file name, relative to the assets/ directory */
        var currentModel: TFLiteModel = DEFAULT_MODEL,
        /** The delegate for running computationally intensive operations*/
        var delegate: Delegate = DEFAULT_DELEGATE,
        /** Number of output classes of the TFLite model.  */
        var resultCount: Int = DEFAULT_RESULT_COUNT,
        /** Number of threads to be used for ops that support multi-threading.
         * threadCount>= -1. Setting numThreads to 0 has the effect of disabling multithreading,
         * which is equivalent to setting numThreads to 1. If unspecified, or set to the value -1,
         * the number of threads used will be implementation-defined and platform-dependent.
         * */
        var threadCount: Int = DEFAULT_THREAD_COUNT
    )

    /** As the result of sound classification, this value emits map of probabilities */
    val probabilities: SharedFlow<Pair<List<Category>, Long>>
        get() = _probabilities
    private val _probabilities = MutableSharedFlow<Pair<List<Category>, Long>>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )


    /** The TFLite interpreter instance.  */
    private var interpreter: Interpreter? = null

    private var job: Job? = null

    private var audioManager: AudioManager? = null

    /** Stop, cancel or reset all necessary variable*/
    fun stop() {
        job?.cancel()
        audioManager?.stopRecord()
        interpreter?.resetVariableTensors()
        interpreter?.close()
        interpreter = null
    }

    suspend fun setupInterpreter() {
        interpreter = try {
            val litertBuffer = FileUtil.loadMappedFile(context, options.currentModel.fileName)
            Log.i(TAG, "Done creating TFLite buffer from ${options.currentModel}")
            Interpreter(litertBuffer, Interpreter.Options().apply {
                numThreads = options.threadCount
                useNNAPI = options.delegate == Delegate.NNAPI
            })
        } catch (e: IOException) {
            throw IOException("Failed to load TFLite model - ${e.message}")
        } catch (e: Exception) {
            throw Exception("Failed to create Interpreter - ${e.message}")
        }
        val inputTensor = interpreter?.getInputTensor(0)
        val outputTensor = interpreter?.getOutputTensor(0)

        if (inputTensor == null || outputTensor == null) {
            throw RuntimeException("Tensor initialization failed")
        }
    }

    /*
    * Starts sound classification, which triggers running on IO Thread
    */
    @SuppressLint("MissingPermission")
    suspend fun startRecord() {
        withContext(Dispatchers.IO) {
            // Inspect input and output specs.
            val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return@withContext

            audioManager = AudioManager(
                options.currentModel.sampleRate,
                inputShape[1] * inputShape[2],
                options.overlapFactor
            ).apply {
                setFeatureStepSize(FEATURE_STEP_SIZE)
            }

            audioManager!!.record().collect {
                val array = convertShortToInt8(it)
                startRecognition(array)
            }
        }
    }

    private val slidingWindow = ArrayDeque<ByteArray>(SLIDING_WINDOW_SIZE)

    private suspend fun startRecognition(audioArray: ByteArray) {
        val inputShape = interpreter?.getInputTensor(0)?.shape() ?: return
        val quantizationParameters = interpreter?.getInputTensor(0)?.quantizationParams() ?: return
        val requiredLength = inputShape[1] * inputShape[2] // 3 * 40 = 120 elements
        val quatizationParametersOutput = interpreter?.getOutputTensor(0)?.quantizationParams() ?: return


        // Add to sliding window
        if (slidingWindow.size >= SLIDING_WINDOW_SIZE) {
            slidingWindow.removeFirst()
        }
        slidingWindow.addLast(audioArray)

        // Only process if we have enough data
        if (slidingWindow.size < SLIDING_WINDOW_SIZE) {
            Log.d(TAG, "Waiting for more windows: ${slidingWindow.size}/$SLIDING_WINDOW_SIZE")
            return
        }

        // Calculate total available data
        val totalAvailableData = slidingWindow.sumOf { it.size }
        if (totalAvailableData < requiredLength) {
            Log.d(TAG, "Insufficient total data: $totalAvailableData/$requiredLength bytes")
            return
        }

        // Combine data from sliding windows
        val combinedArray = ByteArray(requiredLength)
        var position = 0

         for (windowData in slidingWindow) {
            val copyLength = minOf(windowData.size, requiredLength - position)
            if (copyLength > 0) {
                System.arraycopy(windowData, 0, combinedArray, position, copyLength)
                position += copyLength
            }
        }

        if (position < requiredLength) {
            Log.e(TAG, "Failed to fill buffer: $position/$requiredLength bytes")
            return
        }

        // Allocate input buffer with specific tensor arena size
        val inputBuffer = ByteBuffer.allocateDirect(requiredLength).apply {
            order(ByteOrder.nativeOrder())
        }

        // Quantization
        combinedArray.forEach { value ->
            val floatValue = value.toFloat()
            val quantizedValue = (floatValue / quantizationParameters.scale + quantizationParameters.zeroPoint).toInt().coerceIn(-128, 127)
            inputBuffer.put(quantizedValue.toByte())
        }


        // Output buffer (1x1 uint8)
        val outputBuffer = ByteBuffer.allocateDirect(1).apply {
            order(ByteOrder.nativeOrder())
        }

        // Run inference with performance monitoring
        val startTime = SystemClock.uptimeMillis()
        try {
            inputBuffer.rewind()
            outputBuffer.rewind()
            interpreter?.run(inputBuffer, outputBuffer)

            outputBuffer.rewind()
            val rawOutput = outputBuffer.get().toInt() and 0xFF
            //quatizationParametersOutput.scale
            val probability = (rawOutput - quatizationParametersOutput.zeroPoint) * quatizationParametersOutput.scale
            // Apply probability cutoff
            if (rawOutput > 4) {
                Log.d(TAG, "probability:${rawOutput} $probability $PROBABILITY_CUTOFF")
            }

//            if (probability >= PROBABILITY_CUTOFF) {
            val categories = listOf(Category(
                label = "Okay Nabu detected",
                score = probability
            ))

            val inferenceTime = SystemClock.uptimeMillis() - startTime
            _probabilities.emit(Pair(categories, inferenceTime))

                // Early exit on detection
//            }
        } catch (e: Exception) {
            Log.e(TAG, "Inference error: ${e.message}")
        }
    }

    // Convert ShortArray to ByteArray
    private fun convertShortToInt8(shortAudio: ShortArray): ByteArray {
        val max8Bit = Byte.MAX_VALUE.toFloat()
        // Normalize to fit within the Byte range (-128 to 127)
        return ByteArray(shortAudio.size) { i ->
            val normalizedValue = shortAudio[i] / Short.MAX_VALUE.toFloat()

            // Scale to 8-bit signed range [-128, 127]
            val scaledValue = (normalizedValue * max8Bit)

            scaledValue.toInt().toByte() // Convert to Byte
        }
    }

    private fun convertShortToFloat(shortAudio: ShortArray): FloatArray {
        val audioLength = shortAudio.size
        val floatAudio = FloatArray(audioLength)

        // Loop and convert each short value to float
        for (i in 0 until audioLength) {
            floatAudio[i] = shortAudio[i].toFloat() / Short.MAX_VALUE
        }
        return floatAudio
    }

    companion object {
        private const val TAG = "SoundClassifier"
        private const val FEATURE_STEP_SIZE = 5
        private const val SLIDING_WINDOW_SIZE = 5
        private const val PROBABILITY_CUTOFF = 0.85f

        val DEFAULT_MODEL = TFLiteModel.SpeechCommand
        val DEFAULT_DELEGATE = Delegate.CPU
        const val DEFAULT_THREAD_COUNT = 2
        const val DEFAULT_RESULT_COUNT = 3
        const val DEFAULT_OVERLAP = 0f
        const val DEFAULT_PROBABILITY_THRESHOLD = PROBABILITY_CUTOFF
    }

    enum class Delegate {
        CPU, NNAPI
    }

    enum class TFLiteModel(val fileName: String, val labelFile: String, val sampleRate: Int) {
        SpeechCommand(
            "okay_nabu.tflite",
            "speech_label.txt",
            16000
        )
    }
}

data class Category(val label: String, val score: Float)
