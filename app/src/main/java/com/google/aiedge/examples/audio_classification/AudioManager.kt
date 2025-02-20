package com.google.aiedge.examples.audio_classification

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.core.app.ActivityCompat
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.isActive
import java.util.concurrent.atomic.AtomicInteger

class AudioManager(
    private var sampleRate: Int,
    private var bufferSize: Int,
    private var overlap: Float
) {
    private var audioRecord: AudioRecord? = null
    private var audioBuffer: ShortArray? = null
    private var overlapBuffer: ShortArray? = null
    private val stepSize = AtomicInteger(0)
    private var samplesPerStep: Int = 0

    companion object {
        private const val TAG = "AudioManager"
    }

    /**
     * Sets the feature step size for processing
     * @param stepSize Step size in milliseconds
     */
    fun setFeatureStepSize(stepSize: Int) {
        require(stepSize > 0) { "Step size must be positive" }
        
        try {
            synchronized(this) {
                this.stepSize.set(stepSize)
                
                // Calculate samples per step based on sample rate
                samplesPerStep = (sampleRate * (stepSize / 1000.0)).toInt()
                
                // Validate against buffer size
                val frameSize = bufferSize
                if (samplesPerStep > frameSize) {
                    throw IllegalArgumentException(
                        "Step size $stepSize ms requires $samplesPerStep samples, " +
                        "but buffer only holds $frameSize samples"
                    )
                }

                // Update overlap calculations
                val effectiveOverlap = frameSize - samplesPerStep
                if (effectiveOverlap > 0) {
                    overlapBuffer = ShortArray(effectiveOverlap)
                }
                
                Log.d(TAG, "Feature step size configured: ${stepSize}ms, " +
                          "samples per step: $samplesPerStep")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Feature step size configuration failed", e)
            throw IllegalStateException("Failed to configure feature step size", e)
        }
    }

    @SuppressLint("MissingPermission")
    suspend fun record(): Flow<ShortArray> {
        return flow {
            val effectiveBufferSize = (bufferSize * (1 - overlap)).toInt()
            Log.i(TAG, "Effective buffer size = $effectiveBufferSize")

            try {
                initializeAudioRecord(effectiveBufferSize)
                val localBuffer = ShortArray(effectiveBufferSize)

                if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
                    Log.e(TAG, "AudioRecord failed to initialize")
                    return@flow
                }

                audioRecord?.startRecording()

                while (currentCoroutineContext().isActive) {
                    when (val readResult = audioRecord?.read(localBuffer, 0, localBuffer.size) ?: 0) {
                        AudioRecord.ERROR_INVALID_OPERATION,
                        AudioRecord.ERROR_BAD_VALUE,
                        AudioRecord.ERROR_DEAD_OBJECT,
                        AudioRecord.ERROR -> {
                            Log.w(TAG, "AudioRecord error: $readResult")
                            continue
                        }
                        effectiveBufferSize -> {
                            processAndEmitAudio(localBuffer)?.let { emit(it) }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Recording error", e)
                throw e
            }
        }
    }

    @SuppressLint("MissingPermission")
    private fun initializeAudioRecord(bufferSize: Int) {
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        ).apply {
            if (state != AudioRecord.STATE_INITIALIZED) {
                throw IllegalStateException("Failed to initialize AudioRecord")
            }
        }
    }

    private fun processAndEmitAudio(buffer: ShortArray): ShortArray? {
        return if (stepSize.get() > 0) {
            // Process with step size
            processWithStepSize(buffer)
        } else {
            // Direct processing
            buffer.clone()
        }
    }

    private fun processWithStepSize(buffer: ShortArray): ShortArray {
        val processed = ShortArray(samplesPerStep)
        System.arraycopy(buffer, 0, processed, 0, samplesPerStep)
        return processed
    }

    fun stopRecord() {
        try {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
            overlapBuffer = null
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recording", e)
        }
    }
}