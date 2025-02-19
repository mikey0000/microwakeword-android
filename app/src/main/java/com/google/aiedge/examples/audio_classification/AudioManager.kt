package com.google.aiedge.examples.audio_classification

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
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
        private const val MIN_BUFFER_MULTIPLIER = 2
    }

    /**
     * Sets the buffer size with proper validation and optimization
     * @param size Desired buffer size in bytes
     */
    fun setBufferSize(size: Int) {
        require(size > 0) { "Buffer size must be positive" }
        
        try {
            val minBufferSize = AudioRecord.getMinBufferSize(
                sampleRate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )

            // Ensure buffer is adequate for high-quality audio capture
            val optimalBufferSize = maxOf(
                size,
                minBufferSize * MIN_BUFFER_MULTIPLIER
            )

            synchronized(this) {
                // Clean up existing resources
                audioRecord?.release()
                
                // Initialize new AudioRecord with optimal buffer
                audioRecord = AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    sampleRate,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    optimalBufferSize
                ).apply {
                    if (state != AudioRecord.STATE_INITIALIZED) {
                        throw IllegalStateException("Failed to initialize AudioRecord")
                    }
                }

                bufferSize = optimalBufferSize
                audioBuffer = ShortArray(optimalBufferSize / 2)
                
                // Update overlap buffer if needed
                if (overlap > 0) {
                    val overlapSize = (optimalBufferSize * overlap).toInt()
                    overlapBuffer = ShortArray(overlapSize)
                }
            }
            
            Log.d(TAG, "Buffer size set to: $optimalBufferSize bytes")
        } catch (e: Exception) {
            Log.e(TAG, "Buffer size configuration failed", e)
            throw IllegalStateException("Failed to configure audio buffer", e)
        }
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
                val frameSize = audioBuffer?.size ?: 0
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
            audioBuffer = null
            overlapBuffer = null
        } catch (e: Exception) {
            Log.e(TAG, "Error stopping recording", e)
        }
    }
}