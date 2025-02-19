import tensorflow as tf
TFLITE_PATH = "/home/michael/git/litert-samples/examples/audio_classification/android/app/src/main/assets/okay_nabu.tflite" # use absolute path here for tflite model.interpreter = tf.compat.v1.lite.Interpreter(model_path=TFLITE_PATH)
interpreter = tf.compat.v1.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
print("Input shape :\n" , interpreter.get_input_details())
print("Output shape: \n", interpreter.get_output_details())