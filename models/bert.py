import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'
tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

# Load the BERT model and the preprocessing model
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)
bert_model = hub.KerasLayer(tfhub_handle_encoder)


def build_bert_model() -> tf.keras.Model:
    """
    Build a BERT model to extract features from text data.
    """
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    encoder_inputs = bert_preprocess_model(text_input)
    outputs = bert_model(encoder_inputs)
    net = outputs['pooled_output']
    return tf.keras.Model(text_input, net)
