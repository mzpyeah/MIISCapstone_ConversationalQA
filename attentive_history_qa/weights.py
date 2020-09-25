import tensorflow as tf
import numpy as np

reader = tf.train.NewCheckpointReader('./12_batch_output/model_58000.ckpt')
all_variables = reader.get_variable_to_shape_map()
# w1 = reader.get_tensor("attention_weights")
# print(type(w1))
# print(w1.shape)
# print(w1[0])
for key in all_variables:
    if key[:4]=="bert":
        continue
    print("key: ", key)
    print(reader.get_tensor(key).shape)


    # if key == "history_attention_model/kernel":
    #     np.savetxt('attention_weight.txt',reader.get_tensor(key), fmt='%0.8f')
