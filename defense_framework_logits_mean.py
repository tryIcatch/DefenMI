import numpy as np
np.random.seed(1000)
import imp
import input_data_class
import numpy as np
from keras import backend as K
import tensorflow as tf
import os
import configparser
import argparse
from scipy.special import softmax

config = configparser.ConfigParser()
parser = argparse.ArgumentParser()
parser.add_argument('-qt', type=str, default='evaluation')
parser.add_argument('-dataset', default='location')
args = parser.parse_args()
dataset = args.dataset
input_data = input_data_class.InputData(dataset=dataset)
config = configparser.ConfigParser()
config.read('config.ini')

user_label_dim = int(config[dataset]["num_classes"])
num_classes = 1
delta = int(config[dataset]["delta"])
epsilon = int(config[dataset]["epsilon"])
beta = float(config[dataset]["beta"])
max_iteration = int(config[dataset]["max_iteration"])
threshold = int(config[dataset]["threshold"])
user_epochs = int(config[dataset]["user_epochs"])
defense_epochs = int(config[dataset]["defense_epochs"])
result_folder = config[dataset]["result_folder"]
network_architecture = str(config[dataset]["network_architecture"])
fccnet = imp.load_source(str(config[dataset]["network_name"]), network_architecture)

print("Config: ")
print("dataset: {}".format(dataset))
print("result folder: {}".format(result_folder))
print("network architecture: {}".format(network_architecture))

config_gpu = tf.compat.v1.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5
config_gpu.gpu_options.visible_device_list = "0"

tf.compat.v1.disable_eager_execution()

sess = tf.compat.v1.InteractiveSession(config=config_gpu)
sess.run(tf.compat.v1.global_variables_initializer())




print("Loading Evaluation dataset...")

(x_evaluate, y_evaluate, l_evaluate) = input_data.input_data_attacker_evaluate()
print("Loading target model...")
npzdata = np.load(result_folder + "/models/" + "epoch_{}_weights_user.npz".format(user_epochs), allow_pickle=True)
weights = npzdata['x']
input_shape = x_evaluate.shape[1:]
model = fccnet.model_user(input_shape=input_shape, labels_dim=user_label_dim)
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
model.set_weights(weights)
output_logits = model.layers[-2].output
f_evaluate = model.predict(x_evaluate)
predicted_classes = np.argmax(f_evaluate, axis=1)
classes = np.unique(predicted_classes)

classified_samples = {}
x_evaluate_n = x_evaluate.copy()
for cls in classes:
    class_indices = np.where(predicted_classes == cls)[0]
    classified_samples[cls] = x_evaluate[class_indices]

for cls, noisy_samples in classified_samples.items():
    class_indices = np.where(predicted_classes == cls)[0]
    noisy_sample = np.random.laplace(
        loc=0,
        scale=delta / epsilon,
        size=(noisy_samples.shape[1], noisy_samples.shape[0])
    )

    noisy = noisy_sample.T

    x_evaluate_n[class_indices] = x_evaluate[class_indices] + noisy
class_logits = {}
class_confidence = {}
for cls in np.unique(predicted_classes):
    class_indices = np.where(predicted_classes == cls)[0]
    class_samples = x_evaluate[class_indices]
    class_samples_logits= sess.run(output_logits, feed_dict={model.input: class_samples})
    class_mean_logits = np.mean(class_samples_logits, axis=0)
    class_logits[cls] = class_mean_logits
    class_confidence[cls] = softmax(class_mean_logits)

f_evaluate_logits = np.zeros([1, user_label_dim], dtype=np.float)
batch_predict = 100
batch_num = np.ceil(x_evaluate.shape[0] / float(batch_predict))
for i in np.arange(batch_num):
    f_evaluate_logits_temp = sess.run(output_logits, feed_dict={model.input: x_evaluate_n[int(i * batch_predict):int(min((i + 1) * batch_predict, x_evaluate.shape[0])), :]})
    f_evaluate_logits = np.concatenate((f_evaluate_logits, f_evaluate_logits_temp), axis=0)
f_evaluate_logits = f_evaluate_logits[1:, :]  # 获取目标模型在评估数据集上的logits

f_evaluate_logits_origin = np.zeros([1, user_label_dim], dtype=np.float)

batch_num = np.ceil(x_evaluate.shape[0] / float(batch_predict))
for i in np.arange(batch_num):
    f_evaluate_logits_temp = sess.run(output_logits, feed_dict={model.input: x_evaluate_n[int(i * batch_predict):int(min((i + 1) * batch_predict, x_evaluate.shape[0])), :]})
    f_evaluate_logits_origin = np.concatenate((f_evaluate_logits_origin, f_evaluate_logits_temp), axis=0)
f_evaluate_logits_origin = f_evaluate_logits_origin[1:, :]
del model
f_evaluate_origin = np.copy(f_evaluate)

sort_index = np.argsort(f_evaluate, axis=1)
back_index = np.copy(sort_index)
for i in np.arange(back_index.shape[0]):
    back_index[i, sort_index[i, :]] = np.arange(back_index.shape[1])
f_evaluate = np.sort(f_evaluate, axis=1)
f_evaluate_logits = np.sort(f_evaluate_logits, axis=1)


print("f evaluate shape: {}".format(f_evaluate.shape))
print("f evaluate logits shape: {}".format(f_evaluate_logits.shape))

input_shape = f_evaluate.shape[1:]
print("Loading defense model...")
npzdata = np.load(result_folder + "/models/" + "epoch_{}_weights_defense.npz".format(defense_epochs), allow_pickle=True)  # 加载防御模型的权重
model = fccnet.model_defense_optimize(input_shape=input_shape, labels_dim=num_classes)  # 初始化防御模型
model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.001), metrics=['accuracy'])  # 编译模型
weights = npzdata['x']
model.set_weights(weights)
model.trainable = False

scores_evaluate = model.evaluate(f_evaluate_logits, l_evaluate, verbose=0)
print('evaluate loss on model:', scores_evaluate[0])
print('evaluate accuracy on model:', scores_evaluate[1])


output = model.layers[-2].output[:, 0]
c1 = 1.0
c2 = 10.0
c3 = 0.1

origin_value_placeholder = tf.compat.v1.placeholder(tf.float32, shape=(1, user_label_dim))
label_mask = tf.compat.v1.placeholder(tf.float32, shape=(1, user_label_dim))
c1_placeholder = tf.compat.v1.placeholder(tf.float32)
c2_placeholder = tf.compat.v1.placeholder(tf.float32)
c3_placeholder = tf.compat.v1.placeholder(tf.float32)


correct_label = tf.reduce_sum(label_mask * model.input, axis=1)
wrong_label = tf.reduce_max((1 - label_mask) * model.input - 1e8 * label_mask, axis=1)
print("model.input : {}".format(model.input))
print("correct_label : {}".format(correct_label))
print("wrong_label : {}".format(wrong_label) )

loss1 = tf.abs(output)
loss2 = tf.nn.relu(wrong_label - correct_label)
loss3 = tf.reduce_sum(tf.abs(tf.nn.softmax(model.input) - origin_value_placeholder))
loss = c1_placeholder * loss1 + c2_placeholder * loss2 + c3_placeholder * loss3


gradient_targetlabel = K.gradients(loss, model.input)
label_mask_array = np.zeros([1, user_label_dim], dtype=np.float)

result_array = np.zeros(f_evaluate.shape, dtype=np.float)
result_array_logits = np.zeros(f_evaluate.shape, dtype=np.float)
success_fraction = 0.0
np.random.seed(1000)



for test_sample_id in np.arange(0, f_evaluate.shape[0]):
    if test_sample_id % 100 == 0:
        print("test sample id: {}".format(test_sample_id))
    max_label = np.argmax(f_evaluate[test_sample_id, :])
    max_label_origin = np.argmax(f_evaluate_origin[test_sample_id, :])
    origin_value = np.copy(class_confidence[max_label_origin]).reshape(1, user_label_dim)
    origin_value_logits = np.copy(f_evaluate_logits[test_sample_id, :]).reshape(1, user_label_dim)
    label_mask_array[0, :] = 0.0
    label_mask_array[0, max_label] = 1.0
    sample_f = np.copy(origin_value_logits)
    result_predict_scores_initial = model.predict(sample_f)
    if np.abs(result_predict_scores_initial - 0.5) <= 1e-5:
        success_fraction += 1.0
        result_array[test_sample_id, :] = origin_value[0, back_index[test_sample_id, :]]
        result_array_logits[test_sample_id, :] = origin_value_logits[ 0, back_index[test_sample_id, :]]
        continue

    last_iteration_result = np.copy(origin_value)[0, back_index[test_sample_id, :]]
    last_iteration_result_logits = np.copy(origin_value_logits)[
        0, back_index[test_sample_id, :]]
    success = True
    c3 = 0.1
    iterate_time = 1
    while success == True:
        sample_f = np.copy(origin_value_logits)
        j = 1
        result_max_label = -1
        result_predict_scores = result_predict_scores_initial
        while j < max_iteration and (max_label != result_max_label or (result_predict_scores - 0.5) * (
                result_predict_scores_initial - 0.5) > 0):
            gradient_values = sess.run(gradient_targetlabel, feed_dict={
                model.input: sample_f,
                origin_value_placeholder: origin_value,
                label_mask: label_mask_array,
                c3_placeholder: c3,
                c1_placeholder: c1,
                c2_placeholder: c2
            })[0][0]
            gradient_values = gradient_values / np.linalg.norm(gradient_values)
            sample_f = sample_f - beta * gradient_values
            result_predict_scores = model.predict(sample_f)
            result_max_label = np.argmax(sample_f)
            j += 1
        if max_label != result_max_label:
            if iterate_time == 1:
                print("failed sample for label not same for id: {},c3:{} not add noise".format(test_sample_id,
                                                                                               c3))
                success_fraction -= 1.0
            break
        if ((model.predict(sample_f) - 0.5) * (result_predict_scores_initial - 0.5)) > 0:
            if iterate_time == 1:
                print(
                    "max iteration reached with id: {}, max score: {}, prediction_score: {}, c3: {}, not add noise".format(
                        test_sample_id, np.amax(softmax(sample_f)), result_predict_scores, c3))
            break
        last_iteration_result[:] = softmax(sample_f)[0, back_index[test_sample_id, :]]
        last_iteration_result_logits[:] = sample_f[0, back_index[test_sample_id, :]]
        iterate_time += 1
        c3 = c3 * 10

        if c3 > threshold:
            break

    success_fraction += 1.0
    result_array[test_sample_id, :] = last_iteration_result[:]
    result_array_logits[test_sample_id, :] = last_iteration_result_logits[:]

print("Success fraction: {}".format(success_fraction / float(f_evaluate.shape[0])))


if not os.path.exists(result_folder):
    os.makedirs(result_folder)
if not os.path.exists(result_folder + "/attack"):
    os.makedirs(result_folder + "/attack")
del model


input_shape = f_evaluate.shape[1:]
print("Loading defense model...")
npzdata = np.load(result_folder + "/models/" + "epoch_{}_weights_defense.npz".format(defense_epochs),
                  allow_pickle=True)
model = fccnet.model_defense(input_shape=input_shape, labels_dim=num_classes)

weights = npzdata['x']
model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=tf.keras.optimizers.SGD(lr=0.001),
              metrics=['accuracy'])
model.set_weights(weights)
model.trainable = False


predict_origin = model.predict(np.sort(f_evaluate_origin, axis=1))
predict_modified = model.predict(np.sort(result_array, axis=1))
np.savez(result_folder + "/attack/" + "noise_data_noise-l.npz",
         defense_output=result_array,
         defense_output_logits=result_array_logits,
         tc_output=f_evaluate_origin,
         tc_output_logits=f_evaluate_logits_origin,
         predict_origin=predict_origin,
         predict_modified=predict_modified)