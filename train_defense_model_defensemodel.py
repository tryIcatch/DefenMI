import numpy as np
np.random.seed(1000)
import imp
import input_data_class
import keras
import tensorflow as tf
import os
import configparser
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-dataset',default='location')
args = parser.parse_args()
dataset=args.dataset 
input_data=input_data_class.InputData(dataset=dataset)
config = configparser.ConfigParser()
config.read('config.ini')

user_label_dim=int(config[dataset]["num_classes"])
num_classes=1
save_model=True
defense_epochs=int(config[dataset]["defense_epochs"])
user_epochs=int(config[dataset]["user_epochs"])
batch_size=int(config[dataset]["defense_batch_size"])
result_folder=config[dataset]["result_folder"]
network_architecture=str(config[dataset]["network_architecture"])
fccnet=imp.load_source(str(config[dataset]["network_name"]),network_architecture)

config_gpu = tf.compat.v1.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5
config_gpu.gpu_options.visible_device_list = "0"
session = tf.compat.v1.Session(config=config_gpu)
tf.compat.v1.keras.backend.set_session(session)
tf.compat.v1.disable_eager_execution()
sess = tf.compat.v1.InteractiveSession(config=config_gpu)
sess.run(tf.compat.v1.global_variables_initializer())

(x_train,y_train,l_train) =input_data.input_data_defender()

y_train=tf.keras.utils.to_categorical(y_train,user_label_dim)


####load target model
npzdata=np.load(result_folder+"/models/"+"epoch_{}_weights_user.npz".format(user_epochs),allow_pickle=True)
weights=npzdata['x']
input_shape=x_train.shape[1:]
model=fccnet.model_user(input_shape=input_shape,labels_dim=user_label_dim)
model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.SGD(lr=0.01),metrics=['accuracy'])
model.set_weights(weights)
########obtain confidence score on defender's training dataset
output_logits = model.layers[-2].output
f_train=model.predict(x_train)
f_train=np.sort(f_train,axis=1)
input_shape=y_train.shape[1:]


f_evaluate_logits = np.zeros([1, user_label_dim], dtype=np.float)
batch_predict = 100
batch_num = np.ceil(x_train.shape[0] / float(batch_predict))
for i in np.arange(batch_num):
    f_evaluate_logits_temp = sess.run(output_logits, feed_dict={model.input: x_train[int(i * batch_predict):int(min((i + 1) * batch_predict, x_train.shape[0])), :]})
    f_evaluate_logits = np.concatenate((f_evaluate_logits, f_evaluate_logits_temp), axis=0)
f_evaluate_logits = f_evaluate_logits[1:, :]
del model
f_evaluate_logits = np.sort(f_evaluate_logits, axis=1)

model=fccnet.model_defense(input_shape=input_shape,labels_dim=num_classes)
# model=fccnet.model_defense_optimize(input_shape=input_shape,labels_dim=num_classes)
model.compile(loss=tf.keras.losses.binary_crossentropy,optimizer=tf.keras.optimizers.SGD(lr=0.001),metrics=['accuracy'])
model.summary()

b_train=f_train[:,:]
# b_train=f_evaluate_logits[:,:]
label_train=l_train[:]


index_array=np.arange(b_train.shape[0])
batch_num=np.int(np.ceil(b_train.shape[0]/batch_size))
for i in np.arange(defense_epochs):
    np.random.shuffle(index_array)
    for j in np.arange(batch_num):
        b_batch=b_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,b_train.shape[0])],:]
        y_batch=label_train[index_array[(j%batch_num)*batch_size:min((j%batch_num+1)*batch_size,label_train.shape[0])]]
        model.train_on_batch(b_batch,y_batch)

    if (i+1)%100==0:
        print("Epochs: {}".format(i))
        scores_train = model.evaluate(b_train, label_train, verbose=0)
        print('Train loss:', scores_train[0])
        print('Train accuracy:', scores_train[1])

if save_model:
    weights=model.get_weights()
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    if not os.path.exists(result_folder+"/models"):
        os.makedirs(result_folder+"/models")
    np.savez(result_folder+"/models/"+"epoch_{}_weights_defense.npz".format(defense_epochs),x=weights)

