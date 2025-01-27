import os
import configparser


config = configparser.ConfigParser()
config.read('config.ini')


result_folder = "./result/location/code_publish/"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)


config["location"]["result_folder"] = result_folder
with open("config.ini", 'w') as configfile:
    config.write(configfile)
    configfile.close()
 

print("============================user model======================================")
cmd = "python train_user_classification_model.py -dataset location"
os.system(cmd)
print("============================defense model======================================")

cmd = "python train_defense_model_defensemodel.py -dataset location"
os.system(cmd)
print("============================adjust noise======================================")

cmd = "python defense_framework.py -dataset location -qt evaluation "
os.system(cmd)
print("============================shadow model======================================")

cmd = "python train_attack_shadow_model.py -dataset location -adv adv1"
os.system(cmd)
print("============================evaluation======================================")

cmd = "python evaluate_nn_attack.py -dataset location -scenario full -version v0"
os.system(cmd)