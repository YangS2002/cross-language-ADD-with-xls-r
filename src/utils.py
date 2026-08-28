import os
import time
def write_exp_detail(config,test_scores_root_path,is_train=False):
    # 写入本次测试的描述
    if is_train:
        target_file = "loss.txt"
    else:
        target_file = "scores.txt"
    if os.path.exists(os.path.join(test_scores_root_path,config['exp_name'])) is False:
        os.makedirs(os.path.join(test_scores_root_path,config['exp_name']),exist_ok=True)
    with open(os.path.join(test_scores_root_path,config['exp_name'],target_file),'a') as f:
        f.write(f"exp_name: "+config['exp_name']+'\n')
        f.write(f"{os.path.join(config['test_data_path'])}\n")
        f.write(f"time: {str(time.ctime())}\n")
        f.write("exp_detail: "+config["exp_detail"]+'   ')
        f.write("checkpoint_path: "+config["checkpoint_path"]+'\n')
        f.close()

