
########################################################################
# import default python-library
########################################################################
import os
import glob
import csv
import re
import itertools
import sys


########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy
import matplotlib.pyplot as plt
# from import
from tqdm import tqdm
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
# original lib
import common as com
import model  as mdl
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()
#######################################################################


########################################################################
# def
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):
    com.logger.info("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = numpy.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = numpy.ones(len(anomaly_files))
        files = numpy.concatenate((normal_files, anomaly_files), axis=0)
        labels = numpy.concatenate((normal_labels, anomaly_labels), axis=0)
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels
########################################################################

# def calc_train_features(target_dir, id_str):
#     test_file_list_generator


if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)
    os.makedirs(param["figure_directory"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)
    #dirs = [dirs[5], dirs[3], dirs[0]]

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # model
    use_model_list = []
    use_for_decimate_list = []
    if(param['model']['am_model']['use'] == True):
        use_model_list.append(mdl.mahalanobis_dist_model())
        use_for_decimate_list.append(param['model']['am_model']['use_for_decimate'])
    if(param['model']['ab_model']['use'] == True):
        k = param['model']['ab_model']['k']
        use_model_list.append(mdl.subspace_dist_model(k))
        use_for_decimate_list.append(param['model']['ab_model']['use_for_decimate'])
    if(param['model']['al_model']['use'] == True):
        use_model_list.append(mdl.matrix_normal_distribution_model(True, 'mndZ'))
        use_for_decimate_list.append(param['model']['al_model']['use_for_decimate'])
    if(param['model']['abl_model']['use'] == True):
        use_model_list.append(mdl.matrix_normal_distribution_model(False, 'mndX'))
        use_for_decimate_list.append(param['model']['abl_model']['use_for_decimate'])
    f = param['model']['f']
    decimate = param['model']['decimate']
    total_mdl = mdl.total_model(f, use_model_list, decimate, use_for_decimate_list)
    
    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        if mode:
            # results by type
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []

        machine_id_list = get_machine_id_list_for_test(target_dir)

        for id_str in machine_id_list:
            # load test file
            test_files, y_true = test_file_list_generator(target_dir, id_str)
            train_files, _ = test_file_list_generator(target_dir, id_str, dir_name="train")

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                                     result=param["result_directory"],
                                                                               machine_type=machine_type,
                                                                                     id_str=id_str)
            anomaly_score_list = []

            print("\n============== BEGIN TRAIN DATA FEATURE EXTRACTION FOR A MACHINE ID ==============")
            cnt = 0
            for file_idx, file_path in tqdm(enumerate(train_files), total=len(train_files)):
                try:
                    train_X = com.file_to_vector_array(file_path,
                                                    n_mels=param["feature"]["n_mels"],
                                                    frames=param["feature"]["frames"],
                                                    n_fft=param["feature"]["n_fft"],
                                                    hop_length=param["feature"]["hop_length"],
                                                    power=param["feature"]["power"])
                    if(file_idx == 0):
                        train_all = numpy.zeros((len(train_files), train_X.shape[0],  train_X.shape[1]))
                    if(train_X.shape[0] == train_all.shape[1] and
                        train_X.shape[1] == train_all.shape[2]):
                        train_all[cnt, :, :] = train_X
                        cnt += 1
                except:
                    com.logger.error("file broken!!: {}".format(file_path))
            train_all = train_all[:cnt, :]            
            total_mdl.fit(train_all)

            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            y_pred = [0. for k in test_files]
            test_data_scores = numpy.zeros((len(test_files), len(use_model_list)))
        
            for file_idx, file_path in tqdm(enumerate(test_files), total=len(test_files)):
                try:
                    data = com.file_to_vector_array(file_path,
                                                    n_mels=param["feature"]["n_mels"],
                                                    frames=param["feature"]["frames"],
                                                    n_fft=param["feature"]["n_fft"],
                                                    hop_length=param["feature"]["hop_length"],
                                                    power=param["feature"]["power"])
                    testX = data 
                    y_pred[file_idx], test_data_scores[file_idx, :] = total_mdl.score(testX)
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                except:
                    com.logger.error("file broken!!: {}".format(file_path))

            
            # # save anomaly score
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

            # save anormal score scatter figure
            for i in range(len(use_model_list)):
                for j in range(i + 1, len(use_model_list)):
                    plt.figure()
                    plt.scatter(total_mdl.removed_train_score[:, i], 
                                total_mdl.removed_train_score[:, j], color = "c")

                    plt.scatter(total_mdl.leave_train_score[:, i], 
                                total_mdl.leave_train_score[:, j], color = "g")
                    
                    plt.scatter(test_data_scores[:, i], test_data_scores[:, j], c = y_pred, cmap = "bwr", norm=plt.Normalize(vmin=0, vmax=numpy.average(y_pred) * 2))
                    plt.xlabel(use_model_list[i].feature_name)
                    plt.ylabel(use_model_list[j].feature_name)
                    title = machine_type +"_" + id_str + "_" + use_model_list[i].feature_name + "_" + use_model_list[j].feature_name + "_pred_score_map"
                    plt.title(title)
                    plt.savefig(param['figure_directory'] + "//" + title + ".png")
                    plt.close()

            if mode:
                # append AUC and pAUC to lists
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                performance.append([auc, p_auc])
                com.logger.info("AUC : {}".format(auc))
                com.logger.info("pAUC : {}".format(p_auc))

            print("\n============ END OF TEST FOR A MACHINE ID ============")

        if mode:
            # calculate averages for AUCs and pAUCs
            averaged_performance = numpy.mean(numpy.array(performance, dtype=float), axis=0)
            csv_lines.append(["Average"] + list(averaged_performance))
            csv_lines.append([])

    if mode:
        # output results
        result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
        com.logger.info("AUC and pAUC results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines)
