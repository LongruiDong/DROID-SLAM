# -*- coding: utf-8 -*-
import os
import sys
import time
import argparse
import numpy as np
from evo.tools import file_interface
from evo.core import trajectory
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core import metrics
from evo.core.metrics import StatisticsType
import evo.core.geometry as geometry

import xlwt
import xlrd
from xlutils.copy import copy

reload(sys)
sys.setdefaultencoding('utf8')


# xz_indoor_large
sub_gt_file = "lsfb/gba_pose.csv"
# sub_gt_file = "gt/handeye_tool/gt.tum"

ape_th = 0.5

run_index = []
for ind in range(10):
    run_index.append(str(ind))
# run_index = ["1", "2", "3"]
# run_avg = True
run_avg = False


info_title = ['dataset', 
               'ATE(se3)-KF', 
               'ATE(sim3)-KF',
                'scale-KF', 
                'completeness']
use_percentage = [3,4]


def getAvg(info_list_input):
    avg_list = list(info_list_input[0])
    for row_index in range(1, len(info_list_input)):
        row_item = info_list_input[row_index]
        for col_index in range(1, len(row_item)):
            item = row_item[col_index]
            avg_list[col_index] = (avg_list[col_index]*row_index + item)/(row_index+1)
    return avg_list

al = xlwt.Alignment()
al.horz = 0x02      # 设置水平居中
al.vert = 0x01      # 设置垂直居中
style = xlwt.XFStyle()
style.alignment = al
def WriteExcel(sheet, info_list_all, enable_color=False):
    for col_index in range(len(info_title)):
        item = info_title[col_index]
        font0 = xlwt.Font()
        font0.colour_index = 0
        style.font = font0
        sheet.write(0, col_index, item, style)
        sheet.col(col_index).width = max(sheet.col(col_index).width, 256 * (len(item.encode('utf-8'))))

    # avg_list_tmp = getAvg(info_list_all)
    # avg_list_tmp[0] = "avg"
    # info_list_all.append(avg_list_tmp)
    for row_index in range(len(info_list_all)):
        for col_index in range(len(info_list_all[row_index])):
            item = info_list_all[row_index][col_index]
            if col_index == 0:
                font0 = xlwt.Font()
                font0.colour_index = 0
                style.font = font0
                sheet.col(col_index).width = max(sheet.col(col_index).width, 256 * (len(item.encode('utf-8'))))
            else:
                style.num_format_str = "0.000"
                if col_index in use_percentage:
                    style.num_format_str = "0.00%"
                font1 = xlwt.Font()
                font1.colour_index = 0
                style.font = font1
            sheet.write(row_index+1, col_index, item, style)

def iceba_align_estimate_Traj_with_SE3(input_sequence, dataset_name, gt_file, est_file, out_file):
    iceba_path = "/home/pz/develop/module_indep/ICEBA2/"
    cmd  = iceba_path +"bin/back_end " + iceba_path +"config/test.cfg " \
            + input_sequence + " " \
            + dataset_name + " " \
            + gt_file + " " \
            + est_file + " " \
            + out_file
    print("cmd: ", cmd)
    os.system(cmd)

def readTimeConsuming(file):
    with open(file, 'r') as fp:
        lines = fp.readlines()
        last_line = lines[-1]
    strss = last_line.split(' ')[-1].strip()
    print("last_line",float(strss))
    return float(strss)*1000

def SLAMEvaluation(dataset_root_dir, result_path):
    dataset_root_dir = "/mnt/hgfs/pengzhen/TCSVT/tcsvt_release_data/"
    device = "ios"
    device = "glass"
    # device = "android"
    res_name = "OrbSlam3_Res_new_1500_12"

    time_suffix = time.strftime("%Y%m%d-%H%M%S")
    # output_excel = dataset_root_dir + "Benchmark-orbslam3(2)-"+ time_suffix +"-ios.xls"
    output_excel = dataset_root_dir + "xls/" + res_name + "-"+ device + "-tttt-glass-monovi.xls"
    info_list_total = []

    print("output_excel: ", output_excel)
    if os.path.isfile(output_excel):
        xl_old = xlrd.open_workbook(output_excel, formatting_info=True)
        xl = copy(xl_old)
        if len(xl_old.sheets()) >= 1:
            sheet = xl.get_sheet(0)
        else:
            sheet = xl.add_sheet('RMSE Result')
    else:
        xl = xlwt.Workbook(encoding='utf-8')
        sheet = xl.add_sheet('RMSE Result')

    dataset_list = os.listdir(dataset_root_dir)
    dataset_list.sort()
    print("dataset_list: ", dataset_list)
    for folders in dataset_list:
        if "tar.gz" not in folders and "xls" not in folders:
            dir1 = dataset_root_dir + folders
            dataset_list2 = os.listdir(dir1)
            dataset_list2.sort()
            for folders2 in dataset_list2:
                dir2 = dir1 + "/" + folders2
                dataset_list3 = os.listdir(dir2)
                dataset_list3.sort()
                for folders3 in dataset_list3:
                    dir3 = dir2 + "/" + folders3
                    if device not in folders3:
                        continue
                    if "test_tmp"  in folders:
                        continue
                    relative_path = folders + "/" + folders2
                    info_list_tmp = []
                    for run_item in run_index:
                        res_f = dir3 + "/"+ res_name+"/Twi_test" + run_item + ".tum"
                        res_kf = res_f + ".kf"
                        if not os.path.exists(res_f):
                            print(" res_f: ", res_f)
                            continue

                        gt_file = dir3 + "/" + sub_gt_file
                        traj_gt = file_interface.read_euroc_csv_trajectory(gt_file)
                        # traj_gt = file_interface.read_tum_trajectory_file(gt_file)

                        traj_f = file_interface.read_tum_trajectory_file(res_f)
                        traj_kf = file_interface.read_tum_trajectory_file(res_kf)

                        # Umeyama's method
                        traj_ref, traj_2 = sync.associate_trajectories(traj_gt, traj_f, 0.02, 0.0,'ref', 'f')
                        traj_ape = trajectory.align_trajectory(traj_2, traj_ref, False, False)
                        
                        traj_ref2, traj_4 = sync.associate_trajectories(traj_gt, traj_kf, 0.02, 0.0, 'ref2', 'kf')
                        traj_ape_kf = trajectory.align_trajectory(traj_4, traj_ref2, False, False)
                        traj_ape_kf_sim3 = trajectory.align_trajectory(traj_4, traj_ref2, True, False)


                        ape_metric = metrics.APE()
                        data_ape = (traj_ref, traj_ape)
                        ape_metric.process_data(data_ape)
                        ape = ape_metric.get_statistic(StatisticsType.rmse)

                        ape_metric2 = metrics.APE()
                        data_ape2 = (traj_ref2, traj_ape_kf)
                        ape_metric2.process_data(data_ape2)
                        ape2 = ape_metric2.get_statistic(StatisticsType.rmse)

                        ape_metric_sim = metrics.APE()
                        data_ape_sim = (traj_ref2, traj_ape_kf_sim3)
                        ape_metric_sim.process_data(data_ape_sim)
                        ape2_sim3 = ape_metric_sim.get_statistic(StatisticsType.rmse)

                        r_a, t_a, s_KF = geometry.umeyama_alignment(traj_ape_kf.positions_xyz.T, traj_ref2.positions_xyz.T, True)
                        
                        info_tmp = [relative_path+"/"+run_item, ape2, ape2_sim3, s_KF, 
                                    float(traj_ape.positions_xyz.shape[0])/traj_gt.positions_xyz.shape[0] ]
                        # info_tmp = [relative_path+"--"+run_item, ape2, ape2_sim3, s_KF, 
                        #             float(traj_ape.positions_xyz.shape[0])/2274 ]
                        # 2274  2070
                        # print("gt size:", traj_gt.positions_xyz.shape[0])
                        info_list_tmp.append(info_tmp)

                    if run_avg:
                        if info_list_tmp:
                            info_list_total.append(getAvg(info_list_tmp))
                    else:
                        for item in info_list_tmp:
                            info_list_total.append(item)

    WriteExcel(sheet, info_list_total, True)
    xl.save(output_excel)
    print('Write RMSE evaluation result into: ' + output_excel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Evo Costum')
    parser.add_argument('-dp',dest = 'dataset_root_path',metavar='dataset_root_path', default="", help='dataset_root_path')
    parser.add_argument('-rp',dest = 'result_path',metavar='result_path', default="", help='result_path')
    parser.add_argument('-dn',dest = 'dataset_name',metavar='dataset_name', default="", help='dataset_name')
    parser.add_argument('-gt',dest='fileGT',metavar='fileGT', default="",help = 'groundTruth file relative path')
    parser.add_argument('-est',dest='fileEST',metavar='fileEST', default="",help = 'estimate file relative path')
    parser.add_argument('-estRPE',dest='fileEST_RPE',metavar='fileEST_RPE', default="",help = 'estimate file relative path for RPE')
    parsed = parser.parse_args()

    print("dataset_root_path: ", parsed.dataset_root_path)
    print("result_path: ", parsed.result_path)
    print("dataset_name: ", parsed.dataset_name)
    print("fileGT: ", parsed.fileGT)
    print("fileEST: ", parsed.fileEST)
    if len(parsed.fileEST_RPE) == 0:
        parsed.fileEST_RPE = parsed.fileEST
    print("fileEST_RPE: ", parsed.fileEST_RPE)
    

    SLAMEvaluation(parsed.dataset_root_path, parsed.result_path)

