# --------------------------------------------------------
# DenseFusion 6D Object Pose Estimation by Iterative Dense Fusion
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen
# --------------------------------------------------------
# import sys
# print("PATH")
# print(sys.path)
# sys.path.insert(0, "/Users/wuzhelun/Code/DenseFusion_0.4/lib/knn/knn_pytorch")
# print(sys.path)

import argparse
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.network import PoseNet, PoseRefineNet
from lib.loss import Loss
# from lib.loss_refiner import Loss_refine
from tools.datasets import get_bbox as get_bbox
from PIL import Image
import scipy.io as scio
import numpy.ma as ma
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import math

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default = 'ycb', help='ycb or linemod')
parser.add_argument('--dataset_root', type=str, default = '', help='dataset root dir (''YCB_Video_Dataset'' or ''Linemod_preprocessed'')')
parser.add_argument('--batch_size', type=int, default = 8, help='batch size')
parser.add_argument('--workers', type=int, default = 10, help='number of data loading workers')
parser.add_argument('--lr', default=0.0001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03, help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default = 2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=500, help='max number of epochs to train')
parser.add_argument('--resume_posenet', type=str, default = '',  help='resume PoseNet model')
parser.add_argument('--resume_refinenet', type=str, default = '',  help='resume PoseRefineNet model')
parser.add_argument('--start_epoch', type=int, default = 1, help='which epoch to start')
opt = parser.parse_args()

PRED_ON_PLANE_FACTOR = 0.5
PRODUCT_THRESHOLD = math.cos(math.pi / 180 * 20)
title_list=['under10cm-diff_dis-angle20','f10t15cm-diff_dis-angle20','up15cm-diff_dis-angle20']

def generate_obj_file_norm_pred(pred_norm, pred_on_plane, cloud, idx1, idx2, idx3):

    with open('gen_obj_norm_pred/labeled_scene_{}_{}_idx_{}.obj'.format(idx1, idx2, idx3), 'w') as f:
        for i in range(cloud.size(1)):
            if pred_on_plane[:,i,:] > max(0.5,pred_on_plane.max()*PRED_ON_PLANE_FACTOR + pred_on_plane.mean() * (1-PRED_ON_PLANE_FACTOR)):
                f.write("v {} {} {} {} {} {}\n".format(cloud[0,i,0], cloud[0,i,1], cloud[0,i,2], 255*(1+pred_norm[0,i,0].item())/2, 255*(1+pred_norm[0,i,1].item())/2, 255*(1+pred_norm[0,i,2].item())/2))
            else:
                f.write("v {} {} {} {} {} {}\n".format(cloud[0,i,0], cloud[0,i,1], cloud[0,i,2], 0, 0, 0))

def generate_obj_file_norm(pred_norm, cloud, idx1, idx2, idx3, count):

    # import pdb;pdb.set_trace()
    with open('gen_obj_norm_scrub/labeled_scene_{}_{}_idx_{}_scrub_{}.obj'.format(idx1, idx2, idx3, count), 'w') as f:
        for i in range(cloud.shape[0]):
            f.write("v {} {} {} {} {} {}\n".format(cloud[i,0], cloud[i,1], cloud[i,2], 255*(1+pred_norm[i,0])/2, 255*(1+pred_norm[i,1])/2, 255*(1+pred_norm[i,2])/2))

def printCurve(take_idx, criterion):
    conf_tp_or_fn = [[] for i in range(5)]
    conf_fp = [[] for i in range(5)]
    prec = [[] for i in range(5)]
    recall = [[] for i in range(5)]

    file_list = []
    input_file = open('./datasets/ycb/dataset_config/test_data_list.txt', 'r')
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]

        _, test_scene_id, test_frame_id = input_line.split('/')

        input_line = '/'.join(['data_v1',test_scene_id, test_frame_id])
        file_list.append(input_line)
    input_file.close()

    if take_idx is 0:
        take_into_account = {2:[2],3:[1],5:[0,1],7:[1],8:[0,1],9:[0,1,2],12:[0],14:[0],16:[1,2],21:[0]}
    elif take_idx is 1:
        take_into_account = {1:[0],3:[0],7:[0],11:[0],13:[0],19:[0]}
    else:
        take_into_account = {2:[0,1],16:[0],20:[0]}


    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])

    for index in range(len(file_list)):
        img = Image.open('{0}/{1}-color.png'.format(opt.dataset_root, file_list[index]))
        depth = np.array(Image.open('{0}/{1}-depth.png'.format(opt.dataset_root, file_list[index])))
        label = np.array(Image.open('{0}/{1}-label.png'.format(opt.dataset_root, file_list[index])))
        meta = scio.loadmat('{0}/{1}-meta.mat'.format(opt.dataset_root, file_list[index]))

        cam_cx = 312.9869
        cam_cy = 241.3109
        cam_fx = 1066.778
        cam_fy = 1067.487
        mask_back = ma.getmaskarray(ma.masked_equal(label, 0))

        obj = meta['cls_indexes'].flatten().astype(np.int32)
        # import pdb;pdb.set_trace()

        # idx = idx_list[index]
        for idx in range(0, len(obj)):
            if not obj[idx] in take_into_account:
                continue
            
            print('scene index: ',file_list[index])
            print('object index: ', obj[idx])
            mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
            mask_label = ma.getmaskarray(ma.masked_equal(label, obj[idx]))
            mask = mask_label * mask_depth
            if not (len(mask.nonzero()[0]) > 50 and len(opt.symmetry[obj[idx]]['mirror'])>0):
                continue
            st_time = time.time()
            rmin, rmax, cmin, cmax = get_bbox(mask_label)
            img_temp = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

            img_masked = img_temp           
            target_r = meta['poses'][:, :, idx][:, 0:3]
            target_t = np.array(meta['poses'][:, :, idx][:, 3:4].flatten())

            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
            if len(choose) > opt.num_points:
                c_mask = np.zeros(len(choose), dtype=int)
                c_mask[:opt.num_points] = 1
                np.random.shuffle(c_mask)
                choose = choose[c_mask.nonzero()]
            else:
                choose = np.pad(choose, (0, opt.num_points - len(choose)), 'wrap')
            
            depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
            choose = np.array([choose])

            cam_scale = meta['factor_depth'][0][0]
            pt2 = depth_masked / cam_scale
            pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
            cloud = np.concatenate((pt0, pt1, pt2), axis=1)


            target_sym = []
            for sym in opt.symmetry[obj[idx]]['mirror']:
                target_sym.append(np.dot(sym, target_r.T))
            target_sym = np.array(target_sym)

            target_cen = np.add(opt.symmetry[obj[idx]]['center'], target_t)

            print('ground truth norm: ', target_sym)
            print('ground truth center: ', target_cen)
            points_ten, choose_ten, img_ten, target_sym_ten, target_cen_ten, idx_ten = \
            torch.from_numpy(cloud.astype(np.float32)).unsqueeze(0), \
            torch.LongTensor(choose.astype(np.int32)).unsqueeze(0), \
            opt.norm(torch.from_numpy(img_masked.astype(np.float32))).unsqueeze(0), \
            torch.from_numpy(target_sym.astype(np.float32)).unsqueeze(0), \
            torch.from_numpy(target_cen.astype(np.float32)).unsqueeze(0), \
            torch.LongTensor([obj[idx]-1]).unsqueeze(0)

            points_ten, choose_ten, img_ten, target_sym_ten, target_cen_ten, idx_ten = Variable(points_ten), \
                                                                Variable(choose_ten), \
                                                                Variable(img_ten), \
                                                                Variable(target_sym_ten), \
                                                                Variable(target_cen_ten), \
                                                                Variable(idx_ten)

            pred_norm, pred_on_plane, emb = opt.estimator(img_ten, points_ten, choose_ten, idx_ten)

            bs, num_p, _ = pred_on_plane.size()

            # generate_obj_file_norm_pred(pred_norm / (torch.norm(pred_norm, dim=2).view(bs, num_p, 1)), pred_on_plane, points_ten, file_list[index].split('/')[1], file_list[index].split('/')[2], obj[idx])


            pred_norm = pred_norm / (torch.norm(pred_norm, dim=2).view(bs, num_p, 1))
            pred_norm = pred_norm.detach().numpy()
            pred_on_plane = pred_on_plane.detach().numpy()
            points = points_ten.detach().numpy()
        
            clustering_points_idx = np.where(pred_on_plane>max(0.5,pred_on_plane.max()*PRED_ON_PLANE_FACTOR + pred_on_plane.mean() * (1-PRED_ON_PLANE_FACTOR)))[1]
            clustering_norm = pred_norm[0,clustering_points_idx,:]
            clustering_points = points[0,clustering_points_idx,:]
            num_points = len(clustering_points_idx)

            # print(pred_on_plane.max())
            # import pdb;pdb.set_trace()

            close_thresh = 2e-3
            broad_thresh = 3e-3


            sym_conf = np.zeros((5, target_sym.shape[0]))

            count_pred = 0

            # import pdb; pdb.set_trace()

            while True:
                count_pred += 1
                if num_points <= 20 or count_pred > 3:
                    break

                # generate_obj_file_norm(clustering_norm, clustering_points, file_list[index].split('/')[1], file_list[index].split('/')[2], obj[idx], count_pred)

                best_fit_num = 0

                count_try = 0

                for j in range(10):

                    pick_idx = np.random.randint(0,num_points-1)
                    pick_point = clustering_points[pick_idx]
                    # proposal_norm = np.array(Plane(Point3D(pick_points[0]),Point3D(pick_points[1]),Point3D(pick_points[2])).normal_vector).astype(np.float32)
                    proposal_norm = clustering_norm[pick_idx]
                    proposal_norm = proposal_norm[:,np.newaxis]

                    # import pdb;pdb.set_trace()
                    proposal_point = pick_point

                    clustering_diff = clustering_points - proposal_point
                    clustering_dist = np.abs(np.matmul(clustering_diff, proposal_norm))

                    broad_inliers = np.where(clustering_dist < broad_thresh)[0]
                    broad_inlier_num = len(broad_inliers)

                    close_inliers = np.where(clustering_dist < close_thresh)[0]
                    close_inlier_num = len(close_inliers)

                    norm_dist = np.abs(clustering_norm-np.transpose(proposal_norm)).sum(1)
                    close_norm_idx = np.where(norm_dist < 0.6)[0]
                    close_norm_num =  len(close_norm_idx)

                    if close_inlier_num >= best_fit_num and broad_inlier_num >= num_points / (4-count_pred) *0.9 and close_norm_num >= num_points / (4-count_pred) * 0.9:
                        best_fit_num = close_inlier_num
                        best_fit_norm = proposal_norm
                        best_fit_cen = clustering_points[close_inliers].mean(0)
                        best_fit_idx = clustering_points_idx[close_inliers]
                        best_norm_dist = norm_dist
                        best_close_norm_idx = np.where(best_norm_dist < 0.6)[0]


                if best_fit_num == 0 or num_points <= 20:
                    break

                clustering_points_same_sym = clustering_points[best_close_norm_idx]

                clustering_diff_same_sym = clustering_points_same_sym - best_fit_cen
                clustering_dist_same_sym = np.abs(np.matmul(clustering_diff_same_sym, best_fit_norm))

                close_inliers = np.where(clustering_dist_same_sym < close_thresh)[0]
                close_inlier_num= len(close_inliers)

                best_fit_num = close_inlier_num

                broad_inliers = np.where(clustering_dist_same_sym < broad_thresh)[0]
                broad_inlier_num= len(broad_inliers)

                def f(x):
                    dist = 0
                    # import pdb;pdb.set_trace()
                    for point in clustering_points_same_sym[broad_inliers]:
                        dist += np.abs((point * x[0:3]).sum() + x[3]) / np.sqrt(np.sum(np.square(x[0:3]), axis=0))

                    return dist

                start_point = np.zeros(4)
                start_point[0:3] = np.copy(best_fit_norm[:,0])
                start_point[3] = (-best_fit_cen*best_fit_norm[:,0]).sum()

                min_point = fmin(f, start_point, maxiter=50)

                # import pdb;pdb.set_trace()
                min_point = min_point / np.sqrt(np.sum(np.square(min_point[0:3]), axis=0))
                
                x_val = -(min_point[3]+best_fit_cen[1] * min_point[1] + best_fit_cen[2] * min_point[2]) / min_point[0]

                y_val = -(min_point[3]+best_fit_cen[0] * min_point[0] + best_fit_cen[2] * min_point[2]) / min_point[1]

                z_val = -(min_point[3]+best_fit_cen[0] * min_point[0] + best_fit_cen[1] * min_point[1]) / min_point[2]

                if np.abs(x_val) < 1:
                    new_pred_loc = np.array([x_val, best_fit_cen[1], best_fit_cen[2]])
                elif np.abs(z_val) < 1:
                    new_pred_loc = np.array([best_fit_cen[0], best_fit_cen[1], z_val])
                else:
                    new_pred_loc = np.array([best_fit_cen[0], y_val, best_fit_cen[2]])

                new_proposal_norm = min_point[0:3]
                clustering_diff = clustering_points_same_sym - new_pred_loc
                clustering_dist = np.abs(np.matmul(clustering_diff, new_proposal_norm))

                close_inliers = np.where(clustering_dist < close_thresh)[0]
                new_close_inlier_num = len(close_inliers)    

                broad_inliers = np.where(clustering_dist < broad_thresh)[0]
                new_broad_inlier_num = len(broad_inliers)
                # import pdb;pdb.set_trace()
                if new_close_inlier_num >= close_inlier_num:
                    best_fit_num = new_close_inlier_num
                    best_fit_norm = new_proposal_norm[:,np.newaxis]
                    best_fit_cen = new_pred_loc

                if best_fit_num == 0:
                    break
                else:
                    
                    print('predicted norm:{}, predicted point:{}'.format(best_fit_norm, best_fit_cen))

                    max_idx = np.argmax(np.abs(np.matmul(target_sym, best_fit_norm)))
                    sym_product = np.abs(np.matmul(target_sym, best_fit_norm)[max_idx][0])
                    sym_dist = np.abs((target_sym[max_idx] * (best_fit_cen-target_cen)).sum())

                    norm_dist = np.abs(clustering_norm-np.transpose(best_fit_norm)).sum(1)
                    scrub_close_norm_idx = np.where(norm_dist < 1.3)[0]

                    # import pdb;pdb.set_trace()
                    predicted_confidence = best_fit_num / len(best_close_norm_idx) - np.abs(clustering_norm[best_close_norm_idx]-np.transpose(best_fit_norm)).mean() * 3 * 1.5
                    # print('right point: ', best_fit_num / len(best_close_norm_idx))
                    # print('wrong points: ', np.abs(clustering_norm[best_close_norm_idx]-np.transpose(best_fit_norm)).mean() * 2)
                    predicted_confidence = max(0, predicted_confidence)
                    # print(sym_product)
                    # print(sym_dist)

                    for dist_idx in range(5):
                        if sym_product > PRODUCT_THRESHOLD and sym_dist < (dist_idx+1)*0.01:
                            sym_conf[dist_idx, max_idx] = max(sym_conf[dist_idx, max_idx], predicted_confidence)
                        # elif max_idx is take_into_account[obj[idx]]:
                        else:
                            if max_idx in take_into_account[obj[idx]]:
                                conf_fp[dist_idx].append(predicted_confidence)
                                # print('sym_product: ',sym_product)
                                # print('sym_dist: ', sym_dist)

                    clustering_points_idx = np.setdiff1d(clustering_points_idx, clustering_points_idx[scrub_close_norm_idx])
                    
                    clustering_norm = pred_norm[0,clustering_points_idx,:]
                    clustering_points = points[0,clustering_points_idx,:]

                    num_points = len(clustering_points_idx)
                    # import pdb;pdb.set_trace()

            # import pdb;pdb.set_trace()
            print('time spent:',time.time()-st_time)
            for dist_idx in range(5):
                for i in range(target_sym.shape[0]):
                    if i in take_into_account[obj[idx]]:
                        conf_tp_or_fn[dist_idx].append(sym_conf[dist_idx,i])
            # print(conf_tp_or_fn[0])
            # print(conf_fp[0])
            # import pdb;pdb.set_trace()

    print(conf_tp_or_fn)
    print(conf_fp)

    # import pdb;pdb.set_trace()
    
    for dist_idx in range(5):
        for t in range(1,1001):
            conf_thresh = t/1000
            
            true_positives = len(np.where(np.array(conf_tp_or_fn[dist_idx]) >= conf_thresh)[0])
            false_negatives = len(np.where(np.array(conf_tp_or_fn[dist_idx]) < conf_thresh)[0])
            false_positives = len(np.where(np.array(conf_fp[dist_idx]) >= conf_thresh)[0])

            if false_positives + true_positives > 0 and true_positives + false_negatives > 0:
                prec[dist_idx].append(true_positives / (false_positives + true_positives))
                recall[dist_idx].append(true_positives / (true_positives + false_negatives))
    
    return prec, recall

def main():
    # opt.manualSeed = random.randint(1, 10000)
    # # opt.manualSeed = 1
    # random.seed(opt.manualSeed)
    # torch.manual_seed(opt.manualSeed)

    torch.set_printoptions(threshold=5000)

    if opt.dataset == 'ycb':
        opt.num_objects = 21 #number of object classes in the dataset
        opt.num_points = 1000 #number of points on the input pointcloud
        opt.outf = 'trained_models/ycb' #folder to save trained models
        opt.log_dir = 'experiments/logs/ycb' #folder to save logs
        opt.repeat_epoch = 3 #number of repeat times for one epoch training
    elif opt.dataset == 'linemod':
        opt.num_objects = 13
        opt.num_points = 500
        opt.outf = 'trained_models/linemod'
        opt.log_dir = 'experiments/logs/linemod'
        opt.repeat_epoch = 20
    else:
        print('Unknown dataset')
        return

    opt.estimator = PoseNet(num_points = opt.num_points, num_obj = opt.num_objects)
    # estimator.cuda()
    refiner = PoseRefineNet(num_points = opt.num_points, num_obj = opt.num_objects)
    # refiner.cuda()
    
    class_id = 1
    opt.symmetry = {}
    with open('symmetries_ordered.txt', 'r') as f:
        while 1:
            line = f.readline()
            line = line[:-1]
            if not line:
                break
            opt.symmetry[class_id] = {}
            opt.symmetry[class_id]['center'] = list(map(lambda x:float(x), line.split(' ')))
            opt.symmetry[class_id]['mirror'] = []
            for i in range(3):
                line = f.readline()
                line = line[:-1]
                x, y, z = list(map(lambda x:float(x),line.split(' ')))

                if not (x == 0 and y == 0 and z == 0):
                    opt.symmetry[class_id]['mirror'].append((x,y,z))
            f.readline()
            f.readline()
            class_id += 1

    opt.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    criterion = Loss(opt.num_points, opt.symmetry)

    opt.estimator.load_state_dict(torch.load('{0}/pose_model_54_0.0.pth'.format(opt.outf),map_location='cpu'))
    # import pdb;pdb.set_trace()
    print('start parallelization')

    # pool = Pool(4)
    # results = [pool.apply_async(printCurve, [take_idx, criterion]) for take_idx in range(3)]

    for take_idx in range(3):
        # prec, recall = results[take_idx].get()
        prec,recall = printCurve(take_idx, criterion)
        for dist_idx in range(5):
                plt.plot(recall[dist_idx], prec[dist_idx], label = 'dis={:.2f}'.format((dist_idx+1)*0.01))

        plt.axis([0, 1, 0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title_list[take_idx])
        plt.legend()
        plt.savefig('prec-recall-{}.png'.format(title_list[take_idx]))
        plt.clf()

if __name__ == '__main__':
    main()
