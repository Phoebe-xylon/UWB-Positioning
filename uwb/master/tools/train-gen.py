import sys

sys.path.insert(0, "D:/python_project/uwb/master")
import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from UWB_part_loader import PartDataset
from lib.network import UWBNet
from lib.utils import setup_logger
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='UWB', help='ycb or linemod')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--lr', default=0.001, help='learning rate')
parser.add_argument('--lr_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--w', default=0.015, help='learning rate')
parser.add_argument('--w_rate', default=0.3, help='learning rate decay rate')
parser.add_argument('--decay_margin', default=0.016, help='margin to decay lr & w')
parser.add_argument('--refine_margin', default=0.013, help='margin to start the training of iterative refinement')
parser.add_argument('--noise_trans', default=0.03,
                    help='range of the random noise of translation added to the training data')
parser.add_argument('--iteration', type=int, default=2, help='number of refinement iterations')
parser.add_argument('--nepoch', type=int, default=2, help='max number of epochs to train')
parser.add_argument('--resume_UWBnet', type=str, default='pose_model_current.pth',
                    help='resume PoseNet model  pose_model_current.pth')
parser.add_argument('--start_epoch', type=int, default=1, help='which epoch to start')
parser.add_argument('--eval_interval', type=int, default=1, help="the interval of eval")
parser.add_argument('--loss_path', default='../loss', help="the path of loss")
opt = parser.parse_args()


def main():
    torch.set_printoptions(threshold=5000)
    cudnn.benchmark = True
    if opt.dataset == 'UWB':
        opt.outf = '../trained_models/UWB/'  # folder to save trained models
        opt.log_dir = '../experiments/logs/UWB'  # folder to save logs
        opt.repeat_epoch = 3  # number of repeat times for one epoch training
    else:
        print('Unknown dataset')
    estimator = UWBNet()
    if torch.cuda.is_available():
        estimator.cuda()
    if opt.resume_UWBnet != '':
        estimator.load_state_dict(torch.load('{0}/{1}'.format(opt.outf, opt.resume_UWBnet)))
        print('LOADED!!')
    optimizer = optim.Adam(estimator.parameters(), lr=opt.lr)
    dataset = PartDataset(root='D:\data\data-UWB-nonormal/train')
    test_dataset = PartDataset(root='D:\data\data-UWB-nonormal/test')
    val_dataset = PartDataset(root='D:\data\data-UWB-nonormal/valu')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True, pin_memory=True, num_workers=int(opt.workers),
                                             drop_last=True)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True, pin_memory=True,
                                                 num_workers=int(opt.workers), drop_last=True)
    valdataloader = torch.utils.data.DataLoader(val_dataset, batch_size=5, shuffle=True, pin_memory=True,
                                                num_workers=int(opt.workers), drop_last=False)

    estimator.train()
    st_time = time.time()
    Loss_plt = []
    for epoch in range(opt.start_epoch, opt.nepoch):
        logger = setup_logger('epoch%d' % epoch, os.path.join(opt.log_dir, 'epoch_%d_log.txt' % epoch))
        logger.info('Train time {0}'.format(
            time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Training started'))
        train_count = 0
        mse_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
        for i, data in enumerate(dataloader, 0):
            input, label = data
            input = input.to(torch.float32)
            label = label.to(torch.float32)
            if torch.cuda.is_available():
                input, label = Variable(input).cuda(), \
                               Variable(label).cuda()
            else:
                input, label = Variable(input), \
                               Variable(label)
            pred_mark = estimator(input)
            loss = mse_loss(pred_mark, label).requires_grad_()
            Loss_plt.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_count += 1
            # 训练时先新建个列表，然后将loss值存入列表中即可，例如列表Recon_loss，Discriminator_loss...，然后将列表替换train_recon_loss即可。
            plt.switch_backend('Agg')
            plt.figure()
            plt.plot(Loss_plt, 'b', label='loss')
            plt.ylabel('loss')
            plt.xlabel('iter_num')
            plt.legend()
            plt.savefig(os.path.join(opt.loss_path, "2_recon_loss.jpg"))
            logger.info('Train time {0} Epoch {1} Batch {2} Loss {3}'.format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch,
                int(train_count / opt.batch_size), loss.detach().cpu().numpy()))

        torch.save(estimator.state_dict(), '{0}/pose_model_current.pth'.format(opt.outf))
        print('>>>>>>>>----------epoch {0} train finish---------<<<<<<<<'.format(epoch))
        if epoch % opt.eval_interval == 0:
            logger = setup_logger('epoch%d_test' % epoch, os.path.join(opt.log_dir, 'epoch_%d_test_log.txt' % epoch))
            logger.info('Test time {0}'.format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + ', ' + 'Testing started'))
            estimator.eval()
            for i, data in enumerate(testdataloader, 0):
                input, label = data
                input = input.to(torch.float32)
                label = label.to(torch.float32)
                if torch.cuda.is_available():
                    input, label = Variable(input).cuda(), \
                                   Variable(label).cuda()
                else:
                    input, label = Variable(input), \
                                   Variable(label)
                pred_mark = estimator(input)
                loss = mse_loss(pred_mark, label)
                prec = 0
                for pred_son, label_son in zip(pred_mark, label):
                    prec += torch.abs(pred_son - label_son) / label_son
                prec = torch.abs(prec / opt.batch_size)
                print(prec.detach().numpy())

                logger.info('Test time {0} Epoch {1} TEST FINISH Avg dis: {2}'.format(
                    time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)), epoch,
                    loss.cpu().detach().numpy()))
            print(epoch, '>>>>>>>>----------BEST TEST MODEL SAVED---------<<<<<<<<')
    for i, data in enumerate(valdataloader, 0):
        input, label = data
        input = input.to(torch.float32)
        label = label.to(torch.float32)
        if torch.cuda.is_available():
            input, label = Variable(input).cuda(), \
                           Variable(label).cuda()
        else:
            input, label = Variable(input), \
                           Variable(label)
        pred_mark = estimator(input)

        f = open("../experiments/result.txt", 'a')
        for pre in pred_mark:
            pre = 500 * pre.detach().numpy()
            f.write(str(pre[0]) + ',' + str(pre[1]) + ',' + str(pre[2]) + '\n')
        f.close()


if __name__ == '__main__':
    main()
