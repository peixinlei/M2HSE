"""Training script for the local-level subnetwork of our M2HSE approach"""
# ---------------------------------------------------------------
# The Local-level subnetwork of our M2HSE approach
#
# Written by Xin-lei Pei, 2021
# ---------------------------------------------------------------
import os
import time
import shutil
import data_provider as data
from vocab import deserialize_vocab
from model import MMHSE_local, MultiSpringBalanceLoss
from evaluation import eval_local_mAP
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import opts


def logging_func(log_file, message):
    with open(log_file, 'a') as f:
        f.write(message)
    f.close()


def main():
    # Hyper Parameters
    opt = opts.parse_opt()
    print(opt)
    device_id = opt.gpuid
    device_count = len(str(device_id).split(","))
    assert device_count == 1 or device_count == 2
    print("use GPU:", device_id, "GPUs_count", device_count, flush=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    torch.cuda.set_device(0)

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, vocab, opt.batch_size, opt)

    # Construct the model
    model = MMHSE_local(opt)
    model.cuda()
    model = nn.DataParallel(model)

    # Loss and Optimizer
    MSB_criterion_P_L = MultiSpringBalanceLoss(opt=opt)
    MSB_criterion_A1_L = MultiSpringBalanceLoss(opt=opt)
    MSB_criterion_A2_L = MultiSpringBalanceLoss(opt=opt)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    # optionally resume from a checkpoint
    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)

    start_epoch = 0
    best_mAP_sum = 0

    eval_local_mAP(model.module, val_loader, opt)

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        message = "\nepoch: %d, model name: %s \n" % (epoch, opt.model_name)
        log_file = os.path.join(opt.logger_name, "performance.log")
        loss_file = os.path.join(opt.logger_name, "loss.log")
        logging_func(log_file, message)
        print("model name: ", opt.model_name, flush=True)
        adjust_learning_rate(opt, optimizer, epoch)
        run_time = 0
        for i, (cnns, captions, lengths, masks, ids, lab, bovws) in enumerate(train_loader):
            start_time = time.time()
            model.train()
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)

            sim_P, sim_A1, sim_A2 = model(cnns, captions, lengths, bovws)

            loss_P_L = MSB_criterion_P_L(sim_P, lab)
            loss_A1_L = MSB_criterion_A1_L(sim_A1, lab)
            loss_A2_L = MSB_criterion_A2_L(sim_A2, lab)

            loss = loss_P_L + opt.beta_1 * loss_A1_L + opt.beta_2 * loss_A2_L

            with torch.autograd.detect_anomaly():
                loss.backward()

            if opt.grad_clip > 0:
                clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            run_time += time.time() - start_time
            # logging at every log_step
            if (i + 1) % opt.log_step == 0:
                log = "epoch: {0}; batch: {1}/{2}; loss: {3}; time: {4}\n".format(epoch,
                                i + 1, len(train_loader), loss.data.item(), run_time / 100)
                logging_func(loss_file, log)
                print(log, flush=True)
                run_time = 0
        print("-------- performance at epoch: %d --------" % (epoch))
        # evaluate on validation set
        mAP_sum = eval_local_mAP(model.module, val_loader, opt)
        # remember best mAP_sum and save checkpoint
        is_best = mAP_sum > best_mAP_sum
        best_mAP_sum = max(mAP_sum, best_mAP_sum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_mAP_sum': best_mAP_sum,
            'opt': opt,
        }, is_best, prefix=opt.model_name + '/')


def save_checkpoint(state, is_best, filename='model.pth.tar', prefix=''):
    tries = 15
    error = None
    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                message = "\n--------save best model at epoch %d---------\n" % (state["epoch"] - 1)
                print(message, flush=True)
                log_file = os.path.join(prefix, "performance.log")
                logging_func(log_file, message)
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries), flush=True)
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    print("learning rate %f in epoch %d" % (lr, epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
