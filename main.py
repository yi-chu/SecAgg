#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import random
import time
import cProfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import sys
import logging

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdateDP, LocalUpdateDPSerial
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM
from models.Fed import FedAvg, FedWeightAvg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule

from utils import *
from user import User
from server import *
from threading import Thread

clients = []
U_1 = []            # ids of all online users
U_2 = []            # ids of all users sending the ciphertexts
U_3 = []            # ids of all users sending the masked gradients
U_4 = []            # ids of all users sending the consistency check
wait_time = 300     # maximum waiting time for each round
t = 0               # threshold value of Shamir's t-out-of-n Secret Sharing

def masked_input_collection(user_gradients: dict) -> bool:

    server = clients["server"]
    MaskingRequestHandler.U_2_num = len(U_2)

    # start the masked gradients collection socket server
    server_thread = Thread(target=server.masking_server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    def fun(user):
        user.mask_gradients(user_gradients[user.id], server.host, server.masking_port)

    for u in U_2:
        user = clients[u]

        thread = Thread(target=fun, args=[user])
        thread.daemon = True
        thread.start()

    time.sleep(0.2)

    cnt = 0
    while len(MaskingRequestHandler.U_3) != len(U_2) and cnt < wait_time:
        time.sleep(1)
        cnt += 1

    if len(MaskingRequestHandler.U_3) >= t:
        global U_3
        U_3 = MaskingRequestHandler.U_3

        logging.info("{} users have sent masked gradients".format(len(U_3)))

        return True
    else:
        # the number of the received messages is less than the threshold value for SecretSharing, abort
        return False


def consistency_check() -> int:
    server = clients["server"]
    ConsistencyRequestHandler.U_3_num = len(U_3)

    # start the consistency check socket server
    server_thread = Thread(target=server.consistency_server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    status_list = []

    for u in U_3:
        thread = Thread(target=clients[u].consistency_check, args=[
                        server.host, server.consistency_port, status_list])
        thread.daemon = True
        thread.start()

    msg = pickle.dumps(U_3)
    for u in U_3:
        server.send(msg, clients[u].host, clients[u].port)

    time.sleep(0.2)

    cnt = 0
    while len(ConsistencyRequestHandler.U_4) != len(U_3) and cnt < wait_time:
        time.sleep(1)
        cnt += 1

    if len(ConsistencyRequestHandler.U_4) >= t:
        global U_4
        U_4 = ConsistencyRequestHandler.U_4

        logging.info("{} users have sent consistency checks".format(len(U_4)))

        for u in U_4:
            msg = pickle.dumps(ConsistencyRequestHandler.consistency_check_map)
            server.send(msg, clients[u].host, clients[u].port)

        if False in status_list:
            # at least one user failed in consistency check
            return 2
        else:
            return 0
    else:
        # the number of the received messages is less than the threshold value for SecretSharing, abort
        return 1


def unmasking(shape: tuple) -> np.ndarray:
    """Each user sends the shares of offline users' private key and online users' random seed to the server.
       The server unmasks gradients by reconstructing random vectors and private mask vectors.

    Args:
        shape (tuple): the shape of the raw gradients.

    Returns:
        Tuple[np.ndarray, np.ndarray]: the sum of the raw gradients and verification gradients.
    """

    server = clients["server"]
    UnmaskingRequestHandler.U_4_num = len(U_4)

    # start the unmasking socket server
    server_thread = Thread(target=server.unmasking_server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    for u in U_4:
        thread = Thread(target=clients[u].unmask_gradients, args=[server.host, server.unmasking_port])
        thread.daemon = True
        thread.start()

    time.sleep(0.2)

    cnt = 0
    while len(UnmaskingRequestHandler.U_5) != len(U_4) and cnt < wait_time:
        time.sleep(1)
        cnt += 1

    if len(UnmaskingRequestHandler.U_5) >= t:
        logging.info("{} users have sent shares".format(len(UnmaskingRequestHandler.U_5)))

        output, verification = server.unmask(shape)

        return output, verification

    else:
        # the number of the received messages is less than the threshold value for SecretSharing, abort
        return None

if __name__ == '__main__':
    # init logger
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)

    # handler for file
    fh = logging.FileHandler('test.log')
    fh.setLevel(logging.INFO)

    # set formatter
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')

    # add formatter
    fh.setFormatter(formatter)

    # add handler
    logger.addHandler(fh)

    # parse args
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        args.num_channels = 1
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        #trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        args.num_channels = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    elif args.dataset == 'fashion-mnist':
        args.num_channels = 1
        trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset_train = datasets.FashionMNIST('./data/fashion-mnist', train=True, download=True,
                                              transform=trans_fashion_mnist)
        dataset_test  = datasets.FashionMNIST('./data/fashion-mnist', train=False, download=True,
                                              transform=trans_fashion_mnist)
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'femnist':
        args.num_channels = 1
        dataset_train = FEMNIST(train=True)
        dataset_test = FEMNIST(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: femnist dataset is naturally non-iid')
        else:
            print("Warning: The femnist dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif args.dataset == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        args.num_users = len(dict_users)
        if args.iid:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print("Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape
    logger.info("dataset [{}] is loaded".format(args.dataset))

    net_glob = None
    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and (args.dataset == 'mnist' or args.dataset == 'fashion-mnist'):
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.dataset == 'femnist' and args.model == 'cnn':
        net_glob = CNNFemnist(args=args).to(args.device)
    elif args.dataset == 'shakespeare' and args.model == 'lstm':
        net_glob = CharLSTM().to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    logger.info("model [{}] is built".format(args.model))

    # use opacus to wrap model to clip per sample gradient
    if args.protocol == "SecAgg":
        args.dp_mechanism = "no_dp"
    if args.dp_mechanism != 'no_dp':
        net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()
    logger.info("dp mechanism [{}] is set".format(args.dp_mechanism))

    # copy weights
    w_glob = net_glob.state_dict()
    all_clients = list(range(args.num_users))

    # init server
    server = Server()
    server.sig_handler.user_num = args.num_users

    # start the signature socket server
    logger.info("The server has been initialized")

    # init clients
    pub_key_map = {}
    for i in range(args.num_users):
        clients.append(User(id=str(i), args=args, dataset=dataset_train, data_id=dict_users[i]))
        if "Agg" in args.protocol:
            pub_key_map[i] = clients[i].pub_key
            clients[i].upload(pickle.dumps([i, clients[i].pub_key]))
            server.download(pickle.dumps([i, clients[i].pub_key]))

    # init clients' signature
    if "Agg" in args.protocol:
        for i in range(args.num_users):
            clients[i].pub_key_map = pub_key_map
            clients[i].download(pickle.dumps([pub_key_map]))
            server.upload(pickle.dumps([pub_key_map]))

    logger.info("The clients[{}] have been initialized".format(args.num_users))

    # protocal start
    logger.info("protocal [{}] start".format(args.protocol))
    acc_test = []
    m, loop_index = max(int(args.frac * args.num_users), 1), int(1 / args.frac)
    t = int(args.Threshold * args.num_users)

    for iter in range(args.epochs):
        t_start = time.time()
        w_locals, loss_locals, weight_locols = [], [], []

        # select clients (round-robin selection)
        begin_index = (iter % loop_index) * m
        end_index = begin_index + m
        idxs_users = all_clients[begin_index:end_index]

        logger.info("clients {} are selected".format(idxs_users))

        U_1 = []
        U_2 = []
        U_3 = []
        U_4 = []
        server.clean()

        # key agreement
        if "Agg" in args.protocol:
            for id in idxs_users:
                clients[id].gen_DH_pairs()
                signature = clients[id].gen_signature()

                msg = pickle.dumps({
                    "id": clients[id].id,
                    "c_pk": clients[id].c_pk,
                    "s_pk": clients[id].s_pk,
                    "signature": signature
                })

                clients[id].upload(msg)
                server.download(msg)
                server.sig_handler.handle(data=msg)
            
            data = server.broadcast_signatures()
            U_1 = server.sig_handler.U_1
            for id in idxs_users:
                server.upload(data)
                clients[id].download(data)
                clients[id].set_pub_keys_map(data)

            logger.info("advertise keys finished")
            logger.info("U_1 users: " + ','.join(U_1))
        else:
            U_1 = [str(i) for i in idxs_users]

        # share secret
        if args.protocol == "SecAgg":
            server.ss_handler.U_1_num = len(U_1)

            for u in U_1:
                data = clients[int(u)].gen_shares(U_1, t)
                server.ss_handler.handle(data=data)
                clients[int(u)].upload(data)
                server.download(data)

            U_2 = server.ss_handler.U_2

            for u in U_2:
                data = pickle.dumps(server.ss_handler.ciphertexts_map[u])
                clients[int(u)].set_ciphertexts(data)
                clients[int(u)].download(data)
                server.upload(data)

            logger.info("share secret finished")
            logger.info("U_2 users: " + ','.join(U_2))

        else:
            U_2 = [str(i) for i in idxs_users]

        # local model train
        for u in U_2:
            w, loss = clients[int(u)].update.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[int(u)]))

            gradients = dict()
            for k,v in w.items():
                gradients[k] = v.cpu().numpy()
            if args.protocol == "SecAgg":
                data = clients[int(u)].mask_gradients(gradients)
                server.mask_handler.handle(data)
                clients[int(u)].upload(data)
                server.download(data)
        
        logger.info("mask gradients finished")

        # unmasking weights
        if args.protocol == "SecAgg":
            print("clinets num:", int(len(server.mask_handler.U_3)*(1 - args.drop_out)))
            server.mask_handler.U_3 = sorted(random.sample(server.mask_handler.U_3, int(len(server.mask_handler.U_3)*(1 - args.drop_out))))
            U_3 = server.mask_handler.U_3
            print("U_3: ", U_3)
            for u in U_3:
                clients[int(u)].U_3 = U_3


            for u in U_3:
                data = clients[int(u)].unmask_gradients()
                server.unmask_handler.handle(data)
                clients[int(u)].upload(data)
                server.download(data)

            weights = dict()
            for k,v in w.items():
                weights[k] = net_glob.state_dict()[k].numpy()
            
            cProfile.run('server.unmask(weights)')
            new_weights = server.unmask(weights)
            
            w_glob = dict()
            for k,v in new_weights.items():
                w_glob[k] = torch.from_numpy(new_weights[k])

            net_glob.load_state_dict(w_glob)

        logger.info("unmask gradients finished")
        logger.info("U_3 users: " + ','.join(U_3))

        # # update global weights
        # w_glob = FedWeightAvg(w_locals, weight_locols)
        # # copy weight to net_glob
        # net_glob.load_state_dict(w_glob)

        # print accuracy
        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, args)
        t_end = time.time()
        logger.info("Round {:3d},Testing accuracy: {:.2f},Time:  {:.2f}s".format(iter, acc_t, t_end - t_start))

        acc_test.append(acc_t.item())

    rootpath = './log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    accfile = open(rootpath + '/accfile_fed_{}_{}_{}_iid{}_dp_{}_epsilon_{}.dat'.
                   format(args.dataset, args.model, args.epochs, args.iid,
                          args.dp_mechanism, args.dp_epsilon), "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot loss curve
    plt.figure()
    plt.plot(range(len(acc_test)), acc_test)
    plt.ylabel('test accuracy')
    plt.savefig(rootpath + '/fed_{}_{}_{}_C{}_iid{}_dp_{}_epsilon_{}_acc.png'.format(
        args.dataset, args.model, args.epochs, args.frac, args.iid, args.dp_mechanism, args.dp_epsilon))



