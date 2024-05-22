import copy
import pickle
import random
import socket
import logging
import numpy as np
from models.Update import LocalUpdateDP, LocalUpdateDPSerial

from utils.encrypt import *

class User:
    def __init__(self, id: str, args, dataset, data_id):
        self.id = id
        self.host = socket.gethostname()
        self.port = int("1" + id.zfill(4))

        self.args = args
        self.pub_key = None
        self.__priv_key = None

        if "Agg" in self.args.protocol:
            pub_key, priv_key = SIG.gen(nbits=1024)
            self.pub_key = pub_key
            self.__priv_key = priv_key
        self.pub_key_map = []

        self.c_pk = None
        self.__c_sk = None
        self.s_pk = None
        self.__s_sk = None

        self.ka_pub_keys_map = None

        self.__random_seed = None

        self.ciphertexts = None

        self.U_3 = None

        if args.serial:
            self.update = LocalUpdateDPSerial(args=args, dataset=dataset, idxs=data_id)
        else:
            self.update = LocalUpdateDP(args=args, dataset=dataset, idxs=data_id)

        # The amount of uploaded data and downloaded data on the client (byte)
        self.uploaded = 0
        self.downloaded = 0

    def upload(self, package):
        self.uploaded += len(package)

    def download(self, package):
        self.downloaded += len(package)

    def gen_DH_pairs(self):
        self.c_pk, self.__c_sk = KA.gen()
        self.s_pk, self.__s_sk = KA.gen()

    def gen_signature(self):
        msg = pickle.dumps([self.c_pk, self.s_pk])
        signature = SIG.sign(msg, self.__priv_key)

        return signature

    def ver_signature(self) -> bool:
        status = True
        for key, value in self.ka_pub_keys_map.items():
            msg = pickle.dumps([value["c_pk"], value["s_pk"]])

            res = SIG.verify(msg, value["signature"], self.pub_key_map[key])

            if res is False:
                status = False
                logging.error("user {}'s signature is wrong!".format(key))

        return status

    def set_pub_keys_map(self, data):
        self.ka_pub_keys_map = pickle.loads(data)

        logging.info("received all signatures from the server")

    def gen_shares(self, U_1: list, t: int):
        self.__random_seed = random.randint(0, 2**32 - 1)

        n = len(U_1)

        s_sk_shares = SS.share(self.__s_sk, t, n)
        random_seed_shares = SS.share(self.__random_seed, t, n)

        all_ciphertexts = {}       # {id: ciphertext}

        for i, v in enumerate(U_1):
            if v == self.id:
                continue

            info = pickle.dumps([self.id, v, s_sk_shares[i], random_seed_shares[i]])

            v_c_pk = self.ka_pub_keys_map[v]["c_pk"]
            shared_key = KA.agree(self.__c_sk, v_c_pk)

            ciphertext = AE.encrypt(shared_key, shared_key, info)

            all_ciphertexts[v] = ciphertext

        msg = pickle.dumps([self.id, all_ciphertexts])

        return msg

    def set_ciphertexts(self, data):
        self.ciphertexts = pickle.loads(data)

        logging.info("received ciphertext from the server")

    def mask_gradients(self, gradients: dict):
        U_2 = list(self.ciphertexts.keys())

        # generate user's own private mask vector p_u_0 and p_u_1
        priv_mask_vec_0 = dict()
        # priv_mask_vec_1 = dict()

        for key, value in gradients.items():
            rs = np.random.RandomState(self.__random_seed | 0)
            priv_mask_vec_0[key] = rs.random(value.shape)
            # rs = np.random.RandomState(self.__random_seed | 1)
            # priv_mask_vec_1[key] = rs.random(value.shape)

        # generate random vectors p_u_v_0 and p_u_v_1 for each user
        random_vec_0_list = []
        # random_vec_1_list = []
        # alpha = 0
        for v in U_2:
            if v == self.id:
                continue

            v_s_pk = self.ka_pub_keys_map[v]["s_pk"]
            shared_key = KA.agree(self.__s_sk, v_s_pk)

            random.seed(shared_key)
            s_u_v = random.randint(0, 2**32 - 1)
            # alpha = (alpha + s_u_v) % (2 ** 32)

            # expand s_u_v into two random vectors
            p_u_v_0 = dict()
            for key, value in gradients.items():
                rs = np.random.RandomState(s_u_v | 0)
                if int(self.id) > int(v):
                    p_u_v_0[key] = rs.random(value.shape)
                else:
                    p_u_v_0[key] = -rs.random(value.shape)
            # rs = np.random.RandomState(s_u_v | 1)
            # p_u_v_1 = rs.random(gradients.shape)
            random_vec_0_list.append(p_u_v_0)
            # random_vec_1_list.append(p_u_v_1)

        # expand α into two random vectors
        # alpha = 10000
        # rs = np.random.RandomState(alpha | 0)
        # self.__a = rs.random(gradients.shape)
        # rs = np.random.RandomState(alpha | 1)
        # self.__b = rs.random(gradients.shape)

        # verification_code = self.__a * gradients + self.__b

        masked_gradients = copy.deepcopy(gradients)
        for k in masked_gradients.keys():
            masked_gradients[k] += priv_mask_vec_0[k]

        for d in random_vec_0_list:
            for k in d.keys():
                masked_gradients[k] += d[k]
        
        # verification_gradients = verification_code + priv_mask_vec_1 + np.sum(np.array(random_vec_1_list), axis=0)

        # msg = pickle.dumps([self.id, masked_gradients, verification_gradients])
        msg = pickle.dumps([self.id, masked_gradients])

        return msg

    def consistency_check(self):
        return
    
    def unmask_gradients(self):
        """Sends the shares of offline users' private key and online users' random seed to the server.

        Args:
            host (str): the server's host.
            port (str): the server's port used to receive the shares.
        """

        U_2 = list(self.ciphertexts.keys())

        priv_key_shares_map = {}
        random_seed_shares_map = {}

        for v in U_2:
            if self.id == v:
                continue

            v_c_pk = self.ka_pub_keys_map[v]["c_pk"]
            shared_key = KA.agree(self.__c_sk, v_c_pk)

            info = pickle.loads(AE.decrypt(shared_key, shared_key, self.ciphertexts[v]))

            if v not in self.U_3:
                # send the shares of s_sk to the server
                priv_key_shares_map[v] = info[2]
            else:
                # send the shares of random seed to the server
                random_seed_shares_map[v] = info[3]

        msg = pickle.dumps([self.id, priv_key_shares_map, random_seed_shares_map])

        return msg

    # def verify(self, output_gradients, verification_gradients, num_U_3):
    #     gradients_prime = self.__a * output_gradients + num_U_3 * self.__b

    #     return ((gradients_prime - verification_gradients) < np.full(output_gradients.shape, 1e-6)).all()
