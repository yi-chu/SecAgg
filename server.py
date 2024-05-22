import pickle
import random
import logging
import copy
import numpy as np

from utils.encrypt import *


class SignatureRequestHandler():
    user_num = 0
    ka_pub_keys_map = {}    # {id: {c_pk: bytes, s_pk, bytes, signature: bytes}}
    U_1 = []

    def handle(self, data) -> None:
        # receive data from the client
        msg = pickle.loads(data)
        id = msg["id"]
        del msg["id"]

        self.ka_pub_keys_map[id] = msg
        self.U_1.append(id)

        received_num = len(self.U_1)

        logging.info("[%d/%d] | received user %s's signature", received_num, self.user_num, id)


class SecretShareRequestHandler():
    U_1_num = 0
    ciphertexts_map = {}         # {u:{v1: ciphertexts, v2: ciphertexts}}
    U_2 = []

    def handle(self, data) -> None:
        # receive data from the client
        msg = pickle.loads(data)
        id = msg[0]

        # retrieve each user's ciphertexts
        for key, value in msg[1].items():
            if key not in self.ciphertexts_map:
                self.ciphertexts_map[key] = {}
            self.ciphertexts_map[key][id] = value

        self.U_2.append(id)

        received_num = len(self.U_2)

        logging.info("[%d/%d] | received user %s's ciphertexts", received_num, self.U_1_num, id)


class MaskingRequestHandler():
    U_2_num = 0
    masked_gradients_list = []
    verification_gradients_list = []
    U_3 = []

    def handle(self, data) -> None:
        # receive data from the client
        msg = pickle.loads(data)
        id = msg[0]

        self.U_3.append(msg[0])
        self.masked_gradients_list.append(msg[1])
        # self.verification_gradients_list.append(msg[2])

        received_num = len(self.U_3)

        logging.info("[%d/%d] | received user %s's masked gradients and verification gradients",
                     received_num, self.U_2_num, id)


class ConsistencyRequestHandler():
    U_3_num = 0
    consistency_check_map = {}
    U_4 = []

    def handle(self, data) -> None:
        msg = pickle.loads(data)
        id = msg[0]

        self.U_4.append(id)
        self.consistency_check_map[id] = msg[1]

        received_num = len(self.U_4)

        logging.info("[%d/%d] | received user %s's consistency check", received_num, self.U_3_num, id)


class UnmaskingRequestHandler():
    U_4_num = 0
    priv_key_shares_map = {}        # {id: []}
    random_seed_shares_map = {}     # {id: []}
    U_5 = []

    def handle(self, data) -> None:
        msg = pickle.loads(data)
        id = msg[0]

        # retrieve the private key shares
        for key, value in msg[1].items():
            if key not in self.priv_key_shares_map:
                self.priv_key_shares_map[key] = []
            self.priv_key_shares_map[key].append(value)

        # retrieve the ramdom seed shares
        for key, value in msg[2].items():
            if key not in self.random_seed_shares_map:
                self.random_seed_shares_map[key] = []
            self.random_seed_shares_map[key].append(value)

        self.U_5.append(id)

        received_num = len(self.U_5)

        logging.info("[%d/%d] | received user %s's shares", received_num, self.U_4_num, id)


class Server:
    def __init__(self):
        self.sig_handler = SignatureRequestHandler()
        self.ss_handler = SecretShareRequestHandler()
        self.mask_handler = MaskingRequestHandler()
        self.consis_handler = ConsistencyRequestHandler()
        self.unmask_handler = UnmaskingRequestHandler()

        # The amount of uploaded data and downloaded data on the server (byte)
        self.uploaded = 0
        self.downloaded = 0

    def upload(self, package):
        self.uploaded += len(package)

    def download(self, package):
        self.downloaded += len(package)

    def broadcast_signatures(self):
        data = pickle.dumps(self.sig_handler.ka_pub_keys_map)
        self.upload(data)

        logging.info("broadcasted all signatures.")

        return data

    def unmask(self, gradients: dict) -> np.ndarray:
        # reconstruct random vectors p_v_u_0 and p_u_v_1
        recon_random_vec_0_list = []
        # recon_random_vec_1_list = []
        for u in self.ss_handler.U_2:
            if u not in self.mask_handler.U_3:
                # the user drops out, reconstruct its private keys and then generate the corresponding random vectors
                priv_key = SS.recon(self.unmask_handler.priv_key_shares_map[u])
                for v in self.mask_handler.U_3:
                    shared_key = KA.agree(priv_key, self.sig_handler.ka_pub_keys_map[v]["s_pk"])

                    random.seed(shared_key)
                    s_u_v = random.randint(0, 2**32 - 1)

                    # expand s_u_v into two random vectors
                    p_u_v_0 = dict()
                    for key, value in gradients.items():
                        rs = np.random.RandomState(s_u_v | 0)
                        p_u_v_0[key] = rs.random(value.shape)
                        if int(u) < int(v):
                            p_u_v_0[key] = -p_u_v_0[key]

                    recon_random_vec_0_list.append(p_u_v_0)

        # reconstruct private mask vectors p_u_0 and p_u_1
        recon_priv_vec_0_list = []
        # recon_priv_vec_1_list = []
        for u in self.mask_handler.U_3:
            random_seed = SS.recon(self.unmask_handler.random_seed_shares_map[u])
            priv_mask_vec_0 = dict()
            for k,v in gradients.items():
                rs = np.random.RandomState(random_seed | 0)
                priv_mask_vec_0[k] = rs.random(v.shape)
                
            # rs = np.random.RandomState(random_seed | 1)
            # priv_mask_vec_1 = rs.random(shape)

            recon_priv_vec_0_list.append(priv_mask_vec_0)
            # recon_priv_vec_1_list.append(priv_mask_vec_1)

        output = copy.deepcopy(gradients)
        for k in output.keys():
            output[k].fill(0)

        for d in self.mask_handler.masked_gradients_list:
            for k in d.keys():
                output[k] += d[k]
        
        for d in recon_priv_vec_0_list:
            for k in d.keys():
                output[k] -= d[k]

        for d in recon_random_vec_0_list:
            for k in d.keys():
                output[k] += d[k]

        for k in output.keys():
            output[k] /= len(self.mask_handler.U_3)

        # masked_gradients = np.sum(np.array(self.mask_handler.masked_gradients_list), axis=0)
        # recon_priv_vec_0 = np.sum(np.array(recon_priv_vec_0_list), axis=0)
        # recon_random_vec_0 = np.sum(np.array(recon_random_vec_0_list), axis=0)

        # output = masked_gradients - recon_priv_vec_0 + recon_random_vec_0

        # verification_gradients = np.sum(np.array(self.mask_handler.verification_gradients_list), axis=0)
        # recon_priv_vec_1 = np.sum(np.array(recon_priv_vec_1_list), axis=0)
        # recon_random_vec_1 = np.sum(np.array(recon_random_vec_1_list), axis=0)

        # verification = verification_gradients - recon_priv_vec_1 + recon_random_vec_1

        # return output, verification
        return output

    def clean(self):
        self.sig_handler.ka_pub_keys_map = {}
        self.sig_handler.U_1 = []
        self.ss_handler.ciphertexts_map = {}
        self.ss_handler.U_2 = []
        self.mask_handler.masked_gradients_list = []
        self.mask_handler.U_3 = []
        self.consis_handler.consistency_check_map = {}
        self.consis_handler.U_4 = []
        self.consis_handler.status_list = []
        self.unmask_handler.priv_key_shares_map = {}
        self.unmask_handler.random_seed_shares_map = {}
        self.unmask_handler.U_5 = []