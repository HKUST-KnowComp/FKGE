#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np
from Cryptodome import Random
from Cryptodome.PublicKey import RSA
# from arch.api.utils import log_utils
import encrypt.gmpy_math as gmpy_math


# LOGGER = log_utils.getLogger()


class Encrypt(object):
    def __init__(self):
        self.public_key = None
        self.privacy_key = None

    def generate_key(self, n_length=0):
        pass

    def set_public_key(self, public_key):
        pass

    def get_public_key(self):
        pass

    def set_privacy_key(self, privacy_key):
        pass

    def get_privacy_key(self):
        pass

    def encrypt(self, value):
        pass

    def decrypt(self, value):
        pass

    def encrypt_list(self, values):
        result = [self.encrypt(int(msg * (10000000))) for msg in values]
        return result

    def decrypt_list(self, values):
        result = [self.decrypt(int(msg)) / 10000000 for msg in values]
        return result

    def distribute_decrypt(self, X):
        decrypt_table = X.mapValues(lambda x: self.decrypt(x))
        return decrypt_table

    def distribute_encrypt(self, X):
        encrypt_table = X.mapValues(lambda x: self.encrypt(x))
        return encrypt_table

        # decrypt a np.array with arbitrary dimension

    def recursive_decrypt(self, A):
        if not isinstance(A, np.ndarray) and not isinstance(A, list):
            data_dict = A.collect()
            data_dict = dict(data_dict)
            A = list(data_dict.values())
            A = np.array(A)
        # LOGGER.debug("type A is {}".format(type(A)))
        if isinstance(A, list):
            A = np.array(A)
        # LOGGER.debug("shape of A: {}".format(A.shape))
        decrypt_row = []
        if len(A.shape) == 1:
            decrypt_row.extend(self.decrypt_list(A))
            return np.array(decrypt_row, dtype=np.float64)
        for row_index, row in enumerate(A):
            if len(np.array(row).shape) >= 2:
                decrypt_row.append(self.recursive_decrypt(row))
            else:
                decrypted_term = self.decrypt_list(row)
                decrypt_row.append(decrypted_term)
        return np.array(decrypt_row, dtype=np.float64)

    def recursive_encrypt(self, A):
        if not isinstance(A, np.ndarray) and not isinstance(A, list):
            data_dict = A.collect()
            data_dict = dict(data_dict)
            A = data_dict.values()
            A = np.array(A)

        if isinstance(A, list):
            A = np.array(A)

        encrypt_row = []
        if len(A.shape) == 1:
            encrypt_row.extend(self.encrypt_list(A))
            return encrypt_row
 
        for row_index, row in enumerate(A):
            if len(np.array(row).shape) >= 2:
                encrypt_row.append(self.recursive_encrypt(row))
            else:
                encrypted_term = self.encrypt_list(row)
                encrypt_row.append(encrypted_term)
        return np.array(encrypt_row, dtype=np.float64)

class RsaEncrypt(Encrypt):
    def __init__(self):
        super(RsaEncrypt, self).__init__()
        self.e = None
        self.d = None
        self.n = None

    def generate_key(self, rsa_bit=1024):
        random_generator = Random.new().read
        rsa = RSA.generate(rsa_bit, random_generator)
        self.e = rsa.e
        self.d = rsa.d
        self.n = rsa.n

    def get_key_pair(self):
        return self.e, self.d, self.n

    def set_public_key(self, public_key):
        self.e = public_key["e"]
        self.n = public_key["n"]

    def get_public_key(self):
        return self.e, self.n

    def set_privacy_key(self, privacy_key):
        self.d = privacy_key["d"]
        self.n = privacy_key["n"]

    def get_privacy_key(self):
        return self.d, self.n

    def encrypt(self, value):
        if self.e is not None and self.n is not None:
            return gmpy_math.powmod(int(value), self.e, self.n)
        else:
            return None

    def decrypt(self, value):
        if self.d is not None and self.n is not None:
            return gmpy_math.powmod(int(value), self.d, self.n)
        else:
            return None
