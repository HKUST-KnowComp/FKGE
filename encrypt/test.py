import numpy as np
from encrypt import RsaEncrypt as Encrypt


#x = np.array([1, 2, 3, 4, 5, 6, 7], np.int64)
x = [[0.2, 0.3, 0.02, 0.03, 0.002, 0.003, 0.0002, 0.0003]]
print(x)
encrypter = Encrypt()
encrypter.generate_key(37)
encrypted = encrypter.recursive_encrypt(x)
decrypted = encrypter.recursive_decrypt(encrypted)
print(encrypted)
print(decrypted)
