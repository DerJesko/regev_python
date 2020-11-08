from regev import BatchedRegevPublicKey, BatchedRegevCiphertext, BatchedRegevSecretKey
import numpy as np
import time

counter = 0
secret_key_time = 0
public_key_time = 0
encrypt_time = 0
compress_time = 0
decrypt_time = 0
for _ in range(100000):
    before = time.perf_counter()
    sk = BatchedRegevSecretKey(10, 300, bs=6)
    secret_key_time += time.perf_counter()-before

    before = time.perf_counter()
    pk = BatchedRegevPublicKey(sk, 5, bound=10)
    public_key_time += time.perf_counter()-before

    before = time.perf_counter()
    c = BatchedRegevCiphertext.encrypt_raw(pk, np.array(
        [[2, 1, 0, 1, 3, 1]]), mes_mod=2)
    encrypt_time += time.perf_counter()-before

    before = time.perf_counter()
    if (c.decrypt(pk, sk) != np.array([[0, 1, 0, 1, 1, 1]])).any():
        print(c)
        counter += 1

    decrypt_time += time.perf_counter()-before

print(secret_key_time, public_key_time,
      encrypt_time, compress_time, decrypt_time)
