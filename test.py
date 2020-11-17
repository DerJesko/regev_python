from regev import BatchedRegevPublicKey, BatchedRegevCiphertext, BatchedRegevSecretKey, PackedRegevCiphertext
import numpy as np
import time
import unittest
from serialize import serialize_ndarray


class TestKeyGen(unittest.TestCase):
    def test_sk_gen(self):
        for _ in range(100):
            BatchedRegevSecretKey.gen(bs=6)

    def test_seed_gen(self):
        for _ in range(100):
            BatchedRegevSecretKey.gen(bs=6, seed=b"\00"*32)

    def test_sk_ser(self):
        for _ in range(100):
            sk = BatchedRegevSecretKey.gen(bs=6)
            sk1 = BatchedRegevSecretKey.from_bytes(sk.to_bytes())
            assert sk == sk1

    def test_seed_ser(self):
        for _ in range(100):
            sk = BatchedRegevSecretKey.gen(bs=6, seed=b"\00"*32)
            sk1 = BatchedRegevSecretKey.from_bytes(sk.to_bytes())
            assert sk == sk1

    def test_pk_gen(self):
        sk = BatchedRegevSecretKey.gen(bs=6)
        for i in range(50):
            sk.pk_gen(i + 1)

    def test_pk_ser(self):
        sk = BatchedRegevSecretKey.gen(bs=6)
        for i in range(100):
            pk = sk.pk_gen(i + 1)
            pk1 = BatchedRegevPublicKey.from_bytes(pk.to_bytes())
            assert pk == pk1


class TestBatRegev(unittest.TestCase):
    def test_enc(self):
        for bs in range(5):
            bs += 1
            sk = BatchedRegevSecretKey.gen(bs=bs)
            for _ in range(5):
                pk = sk.pk_gen(2)
                for i in range(1, 100):
                    bss = i + 1
                    mes = np.zeros((bss, bs), dtype=int) + i
                    c = BatchedRegevCiphertext.encrypt_raw(pk, mes)
                    mes1 = c.decrypt(pk, sk)
                    if not ((mes % 2) == mes1).all():
                        print(bss, mes1)

    def test_ser_c(self):
        for bs in range(5):
            bs += 1
            sk = BatchedRegevSecretKey.gen(bs=bs)
            for _ in range(5):
                pk = sk.pk_gen(2)
                for i in range(1, 20):
                    bss = i + 1
                    mes = np.zeros((bss, bs), dtype=int) + bss
                    c = BatchedRegevCiphertext.encrypt_raw(pk, mes)
                    c1 = BatchedRegevCiphertext.from_bytes(c.to_bytes(pk), pk)
                    assert c == c1


class TestPackRegev(unittest.TestCase):
    def test_enc(self):
        counter = 0
        for bs in range(5):
            bs += 1
            sk = BatchedRegevSecretKey.gen(bs=bs)
            for _ in range(5):
                pk = sk.pk_gen(1)
                for i in range(100):
                    bss = i + 1
                    mes = np.zeros((bss, bs), dtype=int) + i
                    c = BatchedRegevCiphertext.encrypt_raw(pk, mes).pack(pk)
                    mes1 = c.decrypt(pk, sk)

                    if not ((mes % 2) == mes1).all():
                        counter += 1
        assert counter == 0, f"{counter} ciphertext decrypted wrong"

    def test_ser_c(self):
        for bs in range(5):
            bs += 1
            sk = BatchedRegevSecretKey.gen(bs=bs)
            for _ in range(5):
                pk = sk.pk_gen(2)
                for i in range(1, 20):
                    bss = i + 1
                    mes = np.zeros((bss, bs), dtype=int) + bss
                    c = BatchedRegevCiphertext.encrypt_raw(pk, mes).pack(pk)
                    c1 = PackedRegevCiphertext.from_bytes(c.to_bytes(pk), pk)
                    assert c == c1


if __name__ == '__main__':
    unittest.main()
