import numpy as np
import time
import unittest
from regev import RegevKey, BatchedRegevCiphertext, PackedRegevCiphertext, RegevPublicParameters
from serialize import serialize_ndarray
from utils import mround


class TestKeyGen(unittest.TestCase):
    def test_sk_gen(self):
        pp = RegevPublicParameters(23, 2, 69, 88, 2)
        for _ in range(100):
            RegevKey.gen(pp)

    def test_seed_gen(self):
        pp = RegevPublicParameters(23, 2, 69, 88, 2)
        for i in range(10):
            k1 = RegevKey.gen(pp, i.to_bytes(32, "little"))
            k2 = RegevKey.gen(pp, i.to_bytes(32, "little"))
            assert k1 == k2, "Key Generation is not deterministic"


class TestBatRegev(unittest.TestCase):
    def test_enc(self):
        pp = RegevPublicParameters(23, 2, 69, 88, 2)
        failcounter = 0
        trycounter = 0
        for bs in range(5):
            bs += 1
            k = RegevKey.gen(pp)
            for i in range(1, 100):
                mes = np.zeros(bs, dtype=int) + i
                c = BatchedRegevCiphertext.encrypt_raw(pp, k, mes)
                mes1 = c.decrypt(pp, k)

                trycounter += 1
                if not ((mes % 2) == mes1).all():
                    failcounter += 1
        assert failcounter == 0, f"{failcounter} out of {trycounter} ciphertext decrypted wrong"

    def test_ser_c(self):
        pp = RegevPublicParameters(23, 2, 69, 88, 2)
        for bs in range(5):
            bs += 1
            k = RegevKey.gen(pp)
            for i in range(20):
                i += 1
                mes = np.zeros((bs), dtype=int) + i
                c1 = BatchedRegevCiphertext.encrypt_raw(pp, k, mes)
                c2 = BatchedRegevCiphertext.from_bytes(c1.to_bytes(pp), pp)
                assert c1 == c2


class TestPackRegev(unittest.TestCase):
    def test_enc(self):
        pp = RegevPublicParameters(23, 2, 69, 88, 2)
        failcounter = 0
        trycounter = 0
        for bs in range(5):
            bs += 1
            k = RegevKey.gen(pp)
            for i in range(1, 100):
                mes = np.zeros(bs, dtype=int) + i
                c = BatchedRegevCiphertext.encrypt_raw(pp, k, mes).pack(pp)
                mes1 = c.decrypt(pp, k)

                trycounter += 1
                if not ((mes % 2) == mes1).all():
                    failcounter += 1
        assert failcounter == 0, f"{failcounter} out of {trycounter} ciphertext decrypted wrong"

    def test_ser_c(self):
        pp = RegevPublicParameters(23, 2, 69, 88, 2)
        for bs in range(5):
            bs += 1
            k = RegevKey.gen(pp)
            for i in range(20):
                i += 1
                mes = np.zeros((bs), dtype=int) + i
                c1 = BatchedRegevCiphertext.encrypt_raw(pp, k, mes).pack(pp)
                c2 = BatchedRegevCiphertext.from_bytes(c1.to_bytes(pp), pp)
                assert c1 == c2


if __name__ == '__main__':
    unittest.main()
