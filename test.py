import torch
import torchcsprng as prng
import time
import unittest
from regev import RegevKey, BatchedRegevCiphertext, PackedRegevCiphertext, RegevPublicParameters
from serialize import serialize_ndarray


class TestKeyGen(unittest.TestCase):
    def test_sk_gen(self):
        pp = RegevPublicParameters(23, 2, 69, 88, 2)
        for _ in range(100):
            RegevKey.gen(pp)

    def test_seed_gen(self):
        pp = RegevPublicParameters(23, 2, 69, 88, 2)
        for i in range(10):
            seed = prng.aes128_key_tensor(prng.create_random_device_generator())
            k1 = RegevKey.gen(pp, seed)
            k2 = RegevKey.gen(pp, seed)
            assert k1 == k2, "Key Generation is not deterministic"


class TestBatRegev(unittest.TestCase):
    def test_enc(self):
        mes_mod = 64
        num_mes = 100
        pp = RegevPublicParameters.for_pack(32, 0, num_mes, mes_mod)
        failcounter = 0
        trycounter = 0
        for bs in range(5):
            bs += 1
            k = RegevKey.gen(pp)
            for i in range(1, 100):
                mes = (torch.zeros(num_mes, dtype=int) + i) % mes_mod
                c = BatchedRegevCiphertext.encrypt_raw(
                    pp, k, mes, mes_mod=mes_mod)
                mes1 = c.decrypt(pp, k)

                trycounter += 1
                if not torch.all(mes.eq(mes1)):
                    failcounter += 1
        assert failcounter == 0, f"{failcounter} out of {trycounter} ciphertext decrypted wrong"

    def test_ser_c(self):
        mes_mod = 64
        num_mes = 100
        pp = RegevPublicParameters.for_pack(256, 0, num_mes, mes_mod)
        for bs in range(5):
            bs += 1
            k = RegevKey.gen(pp)
            for i in range(20):
                i += 1
                mes = torch.zeros(num_mes, dtype=int) + i
                c1 = BatchedRegevCiphertext.encrypt_raw(pp, k, mes)
                c2 = BatchedRegevCiphertext.from_bytes(pp, c1.to_bytes(pp))
                assert c1 == c2


class TestPackRegev(unittest.TestCase):
    def test_enc(self):
        mes_mod = 64
        num_mes = 100
        pp = RegevPublicParameters.for_pack(32, 0, num_mes, mes_mod)
        failcounter = 0
        trycounter = 0
        for bs in range(5):
            bs += 1
            k = RegevKey.gen(pp)
            for i in range(1, 100):
                mes = (torch.zeros(num_mes, dtype=int) + i) % mes_mod
                c = BatchedRegevCiphertext.encrypt_raw(
                    pp, k, mes, mes_mod=mes_mod).pack(pp)
                mes1 = c.decrypt(pp, k)

                trycounter += 1
                if not torch.all(mes.eq(mes1)):
                    failcounter += 1
        assert failcounter == 0, f"{failcounter} out of {trycounter} ciphertext decrypted wrong"

    def test_ser_c(self):
        mes_mod = 64
        num_mes = 100
        pp = RegevPublicParameters.for_pack(256, 0, num_mes, mes_mod)
        for bs in range(5):
            bs += 1
            k = RegevKey.gen(pp)
            for i in range(20):
                i += 1
                mes = torch.zeros(num_mes, dtype=int) + i
                c1 = BatchedRegevCiphertext.encrypt_raw(pp, k, mes).pack(pp)
                c2 = PackedRegevCiphertext.from_bytes(pp, c1.to_bytes(pp))
                assert c1 == c2


if __name__ == '__main__':
    unittest.main()
