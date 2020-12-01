import torch
import torchcsprng as prng
import numpy as np
import math
import secrets
from utils import mod_between
from serialize import serialize_ndarray, deserialize_ndarray


class RegevPublicParameters:
    def __init__(self, n: int, m: int, cipher_mod: int, bs: int, bound: int):
        self.n = n
        self.m = m
        self.cipher_mod = cipher_mod
        self.bs = bs
        self.bound = bound

    @classmethod
    def for_pack(self, sec_param: int, num_add: int, num_mes: int, mes_mod: int):
        n = 792
        m = 32
        bound = 20
        cipher_mod = 8 * mes_mod * num_mes * bound * (num_add + 1)
        return RegevPublicParameters(n, m, cipher_mod, num_mes, bound)


class RegevKey:
    def __init__(self, seed: bytes, A: torch.tensor, sec: torch.tensor):
        self.seed = seed
        self.A = A
        self.sec = sec

    @ classmethod
    def gen(cls, pp: RegevPublicParameters, seed: torch.tensor = None):
        """ Generate a key for batched/packed Regev encryption """
        if seed is None:
            seed = prng.aes128_key_tensor(prng.create_random_device_generator())
        rng = prng.create_const_generator(seed)
        sec = torch.empty((pp.n, pp.bs), dtype=torch.int64).random_(pp.cipher_mod, generator=rng)
        A = torch.empty((pp.m, pp.n), dtype=torch.int64).random_(pp.cipher_mod, generator=rng)
        return RegevKey(seed, A, sec)

    def __eq__(self, other):
        return self.seed.eq(other.seed).all()

    def to_bytes(self) -> bytearray:
        """ Returns a byte representation of the Regev key 'self' """
        return serialize_ndarray(self.seed.numpy(), 1<<8)

    @ classmethod
    def from_bytes(cls, pp: RegevPublicParameters, b: bytearray):
        """ Reconstructs a key (in the batched/packed Regev encryption) from its byte representation """
        seed = torch.from_numpy(deserialize_ndarray(b,1,1<<8))
        return RegevKey.gen(pp, seed)

    def __repr__(self):
        return f"Regev Key:\nRandomness Seed: {self.seed}"


class BatchedRegevCiphertext:
    def __init__(self, c1: torch.tensor, c2: torch.tensor, mes_mod: int):
        self.c1 = c1
        self.c2 = c2
        self.mes_mod = mes_mod

    @ classmethod
    def encrypt_raw(cls, pp: RegevPublicParameters, k: RegevKey, mes: torch.tensor, mes_mod: int = 2, seed: torch.tensor=None):
        if seed is None:
            seed = prng.aes128_key_tensor(prng.create_random_device_generator())
        rng = prng.create_const_generator(seed)
        if mes.ndim != 1:
            raise MessageWrongDimensions()
        if mes.shape[0] != pp.bs:
            raise MessageWrongSize(
                f"Expected message size {pp.bs}, got {mes.shape[0]}")
        mes = mes % mes_mod
        R = torch.empty((1, pp.m), dtype=torch.int64).random_(-1,2, generator=rng)
        #print(R)
        c1 = R @ k.A % pp.cipher_mod
        e = torch.empty((1, pp.bs)).normal_(0,pp.bound / 2.5, generator=rng).round().int().clamp(min=-pp.bound, max=pp.bound)
        b = (c1 @ k.sec + e) % pp.cipher_mod
        c2 = (b + round(pp.cipher_mod / mes_mod) * mes) % pp.cipher_mod
        return BatchedRegevCiphertext(c1, c2, mes_mod)

    def __repr__(self):
        return f"bRegev Ciphertext:\nMessage Modulus: {self.mes_mod}\n{self.c1}\n{self.c2}"

    def __add__(self, other):
        c1 = (self.c1 + other.c1) % self.mes_mod
        c2 = (self.c1 + other.c1) % self.mes_mod
        return BatchedRegevCiphertext(c1, c2, self.mes_mod)

    def __eq__(self, other):
        return self.c1.eq(other.c1).all() and self.c2.eq(other.c2).all() and self.mes_mod == other.mes_mod

    def to_bytes(self, pp: RegevKey) -> bytes:
        """ Turns a batched Regev ciphertext 'self' into its byte representation """
        res = bytearray()
        num_batches = self.c1.shape[0]
        res.extend(num_batches.to_bytes(8, "little"))
        # The length of the ciphertext modulus in bytes
        cipher_mod_len = math.ceil(pp.cipher_mod.bit_length()/8)
        res.extend(self.mes_mod.to_bytes(cipher_mod_len, "little"))
        res.extend(serialize_ndarray(self.c1.numpy(), cipher_mod_len))
        res.extend(serialize_ndarray(self.c2.numpy(), cipher_mod_len))
        return res

    @ classmethod
    def from_bytes(cls, pp: RegevPublicParameters, b: bytes):
        """ Recovers a batched Regev ciphertext from its byte representation """
        num_batches = int.from_bytes(b[:8], "little")
        b = b[8:]
        c1 = np.zeros((num_batches, pp.n), dtype=int)
        c2 = np.zeros((num_batches, pp.bs), dtype=int)

        cipher_mod_len = math.ceil(pp.cipher_mod.bit_length()/8)
        mes_mod = int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        c1 = torch.from_numpy(deserialize_ndarray(b, (num_batches, pp.n), cipher_mod_len))
        c2 = torch.from_numpy(deserialize_ndarray(
            b[num_batches*pp.n*cipher_mod_len:], (num_batches, pp.bs), cipher_mod_len))
        return BatchedRegevCiphertext(c1, c2, mes_mod)

    def decrypt(self, pp: RegevPublicParameters, k: RegevKey, mes_mod: int = None) -> torch.tensor:
        """ Decrypts a batched Regev ciphertext 'self' """
        mes_mod = mes_mod or self.mes_mod
        noisy_message = (
            ((self.c2 - self.c1 @ k.sec) % pp.cipher_mod) * mes_mod) / pp.cipher_mod
        return noisy_message.round().int() % mes_mod

    def pack(self, pp: RegevPublicParameters, seed = None):
        """ More densely encodes a  batched Regev ciphertext 'self' """
        if seed is None:
            seed = prng.aes128_key_tensor(prng.create_random_device_generator())
        rng = prng.create_const_generator(seed)
        while True:
            r = torch.empty(1,dtype=torch.int64).random_(pp.cipher_mod)
            if PackedRegevCiphertext._near_mes((r+self.c2) % pp.cipher_mod, pp.bound, pp.cipher_mod, self.mes_mod):
                break

        w = ((((self.c2 + r) % pp.cipher_mod) * self.mes_mod) /
                   pp.cipher_mod).round().to(torch.int64) % self.mes_mod
        return PackedRegevCiphertext(self.c1, w, r, self.mes_mod)


class PackedRegevCiphertext:
    def __init__(self, c1: torch.tensor, w: torch.tensor, r: int, mes_mod: int):
        self.c1 = c1
        self.w = w
        self.r = r
        self.mes_mod = mes_mod

    def __repr__(self):
        return f"pRegev Ciphertext:\n{self.c1}\n{self.w}\n{self.r}"

    def __eq__(self, other):
        return (self.c1 == other.c1).all() and (self.w == other.w).all() and self.r == other.r and self.mes_mod == other.mes_mod

    def to_bytes(self, pp: RegevPublicParameters) -> bytes:
        """ Turns a ciphertext into a byte representation """
        res = bytearray()
        num_batches = self.c1.shape[0]
        res.extend(num_batches.to_bytes(8, "little"))
        # The length of the ciphertext modulus in bytes
        cipher_mod_len = math.ceil(pp.cipher_mod.bit_length()/8)
        res.extend(self.mes_mod.to_bytes(cipher_mod_len, "little"))
        res.extend(int(self.r[0]).to_bytes(cipher_mod_len, "little"))
        res.extend(serialize_ndarray(self.c1.numpy(), cipher_mod_len))
        mes_mod_len = math.ceil(self.mes_mod.bit_length()/8)
        res.extend(serialize_ndarray(self.w.numpy(), mes_mod_len))
        return res

    @ classmethod
    def from_bytes(cls, pp: RegevPublicParameters, b: bytes):
        num_batches = int.from_bytes(b[:8], "little")
        b = b[8:]
        c1 = torch.empty((num_batches, pp.n), dtype=torch.int64)
        w = torch.empty((num_batches, pp.bs), dtype=torch.int64)
        r = torch.empty((1,), dtype=torch.int64)

        cipher_mod_len = math.ceil(pp.cipher_mod.bit_length()/8)
        mes_mod = int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        r[0] = int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        c1 = torch.from_numpy(deserialize_ndarray(b, (num_batches, pp.n), cipher_mod_len))
        b = b[num_batches*pp.n*cipher_mod_len:]
        mes_mod_len = math.ceil(mes_mod.bit_length()/8)
        w = torch.from_numpy(deserialize_ndarray(b, (num_batches, pp.bs), mes_mod_len))
        return PackedRegevCiphertext(c1, w, r, mes_mod)

    @ classmethod
    def _near_mes_scalar(cls, x: int, bound: int, cipher_mod: int, mes_mod: int):
        frac = cipher_mod / mes_mod
        check_around = frac / 2
        for _ in range(mes_mod):
            if mod_between(x, (check_around - bound - 1) % cipher_mod, (check_around + bound + 1) % cipher_mod):
                return False
            check_around += frac
        return True

    @ classmethod
    def _near_mes(cls, arr:torch.tensor, bound: int, cipher_mod: int, mes_mod: int):
        return np.vectorize(lambda x: cls._near_mes_scalar(x, bound, cipher_mod, mes_mod))(arr).all()

    def decrypt(self, pp: RegevPublicParameters, k: RegevKey, mes_mod=None) -> torch.tensor:
        mes_mod = mes_mod or self.mes_mod
        return (self.w - ((((self.c1 @ k.sec + self.r) % pp.cipher_mod) * mes_mod)/pp.cipher_mod).round().int()) % mes_mod

"""
mes_mod = 64
num_mes = 100
pp = RegevPublicParameters.for_pack(32, 0, num_mes, mes_mod)
k = RegevKey.gen(pp)
#print("A:",k.A.shape)
#print("s:",k.sec.shape)
mes1 = torch.empty(num_mes, dtype=int) % 2
c = BatchedRegevCiphertext.encrypt_raw(pp,k,mes1).pack(pp)
mes2 = c.decrypt(pp,k)
print(mes1, mes2)
"""
class MessageWrongDimensions(Exception):
    pass


class MessageWrongSize(Exception):
    pass
