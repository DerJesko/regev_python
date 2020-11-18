import numpy as np
import math
import secrets
from utils import mround, uniform, mod_between, gaussian, SeededRNG
from serialize import serialize_ndarray, deserialize_ndarray


class RegevPublicParameters:
    def __init__(self, n: int = None, m: int = None, cipher_mod: int = None, bs: int = 1, bound: int = None):
        self.n = n
        self.m = m
        self.cipher_mod = cipher_mod
        self.bs = bs
        self.bound = bound


class RegevKey:
    def __init__(self, seed: bytes, A: np.array, sec: np.array):
        self.seed = seed
        self.A = A
        self.sec = sec

    @ classmethod
    def gen(cls, pp: RegevPublicParameters, seed: bytes = None):
        """ Generate a key for batched/packed Regev encryption """
        seed = seed or secrets.token_bytes(32)
        rng = SeededRNG(seed)
        sec = uniform(pp.cipher_mod, rng, (pp.n, pp.bs))
        A = uniform(pp.cipher_mod, rng, (pp.m, pp.n))
        return RegevKey(seed, A, sec)

    def __eq__(self, other):
        return self.seed == other.seed

    def to_bytes(self) -> bytearray:
        """ Returns a byte representation of the Regev key 'self' """
        return self.seed

    @ classmethod
    def from_bytes(cls, pp: RegevPublicParameters, b: bytearray):
        """ Reconstructs a key (in the batched/packed Regev encryption) from its byte representation """
        seed = b
        return RegevKey.gen(pp, seed)

    def __repr__(self):
        return f"Regev Key:\nRandomness Seed: {self.seed}"


class BatchedRegevCiphertext:
    def __init__(self, c1: np.ndarray, c2: np.ndarray, mes_mod: int):
        self.c1 = c1
        self.c2 = c2
        self.mes_mod = mes_mod

    @ classmethod
    def encrypt_raw(cls, pp: RegevPublicParameters, k: RegevKey, mes: np.ndarray, mes_mod: int = 2, seed=None):
        rng = SeededRNG(seed or secrets.token_bytes(32))
        if mes.ndim != 1:
            raise MessageWrongDimensions()
        if mes.shape[0] != pp.bs:
            raise MessageWrongSize()
        mes = mes % mes_mod
        r = uniform(2, rng, lbound=-1, shape=(1, pp.m))
        c1 = r @ k.A % pp.cipher_mod
        b = k.A @ k.sec + gaussian(pp.bound, rng, shape=(pp.m, pp.bs))
        c2 = (r @ b + mround(pp.cipher_mod / mes_mod) * mes) % pp.cipher_mod
        return BatchedRegevCiphertext(c1, c2, mes_mod)

    def __repr__(self):
        return f"bRegev Ciphertext:\nMessage Modulus: {self.mes_mod}\n{self.c1}\n{self.c2}"

    def __add__(self, other):
        c1 = (self.c1 + other.c1) % self.mes_mod
        c2 = (self.c1 + other.c1) % self.mes_mod
        return BatchedRegevCiphertext(c1, c2, self.mes_mod)

    def __eq__(self, other):
        return (self.c1 == other.c1).all() and (self.c1 == other.c1).all() and self.mes_mod == other.mes_mod

    def to_bytes(self, pp: RegevKey) -> bytes:
        """ Turns a batched Regev ciphertext 'self' into its byte representation """
        res = bytearray()
        num_batches = self.c1.shape[0]
        res.extend(num_batches.to_bytes(8, "little"))
        # The length of the ciphertext modulus in bytes
        cipher_mod_len = math.ceil(pp.cipher_mod.bit_length()/8)
        res.extend(self.mes_mod.to_bytes(cipher_mod_len, "little"))
        res.extend(serialize_ndarray(self.c1, cipher_mod_len))
        res.extend(serialize_ndarray(self.c2, cipher_mod_len))
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
        c1 = deserialize_ndarray(b, (num_batches, pp.n), cipher_mod_len)
        c2 = deserialize_ndarray(
            b[num_batches*pp.n*cipher_mod_len:], (num_batches, pp.bs), cipher_mod_len)
        return BatchedRegevCiphertext(c1, c2, mes_mod)

    def decrypt(self, pp: RegevPublicParameters, k: RegevKey, mes_mod: int = None) -> np.ndarray:
        """ Decrypts a batched Regev ciphertext 'self' """
        mes_mod = mes_mod or self.mes_mod
        noisy_message = (
            ((self.c2 - self.c1 @ k.sec) % pp.cipher_mod) * mes_mod) / pp.cipher_mod
        return mround(noisy_message) % mes_mod

    def pack(self, pp: RegevPublicParameters, seed: bytes = None):
        """ More densely encodes a  batched Regev ciphertext 'self' """
        seed = seed or secrets.token_bytes(32)
        rng = SeededRNG(seed)
        while True:
            r = uniform(pp.cipher_mod, rng)
            if PackedRegevCiphertext._near_mes((r+self.c2) % pp.cipher_mod, pp.bound, pp.cipher_mod, self.mes_mod):
                break

        w = mround((((self.c2 + r) % pp.cipher_mod) * self.mes_mod) /
                   pp.cipher_mod) % self.mes_mod
        return PackedRegevCiphertext(self.c1, w, r, self.mes_mod)


class PackedRegevCiphertext:
    def __init__(self, c1: np.ndarray, w: np.ndarray, r: int, mes_mod: int):
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
        res.extend(serialize_ndarray(self.c1, cipher_mod_len))
        mes_mod_len = math.ceil(self.mes_mod.bit_length()/8)
        res.extend(serialize_ndarray(self.w, mes_mod_len))
        return res

    @ classmethod
    def from_bytes(cls, pp: RegevPublicParameters, b: bytes):
        num_batches = int.from_bytes(b[:8], "little")
        b = b[8:]
        c1 = np.zeros((num_batches, pp.n), dtype=int)
        w = np.zeros((num_batches, pp.bs), dtype=int)
        r = np.zeros((1,), dtype=int)

        cipher_mod_len = math.ceil(pp.cipher_mod.bit_length()/8)
        mes_mod = int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        r[0] += int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        c1 = deserialize_ndarray(b, (num_batches, pp.n), cipher_mod_len)
        b = b[num_batches*pp.n*cipher_mod_len:]
        mes_mod_len = math.ceil(mes_mod.bit_length()/8)
        w = deserialize_ndarray(b, (num_batches, pp.bs), mes_mod_len)
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
    def _near_mes(cls, arr: np.ndarray, bound: int, cipher_mod: int, mes_mod: int):
        return np.vectorize(lambda x: cls._near_mes_scalar(x, bound, cipher_mod, mes_mod))(arr).all()

    def decrypt(self, pp: RegevPublicParameters, k: RegevKey, mes_mod=None) -> np.array:
        mes_mod = mes_mod or self.mes_mod
        return (self.w - mround((((self.c1 @ k.sec + self.r) % pp.cipher_mod) * mes_mod)/pp.cipher_mod)) % mes_mod


class MessageWrongDimensions(Exception):
    pass


class MessageWrongSize(Exception):
    pass
