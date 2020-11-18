import numpy as np
import math
from utils import mround, uniform, mod_between, gaussian, SeededRNG
from serialize import serialize_ndarray, deserialize_ndarray
import secrets


class BatchedRegevSecretKey:
    def __init__(self, n: int, cipher_mod: int, bs: int, seed=None, sec: np.array = None):
        self.n = n
        self.cipher_mod = cipher_mod
        self.sec = sec
        self.bs = bs
        self.seed = seed

    @ classmethod
    def gen(cls, n: int = 752, cipher_mod: int = 1 << 16, bs: int = 1, seed: bytes = None):
        """ Generate a secret key for batched/packed Regev encryption """
        seed = seed or secrets.token_bytes(32)
        rng = SeededRNG(seed)
        return BatchedRegevSecretKey(n=n, cipher_mod=cipher_mod, bs=bs, sec=uniform(cipher_mod, rng, (n, bs)), seed=seed)

    def pk_gen(self, m: int, bound: int = None, seed: bytes = None):
        """ Generate a public key for secret key 'self' in the batched/packed Regev encryption """
        seed = seed or secrets.token_bytes(32)
        rng = SeededRNG(seed)
        bound = bound or self.cipher_mod // (40 * self.bs)
        A = uniform(self.cipher_mod, rng, (m, self.n))
        # b = As + e where e is a small gaussian error
        b = (A @ self.sec + gaussian(bound, rng, (m, self.bs))
             ) % self.cipher_mod
        return BatchedRegevPublicKey(self.n, m, bound, self.bs, self.cipher_mod, A, b)

    def __eq__(self, other):
        return self.n == other.n and self.cipher_mod == other.cipher_mod and (self.sec == other.sec).all() and self.bs == other.bs

    def to_bytes(self) -> bytearray:
        """ Returns a byte representation of the secret key 'self' """
        res = bytearray()
        seed_len = len(self.seed)
        res.extend(seed_len.to_bytes(1, "little"))
        res.extend(self.n.to_bytes(8, "little"))
        res.extend(self.bs.to_bytes(8, "little"))
        res.extend(self.cipher_mod.to_bytes(8, "little"))
        res.extend(self.seed)
        return res

    @ classmethod
    def from_bytes(cls, b: bytearray):
        """ Reconstructs a secret key (in the batched/packed Regev encryption) from its byte representation """
        seed_len = int.from_bytes(b[:1], "little")
        n = int.from_bytes(b[1:9], "little")
        bs = int.from_bytes(b[9:17], "little")
        cipher_mod = int.from_bytes(b[17:25], "little")
        b = b[25:]
        seed = bytes(b[:seed_len])
        return BatchedRegevSecretKey.gen(n=n, cipher_mod=cipher_mod, bs=bs, seed=seed)

    def __repr__(self):
        return f"bRegev sk:\n{self.sec}"


class BatchedRegevPublicKey:
    def __init__(self, n: int, m: int, bound: int, bs: int, cipher_mod: int, A: np.array, b: np.array):
        self.n = n
        self.m = m
        self.bound = bound
        self.bs = bs
        self.cipher_mod = cipher_mod
        self.A = A
        self.b = b

    def __eq__(self, other):
        return self.n == other.n and self.m == other.m and self.bound == other.bound and self.bs == other.bs and self.cipher_mod == other.cipher_mod and (self.A == other.A).all() and (self.b == other.b).all()

    def to_bytes(self) -> bytes:
        """ Returns a byte representation of the public key 'self' """
        res = bytearray()
        res.extend(self.n.to_bytes(8, "little"))
        res.extend(self.m.to_bytes(8, "little"))
        res.extend(self.bound.to_bytes(8, "little"))
        res.extend(self.bs.to_bytes(8, "little"))
        res.extend(self.cipher_mod.to_bytes(8, "little"))
        cipher_mod_len = math.ceil(self.cipher_mod.bit_length()/8)
        res.extend(serialize_ndarray(self.A, cipher_mod_len))
        res.extend(serialize_ndarray(self.b, cipher_mod_len))
        return res

    @ classmethod
    def from_bytes(cls, b):
        """ Reconstructs a public key (in the batched/packed Regev encryption) from its byte representation """
        n = int.from_bytes(b[:8], "little")
        m = int.from_bytes(b[8:16], "little")
        bound = int.from_bytes(b[16:24], "little")
        bs = int.from_bytes(b[24:32], "little")
        cipher_mod = int.from_bytes(b[32:40], "little")
        b = b[40:]
        cipher_mod_len = math.ceil(cipher_mod.bit_length()/8)
        A = deserialize_ndarray(b[:cipher_mod_len*n*m], (m, n), cipher_mod_len)
        Ase = deserialize_ndarray(
            b[cipher_mod_len*n*m:], (m, bs), cipher_mod_len)
        return BatchedRegevPublicKey(n, m, bound, bs, cipher_mod, A, Ase)

    def __repr__(self):
        return f"bRegev pk:\n{self.A}\n{self.b}"


class BatchedRegevCiphertext:
    def __init__(self, c1: np.ndarray, c2: np.ndarray, mes_mod: int):
        self.c1 = c1
        self.c2 = c2
        self.mes_mod = mes_mod

    @ classmethod
    def encrypt_raw(cls, pk: BatchedRegevPublicKey, mes: np.ndarray, mes_mod: int = 2, seed=None):
        seed = seed or secrets.token_bytes(32)
        rng = SeededRNG(seed)
        if mes.ndim != 1:
            raise MessageWrongDimensions()
        if mes.shape[0] != pk.bs:
            raise MessageWrongSize()
        r = uniform(2, rng, lbound=-1, shape=(1, pk.m))
        c1 = r @ pk.A % pk.cipher_mod
        c2 = (r @ pk.b + mround(pk.cipher_mod / mes_mod)
              * mes) % pk.cipher_mod
        return BatchedRegevCiphertext(c1, c2, mes_mod)

    def __repr__(self):
        return f"bRegev Ciphertext:\n{self.c1}\n{self.c2}"

    def __add__(self, other):
        c1 = (self.c1 + other.c1) % self.mes_mod
        c2 = (self.c1 + other.c1) % self.mes_mod
        return BatchedRegevCiphertext(c1, c2, self.mes_mod)

    def __eq__(self, other):
        return (self.c1 == other.c1).all() and (self.c1 == other.c1).all() and self.mes_mod == other.mes_mod

    def to_bytes(self, pk: BatchedRegevPublicKey) -> bytes:
        """ Turns a batched Regev ciphertext 'self' into its byte representation """
        res = bytearray()
        num_batches = self.c1.shape[0]
        res.extend(num_batches.to_bytes(8, "little"))
        # The length of the ciphertext modulus in bytes
        cipher_mod_len = math.ceil(pk.cipher_mod.bit_length()/8)
        res.extend(self.mes_mod.to_bytes(cipher_mod_len, "little"))
        res.extend(serialize_ndarray(self.c1, cipher_mod_len))
        res.extend(serialize_ndarray(self.c2, cipher_mod_len))
        return res

    @ classmethod
    def from_bytes(cls, b: bytes, pk: BatchedRegevPublicKey):
        """ Recovers a batched Regev ciphertext from its byte representation """
        num_batches = int.from_bytes(b[:8], "little")
        b = b[8:]
        c1 = np.zeros((num_batches, pk.n), dtype=int)
        c2 = np.zeros((num_batches, pk.bs), dtype=int)

        cipher_mod_len = math.ceil(pk.cipher_mod.bit_length()/8)
        mes_mod = int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        c1 = deserialize_ndarray(b, (num_batches, pk.n), cipher_mod_len)
        c2 = deserialize_ndarray(
            b[num_batches*pk.n*cipher_mod_len:], (num_batches, pk.bs), cipher_mod_len)
        return BatchedRegevCiphertext(c1, c2, mes_mod)

    def decrypt(self, pk: BatchedRegevPublicKey, sk: BatchedRegevSecretKey, mes_mod: int = None) -> np.ndarray:
        """ Decrypts a batched Regev ciphertext 'self' """
        mes_mod = mes_mod or self.mes_mod
        noisy_message = (
            ((self.c2 - self.c1 @ sk.sec) % pk.cipher_mod) * mes_mod) / pk.cipher_mod
        return mround(noisy_message) % mes_mod

    def pack(self, pk: BatchedRegevPublicKey, seed=None):
        """ More densely encodes a  batched Regev ciphertext 'self' """
        seed = seed or secrets.token_bytes(32)
        rng = SeededRNG(seed)
        while True:
            r = uniform(pk.cipher_mod, rng)
            if PackedRegevCiphertext._near_mes(r+self.c2, pk.bound, pk.cipher_mod, self.mes_mod):
                break

        w = mround(((((self.c2 + r) % pk.cipher_mod) * self.mes_mod) /
                    pk.cipher_mod)) % self.mes_mod
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

    def to_bytes(self, pk: BatchedRegevPublicKey) -> bytes:
        """ Turns a ciphertext into a byte representation """
        res = bytearray()
        num_batches = self.c1.shape[0]
        res.extend(num_batches.to_bytes(8, "little"))
        # The length of the ciphertext modulus in bytes
        cipher_mod_len = math.ceil(pk.cipher_mod.bit_length()/8)
        res.extend(self.mes_mod.to_bytes(cipher_mod_len, "little"))
        res.extend(int(self.r[0]).to_bytes(cipher_mod_len, "little"))
        res.extend(serialize_ndarray(self.c1, cipher_mod_len))
        mes_mod_len = math.ceil(self.mes_mod.bit_length()/8)
        res.extend(serialize_ndarray(self.w, mes_mod_len))
        return res

    @ classmethod
    def from_bytes(cls, b: bytes, pk: BatchedRegevPublicKey):
        num_batches = int.from_bytes(b[:8], "little")
        b = b[8:]
        c1 = np.zeros((num_batches, pk.n), dtype=int)
        w = np.zeros((num_batches, pk.bs), dtype=int)
        r = np.zeros((1,), dtype=int)

        cipher_mod_len = math.ceil(pk.cipher_mod.bit_length()/8)
        mes_mod = int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        r[0] += int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        c1 = deserialize_ndarray(b, (num_batches, pk.n), cipher_mod_len)
        b = b[num_batches*pk.n*cipher_mod_len:]
        mes_mod_len = math.ceil(mes_mod.bit_length()/8)
        w = deserialize_ndarray(b, (num_batches, pk.bs), mes_mod_len)
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

    def decrypt(self, pk: BatchedRegevPublicKey, sk: BatchedRegevSecretKey, mes_mod=None) -> np.array:
        mes_mod = mes_mod or self.mes_mod
        return (self.w - mround(((self.c1 @ sk.sec + self.r) * mes_mod)/pk.cipher_mod)) % mes_mod


class MessageWrongDimensions(Exception):
    pass


class MessageWrongSize(Exception):
    pass
