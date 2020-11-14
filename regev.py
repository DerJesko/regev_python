import numpy as np
import math
from utils import mround, uniform, mod_between, deserialize_ndarray, gaussian
from serialize import serialize_ndarray


class BatchedRegevSecretKey:
    def __init__(self, sec: np.array, n: int, cipher_mod: int, bs: int):
        self.n = n
        self.cipher_mod = cipher_mod
        self.sec = sec
        self.bs = bs

    @ classmethod
    def gen(cls, n: int = 752, cipher_mod: int = 1 << 16, bs: int = 1):
        """ Generate a secret key """
        return BatchedRegevSecretKey(uniform(cipher_mod, (n, bs)), n=n, cipher_mod=cipher_mod, bs=bs)

    def pk_gen(self, m: int, bound: int = None):
        bound = bound or self.cipher_mod // (40 * self.bs)
        A = uniform(self.cipher_mod, (m, self.n))
        # b = As + e where e is a small gaussian error
        b = (A @ self.sec + gaussian(bound, (m, self.bs))
             ) % self.cipher_mod
        return BatchedRegevPublicKey(self.n, m, bound, self.bs, self.cipher_mod, A, b)

    def __eq__(self, other):
        return self.n == other.n and self.cipher_mod == other.cipher_mod and (self.sec == other.sec).all() and self.bs == other.bs

    def to_bytes(self) -> bytes:
        res = bytearray()
        res.extend(self.n.to_bytes(8, "little"))
        res.extend(self.bs.to_bytes(8, "little"))
        res.extend(self.cipher_mod.to_bytes(8, "little"))
        cipher_mod_len = math.ceil(math.log2(self.cipher_mod)/8)
        res.extend(serialize_ndarray(self.sec, cipher_mod_len))
        return res

    @ classmethod
    def from_bytes(cls, b: bytes):
        n = int.from_bytes(b[:8], "little")
        bs = int.from_bytes(b[8:16], "little")
        cipher_mod = int.from_bytes(b[16:24], "little")
        b = b[24:]
        cipher_mod_len = math.ceil(math.log2(cipher_mod)/8)
        sec = deserialize_ndarray(b, (n, bs), cipher_mod_len)
        return BatchedRegevSecretKey(sec, n=n, cipher_mod=cipher_mod, bs=bs)

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
        res = bytearray()
        res.extend(self.n.to_bytes(8, "big"))
        res.extend(self.m.to_bytes(8, "big"))
        res.extend(self.bound.to_bytes(8, "big"))
        res.extend(self.bs.to_bytes(8, "big"))
        res.extend(self.cipher_mod.to_bytes(8, "big"))
        cipher_mod_len = math.ceil(math.log2(self.cipher_mod)/8)
        res.extend(serialize_ndarray(self.A, cipher_mod_len))
        res.extend(serialize_ndarray(self.b, cipher_mod_len))
        return res

    @ classmethod
    def from_bytes(cls, b):
        n = int.from_bytes(b[:8], "big")
        m = int.from_bytes(b[8:16], "big")
        bound = int.from_bytes(b[16:24], "big")
        bs = int.from_bytes(b[24:32], "big")
        cipher_mod = int.from_bytes(b[32:40], "big")
        b = b[40:]
        cipher_mod_len = math.ceil(math.log2(cipher_mod)/8)
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
    def encrypt_raw(cls, pk: BatchedRegevPublicKey, mes: np.ndarray, mes_mod: int = 2):
        if mes.ndim != 2:
            raise MessageWrongDimensions()
        if mes.shape[1] != pk.bs:
            raise MessageWrongSize()
        R = uniform(2, (mes.shape[0], pk.m))
        c1 = R @ pk.A % pk.cipher_mod
        c2 = (R @ pk.b + mround(pk.cipher_mod / mes_mod)
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
        """ Turns a ciphertext into a byte representation """
        res = bytearray()
        num_batches = self.c1.shape[0]
        res.extend(num_batches.to_bytes(8, "little"))
        # The length of the ciphertext modulus in bytes
        cipher_mod_len = math.ceil(math.log2(pk.cipher_mod)/8)
        res.extend(self.mes_mod.to_bytes(cipher_mod_len, "little"))
        res.extend(serialize_ndarray(self.c1, cipher_mod_len))
        res.extend(serialize_ndarray(self.c2, cipher_mod_len))
        return res

    @ classmethod
    def from_bytes(cls, b: bytes, pk: BatchedRegevPublicKey):
        num_batches = int.from_bytes(b[:8], "little")
        b = b[8:]
        c1 = np.zeros((num_batches, pk.n), dtype=int)
        c2 = np.zeros((num_batches, pk.bs), dtype=int)

        cipher_mod_len = math.ceil(math.log2(pk.cipher_mod)/8)
        mes_mod = int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        c1 = deserialize_ndarray(b, (num_batches, pk.n), cipher_mod_len)
        c2 = deserialize_ndarray(
            b[num_batches*pk.n*cipher_mod_len:], (num_batches, pk.bs), cipher_mod_len)
        return BatchedRegevCiphertext(c1, c2, mes_mod)

    def decrypt(self, pk: BatchedRegevPublicKey, sk: BatchedRegevSecretKey, mes_mod: int = None) -> np.ndarray:
        mes_mod = mes_mod or self.mes_mod
        noisy_message = (
            ((self.c2 - self.c1 @ sk.sec) % pk.cipher_mod) * mes_mod) / pk.cipher_mod
        return mround(noisy_message) % mes_mod

    def pack(self, pk: BatchedRegevPublicKey):
        r = uniform(pk.cipher_mod)
        while not PackedRegevCiphertext._near_mes(r+self.c2, pk.bound, pk.cipher_mod, self.mes_mod):
            r = uniform(pk.cipher_mod)
        c2 = mround(((((self.c2 + r) % pk.cipher_mod) * self.mes_mod) /
                     pk.cipher_mod)) % self.mes_mod
        return PackedRegevCiphertext(self.c1, c2, r, self.mes_mod)


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
        res = b""
        num_batches = self.c1.shape[0]
        res += num_batches.to_bytes(8, "little")
        # The length of the ciphertext modulus in bytes
        cipher_mod_len = math.ceil(math.log2(pk.cipher_mod)/8)
        res += self.mes_mod.to_bytes(cipher_mod_len, "little")
        res += int(self.r[0]).to_bytes(cipher_mod_len, "little")
        res += serialize_ndarray(self.c1, cipher_mod_len)
        mes_mod_len = math.ceil(math.log2(self.mes_mod)/8)
        res += serialize_ndarray(self.w, mes_mod_len)
        return res

    @ classmethod
    def from_bytes(cls, b: bytes, pk: BatchedRegevPublicKey):
        num_batches = int.from_bytes(b[:8], "little")
        b = b[8:]
        c1 = np.zeros((num_batches, pk.n), dtype=int)
        w = np.zeros((num_batches, pk.bs), dtype=int)
        r = np.zeros((1,), dtype=int)

        cipher_mod_len = math.ceil(math.log2(pk.cipher_mod)/8)
        mes_mod = int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        r[0] += int.from_bytes(b[:cipher_mod_len], "little")
        b = b[cipher_mod_len:]
        c1 = deserialize_ndarray(b, (num_batches, pk.n), cipher_mod_len)
        b = b[num_batches*pk.n*cipher_mod_len:]
        mes_mod_len = math.ceil(math.log2(mes_mod)/8)
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
