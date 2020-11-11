import numpy as np
import math
from utils import mround, uniform, mod_between, binomial, gaussian


class BatchedRegevSecretKey:
    def __init__(self, sec: np.array, n: int, cipher_mod: int, bs: int):
        self.n = n
        self.cipher_mod = cipher_mod
        self.sec = sec
        self.bs = bs

    @classmethod
    def gen(cls, n: int = 752, cipher_mod: int = 2**15, bs: int = 1):
        """ Generate a secret key """
        return BatchedRegevSecretKey(uniform(cipher_mod, (n, bs)), n=n, cipher_mod=cipher_mod, bs=bs)

    def pk_gen(self, m: int, bound: int = None):
        bound = bound or self.cipher_mod // 10
        A = uniform(self.cipher_mod, (m, self.n))
        # b = As + e where e is a small gaussian error
        b = (A @ self.sec + gaussian(bound, (m, self.bs))
             ) % self.cipher_mod
        return BatchedRegevPublicKey(self.n, m, bound, self.bs, self.cipher_mod, A, b)

    def __eq__(self, other):
        return self.n == other.n and self.cipher_mod == other.cipher_mod and (self.sec == other.sec).all() and self.bs == other.bs

    def to_bytes(self):
        res = b""
        res += self.n.to_bytes(8, "big")
        res += self.bs.to_bytes(8, "big")
        res += self.cipher_mod.to_bytes(8, "big")
        cipher_mod_len = math.ceil(math.log2(self.cipher_mod)/8)
        for i in np.nditer(self.sec):
            res += int(i).to_bytes(cipher_mod_len, "big")
        return res

    @classmethod
    def from_bytes(cls, b: bytes):
        n = int.from_bytes(b[:8], "big")
        b = b[8:]
        bs = int.from_bytes(b[:8], "big")
        b = b[8:]
        cipher_mod = int.from_bytes(b[:8], "big")
        b = b[8:]
        sec = np.zeros((n, bs))
        cipher_mod_len = math.ceil(math.log2(cipher_mod)/8)
        for i, _ in np.ndenumerate(sec):
            sec[i] = int.from_bytes(b[:cipher_mod_len], "big")
            b = b[cipher_mod_len:]
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

    def to_bytes(self):
        res = b""
        res += self.n.to_bytes(8, "big")
        res += self.m.to_bytes(8, "big")
        res += self.bound.to_bytes(8, "big")
        res += self.bs.to_bytes(8, "big")
        res += self.cipher_mod.to_bytes(8, "big")
        cipher_mod_len = math.ceil(math.log2(self.cipher_mod)/8)
        for i in np.nditer(self.A):
            res += int(i).to_bytes(cipher_mod_len, "big")
        for i in np.nditer(self.b):
            res += int(i).to_bytes(cipher_mod_len, "big")
        return res

    @classmethod
    def from_bytes(cls, b):
        n = int.from_bytes(b[:8], "big")
        b = b[8:]
        m = int.from_bytes(b[:8], "big")
        b = b[8:]
        bound = int.from_bytes(b[:8], "big")
        b = b[8:]
        bs = int.from_bytes(b[:8], "big")
        b = b[8:]
        cipher_mod = int.from_bytes(b[:8], "big")
        b = b[8:]
        cipher_mod_len = math.ceil(math.log2(cipher_mod)/8)
        A = np.zeros((m, n), dtype=int)
        for i, _ in np.ndenumerate(A):
            A[i] = int.from_bytes(b[:cipher_mod_len], "big")
            b = b[cipher_mod_len:]
        Ase = np.zeros((m, bs), dtype=int)
        for i, _ in np.ndenumerate(Ase):
            Ase[i] = int.from_bytes(b[:cipher_mod_len], "big")
            b = b[cipher_mod_len:]
        return BatchedRegevPublicKey(n, m, bound, bs, cipher_mod, A, Ase)

    def __repr__(self):
        return f"bRegev pk:\n{self.A}\n{self.b}"


class BatchedRegevCiphertext:
    def __init__(self, c1: np.ndarray, c2: np.ndarray, mes_mod: int):
        self.c1 = c1
        self.c2 = c2
        self.mes_mod = mes_mod

    @classmethod
    def encrypt_raw(cls, pk: BatchedRegevPublicKey, mes: np.ndarray, mes_mod: int = 2):
        if mes.ndim != 2:
            raise MessageWrongDimensions()
        if mes.shape[1] != pk.bs:
            raise MessageWrongSize()
        R = uniform(2, (mes.shape[0], pk.m))
        c1 = R @ pk.A % pk.cipher_mod
        c2 = (R @ pk.b + mround(pk.cipher_mod / mes_mod)
              * mes) % pk.cipher_mod
        mes_mod = mes_mod
        return BatchedRegevCiphertext(c1, c2, mes_mod)

    def __repr__(self):
        return f"bRegev Ciphertext:\n{self.c1}\n{self.c2}"

    def __add__(self, other):
        c1 = (self.c1 + other.c1) % self.mes_mod
        c2 = (self.c1 + other.c1) % self.mes_mod
        return BatchedRegevCiphertext(c1, c2, self.mes_mod)

    def __eq__(self, other):
        return (self.c1 == other.c1).all() and (self.c1 == other.c1).all() and self.mes_mod == other.mes_mod

    def to_bytes(self, pk: BatchedRegevPublicKey):
        """ Turns a ciphertext into a byte representation """
        res = b""
        num_batches = self.c1.shape[0]
        res += num_batches.to_bytes(8, "big")
        # The length of the ciphertext modulus in bytes
        cipher_mod_len = math.ceil(math.log2(pk.cipher_mod)/8)
        res += self.mes_mod.to_bytes(cipher_mod_len, "big")
        for i in np.nditer(self.c1):
            res += int(i).to_bytes(cipher_mod_len, "big")
        for i in np.nditer(self.c2):
            res += int(i).to_bytes(cipher_mod_len, "big")
        return res

    @classmethod
    def from_bytes(cls, b: bytes, pk: BatchedRegevPublicKey):
        num_batches = int.from_bytes(b[:8], "big")
        b = b[8:]
        c1 = np.zeros((num_batches, pk.n), dtype=int)
        c2 = np.zeros((num_batches, pk.bs), dtype=int)

        cipher_mod_len = math.ceil(math.log2(pk.cipher_mod)/8)
        mes_mod = int.from_bytes(b[:cipher_mod_len], "big")
        b = b[cipher_mod_len:]
        for i, _ in np.ndenumerate(c1):
            c1[i] = int.from_bytes(b[:cipher_mod_len], "big")
            b = b[cipher_mod_len:]
        for i, _ in np.ndenumerate(c2):
            c2[i] = int.from_bytes(b[:cipher_mod_len], "big")
            b = b[cipher_mod_len:]
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
    def __init__(self, c1: np.ndarray, c2: np.ndarray, r: int, mes_mod: int):
        self.c1 = c1
        self.c2 = c2
        self.r = r
        self.mes_mod = mes_mod

    def __repr__(self):
        return f"pRegev Ciphertext:\n{self.c1}\n{self.c2}\n{self.r}"

    def __eq__(self, other):
        return (self.c1 == other.c1).all() and (self.c2 == other.c2).all() and self.r == other.r and self.mes_mod == other.mes_mod

    def to_bytes(self, pk: BatchedRegevPublicKey):
        """ Turns a ciphertext into a byte representation """
        res = b""
        num_batches = self.c1.shape[0]
        res += num_batches.to_bytes(8, "big")
        # The length of the ciphertext modulus in bytes
        cipher_mod_len = math.ceil(math.log2(pk.cipher_mod)/8)
        res += self.mes_mod.to_bytes(cipher_mod_len, "big")
        res += int(self.r[0]).to_bytes(cipher_mod_len, "big")
        for i in np.nditer(self.c1):
            res += int(i).to_bytes(cipher_mod_len, "big")
        mes_mod_len = math.ceil(math.log2(self.mes_mod)/8)
        for i in np.nditer(self.c2):
            res += int(i).to_bytes(mes_mod_len, "big")
        return res

    @classmethod
    def from_bytes(cls, b: bytes, pk: BatchedRegevPublicKey):
        num_batches = int.from_bytes(b[:8], "big")
        b = b[8:]
        c1 = np.zeros((num_batches, pk.n), dtype=int)
        c2 = np.zeros((num_batches, pk.bs), dtype=int)
        r = np.zeros((1,), dtype=int)

        cipher_mod_len = math.ceil(math.log2(pk.cipher_mod)/8)
        mes_mod = int.from_bytes(b[:cipher_mod_len], "big")
        b = b[cipher_mod_len:]
        r[0] += int.from_bytes(b[:cipher_mod_len], "big")
        b = b[cipher_mod_len:]
        for i, _ in np.ndenumerate(c1):
            c1[i] = int.from_bytes(b[:cipher_mod_len], "big")
            b = b[cipher_mod_len:]
        mes_mod_len = math.ceil(math.log2(mes_mod)/8)
        for i, _ in np.ndenumerate(c2):
            c2[i] = int.from_bytes(b[:mes_mod_len], "big")
            b = b[mes_mod_len:]
        return PackedRegevCiphertext(c1, c2, r, mes_mod)

    @classmethod
    def _near_mes_scalar(cls, x: int, bound: int, cipher_mod: int, mes_mod: int):
        frac = cipher_mod / mes_mod
        check_around = frac / 2
        for _ in range(mes_mod):
            if mod_between(x, (check_around - bound - 1) % cipher_mod, (check_around + bound + 1) % cipher_mod):
                return False
            check_around += frac
        return True

    @classmethod
    def _near_mes(cls, arr: np.ndarray, bound: int, cipher_mod: int, mes_mod: int):
        return np.vectorize(lambda x: cls._near_mes_scalar(x, bound, cipher_mod, mes_mod))(arr).all()

    def decrypt(self, pk: BatchedRegevPublicKey, sk: BatchedRegevSecretKey, mes_mod=None) -> BatchedRegevCiphertext:
        mes_mod = mes_mod or self.mes_mod
        return (self.c2 - mround(((self.c1 @ sk.sec + self.r) * mes_mod)/pk.cipher_mod)) % mes_mod


class MessageWrongDimensions(Exception):
    pass


class MessageWrongSize(Exception):
    pass
