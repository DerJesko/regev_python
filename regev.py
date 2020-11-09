import numpy as np
from utils import mround, uniform, mod_between, binomial, gaussian


class BatchedRegevSecretKey:
    def __init__(self, n: int, cipher_mod: int, bs: int = 1):
        self.n = n
        self.cipher_mod = cipher_mod
        self.sec = uniform(cipher_mod, (n, bs))
        self.bs = bs

    def __repr__(self):
        return f"bRegev sk:\n{self.sec}"


class BatchedRegevPublicKey:
    def __init__(self, sec_key: BatchedRegevSecretKey, m: int, bound: int = 3):
        self.n = sec_key.n
        self.m = m
        self.bound = bound
        self.bs = sec_key.bs
        self.cipher_mod = sec_key.cipher_mod
        self.A = uniform(self.cipher_mod, (m, self.n))
        # b = As + e where e is a small gaussian error
        self.b = (self.A @ sec_key.sec + gaussian(bound, (m, self.bs))
                  ) % self.cipher_mod

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
