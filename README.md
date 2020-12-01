# Packed Regev

A python implementation of the packed Regev encryption scheme.

The Regev encryption scheme is a additivly homomorphic, private key encryption scheme.
It is secure under the LWE assumption (which likely is secure even against attacks from quantum computers).

The packed version of this scheme combines two modifications which improve ciphertext sizes.

## Install Dependencies

To install all necessary dependencies run
``` pip install -r requirements.txt ```

## Usage

Here is how you encrypt with packed Regev

``` python
import numpy as np

# Generate the public parameters to use this scheme
sec_param = 256
mes_mod = 64
num_mes = 100
pp = RegevPublicParameters.for_pack(sec_param, 0, num_mes, mes_mod)
# Generate the key according to these parameters
k = RegevKey.gen(pp)
# The message you want to encrypt
mes = np.random.randint(mes_mod, size=num_mess)
# Encrypt the message using the batched Regev encryption
c = BatchedRegevCiphertext.encrypt_raw(pp, k, mes, mes_mod=mes_mod)
# Pack the ciphertext (convert to packed Regev)
cp = c.pack(pp)
```

You can then decrypt it via

``` python
cp.decrypt(pp,k)
```

You can serialize any object of this module via `.to_bytes` and deserialize via `.from_bytes`.
The specific arguments for these methods are detailed in the documentation

Here should be a paragraph about the parameter choice

## TODOs

- [x] fix correctness of packed regev
- [ ] cli
- [x] easier parameter choice
- [x] serialization
  - [x] keys
  - [x] ciphertext
  - [x] packed ciphertext
- [x] seeded randomness (smaller secret keys)
- [x] speed improvements by implementing parts in Cython
  - [x] serialization
- [ ] addition for packed ciphertexts
- [x] unit testing framework
- [ ] package things
