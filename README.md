# Packed Regev

A python implementation of the packed Regev encryption scheme.

The Regev encryption scheme is a additivly homomorphic, public key encryption scheme.
It is secure under the LWE assumption (which likely is secure even against attacks from quantum computers).

The packed version of this scheme combines two modifications which improve ciphertext sizes.

## Install Dependencies

To install all necessary dependencies run
``` pip install -r requirements.txt ```

## Parameter Choice

Here should be a paragraph about the parameter choice

## TODOs

- [ ] cli
- [x] easier parameter choice
- [x] serialization
  - [x] keys
  - [x] ciphertext
  - [x] packed ciphertext
- [ ] seeded randomness (smaller secret keys)
- [ ] speed improvements by implementing parts in C
  - [ ] serialization
  - [ ] randomness
- [ ] addition for packed ciphertexts
- [x] unit testing framework
- [ ] package things
