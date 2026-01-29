import numpy as np
from Pyfhel import Pyfhel

# 1. Configuração do Objeto e Contexto
HE = Pyfhel() 
# n=tamanho do polinômio, t_bits=tamanho dos inteiros
HE.contextGen(scheme='bfv', n=2**14, t_bits=20) 
HE.keyGen()             # Gera chaves pública e privada

# 2. Criptografia
integer1 = np.array([10], dtype=np.int64)
integer2 = np.array([32], dtype=np.int64)

ctxt1 = HE.encryptInt(integer1) # Criptografa o primeiro número
ctxt2 = HE.encryptInt(integer2) # Criptografa o segundo número

# 3. Operação Homomórfica (Soma)
# Note que estamos somando os valores criptografados!
ctxt_sum = ctxt1 + ctxt2 

# 4. Descriptografia
res = HE.decryptInt(ctxt_sum)
print(f"Resultado da soma: {res[0]}") # Saída: 42