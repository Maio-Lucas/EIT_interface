import time
import os
from dotenv import load_dotenv
import serial, time

# carrega configurações do arquivo .env
load_dotenv()

# acessa variáveis de ambiente
arquivo = os.getenv("ARQUIVO")
host = os.getenv("HOST")
porta = 'COM5'
baudrate = 115200
fps = 5
delay = 1.0 / fps

# Leitura completa do TXT para memória
frames = []
with open(arquivo, "r") as f:
    for linha in f:
        linha = linha.strip()
        if not linha:  # ignora linhas vazias
            continue
        valores = [float(v) for v in linha.split("\t")]
        frames.append(valores)

print(f"Frames carregados: {len(frames)}")
print(f"Valores por frame: {len(frames[0])}")

ser = serial.Serial(porta, baudrate=baudrate, timeout=1)
print(f"Enviando para {porta}...")

i = 0
try:
    while True:
        ser.write(('\t'.join(f'{v:.6f}' for v in frames[i]) + '\n').encode('utf-8'))
        time.sleep(0.2)
        i = (i + 1) % len(frames)
        if i % 5 == 0:
            ser.write(b'\xff\n')
            time.sleep(2)
except KeyboardInterrupt:
    ser.close()

