import socket
import time
import os
from dotenv import load_dotenv

# carrega configurações do arquivo .env
load_dotenv()

# acessa variáveis de ambiente
arquivo = os.getenv("ARQUIVO")
host = os.getenv("HOST")
porta = int(os.getenv("PORTA"))
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

# Cria e configura o socket servidor
servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
servidor.setsockopt(
    socket.SOL_SOCKET, socket.SO_REUSEADDR, 1
)  # evita erro "porta em uso"
servidor.bind((host, porta))
servidor.listen(1)

print(f"Bot aguardando conexão em {host}:{porta} ...")
conexao, endereco = servidor.accept()  # bloqueia até alguém conectar
print(f"Cliente conectado: {endereco}")

# Loop de envio em loop infinito
indice = 0
try:
    while True:
        frame = frames[indice]

        # Serializa: 64 números separados por tab + newline no final
        linha = "\t".join(f"{v:.6f}" for v in frame) + "\n"

        conexao.sendall(linha.encode("utf-8"))

        time.sleep(delay)  # controla a taxa de envio
        indice = (indice + 1) % len(frames)  # volta ao 0 depois do 118

except (BrokenPipeError, ConnectionResetError):
    print("Cliente desconectou.")
finally:
    conexao.close()
    servidor.close()
    print("Bot encerrado.")
