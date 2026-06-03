"""
test_receiver.py — Etapa 2: valida o canal TCP antes de tocar na interface real.

Abre uma janela mínima que:
  - conecta ao bot via botão
  - conta frames recebidos
  - pisca o fundo verde a cada frame
  - mostra os 3 primeiros valores do último frame

Rode com o bot.py já rodando em outro terminal.
"""

import sys
import socket
import os
from dotenv import load_dotenv

from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
)

# carrega configurações do arquivo .env
load_dotenv()
host = os.getenv("HOST")
porta = int(os.getenv("PORTA"))


# ── Thread de leitura ─────────────────────────────────────────────────────────


class ReaderThread(QThread):
    """
    Roda em background, lê o socket linha a linha e emite um signal
    por frame recebido. Nunca bloqueia a thread da GUI.
    """

    frame_recebido = pyqtSignal(list)  # payload: lista de floats
    conexao_encerrada = pyqtSignal(str)  # payload: mensagem de status

    def __init__(self):
        super().__init__()
        self._ativo = True

    def run(self):
        """Ponto de entrada da thread — chamado por self.start()."""
        sock = None
        mensagem = "Desconectado"

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((host, porta))

            buffer = ""
            while self._ativo:
                dados = sock.recv(4096)  # bloqueia até chegar algo
                if not dados:  # servidor fechou a conexão
                    mensagem = "Servidor encerrou a conexão"
                    break

                buffer += dados.decode("utf-8")

                # processa todas as linhas completas disponíveis no buffer
                while "\n" in buffer:
                    linha, buffer = buffer.split("\n", 1)
                    linha = linha.strip()
                    if linha:
                        valores = [float(v) for v in linha.split("\t")]
                        self.frame_recebido.emit(valores)

        except ConnectionRefusedError:
            mensagem = "Conexão recusada — bot está rodando?"
        except Exception as e:
            mensagem = f"Erro: {e}"
        finally:
            if sock:
                sock.close()

        self.conexao_encerrada.emit(mensagem)

    def parar(self):
        """Sinaliza a thread para encerrar no próximo ciclo."""
        self._ativo = False
        self.wait()  # aguarda a thread terminar antes de retornar


# ── Janela mínima de teste ────────────────────────────────────────────────────


class TestWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("EITinterface — Teste de Recepção")
        self.setFixedSize(380, 240)

        self.contador = 0
        self.thread = None

        self._montar_ui()

    def _montar_ui(self):
        raiz = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)

        # status da conexão
        self.lbl_status = QLabel("● Desconectado")
        self.lbl_status.setStyleSheet("color: gray; font-size: 13px;")

        # contador grande
        self.lbl_contador = QLabel("Frames recebidos: 0")
        self.lbl_contador.setFont(QFont("Courier New", 20, QFont.Weight.Bold))

        # preview dos primeiros valores
        self.lbl_preview = QLabel("Último frame: —")
        self.lbl_preview.setStyleSheet("color: #444; font-size: 12px;")

        # botão de conectar / desconectar
        self.btn = QPushButton("Conectar")
        self.btn.setFixedHeight(36)
        self.btn.clicked.connect(self._toggle_conexao)

        layout.addWidget(self.lbl_status)
        layout.addWidget(self.lbl_contador)
        layout.addWidget(self.lbl_preview)
        layout.addStretch()
        layout.addWidget(self.btn)

        raiz.setLayout(layout)
        self.setCentralWidget(raiz)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _toggle_conexao(self):
        if self.thread is None or not self.thread.isRunning():
            self._conectar()
        else:
            self._desconectar()

    def _conectar(self):
        self.thread = ReaderThread()
        self.thread.frame_recebido.connect(self._ao_receber_frame)
        self.thread.conexao_encerrada.connect(self._ao_encerrar)
        self.thread.start()
        self.lbl_status.setText("● Conectando...")
        self.lbl_status.setStyleSheet("color: orange; font-size: 13px;")
        self.btn.setText("Desconectar")

    def _desconectar(self):
        if self.thread:
            self.thread.parar()
        self.btn.setText("Conectar")

    def _ao_receber_frame(self, valores: list):
        self.contador += 1

        # atualiza contador
        self.lbl_contador.setText(f"Frames recebidos: {self.contador}")

        # status verde na primeira recepção
        self.lbl_status.setText("● Conectado")
        self.lbl_status.setStyleSheet("color: green; font-size: 13px;")

        # preview: primeiros 3 valores
        preview = "  ".join(f"{v:.2f}" for v in valores[:3])
        self.lbl_preview.setText(f"Último frame: [{preview} ...]")

        # flash de fundo: verde por 80 ms
        self.centralWidget().setStyleSheet("background-color: #c8f7c5;")
        QTimer.singleShot(80, lambda: self.centralWidget().setStyleSheet(""))

    def _ao_encerrar(self, mensagem: str):
        self.lbl_status.setText(f"● {mensagem}")
        self.lbl_status.setStyleSheet("color: gray; font-size: 13px;")
        self.btn.setText("Conectar")
        self.thread = None

    def closeEvent(self, event):
        """Garante que a thread encerra quando a janela fecha."""
        if self.thread and self.thread.isRunning():
            self.thread.parar()
        event.accept()


# ── Ponto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TestWindow()
    win.show()
    sys.exit(app.exec())
