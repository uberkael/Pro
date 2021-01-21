###################################################################
# Sistema de envio y recepcion simple para enviar arrays de numpy #
###################################################################

import socket
import sys
import numpy as np
import pickle
import Config

class cliente():
	"Clase que crea un socket de envio"
	def __init__(self):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

	def envia_array(self, datos):
		"Envia datos serializados a broadcast"
		# self.s.sendto(datos.encode(),
		# ('255.255.255.255', Config.Network.puerto))
		self.s.sendto(pickle.dumps(datos),
			('255.255.255.255', Config.Network.puerto))


class servidor():
	"Clase que crea un socket para recivir datos"
	def __init__(self, servidor=True):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		if servidor:
			self.s.bind(('', Config.Network.puerto))

	def recibe_array(self):
		"Recibe un array y lo desempaqueta"
		data, address = self.s.recvfrom(4096)
		print(pickle.loads(data))
		# print(data.decode())
		# m = self.s.recv(4096)
		# print(pickle.loads(m))
		# print(m)


def is_port_in_use(port):
	"Devuelve si el puerto esta siendo utilizado"
	# Funcion extraida de
	# https://stackoverflow.com/a/52872579/3052862
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		return s.connect_ex(('localhost', port)) == 0


if __name__ == "__main__":
	if(len(sys.argv) > 1):
		print("Servidor")
		s = servidor()
		while True:
			s.recibe_array()
	else:
		import time
		c = cliente()
		while True:
			c.envia_array(np.array([1, 2, 3]))
			time.sleep(1)
