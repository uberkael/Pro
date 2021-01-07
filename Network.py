###################################################################
# Sistema de envio y recepcion simple para enviar arrays de numpy #
###################################################################

import socket
import sys
import numpy as np
import pickle

PUERTO = 5555


class cliente():
	"Clase que crea un socket de envio"
	def __init__(self):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		self.s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

	def envia_array(self, datos):
		# self.s.sendto(datos.encode(), ('255.255.255.255', PUERTO))
		self.s.sendto(pickle.dumps(datos), ('255.255.255.255', PUERTO))


class servidor():
	"Clase que crea un socket para recivir datos"
	def __init__(self, servidor=True):
		self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
		if servidor:
			self.s.bind(('', PUERTO))

	def recibe_array(self):
		"Recibe un array y lo desempaqueta, "
		data, address = self.s.recvfrom(4096)
		print(pickle.loads(data))
		# print(data.decode())
		# m = self.s.recv(4096)
		# print(pickle.loads(m))
		# print(m)


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
