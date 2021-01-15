#######################################################################
# Selector de orden de objetivos empezando por el centro de la imagen #
#######################################################################

import cv2 as cv
import numpy as np
import Utiles
import Config
import Selector
import serial
from serial.tools import list_ports
import time


pos = [0, 0]  # Posicion


def desplazamiento(arduino, punto):
	"""Ordena a arduino por serial que mueva los motores hacia el punto"""
	global pos
	x = punto[0] - pos[0]
	y = punto[1] - pos[1]
	x_step = x * Config.Motor.pixeles_a_step
	y_step = y * Config.Motor.pixeles_a_step
	pos = [pos[0]+x_step, pos[1]+y_step]
	arduino.write(f"{x_step} {y_step}\n".encode())
	return punto


def disparo(orden_objetivos, objetivos_destruidos):
	"""Dispara cuando esta cerca del objetivo, debería activar un interruptor
	En la implementacion real debe activar el disparo (relay o pulso)"""
	objetivos_destruidos.append(orden_objetivos[0])
	# print(orden_objetivos.shape)
	orden_objetivos = orden_objetivos[1:]
	return orden_objetivos


def conexion_serial():
	"Genera una conexion serial para la comunicacion"
	plist = list_ports.comports()
	print(plist[-1][0])
	return serial.Serial(
			port=plist[-1][0],  # El ultimo de la lista
			baudrate=Config.Serial.baudrate,
			parity=Config.Serial.parity,
			stopbits=Config.Serial.stopbits,
			bytesize=Config.Serial.bytesize)


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Motor Serial"

	arduino = conexion_serial()

	punto = [0, 0]
	# punto = [100, 100]

	desplazamiento(arduino, punto)

	data = arduino.readline()
	print(data)
