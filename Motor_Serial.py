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


posicion = [0 0]

def desplazamiento(p_actual, punto):
	"Ordena a arduino por serial que mueva los motores hacia el punto"
	global arduino
	x = punto[0] - p_actual[0]
	y = punto[1] - p_actual[1]
	x_step = x * 10
	y_step = y * 10
	pos = [pos[0]+x_step, pos[1]+y_step]
	arduino.write(f"{x_step} {y_step}\n".encode())


def disparo(orden_objetivos, objetivos_destruidos):
	"""Dispara cuando esta cerca del objetivo, debería activar un interruptor
	En la implementacion real debe activar el disparo (relay o puso)"""
	objetivos_destruidos.append(orden_objetivos[0])
	# print(orden_objetivos.shape)
	orden_objetivos = orden_objetivos[1:]
	return orden_objetivos


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Motor Serial"

	plist = list_ports.comports()
	print(plist[-1][0])

	arduino = serial.Serial(
			port=plist[-1][0],  # El ultimo de la lista
			baudrate=115200,
			parity=serial.PARITY_ODD,
			stopbits=serial.STOPBITS_ONE,
			bytesize=serial.EIGHTBITS
		)

	punto = [0, 0]
	p_actual = [0, 0]

	desplazamiento(p_actual, punto)

	data = arduino.readline()
	print(data)


