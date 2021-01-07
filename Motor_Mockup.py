#######################################################################
# Selector de orden de objetivos empezando por el centro de la imagen #
#######################################################################

import cv2 as cv
import numpy as np
import Utiles
import Config
import Selector


def desplazamiento(p_actual, punto):
	"""Mueve la torreta hacia el punto, de forma limitada segun Config
	En la implementacion real debe actuar sobre dos motores"""
	dist = np.array([punto])
	dist = punto - p_actual
	mov = Config.Motor.mov
	dist[dist > mov] = mov
	dist[dist < (-mov)] = -mov
	p_actual = p_actual + dist
	return p_actual


def disparo(orden_objetivos, objetivos_destruidos):
	"""Dispara cuando esta cerca del objetivo, debería activar un interruptor
	En la implementacion real debe activar el disparo (relay o puso)"""
	objetivos_destruidos.append(orden_objetivos[0])
	# print(orden_objetivos.shape)
	orden_objetivos = orden_objetivos[1:]
	return orden_objetivos


def imagen_punteada(img, orden_objetivos, objetivos_destruidos):
	"Mockup, dibuja los puntos objetivo y los eliminados"
	if (len(orden_objetivos) > 0):
		img = Utiles.dibuja_puntos(img, orden_objetivos)
	if (len(objetivos_destruidos) > 0):
		img = Utiles.dibuja_puntos(img, objetivos_destruidos, destruidos=True)
	return img


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Motor Mockup"
	num_puntos = 50
	w, h = Config.VidProp.resolu
	img = np.zeros([h, w, 3], np.uint8)
	dims = [img.shape[1], img.shape[0]]

	# Para capturar la salida
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida/{titulo}.avi", VidProp.fourcc,
							VidProp.fps, VidProp.resolu)

	lista_p = Utiles.gen_p_aleatorios(num_puntos, (h, w))
	p_actual = Utiles.punto_centro(dims)
	orden_objetivos = Selector.organiza_objetivos(p_actual, lista_p)
	objetivos_destruidos = []

	imagen = imagen_punteada(img, orden_objetivos, objetivos_destruidos)
	Utiles.dibuja_mira(imagen, p_actual)
	cv.imshow(titulo, imagen)

	# Crea un contador fps
	while True:
		if(len(orden_objetivos) <= 0):
			break
		image = img.copy()
		cerca = Utiles.cerca(p_actual, orden_objetivos[0])
		if cerca:
			orden_objetivos = disparo(orden_objetivos, objetivos_destruidos)
			image = cv.putText(image, "Bang!",
					(p_actual[0]+5, p_actual[1]-5), 0, 1, Config.UI.rojo, 2, 16)
		image = imagen_punteada(image, orden_objetivos, objetivos_destruidos)
		p_actual = desplazamiento(p_actual, orden_objetivos[0])
		image = Utiles.dibuja_mira(image, p_actual)
		Utiles.dibuja_path(image, orden_objetivos)
		if Config.VidProp.guardar:
			Utiles.guardar(out, image)
		cv.imshow(titulo, image)
		if (cv.waitKey(1) & 0xFF == 27):
			break
	# cv.imshow(titulo, img)
	cv.waitKey(0)
