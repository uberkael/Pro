#######################################################################
# Selector de orden de objetivos empezando por el centro de la imagen #
#######################################################################

import cv2 as cv
import numpy as np
import Utiles
import Config
import Selector


def centra_objectivo(p_actual, punto):
	"Mueve la torreta hacia el punto"
	dist = np.array([punto])
	dist = punto - p_actual
	dist[dist > 4] = 4
	dist[dist < (-4)] = -4
	p_actual = p_actual + dist
	return p_actual


def disparo():
	global orden_objetivos
	global objetivos_destruidos
	"Dispara cuando esta cerca del objetivo"
	objetivos_destruidos.append(orden_objetivos[0])
	orden_objetivos = orden_objetivos[1:]


def imagen_punteada(img, orden_objetivos, objetivos_destruidos):
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
	dims_img = [h, w, 3]
	img = np.zeros(dims_img, np.uint8)
	dims = [img.shape[1], img.shape[0]]

	# Para capturar la salida
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida/{titulo}.avi", VidProp.fourcc,
				VidProp.fps, VidProp.resolu)

	lista_p = Utiles.gen_p_aleatorios(num_puntos, (dims_img[0], dims_img[1]))
	orden_objetivos = Selector.organiza_objetivos(dims, lista_p)
	objetivos_destruidos = []
	p_actual = Utiles.punto_centro(dims)

	imagen = imagen_punteada(img, orden_objetivos, objetivos_destruidos)
	Utiles.dibuja_mira(imagen, p_actual)
	cv.imshow(titulo, imagen)

	# Crea un contador fps
	while True:
		if(len(orden_objetivos) <= 0):
			break
		image = img.copy()
		cerca = Utiles.cerca(p_actual, orden_objetivos[0])
		if(cerca):
			disparo()
			image = cv.putText(image, "Bang!",
					(p_actual[0]+5, p_actual[1]-5), 0, 1, Config.UI.rojo, 2, 16)
		image = imagen_punteada(image, orden_objetivos, objetivos_destruidos)
		p_actual = centra_objectivo(p_actual, orden_objetivos[0])
		image = Utiles.dibuja_mira(image, p_actual)
		Utiles.dibuja_path(image, orden_objetivos)
		if Config.VidProp.guardar:
			Utiles.guardar(out, image)
		cv.imshow(titulo, image)
		if (cv.waitKey(1) & 0xFF == 27):
			break
	# cv.imshow(titulo, img)
	cv.waitKey(0)
