#################################################
# Implementacion de una unica torreta solitaria #
#################################################

import cv2 as cv
import numpy as np
import Tracker
import Selector
import Utiles
import Config

dims = (0, 0)


def objetivos_centros(objetivos):
	"Devuelve los centros de los objetivos"
	centros = []
	for o in objetivos:
		M = cv.moments(o)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		centros.append(np.array([cX, cY]))
	return np.array(centros)


def search_destroy(frame):
	"Funcion principal"
	img, objetivos = Tracker.tracker(frame)
	p_objetivos = objetivos_centros(objetivos)
	orden_objetivos = Selector.organiza_objetivos(dims, p_objetivos)
	Selector.dibuja_path(img, orden_objetivos)
	Selector.dibuja_puntos(img, orden_objetivos)
	Utiles.dibuja_contornos(img, objetivos)
	return img


if __name__ == "__main__":
	titulo = "Torreta"
	# Config.Fullscreen(titulo)
	out = None
	# Para capturar la salida
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida {titulo}.avi", VidProp.fourcc,
                    		VidProp.fps, VidProp.resolu)
	# Abre el video y almacena las dimesiones
	cap = cv.VideoCapture("Samples/vtest.avi")
	dims = Utiles.dimensiones_video(cap)

	while cap.isOpened():
		ret, frame = cap.read()
		if not ret: break
		img = search_destroy(frame)
		cv.imshow(titulo, img)
		if Config.VidProp.guardar:
			Utiles.guardar(out, img)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == ord('q')):
			break
	# cv.waitKey(0)
	if Config.VidProp.guardar: out.release()
	out.release()
