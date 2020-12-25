#################################################
# Implementacion de una unica torreta solitaria #
#################################################

import cv2 as cv
import numpy as np
from imutils.video import FPS
import Tracker
import Selector
import Utiles
import Config
import Detector


def centros_de_contornos(objetivos):
	"Devuelve los centros de los objetivos"
	centros = []
	# El primer punto es el centro donde apunta la torreta
	centros.append(np.array([int(dims[0]//2), int(dims[1]//2)]))
	for o in objetivos:
		M = cv.moments(o)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		centros.append(np.array([cX, cY]))
	return np.array(centros)


def centros_de_cuadrados(objetivos):
	"Devuelve los centros de los objetivos"
	centros = []
	# El primer punto es el centro donde apunta la torreta
	centros.append(np.array([int(dims[0]//2), int(dims[1]//2)]))
	for o in objetivos:
		x, y, w, h = o
		cX = (x+x+w)//2
		cY = (y+y+h)//2
		centros.append(np.array([cX, cY]))
	return np.array(centros)


def deteccion(image):
	ROIs = Detector.prediccion(image, modelo, capas_conexion, labels)
	# roi = [ROIs[0][0], [ROIs[1][0]], [ROIs[2][0]]]
	# return Utiles.atencion_blur(image, roi[0])
	# return Detector.dibuja_ROIs(image, ROIs)
	return ROIs


def search_destroy(image):
	"Funcion principal"
	print(objetivos)
	p_objetivos = centros_de_cuadrados(objetivos)
	orden_objetivos = Selector.organiza_objetivos(dims, p_objetivos)
	# img = Utiles.atencion_blur(image, objetivos[0])
	img = Utiles.multi_atencion_blur(image, objetivos)
	# Objetivos son contornos
	# img, objetivos = Tracker.tracker(image)
	# p_objetivos = centros_de_contornos(objetivos)
	# orden_objetivos = Selector.organiza_objetivos(dims, p_objetivos)
	Selector.dibuja_path(img, orden_objetivos)
	# Selector.dibuja_puntos(img, orden_objetivos)
	# Utiles.dibuja_contornos(img, objetivos)
	return img


def search_destroy_antiguo(image):
	"Funcion principal"
	# Objetivos son contornos
	img, objetivos = Tracker.tracker(image)
	p_objetivos = centros_de_contornos(objetivos)
	orden_objetivos = Selector.organiza_objetivos(dims, p_objetivos)
	Selector.dibuja_path(img, orden_objetivos)
	Selector.dibuja_puntos(img, orden_objetivos)
	Utiles.dibuja_contornos(img, objetivos)
	return img


if __name__ == "__main__":
	titulo = "Torreta"
	# Configuracion de video
	Config.Fullscreen(titulo)
	# Declara variables para llenar mas adelante
	out = None
	dims = (0, 0)
	# Para capturar la salida
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida/{titulo}.avi", VidProp.fourcc,
							VidProp.fps, VidProp.resolu)
	# Abre el video y almacena las dimesiones
	cap = cv.VideoCapture(Config.VidProp.source)
	dims = Utiles.dimensiones_video(cap)

	# Setup DNN
	# Crea la red neural
	modelo = Detector.genera_DNN()
	# Extrae layers principales de YoLo
	capas_conexion = Detector.capas_desconectadas(modelo)
	labels = Detector.genera_labels()
	colores = Detector.genera_colores(labels)

	# Modo de operacion
	modo = Config.Modo.deteccion
	objetivos_eliminados = []
	objetivos = []

	# Crea un contador fps
	fps = FPS().start()

	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			break

		if modo == Config.Modo.deteccion:
			ROIs = deteccion(image)
			if(len(ROIs) > 1):
				# Asigna los rectangulos contendores como objetivos
				objetivos = ROIs[0]
				# image = Utiles.atencion_blur(image, objetivos[0])
				image = Utiles.multi_atencion_blur(image, objetivos)
				modo = Config.Modo.search_destroy
		else:
			image = search_destroy(image)
			modo = Config.Modo.deteccion


		if Config.VidProp.show_fps: Utiles.dibuja_FPS(image, fps)
		if Config.VidProp.guardar: Utiles.guardar(out, image)
		cv.imshow(titulo, image)
		if (cv.waitKey(1) & 0xFF == 27): break

	# cv.waitKey(0)
	if Config.VidProp.guardar: out.release()
