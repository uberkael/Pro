##################################################
# Tracker de movimiento por sustraccion de fondo #
##################################################

import cv2 as cv
import numpy as np
from imutils.video import FPS
import Utiles
import Config
import Selector

# MOG
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
# MOG2
# fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)
# KNN
# fgbg = cv.createBackgroundSubtractorKNN(detectShadows=False)


def eliminador_fondo(image):
	"Elimina el fondo para hacer tracking utilizando MOG y adaptative Threshold"
	# OpenCV GPU
	img = cv.UMat(image)
	img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	# Desenfocamos
	img = cv.blur(img, (3, 3))
	# _, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY_INV)
	# cv.imshow("fondo ant", img) # DEBUG
	img = cv.adaptiveThreshold(
		img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 71, 50)
	# Aplica el sustractor de fondos
	fgmask = fgbg.apply(img)
	# Aplica la mascara para ver solo el cambio
	img = cv.bitwise_and(img, img, mask=fgmask)
	# cv.imshow("fondo", img) # DEBUG
	img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
	# Vuelve a una matriz normal no GPU
	img = cv.UMat.get(img)
	return img


def extrae_contornos(image):
	"Devuelve una lista de contornos exteriores de los objetivos de la imagen"
	# Pasamos a gris
	img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
	# Desenfocamos
	img = cv.blur(img, (3, 3))
	# Aplicamos un apertura para eliminar pequeños movimientos
	kernel = np.ones((3, 3), np.uint8)
	img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel,  iterations=1)
	# img = cv.dilate(img, kernel, iterations=2)
	# Detectamos los bordes
	# img = cv.Canny(img, 150, 200)
	# Dilatamos para cerrar contornos
	# img = cv.dilate(img, None, iterations=1)
	# Detectamos contornos
	contornos, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
	# nuevos_contornos = []
	# for c in contornos:
	# 	nuevos_contornos.append(cv.convexHull(c))
	# img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
	# cv.imshow("new", img)
	return contornos


def elimina_contornos_irrelevantes(contornos, area_min=Config.Tracker.area_min):
	"""Elimina los contornos con un area inferior a un umbral"""
	nuevos_contornos = []
	for c in contornos:
		# Elminamos las areas pequeñas
		area = cv.contourArea(c)
		if area > area_min:
			# Los agrega a la lista
			nuevos_contornos.append(c)
	return nuevos_contornos


def actualiza_objetivo(objetivo, objetivos):
	"Actualiza el objetivo si se mueve"
	# No hay nuevos objetivos, nos quedamos con el que se detecto
	if(len(objetivos) < 1):
		return objetivo
	# Elimina las detecciones del tracker con area muy distinta del objetivo
	x, y, w, h = objetivo
	area_objetivo = w * h
	rectangulos = Utiles.genera_rectangulos(objetivos)
	rectangulos = Utiles.elimina_rect_irelevantes(rectangulos, area_objetivo)
	# Los objetivos no eran suficientemente similares en area
	if(len(rectangulos) < 1):
		return objetivo
	# Cambia el objetivo por el mas cercano detectado por el tracker
	distancia_max = max(w, h)
	p_objetivo = Utiles.centro_rectangulo(objetivo)
	objetivo_alt = Selector.objetivo_prioritario(p_objetivo, rectangulos,
		distancia_max)
	if(len(objetivo_alt) > 0):
		return objetivo_alt
	# No habia un objetivo lo suficientemete cercano
	return objetivo


def tracker(image):
	"""
	Funcion principal, extrae objetivos de los objetos en movimiento
	Devuelve el frame pintado y una lista de contornos de objetivos
	"""
	img = eliminador_fondo(image)
	contornos = extrae_contornos(img)
	# identifica_objetivos(img, contornos)
	# OpenCV GPU
	img = cv.UMat(image)
	objetivos = elimina_contornos_irrelevantes(contornos)
	# image = cv.UMat(image)
	# Hacemos los colores oscuros claros
	# return cv.hconcat([image, img]) # # DEBUG
	return img, objetivos


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Tracker"
	# Config.Fullscreen(titulo)
	out = None
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida/{titulo}.avi", VidProp.fourcc,
							VidProp.fps, VidProp.resolu)
	cap = cv.VideoCapture(Config.VidProp.source)
	fps = FPS().start()

	while cap.isOpened():
		ret, image = cap.read()
		if not ret: break
		image, objetivos = tracker(image)
		Utiles.dibuja_contornos(image, objetivos)
		if Config.VidProp.show_fps: Utiles.dibuja_FPS(image, fps)
		cv.imshow(titulo, image)
		if Config.VidProp.guardar:
			Utiles.guardar(out, image)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == 27):
			break
	if Config.VidProp.guardar: out.release()
	cap.release()

