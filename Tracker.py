##################################################
# Tracker de movimiento por sustraccion de fondo #
##################################################

import cv2 as cv
import numpy as np
import Utiles
import Config

# MOG
fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
# MOG2
# fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)
# KNN
# fgbg = cv.createBackgroundSubtractorKNN(detectShadows=False)


def eliminador_fondo(frame):
	"Elimina el fondo para hacer tracking"
	# OpenCV GPU
	img = cv.UMat(frame)
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
	return img


def extrae_contornos(frame):
	"Devuelve una lista de contornos exteriores de los objetivos"
	# Pasamos a gris
	img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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


def identifica_objetivos(contornos):
	"""
	Sera sustituido por una red neuronal
	Identifica objetivos,
	Segun su area y segun su ratio (personas depie)
	"""
	nuevos_contornos = []
	for c in contornos:
		# Elminamos las areas pequeñas
		area = cv.contourArea(c)
		if area > 100:
			# Los agrega a la lista
			nuevos_contornos.append(c)
	return nuevos_contornos


def tracker(frame):
	"""
	Funcion principal, extrae objetivos de los objetos en movimiento
	Devuelve el frame pintado y una lista de contornos de objetivos
	"""
	img = eliminador_fondo(frame)
	contornos = extrae_contornos(img)
	# identifica_objetivos(img, contornos)
	# OpenCV GPU
	img = cv.UMat(frame)
	objetivos = identifica_objetivos(contornos)
	# frame = cv.UMat(frame)
	# Hacemos los colores oscuros claros
	# return cv.hconcat([frame, img]) # # DEBUG
	return img, objetivos


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Tracker"
	# Config.Fullscreen(titulo)
	out = None
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida {titulo}.avi", VidProp.fourcc,
            	        	VidProp.fps, VidProp.resolu)
	cap = cv.VideoCapture("Samples/vtest.avi")

	while cap.isOpened():
		ret, frame = cap.read()
		img, objetivos = tracker(frame)
		if not ret: break
		Utiles.dibuja_contornos(img, objetivos)
		cv.imshow(titulo, img)
		if Config.VidProp.guardar:
			Utiles.guardar(out, img)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == ord('q')):
			break
	cv.waitKey(0)
	if Config.VidProp.guardar: out.release()
	cap.release()

