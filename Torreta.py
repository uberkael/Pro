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
import Motor_Mockup


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


def deteccion(image):
	ROIs = Detector.prediccion(image, modelo, capas_conexion, labels)
	# roi = [ROIs[0][0], [ROIs[1][0]], [ROIs[2][0]]]
	# return Utiles.atencion_blur(image, roi[0])
	# return Detector.dibuja_ROIs(image, ROIs)
	return ROIs


def objetivo_prioritario(punto, objetivos, distancia_max=0):
	"""Devuelve el objetivo prioritario ysegun la distancia a la posicion actual
	de la torreta"""
	puntos_centro = Utiles.centros_rectangulos(objetivos)
	objetivo_anterior = punto
	distancias = [Utiles.distancia(x, objetivo_anterior) for x in puntos_centro]
	menor = np.argmin(distancias)
	# Si el punto mas cercano esta lo suficientemente cerca no se devuelve nada
	if distancia_max and np.min(distancias) > distancia_max:
		return np.array([])
	return objetivos[menor]
	# return objetivos[menor], puntos_centro[menor]


def search_destroy(image):
	"Usa Tracking para seguir al objetivo mientras mueve la torreta y dispara"
	global p_actual
	global objetivo
	global modo

	centro_objetivo = Utiles.centro_rectangulo(objetivo)
	p_actual = Motor_Mockup.desplazamiento(p_actual, centro_objetivo)
	# Utiles.dibuja_puntos(image, [centro])
	# x1, y1, x2, y2 = aume nta_roi(objetivo)

	# cv.imshow("ROI", roi)
	# coordenadas_reducidas(image)
	# cv.waitKey(0)
	# while True:
	image, objetivos = Tracker.tracker(image)
	objetivo = actualiza_objetivo(objetivo, objetivos)
	# image = Utiles.atencion_blur(image, objetivo)
	Utiles.dibuja_contornos(image, objetivos)
	Utiles.dibuja_rectangulo(image, objetivo, Config.UI.cyan2)
	image = Utiles.dibuja_mira(image, p_actual)
	# Comprueba si se puede disparar
	p_objetivo = Utiles.centro_rectangulo(objetivo)
	if(destruccion(p_actual, p_objetivo)):
		image = cv.putText(image, "Bang!",
					(p_actual[0]+5, p_actual[1]-5), 0, 1, Config.UI.rojo, 2, 16)
		modo = Config.Modo.deteccion
	return image



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
	objetivo_alternativo = objetivo_prioritario(
		p_objetivo, rectangulos, distancia_max)
	if(len(objetivo_alternativo) > 0):
		return objetivo_alternativo
	# No habia un objetivo lo suficientemete cercano
	return objetivo


def destruccion(p_actual, p_objetivo):
	"Dispara si el objetivo esta en el rango de disparo y devuelve True"
	cerca = Utiles.cerca(p_actual, p_objetivo)
	if cerca:
		# Motor_Mockup.disparo(orden_puntos, objetivos_destruidos)
		return True
	return False



# def aumenta_roi(objetivo, parte=4):
# 	"Aumenta la zona vigilada alrededor de la roi"
# 	global dims
# 	aumento = dims / parte
# 	# roi = image[y1:y2, x1:x2]
# 	x1, y1, x2, y2 = Utiles.rect_4p(objetivo)
# 	return np.array([x1, y1, x2, y2])



# def reduce_imagen(image, ):
	# "Reduce la zona para hacer tracking"
	# reduce_imagen = image[y1:y2, x1:x2]



#	# Selecciona el objetivo mas cercano
#	img = Utiles.multi_atencion_blur(image, objetivos)
#	# print(objetivos)
#	orden_objetivos, orden_puntos = Selector.organiza_objetivos(dims, objetivos)
#	objetivo = orden_objetivos[0]
#	objetivo_centro = orden_puntos[0]
#	cerca = Utiles.cerca(p_actual, objetivo_centro)
#	if cerca:
#		Motor_Mockup.disparo(orden_puntos, objetivos_destruidos)
#		img = cv.putText(img, "Bang!",
#			(p_actual[0]+5, p_actual[1]-5), 0, 1, Config.UI.rojo, 2, 16)
#	# Mueve la torreta
#	p_actual = Motor_Mockup.centra_objectivo(p_actual, orden_puntos[0])
#	# img = Utiles.atencion_blur(image, objetivos[0])
#	# Objetivos son contornos
#	# img, objetivos = Tracker.tracker(image)
#	# orden_puntos = centros_de_contornos(objetivos)
#	# orden_puntos = Selector.organiza_objetivos(dims, orden_puntos)
#
#	# Utiles.dibuja_path(img, orden_puntos)
#	Utiles.dibuja_puntos(img, orden_puntos)
#	img = Utiles.dibuja_mira(img, p_actual)
#	# Utiles.dibuja_contornos(img, objetivos)
#	return img


if __name__ == "__main__":
	titulo = "Torreta"
	# Configuracion de video
	# Config.Fullscreen(titulo)
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
	objetivos_destruidos = []
	objetivos = []
	objetivo = []

	# Posicion de la torreta
	p_actual = Utiles.punto_centro(dims)

	# Crea un contador fps
	fps = FPS().start()

	count = 0
	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			break
		if modo == Config.Modo.deteccion:
			ROIs = deteccion(image)
			if(len(ROIs) > 1):
				# Asigna solo los rectangulos contendores como objetivos
				objetivos = ROIs[0]
				# image = Utiles.atencion_blur(image, objetivos[0])
				# image = Utiles.multi_atencion_blur(image, objetivos)
				objetivo = objetivo_prioritario(p_actual, objetivos)
				if(objetivo):
					modo = Config.Modo.search_destroy
					Utiles.dibuja_rectangulo(image, objetivo, Config.UI.cyan2)
					image = Utiles.atencion_blur(image, objetivo)
		else:
			image = search_destroy(image)
			cond = False
			if(cond):
				modo = Config.Modo.deteccion

		if Config.VidProp.show_fps: Utiles.dibuja_FPS(image, fps)
		if Config.VidProp.guardar: Utiles.guardar(out, image)
		cv.imshow(titulo, image)
		# TODO solo 10 frames pausa
		# cv.waitKey(0)
		if  count > 100: break
		count += 1
		# TODO solo 10 frames
		if (cv.waitKey(1) & 0xFF == 27): break

	# cv.waitKey(0)
	if Config.VidProp.guardar: out.release()
