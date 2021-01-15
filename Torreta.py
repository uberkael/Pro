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


def search_destroy(image):
	"Usa Tracking para seguir al objetivo mientras mueve la torreta y dispara"
	global p_actual
	global objetivo
	global objetivos_destruidos
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
	objetivo = Tracker.actualiza_objetivo(objetivo, objetivos)
	# image = Utiles.atencion_blur(image, objetivo)
	Utiles.dibuja_contornos(image, objetivos)
	Utiles.dibuja_rectangulo(image, objetivo, Config.UI.cyan2)
	image = Utiles.dibuja_mira(image, p_actual)
	# Comprueba si se puede disparar
	p_objetivo = Utiles.centro_rectangulo(objetivo)
	if(destruccion(p_actual, p_objetivo)):
		image = cv.putText(image, "Bang!",
					(p_actual[0]+5, p_actual[1]-5), 0, 1, Config.UI.rojo, 2, 16)
		# Limpia la lista de destruidos y agrega el ultimo TODO mejorar
		objetivos_destruidos.append(objetivo)
		if(len(objetivos_destruidos)>3):
			objetivos_destruidos.pop(0)
		modo = Config.Modo.deteccion
	return image


# TODO mejorar
def destruccion(p_actual, p_objetivo):
	"Dispara si el objetivo esta en el rango de disparo y devuelve True"
	cerca = Utiles.cerca(p_actual, p_objetivo)
	if cerca:
		# Motor_Mockup.disparo(orden_puntos, objetivos_destruidos)
		return True
	return False


# TODO
# def aumenta_roi(objetivo, parte=4):
# 	"Aumenta la zona vigilada alrededor de la roi"
# 	global dims
# 	aumento = dims / parte
# 	# roi = image[y1:y2, x1:x2]
# 	x1, y1, x2, y2 = Utiles.rect_4p(objetivo)
# 	return np.array([x1, y1, x2, y2])


# TODO
# def reduce_imagen(image, ):
	# "Reduce la zona para hacer tracking"
	# reduce_imagen = image[y1:y2, x1:x2]


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

	# Modo de operacion
	modo = Config.Modo.deteccion
	# Variables importantes para el modo torreta
	objetivos_destruidos = []
	objetivos = []
	objetivo = []

	# Setup DNN
	# Crea la red neural
	modelo = Detector.genera_DNN()
	# Extrae layers principales de YoLo
	capas_conexion = Detector.capas_desconectadas(modelo)
	labels = Detector.genera_labels()
	colores = Detector.genera_colores(labels)

	# Posicion de la torreta
	p_actual = Utiles.punto_centro(dims)

	# Crea un contador fps
	fps = FPS().start()

	# count = 0
	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			break
		# Elimina de la lista de eliminados cada n fotogramas
		# if(count % Config.Tracker.persistencia == 0 and
			# len(objetivos_destruidos) > 0):
			# print("Hola")
		if modo == Config.Modo.deteccion:
			if(len(objetivos_destruidos) > 0):
				Utiles.mascara_destruidos(image, objetivos_destruidos)
			ROIs = deteccion(image)
			if(len(ROIs) > 1):
				# Asigna solo los rectangulos contendores como objetivos
				objetivos = ROIs[0]
				# image = Utiles.atencion_blur(image, objetivos[0])
				# image = Utiles.multi_atencion_blur(image, objetivos)
				objetivo = Selector.objetivo_prioritario(p_actual, objetivos)
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
		# if  count > 100: break
		# count += 1
		# TODO solo 10 frames
		if (cv.waitKey(1) & 0xFF == 27): break

	# cv.waitKey(0)
	if Config.VidProp.guardar: out.release()
