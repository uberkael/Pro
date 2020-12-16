###########################################################
# Clases libreria y funciones auxiliares de configuracion #
###########################################################

import cv2 as cv


class UI():
	"Clase para el aspecto de la interfaz colores basados en CyberPunk 2077"
	verde = (0, 255, 0)
	lima = (20, 230, 150)
	rojo = (84, 7, 220)
	rojo_claro = (66, 69, 255)
	rojo_oscuro = (56, 72, 153)
	rojo_oscuro2 = (25, 28, 67)
	morado = (200, 0, 128)
	purpura = (41, 26, 30)
	cyan = (245, 235, 157)
	cyan2 = (255, 254, 164)


class VidProp():
	"Clase para la configuracion de video de salida"
	source = "vtest.avi"
	# fourcc = cv.VideoWriter_fourcc(*"VP80")
	# fourcc = cv.VideoWriter_fourcc(*"VP90")
	# fourcc = cv.VideoWriter_fourcc(*"LAGS")
	fourcc = cv.VideoWriter_fourcc(*"FFV1")
	fps = 20
	# resolu = (768, 576)
	resolu = (1280, 720)
	mobilenet_10 = (2240, 224)
	guardar = False


class DNN():
	"Parametros para las redes neuronales"
	img_size = 224
	img_margen = 25


def Fullscreen(win_name):
	"Configura el programa para fullscreen"
	cv.namedWindow(win_name, cv.WINDOW_NORMAL)
	cv.setWindowProperty(win_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

