import cv2 as cv


class UI():
	"Clase para el aspecto de la interfaz"
	rojo_claro = (66, 69, 255)
	rojo_oscuro = (56, 72, 153)
	rojo_oscuro2 = (25, 28, 67)
	cyan = (245, 235, 157)


class VideoProp():
	"Clase para la configuracion de video de salida"
	fourcc = cv.VideoWriter_fourcc(*"VP80")
	fps = 30
	resolu = (768, 576)


def fullscreen(win_name):
	"Configura el programa para fullscreen"
	cv.namedWindow(win_name, cv.WINDOW_NORMAL)
	cv.setWindowProperty(win_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
