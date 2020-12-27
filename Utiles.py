###################################
# Funciones de  proposito general #
###################################

import numpy as np
import cv2 as cv
import tensorflow as tf
import Config


def guardar(out, img):
	"Guarda un video"
	# Si no usa la GPU lo trasnforma en UMat para usarla
	# if(isinstance(img, (np.ndarray))):
	if(not isinstance(img, cv.UMat)):
		img = cv.UMat(img)
	# Extrae el size
	h, w, z = img.get().shape
	wc, hc = Config.VidProp.resolu
	# Si la resolucion del frame no es la requerida la cambia
	if wc != w or hc != h:
		img = cv.resize(img, (wc, hc))
	# Escribe en el archivo
	out.write(img)


def dimensiones_video(cap):
	"Devuelve las dimensioones x e y de un video"
	return [int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
			int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))]


def punto_centro(dim):
	"Devuelve el centro de unas dimensiones 2D"
	w, h = dim[0], dim[1]
	return np.array([w//2, h//2])


def centros_rectangulos(rectangulos):
	"Devuelve los centros de los objetivos"
	centros = []
	# El primer punto es el centro donde apunta la torreta
	# centros.append(np.array([int(dims[0]//2), int(dims[1]//2)]))
	for o in rectangulos:
		centro = centro_rectangulo(o)
		centros.append(centro)
	return np.array(centros)


def centro_rectangulo(r):
	"Devuelve el centro de un rectangulo definido con altura y ancho"
	x, y, w, h = r
	cX = (x+x+w)//2
	cY = (y+y+h)//2
	return np.array([cX, cY])


def rect_wh(rect):
	"Devuelve un rectangulo en formato x, y, w, h con otro x1, y1, x2, y2"
	return np.array([rect[0], rect[1], (rect[2]-rect[0]), (rect[3]-rect[1])])


def rect_4p(rect):
	"Devuelve un rectangulo en formato x1, y1, x2, y2 con otro x, y, w, h"
	return np.array([rect[0], rect[1], (rect[0]+rect[2]), (rect[0]+rect[2])])


def distancia(a, b):
	"Calcula la distancia eucidiana entre dos puntos"
	return np.linalg.norm(a-b)
	# dif = a - b
	# return math.sqrt(dif.x*dif.x + dif.y*dif.y)


def cerca(p_actual, punto, distancia_max=Config.Motor.d_disp):
	"Devuelve un booleano si esta lo suficientemente cerca"
	return True if(distancia(p_actual, punto) < distancia_max) else False


def rectangulo_contenedor(contorno, margen=0):
	"Devuelve el rectangulo que contiene el contorno"
	# Calculamos el rectangulo que contiene el elemento
	peri = cv.arcLength(contorno, True)
	# Dibuja un rectangulo con su centro
	approx = cv.approxPolyDP(contorno, 0.02*peri, True)
	x, y, w, h = cv.boundingRect(approx)
	if margen:
		x += margen
		y += margen
		w += margen
		h += margen
	return np.array([x, y, w, h])


def ROI_to_img(frame, ROI):
	"Devuelve una imagen que contiene solo la Region de Interes"
	x, y, w, h = ROI
	return frame[y:y+h, x:x+w]


def genera_rectangulos(contornos):
	"Devuelve todos los rectangulos que contienen a los contornos en una lista"
	rectangulos = []
	for c in contornos:
		rectangulo = rectangulo_contenedor(c)
		rectangulos.append(rectangulo)
	return np.array(rectangulos)


def gen_p_aleatorios(num_puntos, dims):
	"Genera n puntos aleatorios entre unas dimensiones"
	# Para cuadrado
	# return np.random.randint(0, min(dims[0], dims[1]), (num_puntos, 2))
	# Para rectangulo
	return np.random.randint(0, (dims[1], dims[0]), (num_puntos, 2))


def dibuja_puntos(imagen, lista_p, destruidos=False):
	"Dibuja los puntos en la imagen"
	for i, punto in enumerate(lista_p):
		if destruidos:
			cv.circle(imagen, (punto[0], punto[1]), 5, Config.UI.lima, 3)
			cv.putText(imagen, str(i),
				(punto[0]+5, punto[1]-5), 0, 1, Config.UI.lima, 2)
		else:
			cv.circle(imagen, (punto[0], punto[1]), 5, Config.UI.cyan2, 3)
			cv.putText(imagen, str(i),
				(punto[0]+5, punto[1]-5), 0, 1, Config.UI.rojo_claro, 1, 16)
	return imagen


def dibuja_mira(imagen, p_actual):
	"Dibuja una mira en la imagen"
	cv.drawMarker(imagen, (p_actual[0], p_actual[1]), Config.UI.rojo,
					cv.MARKER_CROSS, 30, 2)
	return imagen


def dibuja_path(img, lista_p):
	"Dibuja el path a seguir por la torreta entre distintos objetivos"
	anterior = np.array([])
	for punto in lista_p:
		if anterior.size == punto.size:
			cv.arrowedLine(img, (anterior[0], anterior[1]),
							(punto[0], punto[1]), Config.UI.rojo, 2,
							tipLength=0.05)
		anterior = punto


def dibuja_contornos(frame, contornos):
	"""Dibuja
	El contorno,
	El centro del contorno,
	Un rectangulo alrededor,
	El centro del rectangulo
	"""
	for c in contornos:
		# Dibuja el contorno
		cv.drawContours(frame, [c], 0, Config.UI.rojo_claro, 1)
		# Dibujas el centro del contorno
		M = cv.moments(c)
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])
		cv.circle(frame, (cX, cY), 1, (255, 255, 0), 0)
		rectangulo = rectangulo_contenedor(c)
		dibuja_rectangulo(frame, rectangulo)


def dibuja_rectangulo(frame, rectangulo, color=Config.UI.rojo_oscuro):
	"Dibuja un rectangulo contenedor"
	x, y, w, h = rectangulo
	cv.rectangle(frame, (x, y), (x+w, y+h), color, 1)
	cv.circle(frame, (x+w//2, y+h//2), 2, Config.UI.cyan, 0)


def dibuja_predic(frame, ROIs, predicciones):
	"Escribe la prediccion mas popular encima  del objetivo"
	if len(ROIs) != len(predicciones):
		print(f"Error, no iguales: {len(ROIs)} {len(predicciones)}")
		return
	#Recorre cada region
	for r, p in zip(ROIs, predicciones):
		# Extrae el texto de la prediccion
		print(p)
		text = p[0][1]
		x, y, w, h = r
		cv.putText(frame, text,
                    (x+5, y-5), 0, 1, Config.UI.morado, 1, 16)
		cv.rectangle(frame, (x, y), (x+w, y+h), Config.UI.rojo_oscuro, 1)


def dibuja_FPS(image, fps):
	"Dibuja los FPS para comparar"
	fps.update()
	fps.stop()
	# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
	text = "FPS: {:.2f}".format(fps.fps())
	# text = "Hola que haces"
	cv.putText(image, text, (0, 15),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, Config.UI.rojo, 2)


def elimina_rect_irelevantes(rectangulos, area_obj=Config.Tracker.area_min):
	"Elimina los rectangulos no similares al buscado"
	maximo_admitido = area_obj * (1+Config.Tracker.tolerancia)
	minimo_admitido = area_obj * (1-Config.Tracker.tolerancia)
	nuevos_rectangulos = []
	for r in rectangulos:
		area = r[2] * r[3]
		# Agrega solo los que tienen area similar al del objetivo actual
		if area > minimo_admitido and area < maximo_admitido:
			nuevos_rectangulos.append(r)
	return nuevos_rectangulos


def multi_atencion_blur(image, rectangulos):
	# Config
	img = image.copy()
	kernel = (47, 47)
	# Imagen blur total
	img_soslayo = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img_soslayo = cv.GaussianBlur(img_soslayo, kernel, 0)
	img_soslayo = cv.cvtColor(img_soslayo, cv.COLOR_GRAY2BGR)

	# Creacion de mascara
	mascara = np.full_like(img_soslayo, 255)
	# Rellena de los rectangulos
	for rect in rectangulos:
		x, y, w, h = rect
		x1, y1, x2, y2 = x, y, x+w, y+h
		cv.rectangle(mascara, (x1, y1), (x2, y2), [0], cv.FILLED)
	mascara = cv.GaussianBlur(mascara, kernel, 0)

	# cv.imshow('Imagen centro de atencion', mascara)
	# img[np.where(mascara == 255)] = img_soslayo[np.where(mascara == 255)]

	return alphaBlend(img, img_soslayo, mascara)

def atencion_blur(image, rect):
	# x1, y1, x2, y2 = rect
	x, y, w, h = rect
	x1, y1, x2, y2 = x, y, x+w, y+h
	kernel = (47, 47)
	img = image.copy()

	img_soslayo = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	img_soslayo = cv.GaussianBlur(img_soslayo, kernel, 0)
	img_soslayo = cv.cvtColor(img_soslayo, cv.COLOR_GRAY2BGR)

	mascara = np.full_like(img_soslayo, 255)
	cv.rectangle(mascara, (x1, y1), (x2, y2), [0], cv.FILLED)
	mascara = cv.GaussianBlur(mascara, kernel, 0)

	# cv.imshow('Imagen centro de atencion', mascara)
	# img[np.where(mascara == 255)] = img_soslayo[np.where(mascara == 255)]

	return alphaBlend(img, img_soslayo, mascara)


def alphaBlend(img1, img2, mask):
	# https://stackoverflow.com/a/48274875/3052862
	"alphaBlend img1 and img 2 (of CV_8UC3) with mask (CV_8UC1 or CV_8UC3)"
	if mask.ndim == 3 and mask.shape[-1] == 3:
		alpha = mask/255.0
	else:
		alpha = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)/255.0
	blended = cv.convertScaleAbs(img1*(1-alpha) + img2*alpha)
	return blended
