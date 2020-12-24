#########################################
# Detector y discriminador de objetivos #
#########################################
###############################
# Single Shot Detectors (SSD) #
###############################
# https: // gilberttanner.com/blog/yolo-object-detection-with-opencv
# https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/

import numpy as np
import cv2 as cv
from imutils.video import FPS
import Utiles
import Config
import Tracker


def genera_predicciones(modelo, labels, umbral):
	global image
	(h, w) = image.shape[:2]
	tam = Config.DNN.MobileNet.size
	# blob = cv.dnn.blobFromImage(
	# 	cv.resize(image, Config.DNN.MobileNet.size), 0.007843,
	# 	Config.DNN.MobileNet.size, (127.5, 127.5, 127.5), False)
	# Genera el blob para alimentar a la DNN
	blob = cv.dnn.blobFromImage(cv.resize(image, tam), 0.007843, tam, 127.5)
	modelo.setInput(blob)
	predicciones = modelo.forward()

	for i in np.arange(0, predicciones.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = predicciones[0, 0, i, 2]
		# filter out weak predicciones by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > umbral:
			# extract the index of the class label from the `predicciones`,
			# then compute the (x, y)-coordinates of the bounding box for
			# the object
			indice = int(predicciones[0, 0, i, 1])
			ROI = predicciones[0, 0, i, 3:7] * np.array([w, h, w, h])
			dibuja_ROI(ROI, indice, confidence)


def dibuja_ROI(ROI, indice, confianza):
	"Auxiliar, para DEBUG, Dibuja una ROI en la imagen"
	global image
	(startX, startY, endX, endY) = ROI.astype("int")
	# display the prediction
	# label = "{}: {:.2f}%".format(labels[indice], confidence * 100)
	text = "{}: {:.4f}".format(labels[indice], confianza)
	# print("[INFO] {}".format(label))
	# cv.rectangle(image, (startX, startY), (endX, endY),
	# 				colores[indice], 2)
	# color = [int(c) for c in colores[indice]]
	color = colores[indice]
	cv.rectangle(image, (startX, startY), (endX, endY), color, 2)
	y = startY - 15 if startY - 15 > 15 else startY + 15
	cv.putText(image, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def genera_DNN():
	"Genera una Red Neuronal Profunda Preentrenada con los parametros de Config"
	modelo = cv.dnn.readNetFromCaffe(
		Config.DNN.MobileNet.archivo_prototxt, Config.DNN.MobileNet.archivo_modelo)
	return modelo


def genera_labels():
	# return open(Config.DNN.archivo_labels).read().strip().split('\n')
	return ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]


def genera_colores(labels):
	"Genera colores aleatorios segun los labels"
	# colores = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
	# colores[15] = (84, 7, 220)
	colores = np.random.uniform(0, 255, size=(len(labels), 3))
	colores[15] = [84, 7, 220]
	return colores


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Detector SSD MobileNet"
	dims = (0, 0)
	# Config.Fullscreen(titulo)
	out = None
	# Para capturar la salida
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida/{titulo}.avi", VidProp.fourcc,
							VidProp.fps, VidProp.resolu)
		# print(out.get(2)) # TODO
	# Abre el video y almacena las dimesiones
	cap = cv.VideoCapture(Config.VidProp.source)
	dims = Utiles.dimensiones_video(cap)

	# Crea la red neural
	modelo = genera_DNN()

	# Extrae las capas de YoLo
	# layer_names = modelo.getLayerNames()
	# layer_names = [layer_names[i[0] - 1]
	# 			for i in modelo.getUnconnectedOutLayers()]
	labels = genera_labels()
	colores = genera_colores(labels)
	fps = FPS().start()

	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			print('Video file finished.')
			break

		umbral = 0.1

		genera_predicciones(modelo, labels, umbral)

		if Config.VidProp.show_fps: Utiles.dibuja_FPS(image, fps)
		cv.imshow(titulo, image)

		if (cv.waitKey(1) & 0xFF == 27):
			break

		if Config.VidProp.guardar:
			Utiles.guardar(out, image)

	cap.release()
	if Config.VidProp.guardar: out.release()
