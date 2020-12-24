#########################################
# Detector y discriminador de objetivos #
#########################################
##############################
# You Only Look Once (YOLO)  #
##############################
# https: // gilberttanner.com/blog/yolo-object-detection-with-opencv
# https://www.pyimagesearch.com/2020/01/27/yolo-and-tiny-yolo-object-detection-on-the-raspberry-pi-and-movidius-ncs/

import numpy as np
import tensorflow as tf
import cv2 as cv
from tensorflow.python.keras.api import keras
from tensorflow.keras import datasets, layers, models
from imutils.video import FPS
import Utiles
import Config
import Tracker


def genera_predicciones(modelo, layer_names, labels, image, confidence):
	height, width = image.shape[:2]
	image = cv.UMat(image)
	# Create a blob and pass it through the model
	blob = cv.dnn.blobFromImage(
		image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	modelo.setInput(blob)
	outputs = modelo.forward(layer_names)

	# Extract bounding boxes, confidences and classIDs
	ROIs, predicciones, tipos = genera_ROIs(
		outputs, confidence, width, height)

	# Apply Non-Max Suppression
	# Elimina las detecciones no maximas
	threshold = 0.3
	idxs = cv.dnn.NMSBoxes(ROIs, predicciones, confidence, threshold)

	return ROIs, predicciones, tipos, idxs


def genera_ROIs(outputs, umbral, width, height):
	ROIs = []
	predicciones = []
	tipos = []

	for output in outputs:
		for detection in output:
			# Extract the scores, classid, and the confidence of the prediction
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# Consider only the predictions that are above the confidence threshold
			if confidence > umbral:
				# Scale the bounding box back to the size of the image
				box = detection[0:4] * np.array([width, height, width, height])
				centerX, centerY, w, h = box.astype('int')

				# Use the center coordinates, width and height to get the coordinates of the top left corner
				x = int(centerX - (w / 2))
				y = int(centerY - (h / 2))

				ROIs.append([x, y, int(w), int(h)])
				predicciones.append(float(confidence))
				tipos.append(classID)

	return ROIs, predicciones, tipos


def dibuja_ROIs(ROIs, predicciones, tipos, idxs):
	"Auxiliar, para DEBUG, Genera una imagen con varias ROI"
	global image
	if len(idxs) > 0:
		for i in idxs.flatten():
			dibuja_ROI(ROIs[i], tipos[i], predicciones[i])


def dibuja_ROI(ROI, indice, confianza):
	# extract bounding box coordinates
	x, y, w, h = ROI

	# color = [int(c) for c in colores[tipos[i]]]
	color = colores[indice]
	cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
	text = "{}: {:.4f}".format(labels[indice], confianza)
	cv.putText(image, text, (x, y - 5),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def genera_DNN():
	"Genera una Red Neuronal Profunda Preentrenada con los parametros de Config"
	modelo = []
	modelo = cv.dnn.readNetFromDarknet(
		Config.DNN.YoLo_tiny.archivo_modelo, Config.DNN.YoLo_tiny.archivo_pesos)
	if(Config.DNN.gpu):
		modelo.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
		modelo.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
	return modelo


def genera_labels():
	return open(Config.DNN.archivo_labels).read().strip().split('\n')


def genera_colores(labels):
	"Genera colores aleatorios segun los labels"
	# colores = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
	colores = np.random.uniform(0, 255, size=(len(labels), 3))
	colores[0] = [84, 7, 220]
	return colores


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Detector YoLo-Tiny"
	dims = (0, 0)
	# Config.Fullscreen(titulo)
	out = None
	# Para capturar la salida
	if Config.VidProp.guardar:
		from Config import VidProp
		out = cv.VideoWriter(f"Salida/{titulo}.avi",
							VidProp.fourcc, VidProp.fps, VidProp.resolu)
		# print(out.get(2)) # TODO
	# Abre el video y almacena las dimesiones
	cap = cv.VideoCapture(Config.VidProp.source)
	dims = Utiles.dimensiones_video(cap)

	# Crea la red neural
	modelo = genera_DNN()

	# Extrae las capas de YoLo
	layer_names = modelo.getLayerNames()
	layer_names = [layer_names[i[0] - 1]
				for i in modelo.getUnconnectedOutLayers()]
	labels = genera_labels()
	colores = genera_colores(labels)
	fps = FPS().start()

	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			print('Video file finished.')
			break

		umbral_confianza = 0.5
		ROIs, predicciones, tipos, idxs = genera_predicciones(
			modelo, layer_names, labels, image, umbral_confianza)

		image = cv.UMat(image)
		dibuja_ROIs(ROIs, predicciones, tipos, idxs)

		if Config.VidProp.show_fps: Utiles.dibuja_FPS(image, fps)
		cv.imshow(titulo, image)

		if (cv.waitKey(1) & 0xFF == 27):
			break

		if Config.VidProp.guardar:
			Utiles.guardar(out, image)

	cap.release()
	if Config.VidProp.guardar: out.release()
