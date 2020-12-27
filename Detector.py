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


def prediccion(image, modelo, capas_conexion, labels):
	height, width = image.shape[:2]
	image = cv.UMat(image)
	# Create a blob and pass it through the model
	blob = cv.dnn.blobFromImage(
		image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	modelo.setInput(blob)
	outputs = modelo.forward(capas_conexion)

	# Extract bounding boxes, confidences and classIDs
	rects, probabilidades, tipos = genera_predicciones(outputs, width, height, labels)
	rects, probabilidades, tipos = elimina_nonmax(rects, probabilidades, tipos)

	return [rects, probabilidades, tipos]


def elimina_nonmax(rects, probabilidades, tipos):
	# Apply Non-Max Suppression
	# Elimina las detecciones no maximas
	confidence = Config.DNN.umbral_confianza
	threshold = 0.3
	res_rects = []
	res_probabilidades = []
	res_tipos = []
	idxs = cv.dnn.NMSBoxes(rects, probabilidades, confidence, threshold)
	if len(idxs) > 0:
		for i in idxs.flatten():
			res_rects.append(rects[i])
			res_probabilidades.append(probabilidades[i])
			res_tipos.append(tipos[i])
	return res_rects, res_probabilidades, res_tipos


def genera_predicciones(outputs, width, height, labels):
	"Genera preddiciones en base a un umbral de confianza"
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
			if confidence > Config.DNN.umbral_confianza:
				# Scale the bounding box back to the size of the image
				box = detection[0:4] * np.array([width, height, width, height])
				centerX, centerY, w, h = box.astype('int')

				# Use the center coordinates, width and height to get the coordinates of the top left corner
				x = int(centerX - (w / 2))
				y = int(centerY - (h / 2))

				ROIs.append([x, y, int(w), int(h)])
				predicciones.append(float(confidence))
				tipos.append(labels[classID])
	return ROIs, predicciones, tipos


def dibuja_ROIs(image, ROIs):
	"Auxiliar, para DEBUG, Genera una imagen con varias ROI"
	# img = cv.UMat(image)
	# for rect, probabilidad, tipo in ROIs:
	rects, probabilidades, tipos = ROIs
	for rect, probabilidad, tipo in zip(rects, probabilidades, tipos):
		image = dibuja_ROI(image, rect, probabilidad, tipo)
	return image


def dibuja_ROI(image, rect, probabilidad, tipo):
	# extract bounding box coordinates
	x, y, w, h = rect
	# color = [int(c) for c in colores[tipos[i]]]
	# color = colores[tipo]
	color = Config.UI.rojo
	cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
	text = "{}: {:.4f}".format(tipo, probabilidad)
	cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	return image


def capas_desconectadas(modelo):
	"Devuelve las capas para conectar al modelo DNN"
	capas = modelo.getLayerNames()
	capas = [capas[i[0] - 1] for i in modelo.getUnconnectedOutLayers()]
	return capas


def genera_DNN():
	"Genera una Red Neuronal Profunda Preentrenada con los parametros de Config"
	modelo = []
	modelo = cv.dnn.readNetFromDarknet(
		Config.DNN.YoLo.archivo_modelo, Config.DNN.YoLo.archivo_pesos)
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
	titulo = "Detector YoLo"
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

	# Extrae layers principales de YoLo
	capas_conexion = capas_desconectadas(modelo)

	labels = genera_labels()
	colores = genera_colores(labels)

	fps = FPS().start()

	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			print('Video file finished.')
			break

		ROIs = prediccion(image, modelo, capas_conexion, labels)

		image = dibuja_ROIs(image, ROIs)

		if Config.VidProp.show_fps:
			Utiles.dibuja_FPS(image, fps)
		cv.imshow(titulo, image)

		if (cv.waitKey(1) & 0xFF == 27):
			break

		if Config.VidProp.guardar: Utiles.guardar(out, image)

	cap.release()
	if Config.VidProp.guardar: out.release()
