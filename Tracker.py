import cv2 as cv
import numpy as np


class UI():
	rojo_claro = (66, 69, 255)
	rojo_oscuro = (56, 72, 153)
	rojo_oscuro2 = (25, 28, 67)
	cyan = (245, 235, 157)


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
	# cv.imshow("fondo ant", img)
	# Muy lento
	img = cv.adaptiveThreshold(
		img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 71, 50)
	# Aplica el sustractor de fondos
	fgmask = fgbg.apply(img)
	# Aplica la mascara para ver solo el cambio
	img = cv.bitwise_and(img, img, mask=fgmask)
	# cv.imshow("fondo", img)
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


def identifica_objetivos(frame, contornos):
	"""
	Identifica objetivos, segun su ratio (personas depie)
	Sera sustituido por una red neuronal
	"""
	for c in contornos:
		# Elminamos las areas pequeñas
		area = cv.contourArea(c)
		if area > 100:
			# Dibujamos el contorno
			cv.drawContours(frame, [c], 0, UI.rojo_claro, 1)
			# Calculamos el rectangulo que contiene el elemento
			peri = cv.arcLength(c, True)
			# Pinta un rectangulo con su centro
			approx = cv.approxPolyDP(c, 0.02*peri, True)
			x, y, w, h = cv.boundingRect(approx)
			cv.rectangle(frame, (x, y), (x+w, y+h), UI.rojo_oscuro, 1)
			cv.circle(frame, (x+w//2, y+h//2), 2, UI.cyan, 0)
			# Pinta el centro del contorno
			M = cv.moments(c)
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			cv.circle(frame, (cX, cY), 1, (255, 255, 0), 0)


def tracker(frame):
	img = eliminador_fondo(frame)
	contornos = extrae_contornos(img)
	# identifica_objetivos(img, contornos)
	img = cv.UMat(frame)
	identifica_objetivos(img, contornos)
	# frame = cv.UMat(frame)
	# Hacemos los colores oscuros claros
	# return cv.hconcat([frame, img])
	return img


if __name__ == "__main__":
	guardar = True
	if guardar:
		fourcc = cv.VideoWriter_fourcc(*"VP80")
		fps = 25
		resolu = (768, 576)
		out = cv.VideoWriter("Salida.avi", fourcc, fps, resolu)
	cap = cv.VideoCapture("Samples/vtest.avi")
	cv.namedWindow("Tracker", cv.WINDOW_NORMAL)
	cv.setWindowProperty("Tracker", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

	while cap.isOpened():
		ret, frame = cap.read()
		print("Width:", cap.get(cv.CAP_PROP_FRAME_WIDTH))
		print("Height:", cap.get(cv.CAP_PROP_FRAME_HEIGHT))
		print("Fps:", cap.get(cv.CAP_PROP_FPS))
		img = tracker(frame)
		cv.imshow("Tracker", img)
		if guardar:
			out.write(img)
		# if (cv.waitKey(40) & 0xFF == ord('q')):
		if (cv.waitKey(1) & 0xFF == ord('q')):
			break

cv.waitKey(0)
