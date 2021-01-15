import cv2 as cv
import numpy as np
import Utiles
import Config
import Motor_Serial as Motor


colores = [
	[[20, 55, 110], [35, 135, 220]]
	# [170, 0, 190], [175, 255, 200]
]


def getContours(imgFilter, imgOriginal):
	# Get all contours
	contours, hierarchy = cv.findContours(imgFilter, cv.RETR_EXTERNAL,
							cv.CHAIN_APPROX_NONE)
	img_contour = imgOriginal.copy()
	x, y, w, h = 0, 0, 0, 0
	for cnt in contours:
		area = cv.contourArea(cnt)
		if area > 500:
			# Suaviza los contornos
			contornos_suavizados = []
			for c in cnt:
				contornos_suavizados.append(cv.convexHull(c))
			# Dibuja
			cv.drawContours(img_contour, contornos_suavizados, -1, (0, 255, 255), 3)
			# Calcula el perimetro
			peri = cv.arcLength(cnt, True)
			# Encuentra los vertices
			approx = cv.approxPolyDP(cnt, 0.02*peri, True)
			# Agregamos un rectangulo a cada objeto
			x, y, w, h = cv.boundingRect(approx)
			cv.rectangle(img_contour, (x, y), (x+w, y+h), (0, 255, 0), 2)
			# Dibuja el centro
			cv.circle(img_contour, (x+w//2, y+h//2),
							10, (255, 0, 0), cv.FILLED)
	cv.imshow(titulo, img_contour)
	punto = [x+w//2, y+h//2]
	return Utiles.posicion_relativa(punto, mira)


def find_color(frame, colores):
	frameHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	upper = np.array(colores[0][0])
	lower = np.array(colores[0][1])
	mask = cv.inRange(frameHSV, upper, lower)
	# cv.imshow("Mask", mask)
	return getContours(mask, frame)


if __name__ == "__main__":
	# DEBUG Prueba de las funciones (No se usara, Archivo usado como libreria)
	titulo = "Detector Amarillo Serial"
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
	cap.set(3, 1920)
	cap.set(4, 1080)

	dims = Utiles.dimensiones_video(cap)
	mira = Utiles.punto_centro(dims)

	arduino = Motor.conexion_serial()

	contador = 0
	while cap.isOpened():
		ret, image = cap.read()
		if not ret:
			break
		# Muestra una imagen
		# cv.imshow("Video", frame)
		punto_objetivo = find_color(image, colores)
		if not np.array_equal(punto_objetivo, mira) and contador % 30 == 0:
			print(punto_objetivo)
			Motor.desplazamiento(arduino, punto_objetivo)
			contador = 0
			pass
		contador += 1
		# Espera la pulsacion de una tecla q y sale
		if (cv.waitKey(1) & 0xFF == ord('q')):
			Motor.desplazamiento(arduino, [0, 0])
			break
