import cv2 as cv
import numpy as np


def busca_centro(dim):
	w, h = dim[0], dim[1]
	return (w//2, h//2)


def distancia(a, b):
	"Calcula la distancia eucidiana entre dos puntos"
	return np.linalg.norm(a-b)
	# dif = a - b
	# return math.sqrt(dif.x*dif.x + dif.y*dif.y)


def organiza_objetivos(dim, lista_p):
	print(dim)
	ctr_x, ctr_y = busca_centro(dim)
	centro = np.array((ctr_x, ctr_y))

	orden_objetivos = []
	objetivo_anterior = centro
	for i in range(len(lista_p)):
		distancias = [distancia(x, objetivo_anterior) for x in lista_p]
		menor = np.argmin(distancias)
		objetivo_anterior = lista_p[menor]
		orden_objetivos.append(lista_p[menor])
		lista_p = np.delete(lista_p, menor, axis=0)

	return orden_objetivos


def dibuja_puntos(img, lista_p):
	for i, punto in enumerate(lista_p):
		cv.circle(img, (punto[0], punto[1]), 5, (200, 0, 128), 3)
		cv.putText(img, str(i), (punto[0]+5, punto[1]-5), 0, 1, (20, 230, 150), 2)




def dibuja_paths(img, lista_p):
	"""
	Dibuja el path entre distintos puntos una lista o
	"""
	anterior = np.array([])
	for punto in lista_p:
		if anterior.any():
			cv.arrowedLine(img, (anterior[0], anterior[1]), (punto[0], punto[1]), (0, 255, 0), 3)
		anterior = punto


if __name__ == "__main__":
	lista_p = np.random.randint(0, 500, (30, 2))
	# lista_p = np.array([[20, 400], [500, 40], [200, 150]])
	img = np.zeros((500, 500, 3), np.uint8)
	dim = (img.shape[0], img.shape[1])
	orden_objetivos = organiza_objetivos(dim, lista_p)
	dibuja_paths(img, orden_objetivos)
	dibuja_puntos(img, orden_objetivos)
	cv.imshow("Selector", img)
	cv.waitKey(0)
