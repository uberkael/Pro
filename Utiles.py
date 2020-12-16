import numpy as np


def busca_centro(dim):
	"Devuelve el centro de unas dimensiones 2D"
	w, h = dim[0], dim[1]
	return (w//2, h//2)


def distancia(a, b):
	"Calcula la distancia eucidiana entre dos puntos"
	return np.linalg.norm(a-b)
	# dif = a - b
	# return math.sqrt(dif.x*dif.x + dif.y*dif.y)


def gen_p_aleatorios(num_puntos, dims):
	"Genera n puntos aleatorios entre unas dimensiones (mejor cuadrados)"
	return np.random.randint(0, min(dims[0], dims[1]), (num_puntos, 2))
