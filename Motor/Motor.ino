#include <AccelStepper.h>
#include <MultiStepper.h>

// Define el paso y pasos totales para una vuelta
const int paso = 16;
const int vuelta = 200 * paso;
const int max_velocidad = 1000;
const int aceleracion = 50 * paso;

// Define los PIN de los motores
const int x_step_pin = 12;
const int x_dir_pin = 11;
const int y_step_pin = 10;
const int y_dir_pin = 9;

// Tipo de interface del stepper motor
#define motorInterfaceType 1

// Crea instancias de los stepper motor
AccelStepper x_motor(motorInterfaceType, x_step_pin, x_dir_pin);
AccelStepper y_motor(motorInterfaceType, y_step_pin, y_dir_pin);

// Variables para la posicion
int x_pos;
int y_pos;

// Variables Serial para recibir los  datos
String recivido;
char c;
int indice;

void setup()
{
	// Configura los motores
	x_motor.setMaxSpeed(max_velocidad);
	x_motor.setAcceleration(aceleracion);
	x_motor.setSpeed(vuelta);
	// x_motor.moveTo(vuelta);

	y_motor.setMaxSpeed(max_velocidad);
	y_motor.setAcceleration(aceleracion);
	y_motor.setSpeed(vuelta);
	// y_motor.moveTo(vuelta);

	// Inicia el uso de serial para escuchar
	Serial.begin(115200);
}

void loop()
{
	if(Serial.available() > 0)
	{
		// 3200 1600
		c = Serial.read();
		// Si encuentra el fin del la cadena mira si esta bien formada
		if (c == '\r' || c == '\n') {
			indice = recivido.indexOf(' ');
			// Si no encuentra un espacio es que algo anda mal
			if (indice >= 0) {
				// Extrae las nuevas posiciones
				x_pos = recivido.substring(0, indice).toInt();
				y_pos = recivido.substring(indice).toInt();
				Serial.println();
				Serial.print("X: ");
				Serial.print(x_pos);
				Serial.print(" Y: ");
				Serial.println(y_pos);
				// Mueve el motor a la posicion nueva
				x_motor.moveTo(x_pos);
				y_motor.moveTo(y_pos);
			}
			// Reinicia el buffer
			recivido = "";
		// Escribe en la cadena los nuevos caracteres
		} else {
			recivido += c;
			Serial.print(c);
		}
	}
	// Mueve el motor
	if (x_motor.distanceToGo() != 0) {
		x_motor.run();
	}
	if (y_motor.distanceToGo() != 0) {
		y_motor.run();
	}
}

