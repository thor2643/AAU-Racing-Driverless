import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11, GPIO.OUT)
GPIO.output(11, GPIO.LOW)
#my_pwm = GPIO.PWM(33, 100)


time.sleep(10)

"""
my_pwm.start(25)
time.sleep(5)

print(20)

my_pwm.ChangeDutyCycle(20)
time.sleep(5)

print(50)
my_pwm.ChangeDutyCycle(50)
time.sleep(5)

print(75)
my_pwm.ChangeDutyCycle(75)
time.sleep(5)

print(100)
my_pwm.ChangeDutyCycle(100)
time.sleep(10)

print("stopped")


my_pwm.stop()


"""
GPIO.cleanup()