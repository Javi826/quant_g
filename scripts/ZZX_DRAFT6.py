#!/usr/bin/env python3
import os
import time
import shutil

# ---------------------------------------------------------
# Funciones de sonido
# ---------------------------------------------------------

def beep_terminal():
    """Sonido básico usando el bell del terminal (si está habilitado)."""
    print("\a")

def speaker_tone(freq=800, duration=0.2):
    """Genera un sonido usando speaker-test (viene en la mayoría de distros)."""
    os.system(f'speaker-test -t sine -f {freq} -l 1 >/dev/null 2>&1 &')
    time.sleep(duration)
    os.system('killall speaker-test >/dev/null 2>&1')

def beep_cmd(freq=800, duration=200):
    """Si el comando beep está disponible, lo usa."""
    if shutil.which("beep") is None:
        print("El comando 'beep' no está instalado.")
        return
    os.system(f'beep -f {freq} -l {duration}')

# ---------------------------------------------------------
# Menú
# ---------------------------------------------------------

def menu():
    while True:
        print("\n=== Selecciona un sonido para escuchar ===")
        print("1) Terminal Bell (print '\\a')")
        print("2) Tono generado (speaker-test) - agudo")
        print("3) Tono generado (speaker-test) - grave")
        print("4) beep (si instalado) - agudo")
        print("5) beep (si instalado) - grave")
        print("0) Salir")
        
        choice = input("\nOpción: ").strip()

        if choice == "1":
            print("Reproduciendo bell del terminal...")
            beep_terminal()

        elif choice == "2":
            print("Tono agudo (1000 Hz)...")
            speaker_tone(freq=1000)

        elif choice == "3":
            print("Tono grave (300 Hz)...")
            speaker_tone(freq=300)

        elif choice == "4":
            print("Beep agudo (1000 Hz)...")
            beep_cmd(freq=1000)

        elif choice == "5":
            print("Beep grave (300 Hz)...")
            beep_cmd(freq=300)

        elif choice == "0":
            print("Saliendo.")
            break

        else:
            print("Opción no válida.")

# ---------------------------------------------------------
# Ejecutar menú
# ---------------------------------------------------------

if __name__ == "__main__":
    menu()
