import subprocess

def obtener_dependencias_directas():
    result = subprocess.run(['pipdeptree', '--freeze', '--warn', 'silence'], stdout=subprocess.PIPE)
    paquetes = result.stdout.decode().splitlines()

    # Filtrar solo las dependencias directas
    dependencias_directas = [paquete for paquete in paquetes if ' ' not in paquete]
    
    return dependencias_directas

dependencias = obtener_dependencias_directas()
for dep in dependencias:
    print(dep)
