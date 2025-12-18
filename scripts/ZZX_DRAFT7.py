# Script puro Python para imprimir casi todos los emojis
def is_emoji(char):
    return (
        '\U0001F600' <= char <= '\U0001F64F' or  # Emoticones
        '\U0001F300' <= char <= '\U0001F5FF' or  # Símbolos y pictogramas
        '\U0001F680' <= char <= '\U0001F6FF' or  # Transporte y mapas
        '\U0001F1E6' <= char <= '\U0001F1FF' or  # Banderas
        '\U00002700' <= char <= '\U000027BF' or  # Símbolos misceláneos
        '\U00002600' <= char <= '\U000026FF' or  # Más símbolos
        '\U0001F900' <= char <= '\U0001F9FF' or  # Emoticones recientes
        '\U0001FA70' <= char <= '\U0001FAFF'     # Símbolos adicionales
    )

def generate_emojis():
    emojis = []
    for i in range(0x1F300, 0x1FAFF):
        char = chr(i)
        if is_emoji(char):
            emojis.append(char)
    return emojis

if __name__ == "__main__":
    all_emojis = generate_emojis()
    print("".join(all_emojis))
    print(f"\nTotal aproximado de emojis: {len(all_emojis)}")
