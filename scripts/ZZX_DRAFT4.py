# Instala la librería emoji si no la tienes:
# pip install emoji --upgrade

import emoji

def all_emojis():
    return list(emoji.EMOJI_DATA.keys())

if __name__ == "__main__":
    emojis = all_emojis()
    print("".join(emojis))  # imprime todos los emojis juntos
    print(f"\nTotal de emojis: {len(emojis)}")
