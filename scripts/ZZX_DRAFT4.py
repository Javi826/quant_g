import EmojiStore

# Obtener todos los emojis
all_emojis = list(EmojiStore.get_all())

# Filtrar solo algunas categorías que no quieres (caras, manos, animales, comida, personas)
excluded = {}

# Filtrar los emojis que sí te interesan
filtered = [e.emoji for e in all_emojis if e.category not in excluded]

print("Emojis filtrados:", "".join(filtered))
print("Cantidad:", len(filtered))
