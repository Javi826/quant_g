
# =============================================================================
# # Buscar todos los ficheros resultantes
# result_files = list(output_folder.glob("*.parquet"))
# 
# total_rows = 0
# total_high_same = 0
# total_low_same = 0
# 
# print(f"📂 Analizando {len(result_files)} ficheros...\n")
# 
# for f in result_files:
#     df = pd.read_parquet(f, engine="pyarrow")
# 
#     # Asegurar que timestamp es datetime
#     if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
#         df["timestamp"] = pd.to_datetime(df["timestamp"])
# 
#     same_high = (df["high_time"] == df["timestamp"]).sum()
#     same_low = (df["low_time"] == df["timestamp"]).sum()
#     rows = len(df)
# 
#     total_rows += rows
#     total_high_same += same_high
#     total_low_same += same_low
# 
# # Resumen global
# print("=== 🧮 RESUMEN GLOBAL ===")
# print(f"Total filas analizadas: {total_rows:,}")
# print(f"Coincidencias high_time == timestamp : {total_high_same:,} ({total_high_same/total_rows:.2%})")
# print(f"Coincidencias low_time  == timestamp : {total_low_same:,} ({total_low_same/total_rows:.2%})")
# 
# if total_high_same > 0 or total_low_same > 0:
#     print("⚠️ Hay coincidencias. Algunas velas tienen high_time o low_time igual al timestamp.")
# else:
#     print("✅ Todo correcto: no hay ninguna coincidencia.")
#     
#  #========================================================================================= 
# # =========================================================================================
# 
# # Buscar todos los ficheros resultantes
# result_files = list(output_folder.glob("*.parquet"))
# 
# total_rows = 0
# total_high_next = 0
# total_low_next = 0
# 
# print(f"📂 Analizando {len(result_files)} ficheros para detectar extremos en la vela siguiente...\n")
# 
# for f in result_files:
#     df = pd.read_parquet(f, engine="pyarrow")
# 
#     # Asegurar timestamp datetime
#     if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
#         df["timestamp"] = pd.to_datetime(df["timestamp"])
# 
#     # Ordenar por timestamp (por seguridad)
#     df = df.sort_values("timestamp").reset_index(drop=True)
# 
#     # Crear columna con el timestamp de la siguiente vela
#     df["next_timestamp"] = df["timestamp"].shift(-1)
# 
#     # Comparar si el high_time o low_time coinciden con la siguiente vela
#     same_high_next = (df["high_time"] == df["next_timestamp"]).sum()
#     same_low_next = (df["low_time"] == df["next_timestamp"]).sum()
#     rows = len(df)
# 
#     total_rows += rows
#     total_high_next += same_high_next
#     total_low_next += same_low_next
# 
#     # Opcional: imprimir por archivo
#     if same_high_next > 0 or same_low_next > 0:
#         print(f"⚠️ {f.name}: {same_high_next} highs y {same_low_next} lows coinciden con la vela siguiente")
# 
# # Resumen global
# print("\n=== 🧮 RESUMEN GLOBAL (Coincidencias con la vela siguiente) ===")
# print(f"Total filas analizadas: {total_rows:,}")
# print(f"Coincidencias high_time == timestamp_siguiente : {total_high_next:,} ({total_high_next/total_rows:.2%})")
# print(f"Coincidencias low_time  == timestamp_siguiente : {total_low_next:,} ({total_low_next/total_rows:.2%})")
# 
# if total_high_next > 0 or total_low_next > 0:
#     print("⚠️ Se encontraron casos donde el high_time o low_time coinciden con el inicio de la vela siguiente.")
# else:
#     print("✅ Todo correcto: ningún extremo coincide con la siguiente vela.")
# # =========================================================================================
# sys.exit()
# =============================================================================