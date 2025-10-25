import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parquet_process.Z_parquet_extraction import _call_history_candles, to_dataframe_from_api

MADRID_TZ = ZoneInfo("Europe/Madrid")

def parse_timeframe(timeframe):
    """
    Convierte un timeframe string a minutos.
    Soporta: 1m, 5m, 15m, 30m, 1H, 4H, 1D, etc.
    """
    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('H'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('D'):
        return int(timeframe[:-1]) * 1440
    else:
        raise ValueError(f"Timeframe no reconocido: {timeframe}")

def calculate_expected_candle(now, interval_minutes):
    """
    Calcula cuál debería ser la última vela cerrada basándose en la hora actual.
    
    Args:
        now: datetime actual
        interval_minutes: intervalo de la vela en minutos
    
    Returns:
        datetime de inicio de la última vela cerrada
    """
    # Convertir el tiempo actual a timestamp Unix en minutos
    timestamp_minutes = int(now.timestamp() / 60)
    
    # Calcular el inicio de la vela actual (en curso)
    current_candle_start_minutes = (timestamp_minutes // interval_minutes) * interval_minutes
    
    # La última vela CERRADA es la anterior
    last_closed_candle_minutes = current_candle_start_minutes - interval_minutes
    
    # Convertir de vuelta a datetime
    last_closed_candle = datetime.fromtimestamp(last_closed_candle_minutes * 60, tz=now.tzinfo)
    
    return last_closed_candle

def check_candle_timing(symbol='BTCUSDT', timeframe='5m', limit=10):
    """
    Verifica la coherencia temporal entre la hora actual y la última vela recibida.
    
    Args:
        symbol: Par a consultar (default: BTCUSDT)
        timeframe: Timeframe de las velas (ejemplos: 1m, 5m, 15m, 30m, 1H, 4H, 1D)
        limit: Número de velas a solicitar (default: 10)
    """
    
    # Hora actual en Madrid
    now = datetime.now(MADRID_TZ)
    
    print("=" * 80)
    print(f"🕐 VERIFICACIÓN DE COHERENCIA TEMPORAL DE VELAS")
    print("=" * 80)
    print(f"Símbolo: {symbol}")
    print(f"Timeframe: {timeframe}")
    print(f"Hora actual (Madrid): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Obtener datos
    print(f"\n📡 Solicitando últimas {limit} velas...")
    recent_data = _call_history_candles(symbol=symbol, granularity=timeframe, limit=limit)
    
    if not recent_data:
        print("❌ No se recibieron datos")
        return
    
    # Convertir a DataFrame
    df = to_dataframe_from_api(recent_data)
    
    # Convertir precios a float para evitar errores de formato
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
    
    print(f"✅ Recibidas {len(df)} velas\n")
    
    # Mostrar las últimas 5 velas
    print("📊 Últimas 5 velas recibidas:")
    print("-" * 80)
    for i in range(min(5, len(df))):
        row = df.iloc[-(i+1)]
        ts = row['timestamp']
        # Convertir a timezone aware si no lo está
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(MADRID_TZ)
        else:
            ts = ts.astimezone(MADRID_TZ)
        
        print(f"  Vela {i+1}: {ts.strftime('%Y-%m-%d %H:%M:%S')} | "
              f"O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f}")
    
    # Analizar la última vela
    last_candle = df.iloc[-1]
    last_ts = last_candle['timestamp']
    
    # Asegurar timezone
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=ZoneInfo("UTC")).astimezone(MADRID_TZ)
    else:
        last_ts = last_ts.astimezone(MADRID_TZ)
    
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS DE COHERENCIA")
    print("=" * 80)
    print(f"Última vela recibida: {last_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hora de consulta:     {now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Calcular diferencia
    time_diff = now - last_ts
    minutes_diff = time_diff.total_seconds() / 60
    
    print(f"\nDiferencia temporal: {time_diff}")
    print(f"Diferencia en minutos: {minutes_diff:.2f} min")
    
    # Extraer el intervalo del timeframe
    try:
        interval_minutes = parse_timeframe(timeframe)
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        return
    
    # Calcular cuál debería ser la última vela cerrada
    expected_last_candle_time = calculate_expected_candle(now, interval_minutes)
    
    # También calcular la vela actual (en curso)
    current_candle_start = expected_last_candle_time + timedelta(minutes=interval_minutes)
    current_candle_end = current_candle_start + timedelta(minutes=interval_minutes)
    
    print(f"\n📅 Última vela cerrada esperada: {expected_last_candle_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Vela actual (en curso): {current_candle_start.strftime('%H:%M:%S')} - {current_candle_end.strftime('%H:%M:%S')}")
    
    # Verificar coherencia (tolerancia de 1 minuto)
    is_coherent = abs((last_ts - expected_last_candle_time).total_seconds()) < 120
    
    print("\n" + "=" * 80)
    if is_coherent:
        print("✅ COHERENTE: La última vela recibida corresponde a la última vela cerrada")
        print(f"   ✓ Recibida: {last_ts.strftime('%H:%M:%S')}")
        print(f"   ✓ Esperada: {expected_last_candle_time.strftime('%H:%M:%S')}")
    else:
        print("⚠️  INCOHERENTE: La última vela NO corresponde a la última vela cerrada")
        diff_seconds = (last_ts - expected_last_candle_time).total_seconds()
        print(f"   ✗ Diferencia: {diff_seconds:.0f} segundos ({diff_seconds/60:.1f} minutos)")
        print(f"   ✗ Recibida: {last_ts.strftime('%H:%M:%S')}")
        print(f"   ✗ Esperada: {expected_last_candle_time.strftime('%H:%M:%S')}")
        
        # Verificar si es la vela actual
        if abs((last_ts - current_candle_start).total_seconds()) < 120:
            print(f"\n   ⚠️  PROBLEMA: Estás recibiendo la vela ACTUAL (aún no cerrada)")
            print(f"   Esto causa LOOK-AHEAD BIAS en backtesting!")
    
    print("=" * 80)
    
    # Información adicional
    print("\n📝 Notas:")
    print(f"   • Intervalo de vela: {interval_minutes} minutos")
    if minutes_diff < interval_minutes:
        print(f"   • La última vela es muy reciente (< {interval_minutes} min)")
        if not is_coherent:
            print(f"   • ⚠️  ALERTA: Posiblemente estés recibiendo la vela actual (look-ahead bias)")
    elif minutes_diff > interval_minutes * 2:
        print(f"   • La última vela tiene más de {interval_minutes * 2} min")
        print(f"   • ⚠️  Podría haber un retraso significativo en los datos")
    else:
        print(f"   • El desfase temporal parece normal para datos históricos")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    import time
    
    # Ejemplos de uso con diferentes timeframes
    TIMEFRAMES_TO_TEST = ['5m']
    
    print("\n🚀 Iniciando verificación de coherencia temporal...")
    print("=" * 80)
    print("Timeframes a probar:", ", ".join(TIMEFRAMES_TO_TEST))
    print("=" * 80)
    
    try:
        # Primera ejecución: probar todos los timeframes una vez
        print("\n📊 PRUEBA INICIAL DE TODOS LOS TIMEFRAMES\n")
        for tf in TIMEFRAMES_TO_TEST:
            check_candle_timing(symbol='BTCUSDT', timeframe=tf, limit=10)
            print("\n" + "🔹" * 40 + "\n")
            time.sleep(2)
        
        # Luego monitoreo continuo del timeframe principal
        print("\n" + "=" * 80)
        print("🔄 Iniciando monitoreo continuo del timeframe 5m")
        print("(Presiona Ctrl+C para detener)")
        print("=" * 80 + "\n")
        
        while True:
            check_candle_timing(symbol='BTCUSDT', timeframe='5m', limit=10)
            print("\n⏳ Esperando 30 segundos para la próxima verificación...\n")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n👋 Verificación detenida por el usuario")