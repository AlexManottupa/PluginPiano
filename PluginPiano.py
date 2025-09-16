import numpy as np
import sounddevice as sd
from pedalboard import Pedalboard, Reverb, Chorus, Delay, Phaser, Distortion, HighpassFilter, LowpassFilter
import keyboard
import threading
import time

# Variables globales
active_voices = {}  # note -> {'start_sample': int, 'velocity': int}
sample_rate = 44100
block_size = 512
max_voices = 16
global_volume = 3.0  # Aumentado para mayor volumen
reverb_room = 0.2
chorus_mix = 0.1
delay_time = 0.1
phaser_mix = 0.2
overdrive_gain = 1.5
eq_highpass = 100
eq_lowpass = 8000
flanger_mix = 0.3
tremolo_freq = 5.0
note_duration_sec = 1.0  # Cambiado de 5.0 a 2.0 para notas de máximo 2 segundos
note_cache = {}  # Cache de notas generadas
midi_buffer = []
midi_lock = threading.Lock()

# Inicialización: Genera notas para MIDI 21-108
def init(sample_rate_, block_size_):
    global sample_rate, block_size
    sample_rate = sample_rate_
    block_size = block_size_
    target_len = int(note_duration_sec * sample_rate)
    for note in range(21, 109):
        note_cache[note] = generate_fallback_note(note, target_len, 127)
    print("Inicialización completada: Notas generadas para MIDI 21-108")

# Genera nota sinusoidal
def generate_fallback_note(note, duration_samples, velocity):
    freq = 440 * (2 ** ((note - 69) / 12))
    t = np.linspace(0, duration_samples / sample_rate, duration_samples, False)
    audio = np.sin(2 * np.pi * freq * t)
    for h, amp in zip([2, 3, 5, 7], [0.5, 0.3, 0.2, 0.1]):
        audio += amp * np.sin(2 * np.pi * h * freq * t)
    audio /= np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 1
    audio *= (velocity / 127.0)
    envelope = np.ones(len(audio))
    attack = int(0.01 * sample_rate)
    decay = int(0.1 * sample_rate)
    sustain = 0.7
    envelope[:attack] = np.linspace(0, 1, attack)
    envelope[attack:attack+decay] = np.linspace(1, sustain, decay)
    envelope[attack+decay:] = sustain
    audio *= envelope
    return audio

# Genera audio para una nota
def generate_piano_note(note, duration_samples, velocity):
    if note in note_cache:
        audio = note_cache[note].copy()[:duration_samples]
    else:
        audio = generate_fallback_note(note, duration_samples, velocity)
    audio *= (velocity / 127.0)
    return audio

# Procesa bloques de audio
def process_npy(audio_in, midi_in, num_channels, num_samples):
    global active_voices, global_volume
    audio_out = np.zeros((num_channels, num_samples))
    
    for msg in midi_in:
        if msg['type'] == 'note_on' and msg['velocity'] > 0:
            if len(active_voices) < max_voices:
                active_voices[msg['note']] = {'start_sample': 0, 'velocity': msg['velocity']}
                print(f"Nota tocada: {msg['note']} (C4=60, D4=62, etc.)")
        elif msg['type'] == 'note_off' or (msg['type'] == 'note_on' and msg['velocity'] == 0):
            if msg['note'] in active_voices:
                active_voices[msg['note']]['release'] = True
                print(f"Nota liberada: {msg['note']}")
    
    voices_to_remove = []
    for note, voice in list(active_voices.items()):
        start = voice['start_sample']
        end = start + num_samples
        chunk_size = min(num_samples, int(note_duration_sec * sample_rate) - start)
        if chunk_size <= 0:
            voices_to_remove.append(note)
            continue
        chunk = generate_piano_note(note, chunk_size, voice['velocity'])
        decay = np.exp(-np.linspace(0, note_duration_sec, len(chunk)) / note_duration_sec)
        if 'release' in voice:
            decay *= 0.5
        chunk *= decay * global_volume
        if num_channels == 1:
            audio_out[0, :len(chunk)] += chunk
        else:
            pan = (note - 60) / 48.0
            audio_out[0, :len(chunk)] += chunk * (1 - pan) * 0.7
            audio_out[1, :len(chunk)] += chunk * pan * 0.7
        voice['start_sample'] += num_samples
        if voice['start_sample'] > note_duration_sec * sample_rate:
            voices_to_remove.append(note)
    for note in voices_to_remove:
        del active_voices[note]
    
    # Si no hay audio (notas activas), retornar ceros para silencio
    if not np.any(audio_out):
        print("Silencio: No hay notas activas")
        return audio_out
    
    # Aplica efectos solo si hay audio
    if num_channels == 1:
        audio_out = np.tile(audio_out, (2, 1))
    elif num_channels > 2:
        audio_out = audio_out[:2]
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=eq_highpass),
        LowpassFilter(cutoff_frequency_hz=eq_lowpass),
        Distortion(drive_db=overdrive_gain),
        Reverb(room_size=reverb_room, damping=0.5, wet_level=0.3),
        Chorus(mix=chorus_mix, depth=0.5, rate_hz=0.5),
        Phaser(mix=phaser_mix, depth=0.5, rate_hz=1.0),
        Delay(delay_seconds=delay_time, feedback=0.3, mix=0.2),
    ])
    audio_out = board(audio_out, sample_rate)
    
    # Mezcla con input solo si hay audio
    if np.any(audio_in):
        audio_out += audio_in * 0.5
    
    # Normaliza solo si hay audio significativo
    max_amp = np.max(np.abs(audio_out))
    if max_amp > 0.0001:  # Umbral para evitar amplificar ruido
        audio_out /= max_amp * 1.1
    
    return audio_out

# Callback para el flujo de audio
def audio_callback(outdata, frames, time, status):
    if status:
        print(f"Estado de audio: {status}")
    midi_messages = []
    with midi_lock:
        midi_messages = midi_buffer.copy()
        midi_buffer.clear()
    audio_in = np.zeros((2, frames))
    audio_out = process_npy(audio_in, midi_messages, num_channels=2, num_samples=frames)
    outdata[:, :] = audio_out.T

# Mapeo de teclas a notas MIDI
key_to_note = {
    'a': 60,  # C4
    's': 62,  # D4
    'd': 64,  # E4
    'f': 65,  # F4
    'g': 67,  # G4
    'h': 69,  # A4
    'j': 71,  # B4
    'k': 72,  # C5
    'z': 61,  # C#4
    'x': 63,  # D#4
    'c': 66,  # F#4
    'v': 68,  # G#4
    'b': 70,  # A#4
}

# Función para manejar entrada de teclado
def keyboard_input():
    print("Toca notas con las teclas:")
    print("a (C4), s (D4), d (E4), f (F4), g (G4), h (A4), j (B4), k (C5)")
    print("z (C#4), x (D#4), c (F#4), v (G#4), b (A#4)")
    print("Presiona 'q' para salir")
    pressed_keys = set()
    while not keyboard.is_pressed('q'):
        for key, note in key_to_note.items():
            if keyboard.is_pressed(key) and key not in pressed_keys:
                with midi_lock:
                    midi_buffer.append({'type': 'note_on', 'note': note, 'velocity': 100})
                pressed_keys.add(key)
            elif not keyboard.is_pressed(key) and key in pressed_keys:
                with midi_lock:
                    midi_buffer.append({'type': 'note_off', 'note': note, 'velocity': 0})
                pressed_keys.remove(key)
        time.sleep(0.01)

# Inicialización
init(44100, 512)

# Inicia entrada de teclado
print("Iniciando entrada de teclado...")
threading.Thread(target=keyboard_input, daemon=True).start()

# Inicia el flujo de audio
try:
    print("Verificando dispositivos de audio...")
    print(sd.query_devices())
    with sd.OutputStream(samplerate=sample_rate, blocksize=block_size, channels=2, callback=audio_callback):
        print("Flujo de audio iniciado. Toca notas con las teclas. Presiona 'q' para salir.")
        while not keyboard.is_pressed('q'):
            time.sleep(0.1)
except Exception as e:
    print(f"Error al iniciar el flujo de audio: {e}")
finally:
    print("Flujo de audio detenido.")