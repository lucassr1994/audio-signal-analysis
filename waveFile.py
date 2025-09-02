import wave
import matplotlib.pyplot as plt 
import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import stft
import pandas as pd

wave_audio_object = wave.open('ITEM2.wav', 'rb')

number_of_samples = wave_audio_object.getnframes()
sample_frequency = wave_audio_object.getframerate()
signal_wave = wave_audio_object.readframes(-1)
duration = number_of_samples/sample_frequency
signal_array = np.frombuffer(signal_wave, dtype=np.int16)
channel_1 = signal_array[::2]
channel_2 = signal_array[1::2]
time = np.linspace(0, duration, num=len(signal_array)//2)

# Create a figure with two subplots, stacked vertically (2 rows, 1 column)
fig, (axe_1, axe_2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot Channel 1 on the top subplot (axe_1)
axe_1.plot(time, channel_1, color='blue')
axe_1.set_title('Channel 1 (Left)')
axe_1.set_xlabel('Time (s)')
axe_1.set_ylabel('Signal Wave')
axe_1.grid(True)

# Plot Channel 2 on the bottom subplot (axe_2)
axe_2.plot(time, channel_2, color='red')
axe_2.set_title('Channel 2 (Right)')
axe_2.set_xlabel('Time (s)')
axe_2.set_ylabel('Signal Wave')
axe_2.grid(True)

# Improve layout to prevent titles and labels from overlapping
plt.tight_layout()
plt.show()

# Calculate STFT and Power Spectrum for channel 1
yf = fft(channel_1)
xf = fftfreq(number_of_samples//2, 1 / sample_frequency)
power_spectrum = np.abs(yf)**2

# Plot power spectrum positive half
plt.figure(2)
plt.plot(xf[:number_of_samples//2], power_spectrum[:number_of_samples//2])
plt.title('Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid(True)
plt.show()

# Calculate STFT for channel 1
f, t, Zxx = stft(channel_1, fs=sample_frequency, nperseg=1024)

# Plot spectrogram
plt.figure(3)
plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
plt.title('Signal Spectrogram (Channel 1)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')
plt.ylim([0, 5000]) # Limite para melhor visualização (opcional)
plt.colorbar(label='Amplitude')
plt.show()

# Calcule a FFT do sinal completo


amplitude_spectrum = np.abs(yf)

# --- Criar a Tabela com Pandas SEM ERROS ---

# 1. Crie o DataFrame usando os arrays 'xf' e 'amplitude_spectrum' COMPLETO
#    Eles terão o mesmo tamanho (n), evitando o erro.
df = pd.DataFrame({
    'Frequencia (Hz)': xf,
    'Amplitude': amplitude_spectrum
})

# 2. Agora, filtre o DataFrame para as frequências positivas.
#    Esta é a etapa correta para remover as frequências negativas e o zero.
df = df[df['Frequencia (Hz)'] > 0]

# 3. Ordene o DataFrame pela amplitude para encontrar as frequências mais proeminentes
df = df.sort_values(by='Amplitude', ascending=False)

# 4. Exiba as 10 frequências mais fortes
print("Tabela das Frequências Mais Proeminentes:")
print(df.head(50).round(2))