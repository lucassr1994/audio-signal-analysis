import wave
import matplotlib.pyplot as plt 
import numpy as np

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

