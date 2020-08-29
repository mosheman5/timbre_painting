import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def show_spec(y, nfft=1024, sr=22050, center=True):
    D = np.abs(librosa.stft(y, n_fft=nfft, center=center))
    fig = plt.figure()
    librosa.display.specshow(librosa.amplitude_to_db(D,
                                                            ref=np.max),
                                    y_axis='log', x_axis='time', sr=sr)
    plt.title('Power spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    return fig
