import sys
from functools import cache
from typing import List

import numpy as np
import pyaudio
from reedsolo import RSCodec
from scipy.signal import lfilter

# samples per second, 96000 is commonly supported
sample_rate = 96000

# duration of the handshake sound in seconds
preamble_duration = .02

# Frequency bands for the handshake
preamble = [440 * n for n in range(1, 28, 1)]

# Vinyl records have a frequency response of about 20 Hz to 20 kHz, with most
# LPs performing best in the range of 50 Hz to 15 kHz. However, the high-
# frequency response of vinyl can be particularly limited, and the dynamic
# range is typically lower than that of digital formats.

lp_ideal_frequency_response = (20, 20000)
lp_frequency_response = (50, 15000)
marginal_lp_frequency_response = (125, 12000)

carrier_frequency = (marginal_lp_frequency_response[0] + marginal_lp_frequency_response[1]) / 2
bandwidth = marginal_lp_frequency_response[1] - marginal_lp_frequency_response[0]
baud_rate = 2 * bandwidth  # Nyquist rate
bit_rate = 2 * baud_rate  # QPSK encodes 2 bits per symbol

# Reed-Solomon parameters
n = 255  # Number of data bytes
k = 223  # Number of data bytes that can be recovered from n bytes of data


def generate_handshake(sample_rate: int, duration: float, frequencies: List[float]) -> np.ndarray:
    """Mix sine waves to generate a handshake signal.

    Args:
        sample_rate (int): The sampling rate of the wave, in Hz.
        duration (float): The duration of the wave, in seconds.
        frequencies (list): A list of frequencies, in Hz, to mix together.

    Returns:
        (np.ndarray): A numpy array of shape (n_samples,), representing the handshake signal.
    """

    samples = np.arange(sample_rate * duration)
    signal = np.zeros_like(samples, dtype=np.float32)

    for freq in frequencies:
        sine_wave = np.sin(2 * np.pi * freq * samples / sample_rate)
        signal += sine_wave

    signal /= len(frequencies)
    return signal


@cache
def generate_quadrature_waves(sample_rate: int, bit_rate: int, carrier_frequency: float) -> np.ndarray:
    """
    Generates the four quadrature phase shift keying (QPSK) carriers. The
    sine waves are at 45º, 135º, 225º, and 315º. Many copies of these
    waves will be concatenated together to form the final signal.

    Args:
    - sample_rate (int): The sampling rate of the wave, in Hz.
    - bit_rate (int): The bit rate of the wave, in bits per second (bps).
    - carrier_frequency (float): The frequency of the carrier wave, in Hz.

    Returns:
    - (np.ndarray): A numpy array of shape (4, samples_per_symbol), where samples_per_symbol is the number of
      samples per symbol in the QPSK encoding. The array represents a quadrature wave with two bits of information
      encoded per symbol.

    Example usage:
    >>> wave = generate_quadrature_waves(44100, 2000, 1000)
    """

    samples_per_symbol = sample_rate // (bit_rate // 2)  # QPSK encodes 2 bits per symbol
    cycle_len = sample_rate // carrier_frequency
    return np.sin(2 * np.pi * (np.arange(samples_per_symbol) / cycle_len + np.arange(1, 8, 2)[:, None] / 8)).astype(np.float32)


def qpsk_modulate(data: bytes, sample_rate: int, bit_rate: int, carrier_frequency: float) -> np.ndarray:
    """
    Modulates a binary data stream using quadrature phase shift keying (QPSK) modulation.

    The input data stream is first converted into a sequence of 2-bit symbols, which are then mapped onto
    one of four possible quadrature indices. The quadrature indices are used to select the appropriate quadrature
    waveforms, which are generated using the `generate_quadrature_waves` function. The selected waveforms are then
    concatenated together to form the final modulated signal.

    Args:
    - data (bytes): The binary data stream to be modulated.
    - sample_rate (int): The sampling rate of the wave, in Hz.
    - bit_rate (int): The bit rate of the wave, in bits per second (bps).
    - carrier_frequency (float): The frequency of the carrier wave, in Hz.

    Returns:
    - (np.ndarray): A numpy array of shape (n_samples,), representing the modulated signal.

    Example usage:
    >>> data = b"hello world"
    >>> sample_rate = 44100
    >>> bit_rate = 2000
    >>> carrier_frequency = 1000
    >>> modulated_signal = qpsk_modulate(data, sample_rate, bit_rate, carrier_frequency)
    """
    
    data_array = np.frombuffer(data, dtype=np.uint8)
    data_bits = np.unpackbits(data_array)  # convert bytes to bits

    # Compute quadrature indices as an array
    quadrature_indices = (data_bits[::2] * 2 + data_bits[1::2])

    # Precompute the sine wave for each quadrature by rotating the buffer
    quadrature_waves = generate_quadrature_waves(sample_rate, bit_rate, carrier_frequency)

    # Generate the signal by concatenating the appropriate quadrature_waves
    signal = quadrature_waves[quadrature_indices].flatten()

    return signal


def preemphasis_filter(signal: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """
    Apply a preemphasis filter to a given audio signal to boost high-frequency components.

    The preemphasis filter increases the relative magnitude of high-frequency components in the audio signal,
    which can help to reduce the impact of noise and other sources of interference on the signal during
    transmission. ChatGPT says,
    
        These filters compensate for the non-linear frequency response of the vinyl record.
        
    and that it works by,
     
        ... improving the signal-to-noise ratio of the high-frequency components.

    Args:
    - signal (np.ndarray): A numpy array of shape (n_samples,), representing a time-domain audio signal.
    - alpha (float): The preemphasis coefficient. Defaults to 0.95.

    Returns:
    - (np.ndarray): A numpy array of shape (n_samples,), representing the preemphasized version of the input
      audio signal.

    Example usage:
    >>> signal = np.random.randn(44100)
    >>> filtered_signal = preemphasis_filter(signal, alpha=0.97)
    """

    return lfilter([1, -alpha], [1], signal).astype(np.float32)


def deemphasis_filter(signal: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    """
    Apply a deemphasis filter to a given audio signal to reduce high-frequency components.

    The deemphasis filter decreases the relative magnitude of high-frequency components in the audio signal,
    which can help to reduce the impact of noise and other sources of interference on the signal during
    transmission. ChatGPT says,

        These filters compensate for the non-linear frequency response of the vinyl record.

    and that it works by,

        ... reducing the signal-to-noise ratio of the high-frequency components.

    Args:
    - signal (np.ndarray): A numpy array of shape (n_samples,), representing a time-domain audio signal.
    - alpha (float): The deemphasis coefficient. Defaults to 0.95.

    Returns:
    - (np.ndarray): A numpy array of shape (n_samples,), representing the deemphasized version of the input
      audio signal.

    Example usage:
    >>> signal = np.random.randn(44100)
    >>> filtered_signal = deemphasis_filter(signal, alpha=0.97)
    """
    return lfilter([1], [1, -alpha], signal)


def add_forward_error_correction(data, n, k):
    assert k < n, "k must be less than n"
    rs = RSCodec(n - k)
    encoded_data = rs.encode(bytearray(data))
    return bytes(encoded_data)


def decode_forward_error_correction(encoded_data, n, k):
    rs = RSCodec(n - k)
    decoded_data = rs.decode(bytearray(encoded_data))
    return bytes(decoded_data)


def play_signal(signal, sample_rate):
    p = pyaudio.PyAudio()

    channels = 1 if len(signal.shape) == 1 else signal.shape[1]

    stream = p.open(format=p.get_format_from_width(signal.dtype.itemsize),
                    channels=channels,
                    rate=sample_rate,
                    output=True)

    chunk_size = 1024
    start = 0
    while start < len(signal):
        end = start + chunk_size if start + chunk_size < len(signal) else len(signal)
        chunk = signal[start:end]
        stream.write(chunk.tobytes())
        start += chunk_size

    # Clean up
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    handshake_signal = generate_handshake(sample_rate, preamble_duration, preamble)

    if len(sys.argv) == 1:
        data = b"Hello, world!"
    else:
        if sys.argv[1] == "--help":
            print("Usage: python digital.lp.py [input_file]")
            print("If no input file is specified, the default message 'Hello, world!' is used.")
            sys.exit(0)
        
        with open(sys.argv[1], "rb") as f:
            data = f.read()

    # Add forward error correction
    data = add_forward_error_correction(data, n, k)
    
    # Generate the QPSK modulated signal
    signal = qpsk_modulate(data, sample_rate, bit_rate, carrier_frequency)

    # signal = raised_cosine_filter(signal, sample_rate // bit_rate, roll_off, filter_span)

    signal = np.concatenate((handshake_signal, signal))

    signal = preemphasis_filter(signal)

    # Normalize the filtered signal
    signal = signal / np.max(np.abs(signal))

    # Convert to 16-bit PCM
    signal = (signal * 32767).astype(np.int16)

    play_signal(signal, sample_rate)

    # Save the filtered signal as a .wav file
    if len(sys.argv) > 1:
        from scipy.io.wavfile import write
        write(sys.argv[1] + ".wav", sample_rate, signal)

    print("LP label:")
    print(f"    QPSK Reed-Solomon ({n}, {k})")
    print(f"    frequency: {carrier_frequency} Hz")
    print(f"    baud rate: {baud_rate} bits/s")
