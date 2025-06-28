import unittest
import os
import multiprocessing
import sys
import numpy as np
from scipy.signal import correlate
from scipy.signal import cheby2, freqz, sosfilt, sosfreqz
import matplotlib.pyplot as plt
import pandas as pd
import wave

# Add the parent directory (i.e., one level up from `scripts/`)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Programming\eclipse_workspace\Python_AudioLogger")))

import Python_AudioLogger
from Python_AudioLogger import create_shared_resource_manager
from Python_AudioLogger import create_process_local_common_datadictionary_definition
from Python_AudioLogger import create_shared_memory_resources
from Python_AudioLogger import apply_low_pass
from Python_AudioLogger import nyquist, normal_cutoff, STOPBAND_ATTEN, CUTOFF, RATE, ORDER, SAMPLE_SIZE, CHANNELS, sos, CHUNK


# Unit test howto :
# https://youtu.be/6tNS--WetLI?feature=shared



# generate a sinus signal at 1000Hz in 48000 hz pcm format in  int16 :
def generate_sine_wave_bytes(frequency=100, sample_rate=48000, duration=1.0, amplitude=1.0):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Scale to int16
    samples_int16 = np.int16(samples * 32767)

    # Return as raw bytes
    return samples_int16, t


def signal_energy(signal):
    
    signal = signal.astype(np.float64)
    energy = np.sum(np.abs(signal)**2)
    return energy


def signal_rms(signal):
    signal = signal.astype(np.float64)
    rms = np.sqrt(np.mean(signal**2))
    return rms


def power_spectral_sensity_psd(spectrum, len, signal_mask, freqs):
    
    spectrum = spectrum.astype(np.float64)
    psd = (1/len) * (np.abs(spectrum)**2)
    return psd


def calculate_snr(signal, noisy_signal):
    signal          = signal.astype(np.float32)
    noisy_signal    = noisy_signal.astype(np.float32)
    
    noise = noisy_signal - signal

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    epsilon = 1e-10  # Small constant to avoid divide-by-zero or log(0)

    snr = signal_power / (noise_power + epsilon)
    snr_db = 10 * np.log10(snr + epsilon)  # Ensure you also avoid log(0) here
    
    return snr_db


# ✅ TEST CLASS: always define outside `if __name__ == "__main__"`
class UnitTest_LowPass(unittest.TestCase):
    def setUp(self):
        self.manager = create_shared_resource_manager()
        self.data_dictionary = create_process_local_common_datadictionary_definition(self.manager)

        (
            self._device_index,
            self.chunk_index_i,
            self.is_recording,
            self.is_logging,
            self.is_lowpass,
            self.system_calibration_factor_94db,
            self.frames,
            self.frames_filtered,
            self.chunk_noise_list_index,
            self.chunk_noise_list_spl
        ) = create_shared_memory_resources(self.manager)


    def tearDown(self):
        self.manager.shutdown()



    # a) Plot the frequency response for visual checks
    # b) Check if the attenuation within the passband is within acceptable limits.
    # c) Print filter gain ata specific frequency in the passband and in the stopband, eg. 100Hz and 1000Hz.
    def test_Unit_01_Check_LowPass_Check_TransferFunction(self):
        global sos # import filter coeficients from function under test
        result_H_cutoff     = False
        result_H_passband   = False
        result_H_dc         = False
        result_freqOffset   = False

        

        # Compute frequency response
        frequencies, h = sosfreqz(sos, worN=1024, fs=RATE)  # w: frequency axis in Hz, h: complex gain
        
        
        # Plot on log x-axis
        plt.figure()
        plt.figure(figsize=(10, 6))
        plt.semilogx(frequencies, 20 * np.log10(np.abs(h)), 'b')
        plt.title("Chebyshev Type II Lowpass Filter Frequency Response (Log X-Axis)")
        plt.xlabel("Frequency [Hz] (log scale)")
        plt.ylabel("Amplitude [dB]")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.axvline(CUTOFF, color='red', linestyle='--', label='Cutoff Frequency')
        plt.axhline(-STOPBAND_ATTEN, color='green', linestyle='--', label='Stopband Attenuation')
        plt.legend()
        plt.savefig("filter_response_lowpass.png", dpi=300)
        plt.close()
        
        
        # Find the index closest to normal_cutoff  
        idx_cutoff = np.argmin(np.abs(frequencies - CUTOFF))

        # Check for minim attentiation inside stopband. Consider stop band + 10% offset. 
        # because the specified attentuation is not yet reached at exactly Cutoff frequency 
        idx_cutoffOffset = np.argmin(np.abs(frequencies - (CUTOFF*1.2)))


        H_cutoff_min_dB        = 20 * np.log10(max(np.abs(h[idx_cutoffOffset:])))
        print(f"frequencies[idx_cutoff]: {frequencies[idx_cutoff]}")


        # calculate min frency response at inside passband first section 0 hz to CutOFF/2
        # Because the response starts to drop before CutOff
        idx_dc_min = np.argmin(np.abs(frequencies - (CUTOFF/2)))
        H_dc_min_dB            = 20 * np.log10(min(np.abs(h[0:idx_dc_min])))

        print(f"H_cutoff_min_dB: {H_cutoff_min_dB:.2f}")
        print(f"H_dc_min_dB: {H_dc_min_dB:.2f}")


        # check frequency offset to cuttoff frequency
        offset_prc = 100* frequencies[idx_cutoffOffset]/frequencies[idx_cutoff]
        if 110 <= offset_prc <= 130 :
            result_freqOffset = True

        offset_prc = 100* frequencies[idx_dc_min]/frequencies[idx_cutoff]
        if 30 <= offset_prc <= 80 :
            result_H_passband = True


        # check if attentuation is as expected at the specified cutt off frenquency
        if H_cutoff_min_dB < -STOPBAND_ATTEN :
            result_H_cutoff = True
            
        # check if attentuation is ZERO at 0 Hz
        if -0.1 < H_dc_min_dB < 0.1 :
            result_H_dc = True
        
                # Put into a DataFrame
        df = pd.DataFrame({
            'frequencies': frequencies,
            'h': h,
            'frequencies[idx_cutoff]': frequencies[idx_cutoff],
            'frequencies[idx_cutoffOffset]': frequencies[idx_cutoffOffset],
            'H_dc_min_dB': H_dc_min_dB,
            'H_cutoff_min_dB': H_cutoff_min_dB
        })

        # Write to Excel
        df.to_excel('test_Unit_01.xlsx', index=False)
        
        
        print(f"result_freqOffset: {result_freqOffset}")
        print(f"result_H_passband: {result_H_passband}")
        print(f"result_H_cutoff: {result_H_cutoff}")
        print(f"result_H_dc: {result_H_dc}")
        
        result = result_freqOffset and result_H_passband and result_H_cutoff and result_H_dc 
        
        self.assertTrue(result, "test_Unit_01 : Expected result to be True")
        

        
        

    # a) Check if the minimum attention within the stop band matchs the filter parameter specification.
    # b) Check if the attenuation within the passband is within acceptable limits.
    # c) Print filter gain ata specific frequency in the passband and in the stopband, eg. 100Hz and 1000Hz.
    def test_Unit_02_Check_LowPass_Check_withSinus(self):

        result_gain_100hz   = False
        result_gain_1000hz  = False
        
        test_chunk_duration = CHUNK / RATE


        self.sinus_100hz_s16,time   = generate_sine_wave_bytes(frequency=100, sample_rate=RATE, duration=test_chunk_duration, amplitude=1.0)
        self.sinus_1000hz_s16,time  = generate_sine_wave_bytes(frequency=1000, sample_rate=RATE, duration=test_chunk_duration, amplitude=1.0)
        
        
        self.data_dictionary['audio_data'] = self.sinus_100hz_s16
        lowpass_array_100hz   = apply_low_pass(self.data_dictionary)
        
        self.data_dictionary['audio_data'] = self.sinus_1000hz_s16
        lowpass_array_1000hz   = apply_low_pass(self.data_dictionary)
        
        # Calculate energy
        unfiltered_energy_100hz     = signal_energy(self.sinus_100hz_s16)
        lowpass_energy_100hz        = signal_energy(lowpass_array_100hz )
        unfiltered_energy_1000hz    = signal_energy(self.sinus_1000hz_s16)
        lowpass_energy_1000hz       = signal_energy(lowpass_array_1000hz )
        
        # Calculate gain
        lowpass_gain_100hz =   10 * np.log10(lowpass_energy_100hz / unfiltered_energy_100hz)
        lowpass_gain_1000hz =  10 * np.log10(lowpass_energy_1000hz / unfiltered_energy_1000hz)
        
        print(f"lowpass_gain_100hz: {lowpass_gain_100hz:.2f}")
        print(f"lowpass_gain_1000hz: {lowpass_gain_1000hz:.2f}")
        
        # expected gain to be close to zero+ tolerance
        if -5 <= lowpass_gain_100hz <= 5 :
            result_gain_100hz = True
            
        # expected gain to be arround expected attentuation + tolerance
        if lowpass_gain_1000hz <= -20 :
            result_gain_1000hz = True
        

        # Put into a DataFrame
        df = pd.DataFrame({
            'self.sinus_100hz_s16': self.sinus_100hz_s16,
            'lowpass_array_100hz': lowpass_gain_100hz,
            'self.sinus_1000hz_s16': self.sinus_1000hz_s16,
            'lowpass_array_1000hz': lowpass_array_1000hz,
            'time': time,
            'lowpass_gain_100hz': lowpass_gain_100hz,
            'lowpass_gain_1000hz': lowpass_gain_1000hz
        })

        # Write to Excel
        df.to_excel('test_Unit_02.xlsx', index=False)
    
            
        print(f"result_gain_100hz: {result_gain_100hz}")
        print(f"result_gain_1000hz: {result_gain_1000hz}")
    
        result = result_gain_100hz and result_gain_1000hz
        
        self.assertTrue(result, "test_Unit_02 : Expected result to be True")
    
    
    # a) Calculate SNR for a specific frequency in the passband and in the stopband, eg. 100Hz and 1000Hz.
    # b) Check if the specified filter attentuation is met
    def test_Unit_03_Check_LowPass_Check_OutPut_AudioIntegrity(self):
        result_SNA_100hz_dB = False
        result_SNA_1000hz_dB = False
        
        
        test_chunk_duration = CHUNK / RATE
        
        
        self.sinus_100hz_s16, time  = generate_sine_wave_bytes(frequency=100, sample_rate=RATE, duration=test_chunk_duration, amplitude=1.0)
        self.sinus_1000hz_s16, time = generate_sine_wave_bytes(frequency=1000, sample_rate=RATE, duration=test_chunk_duration, amplitude=1.0)
        self.sinus_zero_s16, time   = generate_sine_wave_bytes(frequency=100, sample_rate=RATE, duration=test_chunk_duration, amplitude=1e-10)
        
        self.data_dictionary['audio_data'] = self.sinus_100hz_s16
        lowpass_array_100hz   = apply_low_pass(self.data_dictionary)


        self.data_dictionary['audio_data'] = self.sinus_1000hz_s16
        lowpass_array_1000hz   = apply_low_pass(self.data_dictionary)
        

        SNA_100hz_dB    = calculate_snr(self.sinus_100hz_s16, lowpass_array_100hz)
        SNA_1000hz_dB   = calculate_snr(self.sinus_zero_s16, lowpass_array_1000hz)
        
        if SNA_100hz_dB >= -0.1 :
            result_SNA_100hz_dB = True
        if SNA_1000hz_dB <= -80.0 :
            result_SNA_1000hz_dB = True
        
        print(f"SNA_100hz_dB: {SNA_100hz_dB:.2f}")
        print(f"SNA_1000hz_dB: {SNA_1000hz_dB:.2f}")
        
        
        # Put into a DataFrame
        df = pd.DataFrame([{
            'SNA_100hz_dB': SNA_100hz_dB,
            'SNA_1000hz_dB': SNA_1000hz_dB
        }])

        # Write to Excel
        df.to_excel('test_Unit_03.xlsx', index=False)



        result = result_SNA_100hz_dB and result_SNA_1000hz_dB
                
        print(f"result_SNA_100hz_dB: {result_SNA_100hz_dB}")
        print(f"result_SNA_1000hz_dB: {result_SNA_1000hz_dB}")  
        

        
        self.assertTrue(result, "test_Unit_03 : Expected result to be True")
    


    # a) Plot unfiltered and filtered sinus and overlay signals
    # b) Save wave files to check by licening if noise is present in the filtered signals
    def test_Unit_04_Check_LowPass_Check_OutPut_AudioIntegrity_Plot_Wave(self):

        chunk_duration = CHUNK / RATE
        
        
        test_duration = 4.0 # seconds
        test_number_chunks = int(test_duration / chunk_duration)
        
        
        self.sinus_100hz_s16, time  = generate_sine_wave_bytes(frequency=100, sample_rate=RATE, duration=test_duration, amplitude=1.0)
        self.sinus_1000hz_s16, time = generate_sine_wave_bytes(frequency=1000, sample_rate=RATE, duration=test_duration, amplitude=1.0)
        self.sinus_zero_s16, time   = generate_sine_wave_bytes(frequency=100, sample_rate=RATE, duration=test_duration, amplitude=1e-10)
        
        # Simulate the chunk wise low pass filtering in order to detect noise due to transients etc.


        lowpass_chunks      = []
        filtered_chunk      = []
        i = 0
        for i in range(0, len(self.sinus_100hz_s16), CHUNK):        
            # Process the chunk
            self.data_dictionary['audio_data']  = self.sinus_100hz_s16[i:i+CHUNK]
            filtered_chunk                      = apply_low_pass(self.data_dictionary)
            lowpass_chunks.append(filtered_chunk)
        
        # Join all filtered chunks into one array
        lowpass_array_100hz = np.concatenate(lowpass_chunks)



        lowpass_chunks          = []
        filtered_chunk          = []
        i = 0
        for i in range(0, len(self.sinus_1000hz_s16), CHUNK):        
            # Process the chunk
            self.data_dictionary['audio_data']  = self.sinus_1000hz_s16[i:i+CHUNK] 
            filtered_chunk                      = apply_low_pass(self.data_dictionary)
            lowpass_chunks.append(filtered_chunk)
        
        # Join all filtered chunks into one array
        lowpass_array_1000hz = np.concatenate(lowpass_chunks)
        
        
        SNA_100hz_dB    = calculate_snr(self.sinus_100hz_s16, lowpass_array_100hz[:len(self.sinus_100hz_s16)])
        SNA_1000hz_dB   = calculate_snr(self.sinus_zero_s16, lowpass_array_1000hz[:len(self.sinus_1000hz_s16)])
        
        if SNA_100hz_dB >= -0.1 :
            result_SNA_100hz_dB = True
        if SNA_1000hz_dB <= -80.0 :
            result_SNA_1000hz_dB = True
        
        print(f"SNA_100hz_dB: {SNA_100hz_dB:.2f}")
        print(f"SNA_1000hz_dB: {SNA_1000hz_dB:.2f}")
        
        
        # Put into a DataFrame
        df = pd.DataFrame([{
            'self.sinus_100hz_s16': self.sinus_100hz_s16,
            'self.sinus_1000hz_s16': self.sinus_1000hz_s16,
            'self.sinus_zero_s16': self.sinus_zero_s16,
            'lowpass_array_100hz': lowpass_array_100hz,
            'lowpass_array_1000hz': lowpass_array_1000hz,                        
            'SNA_100hz_dB': SNA_100hz_dB,
            'SNA_1000hz_dB': SNA_1000hz_dB
        }])

        # Write to Excel
        df.to_excel('test_Unit_04.xlsx', index=False)


        with wave.open('sinus_100hz_s16_unfiltered.wav', 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_SIZE)  # SAMPLE_SIZE should be in bytes (e.g., 2 for int16)
            wf.setframerate(RATE)
            wf.writeframes(self.sinus_100hz_s16.tobytes())
        print(f"Audio saved as {'sinus_100hz_s16_unfiltered.wav'}\n")


        with wave.open('lowpass_array_100hz.wav', 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_SIZE)  # SAMPLE_SIZE should be in bytes (e.g., 2 for int16)
            wf.setframerate(RATE)
            wf.writeframes(lowpass_array_100hz.tobytes())
        print(f"Audio saved as {'lowpass_filtered_100hz.wav'}\n")


        with wave.open('lowpass_array_1000hz.wav', 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(SAMPLE_SIZE)  # SAMPLE_SIZE should be in bytes (e.g., 2 for int16)
            wf.setframerate(RATE)
            wf.writeframes(lowpass_array_1000hz.tobytes())
        print(f"Audio saved as {'lowpass_filtered_1000hz.wav'}\n")


        plt.figure()
        plt.plot(time[:8000], self.sinus_100hz_s16[:8000], color='blue', label='sinus_100hz_s16')
        plt.plot(time[:8000], lowpass_array_100hz[:8000], color='green', label='lowpass_array_100hz')
        plt.plot(time[:8000], lowpass_array_1000hz[:8000], color='orange', label='lowpass_array_1000hz')
        plt.xlabel('time')
        plt.legend()
        plt.ylabel(f"pcm. LowPass Filtered with CutOff freq.: {CUTOFF} ")
        plt.grid(True, which='both', linestyle='--', linewidth=0.1)
        plt.savefig("self.sinus100hz_LowPass100hz_LoPass1000hz_s16.png", dpi=600)
        plt.close()



        result = True
        self.assertTrue(result, "test_Unit_04 : Expected result to be True")



# to call from comand line :
# alternative : use > python -m unittest test_unit_01.py
# ✅ RUN TESTS
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Optional, but good for Windows
    unittest.main()