import unittest
import os
import multiprocessing
import sys
import numpy as np
from scipy.signal import correlate

# Add the parent directory (i.e., one level up from `scripts/`)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r"C:\Programming\eclipse_workspace\Python_AudioLogger")))

import Python_AudioLogger
from Python_AudioLogger import create_shared_resource_manager
from Python_AudioLogger import create_process_local_common_datadictionary_definition
from Python_AudioLogger import create_shared_memory_resources
from Python_AudioLogger import apply_low_pass


# Unit test howto :
# https://youtu.be/6tNS--WetLI?feature=shared



# generate a sinus signal at 1000Hz in 48000 hz raw byte pcm coded format int16 :
def generate_sine_wave_bytes(frequency=1000, sample_rate=48000, duration=3.0, amplitude=0.5):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    samples = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # Scale to int16
    samples_int16 = np.int16(samples * 32767)

    # Return as raw bytes
    return samples_int16


def signal_energy(signal):
    
    signal = signal.astype(np.float64)
    energy = np.sum(np.abs(signal)**2)
    return energy


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

        self.sinus_1000hz_s16 = generate_sine_wave_bytes(frequency=1000)
        self.sinus_100hz_s16 = generate_sine_wave_bytes(frequency=100)

    def tearDown(self):
        self.manager.shutdown()

    def test_Unit_01_LowPass(self):
        
        self.data_dictionary['audio_data'] = self.sinus_1000hz_s16
        lowpass_array_1000hz   = apply_low_pass(self.data_dictionary)
        
        self.data_dictionary['audio_data'] = self.sinus_100hz_s16
        lowpass_array_100hz   = apply_low_pass(self.data_dictionary)

        unfiltered_energy_100hz     = signal_energy(self.sinus_100hz_s16)
        lowpass_energy_100hz        = signal_energy(lowpass_array_100hz )
        unfiltered_energy_1000hz   = signal_energy(self.sinus_1000hz_s16)
        lowpass_energy_1000hz      = signal_energy(lowpass_array_1000hz )
        
        #print(f"Average power: unfiltered_energy_200hz : {unfiltered_energy_200hz}")
        #print(f"Average power: lowpass_energy_200hz : {lowpass_energy_200hz}")
        #print(f"Average power: unfiltered_energy_1000hz : {unfiltered_energy_1000hz}")
        #print(f"Average power: lowpass_energy_1000hz : {lowpass_energy_1000hz}")
        
        
        lowpass_gain_100hz =   10 * np.log10(lowpass_energy_100hz / unfiltered_energy_100hz)
        lowpass_gain_1000hz =  10 * np.log10(lowpass_energy_1000hz / unfiltered_energy_1000hz)
        
        print(f"lowpass_gain_100hz: {lowpass_gain_100hz}")
        print(f"lowpass_gain_1000hz: {lowpass_gain_1000hz}")
        
        result = True  # Replace with real test logic
        self.assertTrue(result, "Expected result to be True")


# to call from comand line :
# alternative : use > python -m unittest test_unit_01.py
# ✅ RUN TESTS
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Optional, but good for Windows
    unittest.main()