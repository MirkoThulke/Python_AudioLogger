'''
# -----------------------------------------------------------------------------
# Author: MIRKO THULKE 
# Copyright (c) 2025, MIRKO THULKE
# All rights reserved.
#
# Date: 2025, VERSAILLES, FRANCE
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING
# FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#
# -----------------------------------------------------------------------------
'''


'''
important commands : 
cmd> python -m pip install --upgrade pip
cmd> pip install pyaudio
cmd> pip install numpy
cmd> where python
cmd> python --version
cmd> python3 --version
cmd> pip install --upgrade -r requirements.txt
cmd> pip freeze > requirements.txt
cmd> pip list --outdated
'''
import wx # click button GUI
import pyaudio
from endolith_weighting_filters import A_weight
import numpy as np
import wave
import matplotlib.pyplot as plt
import configparser
import multiprocessing
import threading
import subprocess
import os
import psutil
import time
import datetime
from viztracer import VizTracer # visual thread debugging


'''
DEBUG == 0 # Default 
DEBUG == 1 # tracer logs for runtime analysis
'''
DEBUG   = 1

# visual thead debugging
if DEBUG == 1:
    tracer = VizTracer()  # Start VizTracer
    tracer.start()

# open results in CMD : "vizviewer Python_AudioLogger_JSON_LogFile.json"
# ZOOM into timeline by pressing CTLR + Mouse wheel
# and check for status update and display update tasks. 
# Should be called every 20ms aprox with chunk size 1024 and 48kHz



# Behringer UMC control panel settings :
# ASIO buffer size 1024   
# Sampling rate 48kHz
# Format : 4 Channel 16 bits  


# Set parameters for audio input
FORMAT = pyaudio.paInt16  # Format for the audio
CHANNELS = 1  # Mono audio (1 channel)
RATE = 48000  # Sampling rate (samples per second)
CHUNK = 1024  # Number of frames per buffer (size of each audio chunk)
DEVICE_INDEX = None  # Set to the correct device index if you have multiple devices
CHUNK_SEC = CHUNK/RATE # Chunk duration in seconds
WAVE_DT_SEC = 1.5 # Delta time duration before and after noise event, that will be added to wave output
CHUNK_DNUM = int(WAVE_DT_SEC/CHUNK_SEC) # number of chunks to be added before and after noise event, that will be added to wave output

REFERENCE_PRESSURE = 20e-6  # in Pa. Reference pressure in Pa (20 µPa)
MIC_SENSITIVITY = 15 # 15mV Sensitivity (mV / Pa) of the Behringer ECM8000 (for better understanding only)
MIC_SPL_MAX_DB = 120   # in dB. virtual SPL max. value that the microphone can measure
MIC_PA_MAX = 20     # in Pa. Maximum Number of Pa that the mcicrophone can measure (120dB converted to Pascal)
MIC_MAX_MVOLT = MIC_PA_MAX * MIC_SENSITIVITY # maximum mV that the microphone can measure (corresponds to 120dB, 20Pa)

PRE_AMP_GAIN_DB = 45  # in dB. Assupmption that the Gain poti on the UMC is on 3h00. Virtuel Pre-Amp Gain Factor for better understanding only
PRE_AMP_GAIN_LIN = np.power(10, (PRE_AMP_GAIN_DB / 20) ) # Converted to linear scale.

CALIB_ITERATION_LENGTH = 50 # 50 chunks are checked during calibration
MAX_INT16 = np.iinfo(np.int16).max

# dB(A) Threshold for indoor noise
# NIGHT : 22h00 - 06h00
SPL_MAX_DAY_DBA     = 50
SPL_MAX_NIGH_DBA    = 35



# Set parameters for audio outut
RECORD_SECONDS = 4        # Duration of the recording in seconds
OUTPUT_FILENAME = "output.wav"  # Output WAV file
OUTPUT_NOISE_FILENAME = "Bruit" #Output filename prefix for logged noise events
OUTPUT_FILE_DIRECTORY = "audio_logfiles"



# Global Variables NOT in shared memory #######################################

# To persist Settings -> Initialise a config object.
config                  = None

# Global variables to Initialize PyAudio in the main Init class
p                       = None
# Open stream to read audio from the microphone
stream                  = None



# Global Funtion definitions ##########################################
def get_commit_version():
    try:
        # Run 'git rev-parse HEAD' to get the commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except subprocess.CalledProcessError:
        return "Not a git repository or error retrieving commit."
 
def func_check_devices(data_dictionary):
    global p
    i=0
    
    # List available devices
    print("Available devices:")
    for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']}")


def func_on_button_setDevices_click(frame, data_dictionary):
    global p

    
    data_dictionary['_device_index'] = 0
    min_range = 0
    max_range = 100

    user_value = int(frame.text_ctrl.GetValue())
    
    # Check if the input is within the valid range
    # obtain user text input for device selection
    if min_range <= user_value  <= max_range:
        data_dictionary['_device_index'] = user_value
        wx.MessageBox(f"Device {data_dictionary['_device_index']}: {p.get_device_info_by_index(data_dictionary['_device_index'])}","Info", wx.OK | wx.ICON_INFORMATION)
    else :
        wx.MessageBox(f"Error: The number must be between {min_range} and {max_range}.","Info", wx.OK | wx.ICON_INFORMATION)


def apply_a_weighting(data_dictionary):
    """Apply the A-weighting filter to the signal"""

    
    # convert to float for filtering
    float_array  = data_dictionary['audio_data'].astype(np.float32)
    

    # Apply A-weighting
    float_array_filt = A_weight(float_array, RATE)
    
    #convert back to integer for further processing
    int_array   = float_array_filt.astype(np.int16)
    
    #print(f'audio_data: {audio_data}')
    #print(f'float_array: {float_array}')
    #print(f'float_array_filt: {float_array_filt}') 
    #print(f'int_array: {int_array}')   
    
    return ( int_array )


def func_calc_SPL(data_dictionary):
 

    # reset output arrays :
    data_dictionary['a_weighted_signal']                        = np.zeros(data_dictionary['a_weighted_signal'].shape)
    data_dictionary['audio_data_pcm_abs']                       = np.zeros(data_dictionary['audio_data_pcm_abs'].shape)
    data_dictionary['audio_data_mV']                            = np.zeros(data_dictionary['audio_data_mV'].shape)
    data_dictionary['audio_data_pressurePa']                    = np.zeros(data_dictionary['audio_data_pressurePa'].shape)
    data_dictionary['audio_data_pressurePa_square']             = np.zeros(data_dictionary['audio_data_pressurePa_square'].shape)
    # reset output variables :
    data_dictionary['audio_data_pressurePa_squareMean']         = 0
    data_dictionary['audio_data_pressurePa_rms']                = 0
    data_dictionary['audio_data_pressurePa_rms_calib']          = 0
    data_dictionary['audio_data_pressurePa_spl']                = 0

    
    # Apply A-weighting to the signal
    # A-weighting does not have an impcat on microphone calibration at 1000Hz, because weighting is 1 at 1000Hz. 
    # Convert to float for A-weighting processing ( scipy.signal requires type 'signal', thus float32)
    data_dictionary['a_weighted_signal'] = apply_a_weighting(data_dictionary)

    # Check  the maximum absolute value
    audio_data_max_pcm_value_new = np.max(np.abs(data_dictionary['a_weighted_signal']))
    
    # save if highest of all chunks
    data_dictionary['audio_data_max_pcm_value'] = max(data_dictionary['audio_data_max_pcm_value'], audio_data_max_pcm_value_new)
    # Example: Process the audio (e.g., calculate RMS for volume level) 
        
    # Absolute values first      
    data_dictionary['audio_data_pcm_abs'] = np.abs(data_dictionary['a_weighted_signal'])
    
    # Convert to mV using the Sensitivy value of the microphone/ Assuming that the microphone uses the full int16 signale range . 
    # Applying a preamp gain factor (only for better understanding)
    data_dictionary['audio_data_mV'] = data_dictionary['audio_data_pcm_abs'] / PRE_AMP_GAIN_LIN
    #print(f"data_dictionary['audio_data_mV']: {data_dictionary['audio_data_mV']}")
    
    
    # Converting mV to Pa using the Sensitivy value
    data_dictionary['audio_data_pressurePa'] = data_dictionary['audio_data_mV'] / MIC_SENSITIVITY
    #print(f"data_dictionary['audio_data_pressurePa']: {data_dictionary['audio_data_pressurePa']}")
    
    # Convert to RMS - Root mean square
    data_dictionary['audio_data_pressurePa_square'] = data_dictionary['audio_data_pressurePa'] ** 2    
    data_dictionary['audio_data_pressurePa_squareMean'] = np.mean(data_dictionary['audio_data_pressurePa_square'])         
    if data_dictionary['audio_data_pressurePa_squareMean']  > 0 :
        data_dictionary['audio_data_pressurePa_rms'] = np.sqrt(data_dictionary['audio_data_pressurePa_squareMean'])
    else :
        data_dictionary['audio_data_pressurePa_rms'] = 0           
   
    data_dictionary['audio_data_pressurePa_rms_calib'] =data_dictionary['audio_data_pressurePa_rms'] * data_dictionary['system_calibration_factor_94db']
    
    # convert RMS to SPL (explained below)
    # Reference pressure in air = 20 µPa
    if data_dictionary['audio_data_pressurePa_rms_calib'] > 0:
        data_dictionary['audio_data_pressurePa_spl'] = 20 * np.log10(data_dictionary['audio_data_pressurePa_rms_calib'] / REFERENCE_PRESSURE)    
    else :
        data_dictionary['audio_data_pressurePa_spl'] = 0     
    #print(f"data_dictionary['audio_data_pressurePa_spl']: {data_dictionary['audio_data_pressurePa_spl']}")      


def func_process_audio_input(data_dictionary, recording_status_queue, recording_dba_queue):

 
    #whole process will run in high priority mode
    p_func_process_audio_input = psutil.Process(os.getpid())
    p_func_process_audio_input.nice(psutil.REALTIME_PRIORITY_CLASS)  

    # Initialize PyAudio
    p_loc                                  = pyaudio.PyAudio()
    # Open stream to read audio from the microphone
    stream_loc = p_loc.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index = data_dictionary['_device_index'] ,
                    frames_per_buffer=CHUNK)
    print("Audio stream opened [recording]\n")
    
    print("recording_process started 1/2\n")
    
    
    #reset audio input related lists and counters
    data_dictionary['chunk_index_i']            = 0    # counter of processed chunks
    data_dictionary['chunk_noise_list_index']   = np.array([], dtype=int)
    data_dictionary['chunk_noise_list_spl']     = np.array([])
    data_dictionary['frames']                   = [] # Wave data defined as byte list !
    
    print(f"is_recording: {data_dictionary['is_recording']}\n")
    while data_dictionary['is_recording']==True :
        # Read a chunk of audio data
        data = stream_loc.read(CHUNK)
      
        # Convert the audio data to a numpy array
        data_dictionary['audio_data']   = np.frombuffer(data, dtype=np.int16)
        
        #for output wave file creation, add to a list
        data_dictionary['frames'].append(data)
        
        # Calculate PCM to SPl ! 
        func_calc_SPL(data_dictionary)
        
        #chunk counter
        data_dictionary['chunk_index_i'] = data_dictionary['chunk_index_i']+1

        recording_status_queue.put(f"Recording running. Number of chunks processed: {data_dictionary['chunk_index_i']}\n")  # Send message to main process
        print(f"chunk_index : {data_dictionary['chunk_index_i']}\n")
        
        # Print dbA on GUI
        recording_dba_queue.put(f"{round(data_dictionary['audio_data_pressurePa_spl'], 2)} [dbA]\n")  # Send message to main process

        
        if data_dictionary['audio_data_pressurePa_spl'] > SPL_MAX_DAY_DBA :
            data_dictionary['chunk_noise_list_index']   = np.append(data_dictionary['chunk_noise_list_index'], data_dictionary['chunk_index_i'])
            data_dictionary['chunk_noise_list_spl']     = np.append(data_dictionary['chunk_noise_list_spl'], data_dictionary['audio_data_pressurePa_spl'])
            print(f"data_dictionary['chunk_noise_list_index'] : {data_dictionary['chunk_noise_list_index']}\n")            
            print(f"data_dictionary['chunk_noise_list_spl'] : {data_dictionary['chunk_noise_list_spl']}\n")
            

    print("Recording thread stopped.\n")
    recording_status_queue.put("Recording stopped ...\n")  # Send message to main process


    p_loc.terminate()
    
def func_run_calibration(data_dictionary):
    global stream

  

    calib_arr = np.array([])
    i= 0 
    data_dictionary['audio_data_max_pcm_value']  = 0
    
    
    while i<CALIB_ITERATION_LENGTH :
        
        i=i+1
    
        # Read a chunk of audio data
        data = stream.read(CHUNK)
      
        # Convert the audio data to a numpy array
        data_dictionary['audio_data']   = np.frombuffer(data, dtype=np.int16)
        
        #for output wave file creation, add to a list
        data_dictionary['frames'].append(data)
        
        # Calculate PCM to SPl ! 
        func_calc_SPL(data_dictionary)
        
        
        if data_dictionary['audio_data_pressurePa_rms'] > 0 : 
            
            # 94 dB != 20 log (rms /p_0) :
            system_calibration_factor_94db_new = (REFERENCE_PRESSURE * (np.power(10, 94/20))) /data_dictionary['audio_data_pressurePa_rms']
            print(f"system_calibration_factor_94db_new: {system_calibration_factor_94db_new}\n") 
            
            # Store new value in array 
            calib_arr   =   np.append(calib_arr, system_calibration_factor_94db_new)
            
        else :
            print("SPL or Pa equal to zero in this chunk. Check sound input ! \n") 
        
         
    # Check if the input PCM coded signal at 94dB calibration db (which is quite loud) is using the full range of the sint16 signal range
    # Check maximum across all chunks, see while loop
    print(f"Maximum PCM  amplitude: {data_dictionary['audio_data_max_pcm_value']}\n")
    if data_dictionary['audio_data_max_pcm_value'] > int(MAX_INT16*0.95) :
        wx.MessageBox(f"Maximum PCM 16bit amplitude: {data_dictionary['audio_data_max_pcm_value']}/{MAX_INT16}. Upper threshold: {int(MAX_INT16*0.95)} . Reduce GAIN on PreAmp !\n","Info", wx.OK | wx.ICON_INFORMATION)       
    elif data_dictionary['audio_data_max_pcm_value'] < int(MAX_INT16*0.8) :
        wx.MessageBox(f"Maximum PCM 16bit amplitude: {data_dictionary['audio_data_max_pcm_value']}/{MAX_INT16}. lower threshold: {int(MAX_INT16*0.8)}  . Increase GAIN on PreAmp !\n","Info", wx.OK | wx.ICON_INFORMATION)      
    else :
        wx.MessageBox(f"Maximum PCM 16bit amplitude: {data_dictionary['audio_data_max_pcm_value']}/{MAX_INT16}. PreAmp GAIN OK !\n","Info", wx.OK | wx.ICON_INFORMATION)
    
    # calculate average calibration factor across all chunks, see while loop
    calib_average = sum(calib_arr) / len(calib_arr)
    
    #store average as new calibration factor
    data_dictionary['system_calibration_factor_94db'] = calib_average
    print(f"Averaged data_dictionary['system_calibration_factor_94db']: {data_dictionary['system_calibration_factor_94db']}\n") 
       



def func_check_calibration(data_dictionary):
    global stream


    spl_error_arr           = np.array([])
    spl_error_arr_square    = np.array([])
    i = 0
   
    while i<CALIB_ITERATION_LENGTH :       
        i = i+1
        # Read a chunk of audio data
        data = stream.read(CHUNK)
      
        # Convert the audio data to a numpy array
        data_dictionary['audio_data']   = np.frombuffer(data, dtype=np.int16)
        
        #for output wave file creation, add to a list
        data_dictionary['frames'].append(data)
        
        # Calculate PCM to SPl ! 
        func_calc_SPL(data_dictionary)
        
        
        if data_dictionary['audio_data_pressurePa_spl'] > 0 : 
            
            # calculate error in dB SPL
            spl_error = 94 - data_dictionary['audio_data_pressurePa_spl']
            print(f"spl_error: {spl_error}\n")
            
            spl_error_square = np.power(spl_error, 2)
                                                
            # Store new value in array 
            spl_error_arr_square =  np.append(spl_error_arr_square, spl_error_square)
            spl_error_arr        =  np.append(spl_error_arr, spl_error)
            
        else :
            print("SPL or Pa equal to zero in this chunk. Check sound input !\n ") 

                          
    # calculate root mean square error 
    spl_error_average = np.sqrt(np.average(spl_error_square ))
    print(f"spl_error_average: {spl_error_average}\n")

       

def func_on_button_start_click(frame, data_dictionary):

    # Enable / Disable buttons
    frame.button_start.Disable()
    frame.button_stop.Enable()
    
    # Start recording process
    wx.CallAfter(frame.update_status,  "Start button pressed...\n")
    print(f"data_dictionary['is_recording']: {data_dictionary['is_recording']}\n")
    if not data_dictionary['is_recording']:
        data_dictionary['is_recording'] = True
        print(f"data_dictionary['is_recording']: {data_dictionary['is_recording']}\n")
        # Create a separate processto run the audio processing task
        # The processes are required to decouple the input stream reading from the GUI app 
        
        print("recording process will be created now.\n")
        if frame.recording_process is None or not frame.recording_process.is_alive():
            try:
                print("Creating and starting recording process...\n")

                frame.recording_process         = multiprocessing.Process(target=func_process_audio_input, args=(data_dictionary,frame.recording_status_queue,frame.recording_dba_queue,))
                print("recording process created\n")
            
                # Argument : frame. Required to create a process from inside the GUI that serves as longrunning
                # background task. And must refresh the GUI (frame instance) from inside the backround task via AppendText
       
                frame.recording_process.start()
                print("recording process started\n")
            
                # Update the status text after the task is complete (safely in the main process)  
                wx.CallAfter(frame.update_status,  "recording process started 2/2\n")
            
            except Exception as e:
                print(f"Error starting recording process : {e}\n")
            
        else:
            wx.CallAfter(frame.update_status,  "recording process is already running 1.\n")
            print("recording process is already running 1.\n")
    else:
        wx.CallAfter(frame.update_status,  "recording process is already running 2.\n")
        print("recording process is already running 2.\n")

    print(f"logging process will be started: {data_dictionary['is_logging']}\n")   
    # Start logging thread
    if not data_dictionary['is_logging']:
        data_dictionary['is_logging'] = True
        print(f"data_dictionary['is_logging']: {data_dictionary['is_logging']}\n")
        # Create a separate thread to run the process
        # The thread is required to decouple the input stream reading from the GUI app 

        if frame.logging_process is None or not frame.logging_process.is_alive():
            print("logging process will be created now.\n")
            
            # Argument : frame. Required to create a process from inside the GUI that serves as longrunning
            # background task. And must refresh the GUI (frame instance) from inside the backround task via callAfter
           
            frame.logging_process = multiprocessing.Process(target=func_saveWave_on_noise_event, args=(data_dictionary,frame.logging_status_queue,))
            print("logging process created\n")
 
            frame.logging_process.start()
            print("logging process started 2/2\n")
            
            # Update the status text after the task is complete (safely in the main process)
            wx.CallAfter(frame.update_status,  "logging process started 2/2\n")
        else:
            wx.CallAfter(frame.update_status,  "logging process is already running.\n")
            print("logging process is already running 1.\n")
    else:
        wx.CallAfter(frame.update_status,  "logging process is already running.\n")
        print("logging Thread is already running 2.\n")


def func_on_button_stop_click(frame, data_dictionary):

    
    # Enable / Disable buttons
    frame.button_start.Enable()
    frame.button_stop.Disable()
    
    print(f"data_dictionary['is_recording']: {data_dictionary['is_recording']}\n")
    data_dictionary['is_recording'] = False
    print(f"data_dictionary['is_recording']: {data_dictionary['is_recording']}\n")
    
    print(f"data_dictionary['is_logging']: {data_dictionary['is_logging']}\n")
    data_dictionary['is_logging'] = False
    print(f"data_dictionary['is_logging']: {data_dictionary['is_logging']}\n")
    
    if frame.recording_process is not None and frame.recording_process.is_alive():
        frame.recording_process.join()  # Wait for the process to finish gracefully
        wx.CallAfter(frame.update_status,  "Recording process stopped.\n")
    else:
        wx.CallAfter(frame.update_status,  "No Recording process is running.\n")
    
    if frame.logging_process is not None and frame.logging_process.is_alive():
        frame.logging_process.join()  # Wait for the process to finish gracefully
        wx.CallAfter(frame.update_status,  "logging process stopped.\n")
    else:
        wx.CallAfter(frame.update_status,  "No logging process is running.\n")
   

def func_on_button_runCalib_click(frame, data_dictionary):
    
    # Create a separate thread to run the process
    # The thread is required to decouple the input stream reading from the GUI app 
    frame.runCalib_thread   =threading.Thread(target=func_run_calibration, args=(data_dictionary,))
    frame.runCalib_thread.daemon = True
    frame.runCalib_thread.start()

       
       
def func_on_button_checkCalib_click(frame, data_dictionary):
    
    # Create a separate thread to run the process
    # The thread is required to decouple the input stream reading from the GUI app 
    frame.checkCalib_thread  =threading.Thread(target=func_check_calibration, args=(data_dictionary,))
    frame.checkCalib_thread.daemon = True
    frame.checkCalib_thread.start()


def func_on_button_exit_click(frame, data_dictionary):

    # Close the parent application and the GUI event loop (-> indicated by self)
    frame.Close()



def func_saveWave_on_noise_event(data_dictionary, recording_status_queue):


    max_spl_in_chunk            = 0
    max_spl_index_in_chunk      = 0
    max_spl_chunk_index         = 0
    start_chunk                 = 0
    stop_chunk                  = 0
    
    noise_frames                = [] # Wave data defined as byte list !
  
    #whole process will run in high priority mode, but lower than the Audio processing task
    p_func_saveWave_on_noise_event = psutil.Process(os.getpid())
    p_func_saveWave_on_noise_event.nice(psutil.HIGH_PRIORITY_CLASS)  
  
    print("logging process started 1/2\n")
    

    #Start with offset of wave output length
    time.sleep(WAVE_DT_SEC)
    
    while data_dictionary['is_logging']:  
        
             
        # check if events are detected and stored in the list
        if data_dictionary['chunk_noise_list_index'] is not None and data_dictionary['chunk_noise_list_index'].size > 0 :
            print("\nNew Recording event: \n")
            
            #identify max spl value and repsective chunk number 
            max_spl_in_chunk = np.max(data_dictionary['chunk_noise_list_spl'])
            max_spl_index_in_chunk = np.where(data_dictionary['chunk_noise_list_spl'] == max_spl_in_chunk)[0][0]
 
            #index of the chunk with maximum spl 
            max_spl_chunk_index = data_dictionary['chunk_noise_list_index'][max_spl_index_in_chunk]
        
            #extract the relevant noise frames + some delta
            start_chunk = max(int(max_spl_chunk_index-CHUNK_DNUM), 0)
            stop_chunk = int(max_spl_chunk_index+CHUNK_DNUM)

            
            # Wait for minimum time CHUNK_DNUM before saveing to add delta to the wave. 
            while data_dictionary['chunk_index_i'] < stop_chunk :
                pass  # This does nothing, just a placeholder
            
            print("Noise events will be written: \n")
            print(f"data_dictionary['chunk_noise_list_index'] : {data_dictionary['chunk_noise_list_index']}\n")
            print(f"data_dictionary['chunk_noise_list_spl'] : {data_dictionary['chunk_noise_list_spl']}\n")
            print(f"start_chunk is : {start_chunk}\n")
            print(f"stop_chunk is : {stop_chunk}\n")
            
            noise_frames = data_dictionary['frames'][start_chunk:stop_chunk]
        
            
            # Get the current local time
            current_time = datetime.datetime.now()
            # Round to the nearest second (remove microseconds)
            rounded_time = current_time.replace(microsecond=0)
        
            # construct file name with relevant data
            noise_file_name = f"{OUTPUT_NOISE_FILENAME}_DataID{max_spl_chunk_index}_dBA{round(max_spl_in_chunk,2)}_Date{rounded_time}.wav"
            # make it compatible with windows filename rules
            noise_file_name = noise_file_name.replace(' ', '_').replace(':', '-')
            print(noise_file_name)

            # Ensure the directory exists (optional, for better handling)
            os.makedirs(OUTPUT_FILE_DIRECTORY, exist_ok=True)

            # Combine directory and file name
            full_path = os.path.join(OUTPUT_FILE_DIRECTORY, noise_file_name)
            print(full_path)
                        
            # Write the recorded data to a WAV file

            with wave.open(full_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(data_dictionary['sample_size'])
                wf.setframerate(RATE)
                wf.writeframes(b''.join(noise_frames))
                print(f"Audio saved as {noise_file_name}\n")


            #erase noise event arrays
            data_dictionary['chunk_noise_list_index']   = np.array([], dtype=int)
            data_dictionary['chunk_noise_list_spl']     = np.array([])
            
            max_spl_in_chunk = 0
            max_spl_index_in_chunk = 0
            max_spl_chunk_index = 0
            start_chunk = 0
            stop_chunk = 0
            noise_frames                                = [] # Wave data defined as byte list !
        
            #remove chunks from wave output which are already treated. To free local resources.
            #data_dictionary['frames'] = data_dictionary['frames'][start_chunk:]
            
            recording_status_queue.put(f"DataID_{max_spl_chunk_index}__dB_{max_spl_in_chunk}__Horaire:_{rounded_time}.wav")  # Send message to main process




def func_on_saveWave_exit_click(data_dictionary):

    # Write the recorded data to a WAV file
    with wave.open(OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(data_dictionary['sample_size'])
        wf.setframerate(RATE)
        wf.writeframes(b''.join(data_dictionary['frames']))
        print(f"Audio saved as : {OUTPUT_FILENAME}")
        print(f"CHANNELS : {CHANNELS}")
        print(f"RATE : {RATE}")
        print(f"data_dictionary['sample_size']: {data_dictionary['sample_size']}")
        print(f"data_dictionary['frames']: {data_dictionary['frames']}")
        
    # open the recorded data to a WAV file
    with wave.open(OUTPUT_FILENAME, 'rb') as wav_file:
        sample_rate = wav_file.getframerate()  # Sample rate (samples per second)
        num_frames = wav_file.getnframes()
        num_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        num_frames = wav_file.getnframes()
           
        raw_data = wav_file.readframes(num_frames)
 
    
    # read audio data and apply weighting filter    
    # Convert raw byte data into a numpy array
    # For 16-bit audio (common for WAV), use np.int16
    data_dictionary['audio_data'] = np.frombuffer(raw_data, dtype=np.int16)
    # A-weighted audio data
    audio_data_weighted =   apply_a_weighting(data_dictionary)
    # Create a time axis for plotting
    time = np.linspace(0, num_frames / sample_rate, num_frames)
   
   
   
    # FFT  
    # Perform FFT on the audio signal
    fft_signal = np.fft.fft(data_dictionary['audio_data'])
    # Perform FFT on the audio signal
    fft_signal_weighted = np.fft.fft(audio_data_weighted)

    # Compute the corresponding frequencies
    frequencies = np.fft.fftfreq(len(fft_signal), d=1/RATE)
    
    # Get the magnitude of the FFT
    fft_magnitude = np.abs(fft_signal)
        # Get the magnitude of the FFT
    fft_magnitude_weighted = np.abs(fft_signal_weighted)
    
    # We only want the positive frequencies
    positive_frequencies = frequencies[:len(frequencies)//2]
    positive_magnitude = fft_magnitude[:len(frequencies)//2]
    positive_magnitude_weighted = fft_magnitude_weighted[:len(frequencies)//2]    
 
    '''
    # Plot the waveform
    plt.figure(figsize=(10, 6))
    plt.plot(time, data_dictionary['audio_data'], color='blue')
    plt.title('Raw audio time domain')
    plt.xlabel("Time [s]")
    plt.ylabel('PCM encoded audio [sint16]')
    plt.grid(True)
    plt.show()        
    '''
    
    # plot frames in time domaine
    # plot process frames in time domaine
    # plot frames FFT
    # plot process frames FFT
    
    # Create some data for plotting
    x1 = time
    y1 = data_dictionary['audio_data']

    x2 = time
    y2 = audio_data_weighted
    
    x3 = positive_frequencies
    y3 = positive_magnitude
    
    x4 = positive_frequencies
    y4 = positive_magnitude_weighted
    
    

 
    ymin_t = MAX_INT16 = np.iinfo(np.int16).min
    ymax_t = MAX_INT16 = np.iinfo(np.int16).max
    
    
    xmin_f = 0
    xmax_f = 16000
    
    ymin_f = 0
    ymax_f = max(max(positive_magnitude_weighted),max(positive_magnitude))


    
    # Create a 2x2 grid of subplots (2 rows, 2 columns)
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    
    # First plot (top-left)
    axs[0, 0].plot(x1, y1)
    axs[0, 0].set_title('Raw audio time domain')
    axs[0, 0].set_xlabel("Time [s]")
    axs[0, 0].set_ylabel('PCM encoded audio [sint16]')
    axs[0, 0].set_ylim(ymin_t, ymax_t)
    
    # Third plot (bottom-left)
    axs[1, 0].plot(x2, y2)
    axs[1, 0].set_title('A-weighted audio time domain')
    axs[1, 0].set_xlabel("Time [s]")
    axs[1, 0].set_ylabel('PCM encoded audio [sint16]')
    axs[1, 0].set_ylim(ymin_t, ymax_t)
    
    # Second plot (top-right)
    axs[0, 1].plot(x3, y3)
    axs[0, 1].set_title('Raw audio frequency domain')
    axs[0, 1].set_xlabel("Freq. [Hz]")
    axs[0, 1].set_ylabel('PCM encoded audio [sint16]')
    axs[0, 1].set_xlim(xmin_f, xmax_f)
    axs[0, 1].set_ylim(ymin_f, ymax_f)
    
    # Fourth plot (bottom-right) with a different x-axis range
    axs[1, 1].plot(x4, y4)
    axs[1, 1].set_title('A-weighted audio frequency domain')
    axs[1, 1].set_xlabel("Freq. [Hz]")
    axs[1, 1].set_ylabel('PCM encoded audio [sint16]')
    axs[1, 1].set_xlim(xmin_f, xmax_f)
    axs[1, 1].set_ylim(ymin_f, ymax_f)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Show the plots
    plt.show()
    
    
    
# GUI ############################################"
# Define the main application frame
class MyFrame(wx.Frame):
    def __init__(self, parent, title):
        super().__init__(parent, title=title, size=(350, 550))
        
        # Load settings from previous sessions
        global config

        global p
        global stream

        
        print(f"data_dictionary['_device_index'] [default]:  {data_dictionary['_device_index']}")
        print(f"data_dictionary['system_calibration_factor_94db'][default]:  {data_dictionary['system_calibration_factor_94db']}")
        
        
        # To persist Settings -> Initialise a config object.
        config  = configparser.ConfigParser()
        # Read Settings
        config.read('config.ini')
        
        # Access values from the config
        data_dictionary['_device_index']                     = config.getint('Settings', "_device_index")
        data_dictionary['system_calibration_factor_94db']    = config.getfloat('Settings', "system_calibration_factor_94db") 

        # getint is used for integers
        print(f"data_dictionary['_device_index'][loaded from config file]:  {data_dictionary['_device_index']}")
        print(f"data_dictionary['system_calibration_factor_94db'][loaded from config file]:  {data_dictionary['system_calibration_factor_94db']}")

        # Print the commit hash
        print("Commit version:", get_commit_version())
        
        # Initialize PyAudio
        p                   = pyaudio.PyAudio()
        # Open stream to read audio from the microphone
        stream              = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index = data_dictionary['_device_index'] ,
                    frames_per_buffer=CHUNK)
        data_dictionary['sample_size']     =   p.get_sample_size(FORMAT)
        print("Audio stream opened [Init]\n")
        
        
        # Create a panel inside the frame
        panel = wx.Panel(self)
   
        # To store reference to the thread or process (optional choice)
        self.recording_process          = None
        self.recording_status_queue     = multiprocessing.Queue()
        self.recording_dba_queue        = multiprocessing.Queue()

        self.logging_process            = None  
        self.logging_status_queue       = multiprocessing.Queue() 
        
        # Timer to  trigger queue reading
        self.timerDba                   = wx.Timer(self)  
        self.timerStatus                = wx.Timer(self)  



        self.runCalib_thread            = None
        self.checkCalib_thread          = None
            
        # Create a text box for user input
        self.text_ctrl   = wx.TextCtrl(panel, value=str(data_dictionary['_device_index']), pos=(290, 40), size=(30, 25))

        # Create a button on the panel
        self.button_checkDevices = wx.Button(panel, label="CheckDevices", pos=(200, 10))

        # Create a button on the panel
        self.button_setDevices = wx.Button(panel, label="SetDevices", pos=(200, 42))

        # Create a button on the panel
        self.button_start = wx.Button(panel, label="Start Measurement!", pos=(10, 10))

        # Create a button on the panel
        self.button_stop = wx.Button(panel, label="Stop Measurement!", pos=(10, 40))
        
        # Create a button on the panel
        self.button_runCalib = wx.Button(panel, label="Calibrate!", pos=(10, 80))
 
        # Create a button on the panel
        self.button_checkCalib = wx.Button(panel, label="Check Calibration!", pos=(10, 110))

        # Create a button on the panel
        self.button_saveWave = wx.Button(panel, label="Save and Plot Wave File [check signal]", pos=(10, 140))
  
  
        # Create text output field
        self.status_text = wx.StaticText(panel, label="Status:", pos=(10, 180), size=(300, 50))
        
        # Create text output field
        self.dba_display = wx.StaticText(panel, label="dbA:", pos=(10, 240), size=(33, 50))
        # Set a larger font
        font_large = wx.Font(18, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.dba_display.SetFont(font_large)
        
        # Create a button on the panel
        self.button_exit = wx.Button(panel, label="Close Application", pos=(190, 400))
 

        
        ##################################################################
        self.Bind(wx.EVT_TIMER, self.update_dba_display, self.timerDba)
        self.Bind(wx.EVT_TIMER, self.update_status, self.timerStatus)
        # Check every 10ms
        self.timerDba.Start(10)
        self.timerStatus.Start(10)
        
        # Bind the button click event to an event handler function
        self.button_checkDevices.Bind(wx.EVT_BUTTON, self.on_button_checkDevices_click)
 
        # Bind the button click event to an event handler function
        self.button_setDevices.Bind(wx.EVT_BUTTON, self.on_button_setDevices_click)
        
        # Bind the button click event to an event handler function
        self.button_start.Bind(wx.EVT_BUTTON, self.on_button_start_click)
        
        # Bind the button click event to an event handler function
        self.button_stop.Bind(wx.EVT_BUTTON, self.on_button_stop_click)
        
        # Bind the button click event to an event handler function
        self.button_runCalib.Bind(wx.EVT_BUTTON, self.on_button_runCalib_click)
 
        # Bind the button click event to an event handler function
        self.button_checkCalib.Bind(wx.EVT_BUTTON, self.on_button_checkCalib_click)
   
        # Bind the button click event to an event handler function
        self.button_exit.Bind(wx.EVT_BUTTON, self.on_button_exit_click)
   
        # Bind the button click event to an event handler function
        self.button_saveWave.Bind(wx.EVT_BUTTON, self.on_button_saveWave_click)
   
    
        # Show the window
        self.Show()


    def update_status(self, event):
        # Safely append text to the TextCtrl
        while not self.recording_status_queue.empty():
            msg = self.recording_status_queue.get()
            wx.CallAfter(self.status_text.SetLabel, msg)
        # Safely append text to the TextCtrl
        while not self.logging_status_queue.empty():
            msg = self.logging_status_queue.get()
            wx.CallAfter(self.status_text.SetLabel, msg)
        
    def update_dba_display(self, event):
        # Safely append text to the TextCtrl
        while not self.recording_dba_queue.empty():
            msg = self.recording_dba_queue.get()
            wx.CallAfter(self.dba_display.SetLabel, msg)


    def on_button_checkDevices_click(self, event):
        """Event handler function for the button click."""
        wx.MessageBox("Check Device List in console and set device number in the text box!!", "Info", wx.OK | wx.ICON_INFORMATION)
        # Call the check device function
        func_check_devices(data_dictionary)

            
    def on_button_setDevices_click(self, event):  
        # Call function
        func_on_button_setDevices_click(self, data_dictionary)


    def on_button_start_click(self, event):
        # Call function
        func_on_button_start_click(self, data_dictionary)

            
    def on_button_stop_click(self, event):       
        # Call function
        func_on_button_stop_click(self, data_dictionary)


    def on_button_runCalib_click(self, event):
        # Call function
        func_on_button_runCalib_click(self, data_dictionary)


    def on_button_checkCalib_click(self, event):
        # Call function
        func_on_button_checkCalib_click(self, data_dictionary)
        

    def on_button_exit_click(self, event):
        # Call function
        # need self arguments to know which class instance to close
        func_on_button_exit_click(self, data_dictionary)
        
        
    def on_button_saveWave_click(self, event):
        # Call function
        # need self arguments to know which class instance to close
        func_on_saveWave_exit_click(data_dictionary)
        
    


class MyApp(wx.App):
    def OnInit(self):
        self.frame = MyFrame(None, title="Task Scheduler GUI")      
        return True

# Main GUI application loop ###################################
if __name__ == "__main__":
    
    # required to start new process under windows systems
    multiprocessing.set_start_method("spawn", force=True)  # optional but clear
    
    # Defined in shared memory to allow several processes to work on the data : multiprocessing."
    # values must be accessed via .value postfix ! 
    # arrays do not require to have a postfix.

    # DataDictionary
    manager = multiprocessing.Manager()

    data_dictionary = manager.dict(
        
        # Device list user input 
        _device_index                           = 0,
        is_recording                            = False,
        is_logging                              = False,
        
        # Microphone calibration factor to obtain 94dB at 1kHz. Default value 1
        system_calibration_factor_94db          = 1.0, 
        
        # audio data extraceted from chunk in np format.
        audio_data                          = np.array([]),
        a_weighted_signal                   = np.array([]),
        audio_data_pcm_abs                  = np.array([]),
        audio_data_mV                       = np.array([]),
        audio_data_mV_calib                 = np.array([]),
        audio_data_pressurePa               = np.array([]),
        audio_data_pressurePa_square        = np.array([]),  
        
        audio_data_pressurePa_squareMean    = 0.0,
        audio_data_pressurePa_rms           = 0.0,
        audio_data_pressurePa_rms_calib     = 0.0,
        audio_data_pressurePa_spl           = 0.0,
        audio_data_max_pcm_value            = 0.0,

        #for output wave file creation / all chunks, complete measurement
        # The wave raw data must be handled as byte list. Not numpy array !
        frames                              = [],

        chunk_index_i                       = 0,
        chunk_noise_list_index              = np.array([], dtype=int),
        chunk_noise_list_spl                = np.array([]),

        sample_size                         = 2
    )
      

    app = MyApp()

    # Wrap the main event loop in a try-except block
    try:

        
        app.MainLoop()
        
        
    except Exception as e:
        print(f"An error occurred: {e}")
        # You can add additional cleanup or logging here if needed
    
    finally:
        # Add settings to the config
        # Add USB device index
        # Add calibration value
        config.set("Settings","_device_index", f"{data_dictionary['_device_index']}")
        config.set("Settings","system_calibration_factor_94db", f"{data_dictionary['system_calibration_factor_94db']}") 
        print(f"data_dictionary['_device_index'] [saved]:  {data_dictionary['_device_index']}")
        print(f"data_dictionary['system_calibration_factor_94db'][saved]:  {data_dictionary['system_calibration_factor_94db']}")
    
        # Save the program state (configuration) to a file
        with open('config.ini', 'w') as configfile:
            config.write(configfile)
        

    
        # Close the stream and terminate PyAudio
        stream.stop_stream()
        stream.close()
    
        # Close audio interface
        p.terminate()
        data_dictionary['frames']                               = [] # Wave data defined as byte list !
        
        data_dictionary['chunk_index_i']                        = 0    # counter of processed chunks
        data_dictionary['chunk_noise_list_index']               = np.array([], dtype=int),
        data_dictionary['chunk_noise_list_spl']                 = np.array([])
        data_dictionary['is_recording']                         = False
        data_dictionary['is_logging']                           = False
        data_dictionary['audio_data']                           = np.array([])
        data_dictionary['audio_data_pcm_abs']                   = np.array([])
        data_dictionary['audio_data_mV']                        = np.array([])
        data_dictionary['audio_data_mV_calib']                  = np.array([])
        data_dictionary['audio_data_pressurePa']                = np.array([])
        data_dictionary['audio_data_pressurePa_square']         = np.array([])
        data_dictionary['audio_data_pressurePa_squareMean']     = 0
        data_dictionary['audio_data_pressurePa_rms']            = 0
        data_dictionary['audio_data_pressurePa_rms_calib']      = 0
        data_dictionary['audio_data_pressurePa_spl']            = 0
        data_dictionary['audio_data_max_pcm_value']             = 0
        
        # Terminate running process, in case they are not closed already.
        #if recording_process.is_alive() :
        #    print("Timeout reached, terminating process...")
        #    app.recording.terminate()
        #   app.recording.join()
    
        #if logging_process.is_alive() :
        #   print("Timeout reached, terminating process...")
        #   app.logging.terminate()
        #   app.logging.join()
    
        print("Application has finished.")

# visual thead debugging
if DEBUG == 1:
    tracer.stop()  # Stop VizTracer
    tracer.save("Python_AudioLogger_JSON_LogFile.json")  # Save trace data to a file

##############################################################################