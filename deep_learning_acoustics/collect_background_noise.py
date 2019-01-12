'''
Script to collect 5 seconds of background noise

'''
import sounddevice as sd
import soundfile as sf



def record_background_noise(sec):
    sr = 16000
    print("Now recording for {} seconds...".format(sec))
    background = sd.rec(int(sec*sr),samplerate=sr,channels=1)
    sd.wait()
    return background, sr

if __name__=="__main__":
    
    #conn = sqlite3.connect("male_female_speech.db")
    #c = conn.cursor()
    
    #record background noise? w computer?
    print("Are you ready to record background noise? (Y/N): ")
    ready = input()
    
    if isinstance(ready, str) and "y" in ready.lower():
        noise_samples, sr = record_background_noise(5)
        
        filename = "./data/background_noise.wav"
        sf.write(filename,noise_samples,sr)
        print("Background noise successfully saved at the following location:\n\n{}".format(filename))
