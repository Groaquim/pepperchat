import dotenv
dotenv.load_dotenv()
import threading, time, json, os
from datetime import datetime

import oai_dialogue.pepper_command as pepper_command
from oai_dialogue.speech_to_text.pepper_text_speaker import PepperTextSpeaker
from oai_dialogue.speech_to_text import pcm_utils, subtitles, silerovad
from oai_dialogue.speech_to_text.groq_transcriber import GroqTranscriber
from oai_dialogue.speech_to_text.transcriber import TranscriberResult
from oaichat.openaichat import OaiChat
import oai_dialogue.comm as comm

dotenv.load_dotenv(os.getenv('DIALOGUE_ENV', 'dialogue.env'))

def main():
    command_sender = pepper_command.CommandSender()
    
    def init_robot():
        command_sender.send(pepper_command.ConfigSpeech(language=os.getenv('LANGUAGE', 'French'), animated=True))
        command_sender.send(pepper_command.ConfigAudio(output_volume=70))
        wifi_ssid = os.getenv('TABLET_WIFI_SSID')
        if wifi_ssid:
            print("Configuring tablet wifi:", wifi_ssid)
            command_sender.send(pepper_command.ConfigTabletWifi(
                ssid=wifi_ssid,
                pwd=os.getenv('TABLET_WIFI_PWD', ''),
                security_type=os.getenv('TABLET_WIFI_SECURITY', 'wpa')
            ))
    
    init_robot()
    pts = PepperTextSpeaker(
        command_sender=command_sender,
        subtitle_server=subtitles.SubtitleServer()
    )
    pts.push_text(os.getenv('SAY', ''))

    # Initialisation du cerveau (Groq Llama 3.3) et des oreilles (Groq Whisper)
    chat = OaiChat(user="pepper_jpo", prompt=os.getenv('OPENAI_PROMPTFILE'))
    transcriber = GroqTranscriber(language="fr")
    is_bot_busy = False

    def on_robot_state_change(state: comm.RobotState):
        print(state)
        if state.just_started:
            init_robot()
        pts.on_robot_state_change(state)

    robot_state_listener = comm.RobotStateListener(on_robot_state_change)

    logdir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, datetime.now().strftime('dialogue_%Y-%m-%d_%H%M%S.log'))

    def on_transcript(result: TranscriberResult):
        nonlocal is_bot_busy
        if robot_state_listener.state.head_touched:
            return
        if not result.is_final or not result.transcripts:
            return
        user_text = result.transcripts[0].transcript.strip()
        if not user_text:
            return

        print("USER:", user_text)
        is_bot_busy = True
        try:
            response = chat.respond(user_text)
            reply_text = response.getText().strip()
            print("PEPPER:", reply_text)
            pts.push_text(reply_text)

            with open(logfile, 'a', encoding='utf-8') as f:
                json.dump({
                    'time': datetime.now().isoformat(),
                    'user': user_text,
                    'response': reply_text
                }, f, ensure_ascii=False)
                f.write(',\n')
        finally:
            is_bot_busy = False

    transcriber.add_transcript_callback(on_transcript)

    speech_start_time = 0.0
    def on_speech_end(_sample_rate, _channel_cnt, _pcm16_chunks):
        duration = time.time() - speech_start_time if speech_start_time else 1.0
        transcriber.flush_and_transcribe(speech_start_time, duration)

    silero = silerovad.SileroVad(
        threshold=0.5,
        head_millis=500,
        speech_stream_callback=transcriber.push_pcm16_frames,
        speech_end_callback=on_speech_end
    )

    def mic_callback(sample_rate, channel_cnt, frames):
      nonlocal speech_start_time
      # Coupe le micro si PepperChat est arrêté sur la tablette (head_touched == True),
      # ou pendant que Pepper réfléchit/parle
      if (
          robot_state_listener.state.head_touched
          or is_bot_busy
          or robot_state_listener.state.talking
      ):
        return
      if not transcriber._buffer:
        speech_start_time = time.time()
      silero.push_pcm16_frames(sample_rate, channel_cnt, frames)

    pcm_utils.listen_on_local_mic(48000, [mic_callback], channel_cnt=1)

if __name__ == "__main__":
    main()