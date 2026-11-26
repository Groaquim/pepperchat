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
import socket
import numpy as np

dotenv.load_dotenv(os.getenv('DIALOGUE_ENV', 'dialogue.env'))


def main():
  command_sender = pepper_command.CommandSender()

  def init_robot():
    command_sender.send(
        pepper_command.ConfigSpeech(
            language=os.getenv('LANGUAGE', 'French'), animated=True
        )
    )
    command_sender.send(pepper_command.ConfigAudio(output_volume=70))
    wifi_ssid = os.getenv('TABLET_WIFI_SSID')
    if wifi_ssid:
      print('Configuring tablet wifi:', wifi_ssid)
      command_sender.send(
          pepper_command.ConfigTabletWifi(
              ssid=wifi_ssid,
              pwd=os.getenv('TABLET_WIFI_PWD', ''),
              security_type=os.getenv('TABLET_WIFI_SECURITY', 'wpa'),
          )
      )

  init_robot()
  pts = PepperTextSpeaker(
      command_sender=command_sender, subtitle_server=subtitles.SubtitleServer()
  )
  # pts.push_text(os.getenv('SAY', ''))  # Désactivé pour attendre le bouton sur la tablette

  # Initialisation du cerveau (Groq Llama 3.3) et des oreilles (Groq Whisper)
  chat = OaiChat(user='pepper_jpo', prompt=os.getenv('OPENAI_PROMPTFILE'))
  transcriber = GroqTranscriber(language='fr')
  is_bot_busy = False

  def on_robot_state_change(state: comm.RobotState):
    print(state)
    if state.just_started:
      init_robot()
    pts.on_robot_state_change(state)

  robot_state_listener = comm.RobotStateListener(on_robot_state_change)

  logdir = os.path.join(os.path.dirname(__file__), 'logs')
  os.makedirs(logdir, exist_ok=True)
  logfile = os.path.join(
      logdir, datetime.now().strftime('dialogue_%Y-%m-%d_%H%M%S.log')
  )

  def on_transcript(result: TranscriberResult):
    nonlocal is_bot_busy
    # Ignore toute transcription si le bouton tablette n'est pas actif
    if not robot_state_listener.state.pepperchat_active:
      return
    if not result.is_final or not result.transcripts:
      return
    user_text = result.transcripts[0].transcript.strip()
    if not user_text:
      return

    # Filtre les hallucinations classiques de Whisper sur le silence ou les bruits moteurs
    hallucinations = {
        'merci',
        'merci.',
        'merci !',
    }
    if user_text.lower() in hallucinations:
      return

    print('USER:', user_text)
    is_bot_busy = True
    try:
      response = chat.respond(user_text)
      reply_text = response.getText().strip()
      if "[IGNORE]" in reply_text:
        print("PEPPER (ignoré - discussion externe) :", user_text)
        return
      print('PEPPER:', reply_text)
      pts.push_text(reply_text)

      with open(logfile, 'a', encoding='utf-8') as f:
        json.dump(
            {
                'time': datetime.now().isoformat(),
                'user': user_text,
                'response': reply_text,
            },
            f,
            ensure_ascii=False,
        )
        f.write(',\n')
    finally:
      is_bot_busy = False

  transcriber.add_transcript_callback(on_transcript)

  speech_start_time = 0.0

  def on_speech_end(_sample_rate, _channel_cnt, _pcm16_chunks):
    if not robot_state_listener.state.pepperchat_active:
      if hasattr(transcriber, "_buffer"):
        transcriber._buffer.clear()
      return
    duration = time.time() - speech_start_time if speech_start_time else 1.0
    transcriber.flush_and_transcribe(speech_start_time, duration)

  silero = silerovad.SileroVad(
      threshold=0.5,
      head_millis=500,
      speech_stream_callback=transcriber.push_pcm16_frames,
      speech_end_callback=on_speech_end,
  )

  def mic_callback(sample_rate, channel_cnt, frames):
    nonlocal speech_start_time
    if not robot_state_listener.state.pepperchat_active:
      # Vide tout résidu audio accumulé avant la coupure
      if hasattr(transcriber, "_buffer"):
        transcriber._buffer.clear()
      return

    if is_bot_busy or robot_state_listener.state.talking:
      return

    if not transcriber._buffer:
      speech_start_time = time.time()
    silero.push_pcm16_frames(sample_rate, channel_cnt, frames)

  def listen_on_pepper_mic(callbacks, port=50008):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", port))
    print(f"[Audio] En écoute du micro de Pepper sur le port UDP {port}...")
    first_packet = True
    while True:
      data, addr = sock.recvfrom(16384)
      if first_packet:
        print(
            f"[Audio PC] Premier paquet reçu de {addr}, taille: {len(data)}"
            " octets"
        )
        first_packet = False
      frames = np.frombuffer(data, dtype=np.int16)
      for cb in callbacks:
        cb(16000, 1, frames)

  listen_on_pepper_mic([mic_callback])


if __name__ == '__main__':
  main()