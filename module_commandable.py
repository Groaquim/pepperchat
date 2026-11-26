# -*- coding: utf-8 -*-

ROBOT_PORT = 9559  # Robot
ROBOT_IP = "10.138.204.219"  # Pepper default

import codecs
from optparse import OptionParser
import os, sys
import threading
import time
import traceback
import naoqi
from naoqi import ALProxy
from oai_dialogue.comm import RobotStateReporter
import oai_dialogue.pepper_command as pepper_command


def start_thread(target):
  t = threading.Thread(target=target)
  t.setDaemon(True)
  t.start()
  return t


def encode(s):
  return codecs.encode(s, "utf-8", "ignore")


class ModuleCommandable(naoqi.ALModule):

  def __init__(self, robot_ip, robot_port):
    naoqi.ALModule.__init__(self, mod_name)
    self.memory = ALProxy("ALMemory", robot_ip, robot_port)
    self.posture = ALProxy("ALRobotPosture", robot_ip, robot_port)
    self.autonomous_life = ALProxy("ALAutonomousLife")
    self.tts = ALProxy("ALTextToSpeech", robot_ip, robot_port)
    self.aup = ALProxy("ALAnimatedSpeech", robot_ip, robot_port)
    self.tablet = ALProxy("ALTabletService", robot_ip, robot_port)
    self.audio = ALProxy("ALAudioDevice")
    self.awareness = ALProxy("ALBasicAwareness", robot_ip, robot_port)
    self.bg_movement = ALProxy("ALBackgroundMovement", robot_ip, robot_port)

    # Active la respiration naturelle et le suivi du regard même sans Autonomous Life
    self.bg_movement.setEnabled(True)
    self.awareness.setStimulusDetectionEnabled("People", True)
    self.awareness.setStimulusDetectionEnabled("Sound", True)
    self.awareness.setEngagementMode("FullyEngaged")
    self.awareness.startAwareness()

    self.state_reporter = RobotStateReporter()
    self.command_receiver = pepper_command.CommandReceiver(self.on_command)

    # État d'activation piloté par le bouton HTML sur la tablette
    self.pepperchat_active = False
    try:
      self.memory.insertData("ENSC/PepperChat/Active", False)
    except Exception:
      pass

    # Abonnement aux capteurs tactiles et au bouton de la tablette
    self.memory.subscribeToEvent(
        "TouchChanged", self.getName(), "on_touch_changed"
    )
    self.memory.subscribeToEvent(
        "ENSC/PepperChat/Active", self.getName(), "on_pepperchat_toggled"
    )

    self.touched = False
    self.running = True
    self.pending_speech = ""
    self.tablet_wifi_config = None
    self.speech_config = None

    def speech_loop():
      last_state = None
      last_sync = 0
      while self.running:
        try:
          try:
            val = self.memory.getData("ENSC/PepperChat/Active")
            self.pepperchat_active = bool(val)
          except Exception:
            self.pepperchat_active = False

          now = time.time()
          if self.pepperchat_active != last_state or (now - last_sync > 1.0):
            if self.pepperchat_active != last_state:
              print(
                  "[Choregraphe] État PepperChat :", self.pepperchat_active
              )
              if not self.pepperchat_active:
                self.stop_talking()
              last_state = self.pepperchat_active
            self.state_reporter.state.pepperchat_active = (
                self.pepperchat_active
            )
            self.state_reporter.report_cur_state()
            last_sync = now

          if self.pending_speech and self.pepperchat_active:
            data = encode(self.pending_speech)
            self.pending_speech = ""
            self.state_reporter.report_talking(True)
            if self.speech_config and self.speech_config.animated:
              self.aup.say(data)
            else:
              self.tts.say(data)
          elif self.pending_speech and not self.pepperchat_active:
            self.pending_speech = ""

          self.state_reporter.report_talking(False)
        except:
          traceback.print_exc()
        time.sleep(0.05)

    start_thread(speech_loop)

  def on_pepperchat_toggled(self, key, value, message):
    """Déclenché instantanément quand on clique sur le bouton HTML de la tablette"""
    self.pepperchat_active = bool(value)
    print(
        "[Choregraphe] Bouton PepperChat -> Active :", self.pepperchat_active
    )
    if not self.pepperchat_active:
      self.stop_talking()

  def on_command(self, command):
    try:
      if isinstance(command, pepper_command.Say):
        if self.pepperchat_active:
          self.stop_talking()
          self.pending_speech = command.text

      elif isinstance(command, pepper_command.ConfigSpeech):
        # DÉSACTIVÉ : On laisse Choregraphe gérer ALAutonomousLife et la posture
        # pour ne pas couper l'activité tablette en cours !
        if (
            not self.speech_config
            or self.speech_config.language != command.language
        ):
          print(command.language, self.tts.getLanguage())
          self.tts.setLanguage(command.language.encode("utf-8"))
        self.speech_config = command

      elif isinstance(command, pepper_command.ConfigAudio):
        self.audio.setOutputVolume(command.output_volume)

      elif isinstance(command, pepper_command.ConfigTabletWifi):
        self.tablet_wifi_config = command

      elif isinstance(command, pepper_command.OpenUrlOnTablet):
        # DÉSACTIVÉ : On garde notre page html/index.html affichée par Choregraphe
        print("[Info] OpenUrlOnTablet ignoré (géré par Choregraphe) :", command.url)
    except:
      traceback.print_exc()

  def on_touch_changed(self, name, touches):
    touched = any([len(touch) > 1 and touch[1] == True for touch in touches])
    self.state_reporter.report_head_touched(touched)
    if touched:
      self.stop_talking()

  def __del__(self):
    self.stop()

  def stop(self):
    try:
      self.memory.unsubscribeToEvent("TouchChanged", self.getName())
      self.memory.unsubscribeToEvent("ENSC/PepperChat/Active", self.getName())
    except Exception:
      pass
    self.running = False
    self.stop_talking()

  def version(self):
    return "1.0"

  def stop_talking(self):
    self.pending_speech = ""
    self.tts.stopAll()
    self.state_reporter.report_talking(False)


def main():
  parser = OptionParser()
  parser.add_option(
      "--pip", help="Parent broker port. The IP address or your robot", dest="pip"
  )
  parser.add_option(
      "--pport",
      help="Parent broker port. The port NAOqi is listening to",
      dest="pport",
      type="int",
  )
  parser.set_defaults(pip=ROBOT_IP, pport=ROBOT_PORT)

  (opts, args_) = parser.parse_args()
  pip = opts.pip
  pport = opts.pport

  myBroker = naoqi.ALBroker("myBroker", "0.0.0.0", 0, pip, pport)

  global mod_name
  mod_name = "modcomm"
  global modcomm
  modcomm = ModuleCommandable(pip, pport)

  try:
    while True:
      time.sleep(1)
  except KeyboardInterrupt:
    pass
  myBroker.shutdown()


if __name__ == "__main__":
  main()