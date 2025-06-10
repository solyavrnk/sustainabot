import os
import json

class LogWriter:
    def __init__(self):
        self.conversation_logfile = "conversation.jsonp"
        if os.path.exists(self.conversation_logfile):
            os.remove(self.conversation_logfile)

    def make_json_safe(self, value):
        if type(value) == list:
            return [self.make_json_safe(x) for x in value]
        elif type(value) == dict:
            return {key: self.make_json_safe(value) for key, value in value.items()}
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def write(self, log_message):
        # Add slots info if available from consultant
        if "consultant" in log_message:
            consultant = log_message["consultant"]
            slots = consultant.slots.slots if hasattr(consultant, "slots") else {}
            log_message["slots"] = {k: v if v is not None else "" for k, v in slots.items()}
            del log_message["consultant"]
        elif "slots" not in log_message and hasattr(self, "consultant") and hasattr(self.consultant, "slots"):
            slots = self.consultant.slots.slots
            log_message["slots"] = {k: v if v is not None else "" for k, v in slots.items()}
        # ...otherwise, do not add slots

        with open(self.conversation_logfile, "a") as f:
            f.write(json.dumps(self.make_json_safe(log_message), ensure_ascii=False, indent=2))
            f.write("\n")
