from django import forms

from .models import Camera


class CameraForm(forms.ModelForm):
    class Meta:
        model = Camera
        fields = ["name", "source", "rotate", "long_side",
                  "enable_face_recognition", "enable_vlm_par", "enable_direction", "should_run"]
        widgets = {"source": forms.TextInput(attrs={"placeholder": "rtsp://user:pass@192.168.1.13:554/media/video2",
                                                    "autocomplete": "off", "spellcheck": "false"})}

    def clean_source(self):
        return self.cleaned_data["source"].strip()
