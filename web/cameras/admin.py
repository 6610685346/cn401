from django.contrib import admin

from .models import Camera


@admin.register(Camera)
class CameraAdmin(admin.ModelAdmin):
    list_display = ("name", "masked_source", "long_side", "should_run")
    list_filter = ("should_run",)


from .models import EvalVideo  # noqa: E402


@admin.register(EvalVideo)
class EvalVideoAdmin(admin.ModelAdmin):
    list_display = ("name", "split", "status", "created_at")
    list_filter = ("split", "status")
