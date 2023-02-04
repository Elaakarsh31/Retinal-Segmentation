from django import forms
from basic.models import ImageField

class ImageForm(forms.ModelForm):
    # image = forms.ImageField(label="image", widget=forms.ClearableFileInput(attrs={"mutiple": True}))
    class Meta:
        model = ImageField
        fields = ("image",)