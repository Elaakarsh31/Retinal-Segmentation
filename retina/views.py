from django.views.generic import TemplateView, CreateView, FormView
from django.http import HttpResponseRedirect, HttpResponse
from basic import forms
import os, glob
import sys
# sys.path.insert(0,'C://Users/lenovo//Desktop//VSCODE//django//retina_project//machine')
from machine.module import predict
from basic.models import ImageField
from django.shortcuts import render, redirect

IMG_PATH = glob.glob("C:/Users/lenovo/Desktop/VSCODE/django/retina_project/retina/machine/data/*")

# class Home(CreateView):
#     form_class = forms.ImageForm
#     template_name = 'index.html'
#     success_url = reverse_lazy("home")

#     def post(self, request):
#         self.form_class = self.form_class(request.FILES)
    
#     def form_valid(self):
#         self.form_class.save()
#         return super().form_valid(self.form)

def Home(request):
    form = forms.ImageForm(request.POST, request.FILES)
    if request.method == 'POST':
        if form.is_valid():
            image = form.save(commit=False)
            form.save()
            return redirect('/view/')
    return render(request, "index.html", {'form': form})
    

def Result(request):
    
    image = ImageField.objects.all()
    # predict(IMG_PATH[0])
    return render(request, "view.html")

def delete(request):
    image = ImageField.objects.all()
    image.delete()
    files = glob.glob("images/*")
    for f in files:
        os.remove(f)
    results = glob.glob("static/results/*")
    for r in results:
        os.remove(r)
    return redirect('/home/')


    