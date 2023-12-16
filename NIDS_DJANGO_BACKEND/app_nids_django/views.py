from django.views.decorators.csrf import csrf_protect
from django.shortcuts import render
from django.http import HttpResponse,request,response
from django.template import loader,RequestContext
import numpy as np
import joblib as jb


model=jb.load('model.pkl')

def index(request):
    return render(request,'index.html')

def predict(request):
    if request.method=='POST':
        int_features = [float(request.POST.get('attack', 0)),
                        float(request.POST.get('count', 0)),
                        float(request.POST.get('dst_host_diff_srv_rate', 0)),
                        float(request.POST.get('dst_host_same_src_port_rate', 0)),
                        float(request.POST.get('dst_host_same_srv_rate', 0)),
                        float(request.POST.get('dst_host_srv_count', 0)),
                        float(request.POST.get('flag', 0)),
                        float(request.POST.get('last_flag', 0)),
                        float(request.POST.get('logged_in', 0)),
                        float(request.POST.get('same_srv_rate', 0)),
                        float(request.POST.get('serror_rate', 0)),
                        float(request.POST.get('service_http', 0))]

        if int_features[0]==0:
            f_features=[0,0,0]+int_features[1:]
        elif int_features[0]==1:
            f_features=[1,0,0]+int_features[1:]
        elif int_features[0]==2:
            f_features=[0,1,0]+int_features[1:]
        else:
            f_features=[0,0,1]+int_features[1:]

        if f_features[6]==0:
            fn_features=f_features[:6]+[0,0]+f_features[7:]
        elif f_features[6]==1:
            fn_features=f_features[:6]+[1,0]+f_features[7:]
        else:
            fn_features=f_features[:6]+[0,1]+f_features[7:]

        final_features = [np.array(fn_features)]
        predict = model.predict(final_features)

        if predict==0:
            output='Normal'
        elif predict==1:
            output='DOS'
        elif predict==2:
            output='PROBE'
        elif predict==3:
            output='R2L'
        else:
            output='U2R'

        context={'prediction':output}

        return render(request, 'index.html', context)

    