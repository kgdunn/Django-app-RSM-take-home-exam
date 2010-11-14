from django.conf.urls.defaults import *
from takehome.question5 import views

urlpatterns = patterns('',
    (r'^$', views.sign_in),
    (r'^/$', views.sign_in),
    (r'^not-registered$', views.not_registered_student),
    (r'^run-experiment-(.*)/', views.run_experiment),
    (r'^download-values-(.*)/', views.download_values),

)


