from django.conf.urls.defaults import *
from takehome.question5 import views
from django.conf import settings

urlpatterns = patterns('',
    (r'^$', views.sign_in),
    (r'^/$', views.sign_in),
    (r'^not-registered$', views.not_registered_student),
    (r'^run-experiment-(.*)/', views.run_experiment),
    (r'^download-values-(.*)/', views.download_values),
)

if settings.DEBUG:

    urlpatterns += patterns('',
        # For example, files under media/file.jpg will be retrieved from
        # settings.MEDIA_ROOT/file.jpg
        (r'^media/(?P<path>.*)$', 'django.views.static.serve',
         {'document_root': settings.MEDIA_ROOT, 'show_indexes': True}),
        )