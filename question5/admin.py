from django.contrib import admin
from takehome.question5.models import Student, Experiment, Token

class StudentAdmin(admin.ModelAdmin):
    list_display = ('first_name', 'category', 'runs_used_so_far', 'student_number',)
    search_fields = ('student_number', 'first_name', 'category',)
    list_filter = ('student_number', 'category', )

class ExperimentAdmin(admin.ModelAdmin):
    list_per_page = 2000
    list_display = ('student', 'factor_A', 'response', 'response_noisy', 'date_time')


admin.site.register(Student, StudentAdmin)
admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Token)
