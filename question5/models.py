from django.db import models

class Student(models.Model):
    first_name = models.CharField(max_length=250)
    last_name = models.CharField(max_length=250)
    student_number = models.CharField(max_length=7, unique=True, primary_key=True)
    email_address = models.EmailField()
    grad_student = models.BooleanField()
    runs_used_so_far = models.IntegerField(default=0)
    # Category A, B, C, or D: determines which function the student sees.
    category = models.CharField(max_length=1)

    def __unicode__(self):
        return u'%s, %d, %s ' % (self.first_name, self.runs_used_so_far,
                                 self.category)

class Experiment(models.Model):
    student = models.ForeignKey(Student)
    factor_A = models.FloatField()
    response = models.FloatField()
    response_noisy = models.FloatField()
    date_time = models.DateTimeField(auto_now=False, auto_now_add=True)

    def __unicode__(self):
        return u'%s, %s, %s, %s' % (str(self.student), str(self.factor_A),
                                str(self.response_noisy), str(self.date_time))


class Token(models.Model):
    student = models.ForeignKey(Student)
    token_string = models.CharField(max_length=250)
    active = models.BooleanField(default=True)

    def __unicode__(self):
        return u'%s, %s, %s' % (str(self.student), self.token_string, str(self.active))

