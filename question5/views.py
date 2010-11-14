from django.template import loader, Context
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render_to_response
from django.contrib.auth.decorators import login_required
from django.contrib import auth
from django.contrib.auth.models import User

import numpy as np
import hashlib as hashlib

# Command line use
#import sys, os
#sys.path.extend(['/var/django/projects/'])
#os.environ['DJANGO_SETTINGS_MODULE'] = 'rsm.settings'

from takehome.question5.models import Student, Token, Experiment

# Improvements for next time
#
# * sign in, generate a token, email that token: they have to access all their
#   experiments from that tokenized page: prevents other students look at each
#   others experiments.
# * factors is a variable, so you can have unlimited number of factors
# * PDF download link (using ReportLab to create the PDF)


# Logging
LOG_FILENAME = 'logfile.log'
import logging.handlers
my_logger = logging.getLogger('MyLogger')
my_logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler(LOG_FILENAME, maxBytes=2000000, backupCount=5)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
my_logger.addHandler(fh)
my_logger.debug('A new call to the views.py file')

# TO DO LIST
# Put base-line hard coded into a variable

# Settings
token_length = 10
max_experiments_allowed = 8
show_result = True

def generate_random_token():
    import random
    return ''.join([random.choice('ABCEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz2345689') for i in range(token_length)])

def generate_result(student_number, factors, bias):
    """
    Generates an experimental result for the student.
    The first error added is always the same, the second error added is proportional to the number of runs.
    """
    x1s, x2s, x3s = factors
    my_logger.debug('Generating a new experimental result for student number ' + student_number)

    x1off = 400.0
    x1scale = 5.0
    x = np.array((np.array(x1s) - x1off)/(x1scale+0.0))

    x2off = 16.0
    x2scale = 2.0
    y = np.array((np.array(x2s) - x2off)/(x2scale+0.0))

    r = np.sqrt(0.9*x**2 + y**2 + 0.2*x*y)
    z = np.sin(x**2 + x*y + y**2) / (0.1 + r**2) + (x**2 + x + y**2) * 0.5 * np.exp(1-r + np.sign(x) * np.abs(x)**0.2)
    z = z*100.0

    np.random.seed(int(student_number))
    noise_sd = 0.03 * np.abs(np.max(z)) + 0.001
    z_noisy = z + np.random.normal(loc=0.0, scale=noise_sd) + np.random.normal(loc=0.0, scale=noise_sd, size=bias)[-1]
    return (z, z_noisy)


def plot_results(expts):
    """Plots the data into a PNG figure file"""
    factor_A = []
    factor_B = []
    factor_C = []
    response = []
    for entry in expts:
        factor_A.append(entry['factor_A'])
        factor_B.append(entry['factor_B'])
        factor_C.append(entry['factor_C'])
        response.append(entry['response'])

    data_string = str(factor_A) + str(factor_B) + str(factor_C) + str(response)
    filename = hashlib.md5(data_string).hexdigest() + '.png'
    full_filename = '/var/django-projects/media/rsm/' + filename

    # Baseline and limits
    baseline_xA = 440
    baseline_xB = 20
    baseline_xC = 'Absent'
    limits_A = [390, 480]
    limits_B = [5, 30]

    # Start and end point of the linear constraint region
    constraint_a = [440, 30]
    constraint_b = [480, 10]

    # Offsets for labeling points
    dx = 1.2
    dy = 0.05

    # Create the figure
    import matplotlib as matplotlib
    from matplotlib.figure import Figure  # for plotting
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    fig = Figure(figsize=(9,6))
    rect = [0.15, 0.1, 0.80, 0.85] # Left, bottom, width, height
    ax = fig.add_axes(rect, frameon=True)
    ax.set_title('Response surface: experiments performed', fontsize=16)
    ax.set_xlabel('Cooking temperature [K]', fontsize=16)
    ax.set_ylabel('Sodium hydroxide [kg]', fontsize=16)

    my_logger.debug('Show_result status = ' + str(show_result))
    if show_result:
        r = 70         # resolution of surface
        x1 = np.arange(limits_A[0], limits_A[1], step=(limits_A[1] - limits_A[0])/(r+0.0))
        x2 = np.arange(limits_B[0], limits_B[1], step=(limits_B[1] - limits_B[0])/(r+0.0))
        X1, X2 = np.meshgrid(x1, x2)
        Z, Z_noisy = generate_result('0000000', (X1, X2, None), 1)
        levels = np.array([5,15,  50, 75, 100, 150, 200, 300, 350, 375])
        CS = ax.contour(X1, X2, Z, colors='#006600', levels=levels, linestyles='dotted',linewidths=1)
        ax.clabel(CS, inline=1, fontsize=10, fmt='%1.0f' )

    # Plot constraint
    ax.plot([constraint_a[0], constraint_b[0]], [constraint_a[1], constraint_b[1]], color="#FF0000", linewidth=2)

    # Baseline marker and label
    ax.text(baseline_xA, baseline_xB, "    Baseline", horizontalalignment='left', verticalalignment='center', color="#0000FF")
    ax.plot(baseline_xA, baseline_xB, 'b.', linewidth=2, ms=20)

    for idx, entry_A in enumerate(factor_A):
        jitter_x = np.random.normal(loc=0.0, scale=0.3)
        jitter_y = np.random.normal(loc=0.0, scale=0.05)
        if factor_C[idx] == 'Absent':
            ax.plot(entry_A+jitter_x, factor_B[idx]+jitter_y, 'k.', ms=20)
        else:
            ax.plot(entry_A+jitter_x, factor_B[idx]+jitter_y, 'r.', ms=20)
        ax.text(entry_A+jitter_x+dx, factor_B[idx]+jitter_y+dy, str(idx+1))

    ax.plot(455, 28, 'k.', ms=20)
    ax.text(455+dx, 28, 'Cellulose: absent', va='center', ha='left')

    ax.plot(455, 26, 'r.', ms=20)
    ax.text(455+dx, 26, 'Cellulose: present', va='center', ha='left')

    ax.set_xlim(limits_A)
    ax.set_ylim(limits_B)

    # Grid lines
    ax.grid(color='k', linestyle=':', linewidth=1)
    #for grid in ax.yaxis.get_gridlines():
    #     grid.set_visible(False)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas=FigureCanvasAgg(fig)
    my_logger.debug('Saving figure: ' + full_filename)
    fig.savefig(full_filename, dpi=150, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=True)

    return filename

def not_registered_student(request):
    """ Invalid student number received"""
    t = loader.get_template("not_registered_student.html")
    c = Context({})
    return HttpResponse(t.render(c))

def sign_in(request):
    """
    Verifies the user. If they are registered, then proceed with the experimental results
    """
    my_logger.debug('Sign-in page activated')
    if request.method == 'POST':
        form_student_number = request.POST.get('student_number', '')
        my_logger.debug('Student number (POST: sign_in) = '+ str(form_student_number))

        # Must return an HttpResponseRedirect object by the end of this
        try:
            the_student = Student.objects.get(student_number=form_student_number)
        except Student.DoesNotExist:
            # If student number not in list, tell them they are not registered
            return HttpResponseRedirect('/take-home-final/not-registered')
        else:
            return setup_experiment(request, form_student_number)

    # Non-POST access of the sign-in page: display the login page to the user
    else:
        my_logger.debug('Non-POST sign-in')
        return render_to_response('sign_in_form.html')

def get_experiment_list(the_student):
    """ Returns a list of experiments associates with `the_student` (a Django record)"""

    # Create a list of dictionaries: contains their previous experiments
    prev_expts = []
    counter = 1
    for item in Experiment.objects.select_related().filter(student=the_student.student_number):
        prev_expts.append({'factor_A': item.factor_A, 'factor_B': item.factor_B, 'factor_C': item.factor_C,
                            'response': item.response_noisy, 'date_time': item.date_time, 'number': counter})
        counter += 1
    return prev_expts

def render_next_experiment(the_student):
    """ Setup the dictionary and HTML for the student to enter their next experiment.

    the_student: Django record for the student
    """
    # Get the student's details into the template format
    if the_student.grad_student:
        level = '600'
    else:
        level = '400'
    student = {'name': the_student.first_name + ' ' + the_student.last_name,
              'level': level, 'number': the_student.student_number, 'email': the_student.email_address,
              'runs_used_so_far': the_student.runs_used_so_far}

    prev_expts = get_experiment_list(the_student)

    # Calculate bonus marks
    response = [-10000.0]
    for entry in prev_expts:
        response.append(entry['response'])
    highest_profit = np.max(response)
    my_logger.debug('profit = ' + str(highest_profit))
    max_profit = 420.0
    baseline = 13.85
    student['profit_bonus'] =  1.5 * (highest_profit - baseline) / (max_profit - baseline)
    student['runs_bonus'] = -0.15 * the_student.runs_used_so_far + 3.0


    # Generate a picture of previous experiments
    filename = plot_results(prev_expts)

    token_string = generate_random_token()
    Token.objects.get_or_create(token_string=token_string, student=the_student, active=True)

    settings = {'max_experiments_allowed': max_experiments_allowed,
                'token': token_string,
                'figure_filename': filename}

    my_logger.debug('Dealing with student = ' + str(the_student.student_number) + '; has run ' + str(len(prev_expts)) + ' already.')
    t = loader.get_template("deal-with-experiment.html")
    c = Context({'PrevExpts': prev_expts, 'Student': student, 'Settings': settings})
    return HttpResponse(t.render(c))


def setup_experiment(request, student_number):
    """
    Returns the web-page where the student can request a new experiment.
    We can assume the student is already registered.
    """
    my_logger.debug('About to run experiment for student = ' + str(student_number))
    the_student = Student.objects.get(student_number=student_number)
    return render_next_experiment(the_student)

def run_experiment(request, token):
    """
    Returns the web-page for the student if the token is valid
    """
    my_logger.debug('Running experiment with token=' + str(token))
    if request.method != 'POST':
        my_logger.debug('Non-POST access to `run_experiment` - not sure how that is possible')
        return render_to_response('sign_in_form.html')

    # This is a hidden field
    student_number = request.POST.get('_student_number_', '')
    my_logger.debug('Student number (POST:run_experiment) = '+ str(student_number))
    the_student = Student.objects.get(student_number=student_number)

    # Check if the user had valid numbers
    factor_A = request.POST.get('factor_A', '')
    factor_B = request.POST.get('factor_B', '')
    factor_C = request.POST.get('factor_C', '')
    try:
        factor_A, factor_B = np.float(factor_A), np.float(factor_B)
    except ValueError:
        my_logger.debug('Invalid values for factors received from student ' + student_number)
        t = loader.get_template("invalid-factor-values.html")
        c = Context({})
        return HttpResponse(t.render(c))
    else:
        my_logger.debug('factor_A = '+ str(factor_A) + '; factor_B = ' + str(factor_B) + '; factor_C = ' + str(factor_C))


    # Check constraints:
    satisfied = True
    if factor_A > 480.0 or factor_A < 390.0:
        satisfied = False
    if factor_B > 30.0 or factor_B < 5.0:
        satisfied = False

    m = (10.0 - 30.0)/(480.0 - 440.0)
    c = 30.0 - m * 440.0
    if factor_A*m + c < factor_B:    # predicted_B > actual_B: then you are in the constraint region
        satisfied = False

    if not satisfied:
        my_logger.debug('Invalid values for factors received from student ' + student_number)
        t = loader.get_template("invalid-factor-values.html")
        c = Context({})
        return HttpResponse(t.render(c))


    # Check if the user has enough experiments remaining
    if the_student.runs_used_so_far >= max_experiments_allowed:
        # Used token
        my_logger.debug('Limit reached for student number ' + student_number)
        t = loader.get_template("experiment-limit-reached.html")
        c = Context({})
        return HttpResponse(t.render(c))

    # Check that the token matches the student number and hasn't been used already
    token_item = Token.objects.filter(token_string=token)
    token_pk = token_item[0].pk
    if not token_item[0].active:
        # Used token
        my_logger.debug('Used token received: ' + token)
        t = loader.get_template("experiment-already-run.html")
        c = Context({})
        return HttpResponse(t.render(c))

    if token_item[0].student != the_student:
        my_logger.debug('Token does not belong to student')
        t = loader.get_template("token-spoofing.html")
        c = Context({})
        return HttpResponse(t.render(c))

    response, response_noisy = generate_result(student_number, [factor_A, factor_B, factor_C], bias=the_student.runs_used_so_far+1)
    expt = Experiment.objects.get_or_create(student=the_student, factor_A=factor_A, factor_B=factor_B, factor_C=factor_C, response=response,
                                            response_noisy=response_noisy)

    the_student.runs_used_so_far = the_student.runs_used_so_far + 1
    the_student.save()

    token = Token.objects.get(pk=token_pk)
    token.active = False
    token.save()

    return render_next_experiment(the_student)

def download_values(request, token):
    """ From the download link on the output"""
    token_item = Token.objects.filter(token_string=token)
    the_student = token_item[0].student
    my_logger.debug('Generating CSV file for token = ' + str(token) + '; student number = ' + the_student.student_number)

    prev_expts = get_experiment_list(the_student)

    # Create CSV response object
    import csv
    response = HttpResponse(mimetype='text/csv')
    response['Content-Disposition'] = 'attachment; filename=rsm-results-' + the_student.student_number + '.csv'
    writer = csv.writer(response)
    writer.writerow(['Number', 'DateTime', 'Cooking temperature', 'Sodium hydroxide', 'Cellulose', 'Response'])
    writer.writerow(['0','Baseline','440.0','20.0','Absent','13.85'])
    for expt in prev_expts:
        writer.writerow([str(expt['number']),
                         expt['date_time'].strftime('%d %B %Y %H:%M:%S'),
                         str(expt['factor_A']),
                         str(expt['factor_B']),
                         str(expt['factor_C']),
                         str(round(expt['response'],2)),
                         ])
    return response


