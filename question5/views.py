from django.template import loader, Context
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render_to_response
from django.core.context_processors import csrf

import os
import numpy as np
import hashlib as hashlib

# Command line use
#import sys, os
#sys.path.extend(['/var/django/projects/'])
#os.environ['DJANGO_SETTINGS_MODULE'] = 'question5.settings'

from takehome.question5.models import Student, Token, Experiment

# Improvements for next time
# --------------------------
# * factors is a variable, so you can have unlimited number of factors
# * PDF download link (using ReportLab to create the PDF)

# SETTINGS
# --------
# Django application location (must end with trailing slash)
base_dir = '/home/kevindunn/webapps/takehome/takehome/question5/'
MEDIA_DIR = base_dir + 'media/'
MEDIA_URL = '/media/takehome/'

# Logging
LOG_FILENAME = base_dir + 'logfile.log'
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
show_result = False

def lagrange(x, xx, yy):
    """
    Returns the p_3(x) Lagrange polynomial for 4 points.
    >>> y = lagrange(x, xx, yy)

    INPUTS:
        x:  New value(s) of x at which to evaluate the
            polynomial.  Can be a scalar, or a vector.
        xx: A vector of x-values, 4 data points
        yy: A vector of y-values, 4 data points
    """

    # Sub-functions used inside the ``lagrange`` function
    def l1(x, xx):
        return (x-xx[1])*(x-xx[2])*(x-xx[3]) / ( (xx[0]-xx[1])*(xx[0]-xx[2])*(xx[0]-xx[3]) )
    def l2(x, xx):
        return (x-xx[0])*(x-xx[2])*(x-xx[3]) / ( (xx[1]-xx[0])*(xx[1]-xx[2])*(xx[1]-xx[3]) )
    def l3(x, xx):
        return (x-xx[0])*(x-xx[1])*(x-xx[3]) / ( (xx[2]-xx[0])*(xx[2]-xx[1])*(xx[2]-xx[3]) )
    def l4(x, xx):
        return (x-xx[0])*(x-xx[1])*(x-xx[2]) / ( (xx[3]-xx[0])*(xx[3]-xx[1])*(xx[3]-xx[2]) )

    # Calculates the Lagrange polynomial
    return yy[0]*l1(x, xx) + yy[1]*l2(x, xx) + yy[2]*l3(x, xx) + yy[3]*l4(x, xx)

def simulate_process(f_input, category):
    """
    Returns the process output at the point(s) given by ``f_input``.
    The noise-free output is returned
    """
    lo, hi = 300.0, 400.0
    top = 25.0
    bottom = 2.0
    a, b, c, d = 0.0, 45.0, 60.0, 95.0
    fa, fb, fc, fd = 18.7, 14.0, 6.2, 14.1

    if category.upper() == 'A':
        # A: up-down-fastup, min at 334.7
        nodes = np.array([lo+a, lo+b, lo+c, lo+d])
        f_nodes = np.array([fa, fb, fc, fd]) + bottom
    elif category.upper() == 'B':
        # B = A flipped LR: fastdown-up-down, min at 298.1
        nodes = np.array([hi-d, hi-c, hi-b, hi-a])
        f_nodes = np.array([fd, fc, fb, fa]) + bottom
    elif category.upper() == 'C':
        # C = A flipped TB = 25-A = down-up-fastdown, min at 290.5
        nodes = np.array([lo+a, lo+b, lo+c, lo+d])
        f_nodes = np.array([top-fa, top-fb, top-fc, top-fd])+ bottom
    else:
        # D = B flipped TB = 25-B = fastup-down-up, min at 342.5
        nodes = np.array([hi-d, hi-c, hi-b, hi-a])
        f_nodes = np.array([top-fd, top-fc, top-fb, top-fa]) + bottom

    # Shrink/stretch the nodes:
    lo_new, hi_new = 278, 355.0
    slope = (hi_new-lo_new) / (hi - lo)
    nodes = (nodes - lo)*slope + lo_new

    return lagrange(f_input, nodes, f_nodes)

def generate_random_token():
    import random
    return ''.join([random.choice('ABCEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz2345689') for i in range(token_length)])

def generate_result(student_number, factor, bias, category):
    """
    Generates an experimental result for the student.
    The first error added is always the same, the second error added is proportional to the number of runs.
    """
    my_logger.debug('Generating new experimental result for student number %s' %\
                                                            student_number)
    y = simulate_process(factor, category)

    # Add random distrubance to the output
    np.random.seed(int(student_number))

    error = np.random.normal(loc=0.0, scale=1.0, size=bias*2)[-1]
    while np.abs(error) >= 1.0:
        error = np.random.normal(loc=0.0, scale=1.0, size=bias)[-1]
    y_noisy = np.max([0.0, y+error])

    return (y, y_noisy)

def plot_results(expts, category):
    """Plots the data into a PNG figure file"""
    factor_A = []
    response = []
    for entry in expts:
        factor_A.append(entry['factor_A'])
        response.append(entry['response'])

    if len(response)==0:
        response = [0]

    data_string = str(factor_A) + str(response) + str(category)
    filename = hashlib.md5(data_string).hexdigest() + '.png'
    full_filename = MEDIA_DIR + filename

    # Don't regenerate plot if it already exists
    if os.path.exists(full_filename):
        return filename

    # Baseline and limits
    plot_limits_A = [270.0, 360.0]
    limits_response = [0.0, np.max([35.0, np.max(response)])]

    # Offsets for labeling points
    dx = 1.5
    dy = 0.5

    # Create the figure
    import matplotlib as matplotlib
    from matplotlib.figure import Figure  # for plotting
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    fig = Figure(figsize=(9,6))
    rect = [0.15, 0.1, 0.80, 0.85] # Left, bottom, width, height
    ax = fig.add_axes(rect, frameon=True)
    ax.set_title('Experiments performed', fontsize=16)
    ax.set_xlabel('Jacket temperature [K]', fontsize=16)
    ax.set_ylabel('Concentration of D [mol/L]', fontsize=16)

    my_logger.debug('Show_result status = ' + str(show_result))
    if show_result:
        lo_new = 278.0
        hi_new = 355.0
        T = np.arange(lo_new, hi_new, 0.1)
        y = simulate_process(T, category)
        ax.plot(T, y, 'r-')

    for idx, entry_A in enumerate(factor_A):
        ax.plot(entry_A, response[idx], 'k.', ms=20)
        ax.text(entry_A+dx, response[idx]+dy, str(idx+1))

    ax.set_xlim(plot_limits_A)
    ax.set_ylim(limits_response)

    # Grid lines
    ax.grid(color='k', linestyle=':', linewidth=1)
    #for grid in ax.yaxis.get_gridlines():
    #     grid.set_visible(False)

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    my_logger.debug('Saving figure: ' + full_filename)
    canvas=FigureCanvasAgg(fig)
    fig.savefig(full_filename, dpi=150, facecolor='w',
                edgecolor='w', orientation='portrait',
                papertype=None, format=None, transparent=True)

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
    if request.method == 'POST':
        form_student_number = request.POST.get('student_number', '')
        my_logger.debug('Student number (POST: sign_in) = '+ str(form_student_number))

        # Must return an HttpResponseRedirect object by the end of this
        try:
            _ = Student.objects.get(student_number=form_student_number)
        except Student.DoesNotExist:
            # If student number not in list, tell them they are not registered
            return HttpResponseRedirect('/not-registered')
        else:
            return setup_experiment(request, form_student_number)

    # Non-POST access of the sign-in page: display the login page to the user
    else:
        my_logger.debug('Non-POST sign-in')
        c = {}
        c.update(csrf(request))
        return render_to_response('sign_in_form.html', c)

def get_experiment_list(the_student):
    """ Returns a list of experiments associates with `the_student` (a Django record)"""

    # Create a list of dictionaries: contains their previous experiments
    prev_expts = []
    counter = 1
    for item in Experiment.objects.select_related().filter(student=the_student.student_number):
        prev_expts.append({'factor_A': item.factor_A,
                            'response': item.response_noisy,
                            'date_time': item.date_time,
                            'number': counter})
        counter += 1
    return prev_expts

def render_next_experiment(the_student, request):
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
    student['profit_bonus'] = 1.0
    student['runs_bonus'] = np.max([4.0-the_student.runs_used_so_far,0])/4.0

    # Generate a picture of previous experiments
    filename = MEDIA_URL + plot_results(prev_expts, the_student.category)

    token_string = generate_random_token()
    Token.objects.get_or_create(token_string=token_string, student=the_student, active=True)

    settings = {'max_experiments_allowed': max_experiments_allowed,
                'token': token_string,
                'figure_filename': filename}

    my_logger.debug('Dealing with student = ' + str(the_student.student_number) + '; has run ' + str(len(prev_expts)) + ' already.')
    t = loader.get_template("deal-with-experiment.html")
    context_dict = {'PrevExpts': prev_expts, 'Student': student, 'Settings': settings}
    context_dict.update(csrf(request))
    return HttpResponse(t.render(Context(context_dict)))


def setup_experiment(request, student_number):
    """
    Returns the web-page where the student can request a new experiment.
    We can assume the student is already registered.
    """
    #my_logger.debug('About to run experiment for student = ' + str(student_number))
    the_student = Student.objects.get(student_number=student_number)
    return render_next_experiment(the_student, request)

def run_experiment(request, token):
    """
    Returns the web-page for the student if the token is valid
    """
    my_logger.debug('Running experiment with token=' + str(token))
    if request.method != 'POST':
        my_logger.debug('Non-POST access to `run_experiment`: user is page refreshing an experiment page.')
        return HttpResponseRedirect('/')

    # This is a hidden field
    student_number = request.POST.get('_student_number_', '')
    my_logger.debug('Student number (POST:run_experiment) = '+ str(student_number))
    the_student = Student.objects.get(student_number=student_number)

    # Check if the user had valid numbers
    factor_A = request.POST.get('factor_A', '')
    try:
        factor_A = np.float(factor_A)
    except ValueError:
        my_logger.debug('Invalid values for factors received from student ' + student_number)
        t = loader.get_template("invalid-factor-values.html")
        c = Context({})
        return HttpResponse(t.render(c))
    else:
        my_logger.debug('factor_A = %s' % str(factor_A))

    # Check constraints:
    satisfied = True
    if factor_A < 278.0 or factor_A > 355.0:
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

    response, response_noisy = generate_result(student_number, factor_A,
                                        bias=the_student.runs_used_so_far+1,
                                        category = the_student.category)
    _ = Experiment.objects.get_or_create(student=the_student,
                                          factor_A=factor_A,
                                          response=response,
                                          response_noisy=response_noisy)

    the_student.runs_used_so_far = the_student.runs_used_so_far + 1
    the_student.save()

    token = Token.objects.get(pk=token_pk)
    token.active = False
    token.save()

    return render_next_experiment(the_student, request)

def download_values(request, token):
    """ From the download link on the output"""
    token_item = Token.objects.filter(token_string=token)
    the_student = token_item[0].student
    my_logger.debug('Generating CSV file for token = ' + str(token) + '; student number = ' + the_student.student_number)

    prev_expts = get_experiment_list(the_student)

    # Create CSV response object
    import csv
    response = HttpResponse(mimetype='text/csv')
    response['Content-Disposition'] = 'attachment; filename=experiment-results-' + the_student.student_number + '.csv'
    writer = csv.writer(response)
    writer.writerow(['Experiment_Number', 'Date_Time', 'Jacket_Temperature', 'Concentration_of_D'])
    for expt in prev_expts:
        writer.writerow([str(expt['number']),
                         expt['date_time'].strftime('%d %B %Y %H:%M:%S'),
                         str(expt['factor_A']),
                         str(round(expt['response'],2)),
                         ])
    return response


