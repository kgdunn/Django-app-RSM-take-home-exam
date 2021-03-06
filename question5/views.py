#!/usr/local/bin/python2.6

# Command line use
import sys, os
sys.path.extend(['/home/kevindunn/webapps/takehome/'])
os.environ['DJANGO_SETTINGS_MODULE'] = 'takehome.settings'

from django.template import loader, Context
from django.http import HttpResponseRedirect, HttpResponse
from django.shortcuts import render_to_response
from django.core.context_processors import csrf

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (Table, Frame, TableStyle, Image)

import matplotlib as matplotlib
from matplotlib.figure import Figure  # for plotting

import os
import numpy as np
import hashlib as hashlib


from takehome.question5.models import Student, Token, Experiment

# Improvements for next time
# --------------------------
# * factors is a variable (list/dict), so you can have unlimited number of factors
# * registration is semi-automatic: they fill in a form with their names and 
#   student numbers, an alternative email address and it sends me an email.  I will
#   click in a link in the email that will create a student profile, in the desired
#   category and send them notification.

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
max_experiments_allowed = 0
show_result = True

def spline(x, xx, yy):
    """
    Fits the cubic spline through the 4 points
    """
    if isinstance(x, list):
        x = np.array(x)
    elif isinstance(x, (float, int)):
        x = np.array([float(x)])

    A = np.zeros((12,12))
    A[0,2] = 2
    A[0,3] = 6*xx[0]
    A[1,0] = 1
    A[1,1] = xx[0]
    A[1,2] = xx[0]**2
    A[1,3] = xx[0]**3
    A[2,0] = 1
    A[2,1] = xx[1]
    A[2,2] = xx[1]**2
    A[2,3] = xx[1]**3
    A[3,1] = 1
    A[3,2] = 2*xx[1]
    A[3,3] = 3*xx[1]**2
    A[3,5] = -1
    A[3,6] = -2*xx[1]
    A[3,7] = -3*xx[1]**2
    A[4,2] = 2
    A[4,3] = 6*xx[1]
    A[4,6] = -2
    A[4,7] = -6*xx[1]
    A[5,4] = 1
    A[5,5] = xx[1]
    A[5,6] = xx[1]**2
    A[5,7] = xx[1]**3
    A[6,4] = 1
    A[6,5] = xx[2]
    A[6,6] = xx[2]**2
    A[6,7] = xx[2]**3
    A[7,5] = 1
    A[7,6] = 2*xx[2]
    A[7,7] = 3*xx[2]**2
    A[7,9] = -1
    A[7,10] = -2*xx[2]
    A[7,11] = -3*xx[2]**2
    A[8,6] = 2
    A[8,7] = 6*xx[2]
    A[8,10] = -2
    A[8,11] = -6*xx[2]
    A[9,8] = 1
    A[9,9] = xx[2]
    A[9,10] = xx[2]**2
    A[9,11] = xx[2]**3
    A[10,8] = 1
    A[10,9] = xx[3]
    A[10,10] = xx[3]**2
    A[10,11] = xx[3]**3
    A[11,10] = 2
    A[11,11] = 6*xx[3]

    b = np.zeros((12,1))
    b[1] = yy[0]
    b[2] = yy[1]
    b[5] = yy[1]
    b[6] = yy[2]
    b[9] = yy[2]
    b[10] = yy[3]

    coef = np.linalg.solve(A, b)
    coefsp = coef.flatten().tolist()
    coefsp.insert(0,np.nan)

    def s1(x):
        """ Spline connecting nodes 1 and 2 """
        return np.polyval(coefsp[4:0:-1], x)

    def s2(x):
        """ Spline connecting nodes 2 and 3 """
        return np.polyval(coefsp[8:4:-1], x)

    def s3(x):
        """ Spline connecting nodes 3 and 4 """
        return np.polyval(coefsp[12:8:-1], x)

    # Since ``x`` can be a vector, we must iterate through the vector
    # and evalute the polynomial at each entry in ``x``.  Use the
    # ``enumerate`` function in Python.
    y = np.zeros(x.shape)
    for k, val in enumerate(x):

        # Find which polynomial to use, based on the value of ``val``:
        if val < xx[1]:
            y[k] = s1(val)
        elif val < xx[2]:
            y[k] = s2(val)
        else:
            y[k] = s3(val)

    return y

def simulate_process(f_input, category, find_min=False):
    """
    Returns the process output at the point(s) given by ``f_input``.
    The noise-free output is returned
    """
    lo, hi = 300.0, 400.0
    top = 25.0
    bottom = 2.0
    a, b, c, d = 0.0, 49.0, 66.0, 91.0
    fa, fb, fc, fd = 16.7, 14.0, 4.2, 12.1

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
    
    if not find_min:
        return spline(f_input, nodes, f_nodes)
    else:
        temp_range = np.arange(lo_new, hi_new, 0.01)
        conc_D = spline(temp_range, nodes, f_nodes)
        min_D_idx = np.argmin(conc_D)
        return (conc_D[min_D_idx], temp_range[min_D_idx])

def get_IP_address(request):
    """
    Returns the visitor's IP address as a string.
    """
    # Catchs the case when the user is on a proxy
    try:
        ip = request.META['HTTP_X_FORWARDED_FOR']
    except KeyError:
        ip = ''
    else:
        # HTTP_X_FORWARDED_FOR is a comma-separated list; take first IP:
        ip = ip.split(',')[0]

    if ip == '' or ip.lower() == 'unkown':
        ip = request.META['REMOTE_ADDR']      # User is not on a proxy
    return ip

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

    data_string = str(factor_A) + str(response) + str(category) + str(show_result) + 'with-star-error'
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
        
    if show_result:
        min_D, min_T = simulate_process(300.0, category, find_min=True)
        ax.errorbar(min_T, min_D, yerr=1.0, marker='None', ecolor='red', mew=3)
        ax.plot(min_T, min_D, 'r*', markersize=15)
        my_logger.debug('min_T = %s and min_D = %s' % (str(min_T), str(min_D)))
        ax.text(275, limits_response[1]-2.5, "True solution at T = %s" % str(min_T))

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
        my_logger.debug('Non-POST sign-in from %s' % get_IP_address(request))
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
              'level': level, 
              'number': the_student.student_number, 
              'email': the_student.email_address,
              'runs_used_so_far': the_student.runs_used_so_far,
              'category': the_student.category}

    prev_expts = get_experiment_list(the_student)

    

    # Generate a picture of previous experiments
    filename = MEDIA_URL + plot_results(prev_expts, the_student.category)

    token_string = generate_random_token()
    Token.objects.get_or_create(token_string=token_string, student=the_student, active=True)

    # Get true minimum:
    if show_result:
        min_D, min_T = simulate_process(300.0, the_student.category, find_min=True)
        
    # Calculate bonus marks
    student['profit_bonus'] = 1.0
    student['runs_bonus'] = np.max([4.0-the_student.runs_used_so_far,0])/4.0

    settings = {'max_experiments_allowed': max_experiments_allowed,
                'token': token_string,
                'figure_filename': filename,
                'min_D': min_D,
                'min_T': min_T}

    my_logger.debug('Dealing with student = ' + str(the_student.student_number) + '(%s)' % get_IP_address(request) + '; has run ' + str(len(prev_expts)) + ' already.')
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
    my_logger.debug('Generating PDF file for token = ' + str(token) + '; student number = ' + the_student.student_number)
    PDF_filename = 'question5-takehome-%s-%s.pdf' % (token, the_student.student_number)

    response = HttpResponse(mimetype='application/pdf')
    response['Content-Disposition'] = 'attachment; filename=%s' % PDF_filename

    c = canvas.Canvas(response, pagesize=letter)
    W, H = letter
    RMARGIN = LMARGIN = 20*mm
    TMARGIN = 35*mm
    BMARGIN = 20*mm

    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(W/2, H-TMARGIN, '3E4 take-home exam: question 5 report')
    text = c.beginText(LMARGIN, H-TMARGIN-15*mm)
    text.setFont("Helvetica-Bold", 15)
    text.textLines('Student name(s): %s\n' % the_student.first_name)
    text.textLines('Group number: %s\n' % the_student.student_number)
    c.drawText(text)

    # Collect experimental results together:
    frameWidth = W - (LMARGIN + RMARGIN)
    frameHeight = H - (TMARGIN + BMARGIN+30*mm)
    frame = Frame(LMARGIN, BMARGIN, frameWidth, frameHeight, showBoundary=0)
    table_data = [['Experiment number', 'Date/Time of experiment', 'Jacket temperature [K]', 'Concentration of D [mol/L]']]

    prev_expts = get_experiment_list(the_student)
    for expt in prev_expts:
        table_data.append([str(expt['number']),
                           expt['date_time'].strftime('%d %B %Y %H:%M:%S'),
                           str(expt['factor_A']),
                           str(round(expt['response'],2))])

    tblStyle = TableStyle([('BOX',(0,0), (-1,-1), 2,colors.black),
                            ('BOX',(0,0), (-1,0), 1,colors.black),
                            ('FONT',(0,0), (-1,0), 'Helvetica-Bold',10)])
    table_obj = Table(table_data, style=tblStyle)


    frame.addFromList([table_obj], c)

    filename = MEDIA_DIR + plot_results(prev_expts, the_student.category)
    c.drawImage(filename,LMARGIN,BMARGIN,
                width=0.75*W, preserveAspectRatio=True, anchor='sw')
    c.showPage()
    c.save()

    return response

if __name__ == '__main__':
    threshold = 1.0
    true_mins = {}
    
    for student in Student.objects.all():
    
        response = "Student: %s; Category: %s" % (student.student_number, student.category)
        if student.category not in true_mins.keys():
            min_D, min_T = simulate_process(300.0, student.category, find_min=True)
            true_mins[student.category] = min_T
        min_T = true_mins[student.category]
      
    
        for item in Experiment.objects.select_related().filter(student=student.student_number):
            if np.abs(item.factor_A - min_T) < threshold: #or np.abs(item.response_noisy - min_D) < threshold:
                print(response)

    print('Number of groups = %d' % len(Student.objects.all()))

