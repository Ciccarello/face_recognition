# -*- coding: utf-8 -*-
from __future__ import print_function
import click
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np
import cv2
from array import array
import json

results_list = []

def scan_known_people(known_people_folder):
    known_names = []
    known_face_encodings = []
    for file in image_files_in_folder(known_people_folder):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)

        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 1:
            click.echo('WARNING: More than one face found in {}. Only considering the first face.'.format(file))

        if len(encodings) == 0:
            click.echo('WARNING: No faces found in {}. Ignoring file.'.format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])
    return known_names, known_face_encodings


def print_result(filename, name, distance, show_distance=False):
    add_to_results(filename, name, distance)
    #print (results_list)

    if show_distance:
        print('{}, {}, {}'.format(filename, name, distance))
    else:
        print('{}, {}'.format(filename, name))

def get_best_result():
    try: return results_list[0][1] if results_list[0][2] < 0.75 else 'Face does not match any in the database'
    except: return ('No faces have been tested or there was an error in testing.')

def print_top_results(amount_to_display, known_people_folder):
    try:
        print ('\nTest file', str(results_list[0][0]), 'is most likely', str(results_list[0][1]), '\n') #Display the top result
        print('Best Guess: ' + results_list[0][1])
        try: #Try the best guess as a png
            display_new_image(known_people_folder+'/'+results_list[0][1]+'.png', 'Best Guess: ' + results_list[0][1]) #Display the best guess photo
        except:
            try: #Try the best guess as a jpg
                display_new_image(known_people_folder+'/'+results_list[0][1]+'.jpg', 'Best Guess: ' + results_list[0][1]) #Display the best guess photo
            except:
                print('It wasnt jpg or png, what even is it then?')

        print ('The top', str(amount_to_display), 'results were:') #Display results of comparison
        for i in range(amount_to_display):
            try: #Try each incorrect image as png
                print_result('', results_list[i][1], results_list[i][2], True)
                if i != 0: display_new_image(known_people_folder+'/'+results_list[i][1]+'.png', 'Guess' + str(i+1) + ': ' + results_list[i][1])
            except:
                try: #Try each incorrect image as jpg
                    print_result('', results_list[i][1], results_list[i][2], True)
                    if i != 0: display_new_image(known_people_folder+'/'+results_list[i][1]+'.jpg', 'Guess' + str(i+1) + ': ' + results_list[i][1])
                except:
                    print('It wasnt jpg or png, what even is it then? Version 2.0')
    except:
        print('Error printing results')


def add_to_results(filename, name, distance):
    results_list.append((filename, name, distance))


def test_image(image_to_check, known_names, known_face_encodings, known_people_folder, tolerance=0.6, show_distance=False, verbose=False, number_of_results_to_display=3):
    unknown_image = face_recognition.load_image_file(image_to_check)

    # Scale down image if it's giant so things run a little faster
    if max(unknown_image.shape) > 1600:
        pil_img = PIL.Image.fromarray(unknown_image)
        pil_img.thumbnail((1600, 1600), PIL.Image.LANCZOS)
        unknown_image = np.array(pil_img)

    unknown_encodings =( face_recognition.face_encodings(unknown_image))

    for unknown_encoding in unknown_encodings:
        distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
        result = list(distances <= tolerance)

        if True in result:

            for is_match, name, distance in zip(result, known_names, distances):
                add_to_results(image_to_check, name, distance)
#            [print_result(image_to_check, name, distance, show_distance) for is_match, name, distance in zip(result, known_names, distances) if is_match]
        else:
            print_result(image_to_check, 'unknown_person', None, show_distance)

    if not unknown_encodings:
        # print out fact that no faces were found in image
        print_result(image_to_check, 'no_persons_found', None, show_distance)
    results_list.sort(key=lambda x: x[2]) #Sorts list using the distance value

    if (verbose): print_top_results(number_of_results_to_display, known_people_folder)

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

def process_images_in_process_pool(images_to_check, known_names, known_face_encodings, number_of_cpus, tolerance, show_distance):
    if number_of_cpus == -1:
        processes = None
    else:
        processes = number_of_cpus

    # macOS will crash due to a bug in libdispatch if you dont use 'forkserver'
    context = multiprocessing
    if 'forkserver' in multiprocessing.get_all_start_methods():
        context = multiprocessing.get_context('forkserver')

    pool = context.Pool(processes=processes)

    function_parameters = zip(
        images_to_check,
        itertools.repeat(known_names),
        itertools.repeat(known_face_encodings),
        itertools.repeat(tolerance),
        itertools.repeat(show_distance)
    )

    pool.starmap(test_image, function_parameters)

def display_new_image(image_location, window_name):
    img = cv2.imread(image_location,1)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name,img)

def save_training_data(known_names, known_face_encodings, names_output_file, encodings_output_file):
    #names saving
    output_string = ''
    for i in known_names:
        output_string += i + '\n' #\n is a delimiter to allow later extraction
    file = open(names_output_file, "w")
    file.write(output_string)
    file.close()

    #encodings saving
    for i in range(len(known_face_encodings)):
        known_face_encodings[i] = known_face_encodings[i].tolist()
    file = open(encodings_output_file, "w")
    json.dump(known_face_encodings, file)
    file.close()

def load_training_data(load_names_file, load_encodings_file):
    #names loading
    file = open(load_names_file, "r")
    known_names = []
    for i in file.readlines():
        i = i.replace("\n","")
        known_names.append(i)
    file.close()

    #encodings saving
    file = open(load_encodings_file, "r")
    known_face_encodings = json.load(file)
    file.close()

    return known_names, known_face_encodings


def fit(cpus, tolerance, show_distance, number_of_results_to_display, train, names_output_file, encodings_output_file, image_to_check):
    known_names, known_face_encodings = scan_known_people(train)
    return known_names, known_face_encodings

def predict(cpus, tolerance, show_distance, number_of_results_to_display, known_people_folder, image_to_check, known_names, known_face_encodings):
    if(image_to_check != ''):
        if (number_of_results_to_display < 0):
            number_of_results_to_display = 0
        if (number_of_results_to_display == 0):
            verbose = False
        else:
            verbose = True


        display_new_image(image_to_check, 'Unknown Person')

        # Multi-core processing only supported on Python 3.4 or greater
        if os.path.isdir(image_to_check):
            if cpus == 1:
                [test_image(image_file, known_names, known_face_encodings, known_people_folder, tolerance, show_distance, verbose, number_of_results_to_display) for image_file in image_files_in_folder(image_to_check)]
            else:
                process_images_in_process_pool(image_files_in_folder(image_to_check), known_names, known_face_encodings, cpus, tolerance, show_distance)
        else:
            test_image(image_to_check, known_names, known_face_encodings, known_people_folder, tolerance, show_distance, verbose, number_of_results_to_display)

        print(get_best_result())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 2



@click.command()
# @click.argument('image_to_check')
# @click.argument('known_people_folder')
@click.option('--cpus', default=1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means use all in system')
@click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
@click.option('--show-distance', default=False, type=bool, help='Output face distance. Useful for tweaking tolerance setting.')
@click.option('-n', '--number-of-results-to-display', default=0, type=int, help='If verbose move is active, displays the number top X results. This does not affect the returned valued.')
@click.option('-t', '--train', default='', type=str, help="Enables training. Location of training material.")
@click.option('-o', '--names_output_file', default='', type=str, help="Enables saving of training output data.")
@click.option('-e', '--encodings_output_file', default='', type=str, help="Enables saving of training output data.")
@click.option('-l', '--load-names-file', default='', type=str, help="Loads pre-calculated training results")
@click.option('-k', '--load-encodings-file', default='', type=str, help="Loads pre-calculated training results")
@click.option('-m', '--not-trained-known-person-folder', default='', type=str, help="known person folder if using pre-calculated data. This is used because the normal known person folder triggers training.")
@click.option('-i', '--image_to_check', default='', type=str, help="Enables test for image recognition.")

def main(cpus, tolerance, show_distance, number_of_results_to_display, train, names_output_file, encodings_output_file, load_names_file, load_encodings_file, not_trained_known_person_folder, image_to_check):
    if (sys.version_info < (3, 4)) and cpus != 1:
        click.echo('WARNING: Multi-processing support requires Python 3.4 or greater. Falling back to single-threaded processing!')
        cpus = 1

    if((load_names_file != '') and (train != '')):
        print("ERROR: -t and -l cannot both be selected.")
    elif((load_names_file == '') and (train == '')):
        print("ERROR: -t or -l must be selected.")
    else:
        if(train != ''): #training instead of loading data
            known_people_folder = train
            known_names, known_face_encodings = fit(cpus, tolerance, show_distance, number_of_results_to_display, train, names_output_file, encodings_output_file, image_to_check)

            #save training data if -o and -e are used
            if((names_output_file != '') and (encodings_output_file != '')):
                save_training_data(known_names, known_face_encodings, names_output_file, encodings_output_file)
        elif(load_names_file != ''): #loading data instead of training
            known_names, known_face_encodings = load_training_data(load_names_file, load_encodings_file)
        else:
            print("ERROR: How did you get here. I'm going to break thanks to your bad judgement.")

        print(image_to_check)
        if(image_to_check != ''):
            if (not_trained_known_person_folder != ''): known_people_folder = not_trained_known_person_folder
            else: known_people_folder = train
            predict(cpus, tolerance, show_distance, number_of_results_to_display, known_people_folder, image_to_check, known_names, known_face_encodings)

if __name__ == '__main__':
    print(main())
