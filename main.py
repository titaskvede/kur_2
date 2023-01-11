import os
import statistics
import warnings
import math
import shutil
import numpy as np
import argparse
import time
import sys
import json
from matplotlib import pyplot as plt


class Main:
    def __init__(self):
        self.args = None
        self.average_length = 0
        self.data = []
        self.filtered_data = []
        self.length = []
        self.saved_json_location = os.path.join(os.getcwd(), 'data', 'data.json')

    def open_data_file(self, name=os.path.join('data', 'hairpin.fa'), **kwargs):
        """
        Opens data file, is able to save the data to class or return it.
        :param name: name of data file to open
        :param kwargs: save_to_data used to specify what to do with gathered data
        :return:
        """
        with open(f"{name}") as file:
            data_list = []
            for index, i in enumerate(file):
                i = i.upper()
                if "NNNNNNNN" in i:
                    # remove useless lines
                    continue
                if "hard_limit" in kwargs:
                    # Some files are large, really large, I MEAN REALLY LARGE, lets not use the whole file :)
                    if index >= kwargs["hard_limit"]:
                        break
                data_list.append(i)
            if 'save_to_data' in kwargs and kwargs['save_to_data']:
                self.data = data_list
            else:
                return data_list

    def calculate_average(self, calculate_average=True, **kwargs):
        """
        Calculates average length of sequences and filters data
        :param calculate_average: Bool
        :param kwargs:
        :return: None
        """
        size = 0
        filtered_data = ""
        for line in self.data:
            if '>' in line:
                if len(filtered_data) == 0:
                    continue
                new_line_count = filtered_data.count("\n")
                filtered_data = filtered_data.replace('\n', "")
                self.length.append(size - new_line_count)
                self.filtered_data.append(filtered_data)
                filtered_data = ""
                size = 0
                continue
            size = size + len(line)
            filtered_data = filtered_data + line
        else:
            # Don't leave last argument behind
            self.filtered_data.append(filtered_data)

        if calculate_average:
            if sum(self.length) / float(len(self.length)) != statistics.fmean(self.length):
                warnings.warn(message="Average length calculation is not accurate")
            self.average_length = round(statistics.fmean(self.length))
            if 'save_average' in kwargs and kwargs['save_average']:
                self.save_average(self.average_length)
        else:
            if os.path.exists(self.saved_json_location):
                json_data = self.open_data_file(name=self.saved_json_location)
                for i in json_data:
                    try:
                        if "length" in i.lower():
                            i = i.replace("\n", "")
                            self.average_length = int(i.split(':')[1])
                    except Exception:
                        warnings.warn("Failed to read average length, proceeding to calculate")
                        self.calculate_average(calculate_average=True, save_average=False)
            else:
                # you must calculate average length, if there is no data json file
                self.calculate_average(calculate_average=True, save_average=True)

    def save_average(self, length):
        """
        Saves average length to data json file
        :param length: average length
        :return:
        """

        if os.path.exists(self.saved_json_location):
            json_data = self.open_data_file(name=self.saved_json_location)
            found = False
            for index, i in enumerate(json_data):
                if 'length' in i:
                    json_data[index] = f'    "length":{length}\n'
                    found = True
                    break
            if not found:
                warnings.warn("Length not found in json, recreate data json, or add variable manually")
        else:
            # First time json doesn't exist, please create it
            json_data = {
                "length": length
            }
            with open(self.saved_json_location, 'w') as file:
                json.dump(json_data, file, indent=4)

    def make_average_sequences(self, **kwargs):
        """
        Due to learning models requiring same format images, we need to align all sequences to same size
        :return: None
        """

        if len(self.filtered_data) == 1:
            # if only a single long sequence is in data, use different functionality
            self.filtered_data[0] = self.filtered_data[0].replace("\n", '')
            while len(self.filtered_data[0]) >= self.average_length:
                self.filtered_data.append(self.filtered_data[0][:self.average_length])
                self.filtered_data[0] = self.filtered_data[0][self.average_length:]
            del(self.filtered_data[0])
        else:
            # For hairpin data, a lot of different sequences are present, we need to deal with them differently
            for index, i in enumerate(self.filtered_data):
                diff = len(i) - self.average_length
                if diff == 0:
                    # data is at perfect size do nothing
                    continue
                elif diff > 0:
                    # Sequence is too long, shorten both sides
                    if diff % 2 == 0:
                        # Even number of nucleotides are needed to be shortened.
                        self.filtered_data[index] = self.filtered_data[index][int(diff / 2): int((diff / 2) * -1)]
                    else:
                        # Odd number of nucleotides are needed to be shortened.
                        # Remove 1 additional nucleotide from front
                        cut_amount = math.floor(diff)
                        if cut_amount - 1 == 0:
                            self.filtered_data[index] = self.filtered_data[index][int(cut_amount):]
                        else:
                            self.filtered_data[index] = self.filtered_data[index][int((cut_amount+1)/2):int((cut_amount-1)/2)*-1]
                elif diff < 0:
                    # Sequence to short, append with element which will be ignored
                    element_to_append = "X"
                    if diff % 2 == 0:
                        # Even number of nucleotides are needed to be added.
                        string_to_append = ""
                        for y in range((int(diff / 2)) * -1):
                            string_to_append = f"{string_to_append}{element_to_append}"
                        self.filtered_data[index] = f"{string_to_append}{self.filtered_data[index]}{string_to_append}"
                    else:
                        # Odd number of nucleotides are needed to be added.
                        # Add additional in the front.
                        append_front = ""
                        append_back = ""
                        for y in range(diff * -1):
                            if y % 2 == 0:
                                append_front = append_front + element_to_append
                            elif y % 2 == 1:
                                append_back = append_back + element_to_append
                        self.filtered_data[index] = f"{append_front}{self.filtered_data[index]}{append_back}"
                if len(self.filtered_data[index]) != self.average_length:
                    warnings.warn(message=f"Failure to make average length in index {index}")

    def prepare_folders(self, name):
        """
        Create folders, ensure that no previous run data contaminates current run.
        :return: None
        """
        if os.path.exists(name):
            if self.args['ignore_delete']:
                os.chdir(name)
            else:
                shutil.rmtree(name)
                self.prepare_folders(name)
        else:
            os.mkdir(name)
            os.chdir(name)

    def create_image(self):
        """
        create from single list, 2 dimensional array, which then is transformed into black/white image
        :return:
        """
        for name, i in enumerate(self.filtered_data):
            single_list = []
            single_list[:0] = i
            dimensional_array = []

            for index, y in enumerate(single_list):
                line = []
                for second_index, z in enumerate(single_list):
                    line.append(f"{y}{z}")
                dimensional_array.append(line)

            self.array_to_image(dimensional_array, name)

    def array_to_image(self, array, name):
        """
        creates image based on given array
        :param array:
        :param array:
        :return:
        """
        map_element_to_rgb = {
            "AA": 0,
            "AC": 15,
            "AT": 30,
            "AU": 30,
            "AG": 45,
            "CA": 60,
            "CC": 75,
            "CT": 90,
            "CU": 90,
            "CG": 105,
            "TA": 120,
            "TC": 135,
            "TT": 150,
            "TG": 165,
            "UA": 120,
            "UC": 135,
            "UU": 150,
            "UG": 165,
            "GA": 180,
            "GC": 195,
            "GT": 210,
            "GU": 210,
            "GG": 225,
            "XA": 255,
            "XC": 255,
            "XT": 255,
            "XU": 255,
            "XG": 255,
            "AX": 255,
            "TX": 255,
            "UX": 255,
            "CX": 255,
            "GX": 255,
            "XX": 255,
        }
        size = self.average_length * 10
        grayscale = []
        for i in array:
            line = []
            for y in i:
                if y in map_element_to_rgb:
                    line.append(map_element_to_rgb[y])
                else:
                    exit(f"Error in mapping conversion with element {y}, please update map and try again.")
            grayscale.append(line)
        grayscales = np.array(grayscale, dtype=np.uint8)
        plt.imshow(grayscales, cmap='gray')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(f'{name}.png')
        print(f"Succesfully generated {name} image out of predicted count {len(self.filtered_data)}")

    def parse_args(self):
        parser = argparse.ArgumentParser(description="This program is responsible for prediction sequence format")
        parser.add_argument('-o', '--output', type=str, default=os.getcwd(),
                            help="location to create images and all other files")
        parser.add_argument('-id', '--ignore-delete', default=False, action="store_true",
                            help="does not delete previous run images")
        parser.add_argument('-fl', '--file-location', type=str,
                            help='specify which file to prepare to images.')
        parser.add_argument('-hsa', '--hard-set-average', type=int,
                            help='specify average length')
        parser.add_argument('-fn', '--folder-name', type=str,
                            help='specify image folder name')
        parser.add_argument('-usa', '--use-saved-average', default=False, action="store_true",
                            help="After first run, to keep the same average, use previous run average")
        parser.add_argument('-ca', '--calculate-average', default=True, action="store_false",
                            help='specify if calculate average or use saved one')
        parser.add_argument('-dsa', '--dont-save-average', default=True, action='store_false',
                            help='specify if to store current average length')

        self.args = vars(parser.parse_known_args()[0])


if __name__ == "__main__":
    data_dict = {
        "data_folder": "run_0",
        "list": [
            {"file_location": f"{os.path.join('data', 'hairpin.fn')}",
             "folder_name": "not_hairpin"},
        ]
    }

    # Don't forget to delete hard coded parser flags
    sys.argv.append("-id")
    sys.argv.append('-dsa')
    sys.argv.append('-usa')
    sys.argv.append('-ca')

    main = Main()
    main.parse_args()
    if main.args["file_location"]:
        main.open_data_file(main.args["file_location"], save_to_data=True, hard_limit=2000)
        # main.open_data_file(main.args["file_location"], save_to_data=True)
    else:
        main.open_data_file(data_dict["list"][0]["file_location"], save_to_data=True, hard_limit=2000)
        # main.open_data_file(data_dict["list"][0]["file_location"], save_to_data=True)

    if "hard_saved_average" in main.args and main.args["hard_saved_average"]:
        main.average_length = main.args["hard_set_average"]
    else:
        main.calculate_average(calculate_average=main.args['calculate_average'],
                               save_average=main.args['dont_save_average'])
    main.make_average_sequences()
    main.prepare_folders(data_dict["data_folder"])
    if "folder_name" in main.args:
        if main.args["folder_name"] is not None:
            main.prepare_folders(main.args["folder_name"])
        else:
            main.prepare_folders(data_dict["list"][0]["folder_name"])
    else:
        main.prepare_folders(data_dict["list"][0]["folder_name"])
    main.create_image()
