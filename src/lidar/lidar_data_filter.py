import hashlib

class Lidar_Data_Filter():
    def __init__(self, data_filename, labeled_data_filename, filtered_data_filename):

        # create the filenames and the stale reading dictionary
        self.data_filename = data_filename
        self.labeled_data_filename = labeled_data_filename
        self.filtered_data_filename = filtered_data_filename
        self.stales = dict()


    def mark_stale_values(self):

        # open the files
        fin = open(self.data_filename, 'r')
        f_labeled = open(self.labeled_data_filename, 'w')
        f_filtered = open(self.filtered_data_filename, 'w')

        # read in the headers a write to the new files
        header1 = fin.readline()
        header2 = fin.readline()
        f_labeled.write(header1+header2)
        f_filtered.write(header1+header2)

        # add a column to the labeled datafile with the count of "staleness"
        header3 = fin.readline()
        f_filtered.write(header3)
        header3 = header3[:-1] + ',staleness\n'
        f_labeled.write(header3)

        #start processing the data
        last_print_frame = 0
        while True:
            #read a line in
            line = fin.readline()
            if not line:
                break
            vals = line[:-1].split(',')
            # create a string from the segment, the distance, the amplitude and the flags readings
            valstring = vals[3]+vals[4]+vals[5]+vals[6]
            # make this into a hash
            hashval = hashlib.md5(valstring.encode()).hexdigest()

            # if the hashed value already exists, add one to the staleness otherwise make it zero
            if hashval in self.stales.keys():
                staleness = self.stales[hashval] + 1
            else:
                # write values with zero staleness to the filtered file
                f_filtered.write(line)
                staleness = 0

            # save the new value for staleness to the dictionary
            self.stales[hashval] = staleness
            # write all values to the labeled data file
            f_labeled.write(line[:-1]+','+str(staleness)+'\n')

            # print a progress line to the console every 100 frames
            printframe = int(vals[0])
            if printframe % 100 == 0 and printframe != last_print_frame:
                last_print_frame = printframe
                print('Processing frame: {}'.format(printframe))

        # cleanup
        fin.close()
        f_filtered.close()
        f_labeled.close()

if __name__ == '__main__':
    data_filename = '../../data/2018-12-17/0003.csv'
    labeled_filename = '../../data/2018-12-17/0003_labeled.csv'
    filtered_filename = '../../data/2018-12-17/0003_filtered.csv'

    ldf = Lidar_Data_Filter(data_filename, labeled_filename, filtered_filename)
    ldf.mark_stale_values()
    a = 1