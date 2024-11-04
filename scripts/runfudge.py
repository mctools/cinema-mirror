import os
import subprocess
import numpy as np
import logging
from multiprocessing import Pool
import argparse

class FileProcessor:
    def __init__(self, dirname, output_dir, log_file='process.log', num_processes=40, temperatures=None, ext='endf'):
        self.dirname = dirname + '/'
        self.output_dir = output_dir + '/'
        self.num_processes = num_processes
        self.ext = ext
        self.fn = []
        self.const_boltzmann = 8.6173303e-11
        self.temp = np.array(temperatures) if temperatures else np.array([300])
        self.setup_logging(log_file)
        self.collect_files()

    def setup_logging(self, log_file):
        logging.basicConfig(filename=log_file, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()

    def collect_files(self):
        for files in os.listdir(self.dirname):
            if files.endswith(self.ext):
                self.fn.append(files)
        self.fn.sort()
        self.logger.info(f'Collected {len(self.fn)} files with extension {self.ext}')

    def run_file(self, f):
        try:
            if os.path.exists("TEM"):
                self.logger.info("File TEM found. Exiting loop.")
                return

            # Run the first subprocess
            result1 = subprocess.run(
                f'endf2gnds.py --skipCovariances --skipBadData --ignoreBadDate --continuumSpectraFix {self.dirname+f} {self.output_dir+f+".xml"}',
                shell=True, check=True, stderr=subprocess.PIPE, text=True)

            substr = " "
            for t in self.temp:
                substr += ' -t ' + str(t * self.const_boltzmann) + '  '

            # Run the second subprocess
            result2 = subprocess.run(
                f'processProtare.py {substr} -mc --hybrid {self.output_dir+f+".xml"} {self.output_dir+f+".gnds.xml"}',
                shell=True, check=True, stderr=subprocess.PIPE, text=True)

            self.logger.info(f'{f} is done @ {self.temp} Kelvin')
            print(f'\033[92m{f} is done @ {self.temp} kelvin\033[0m')

        except subprocess.CalledProcessError as e:
            error_message = e.stderr.strip()
            self.logger.error(f'{f} failed: {e} \n {error_message}')
            print(f'\033[91m{f} failed: {error_message}\033[0m')

    def process_files(self):
        with Pool(self.num_processes) as p:
            p.map(self.run_file, self.fn)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process files in a directory with multiprocessing.')
    parser.add_argument('-d', '--dirname', type=str, required=True, help='Directory containing input files.')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='Directory for storing output files.')
    parser.add_argument('-l', '--log_file', type=str, default='process.log', help='Log file name.')
    parser.add_argument('-n', '--num_processes', type=int, default=40, help='Number of processes in the pool.')
    parser.add_argument('-t', '--temperatures', type=float, nargs='+', default=[300], help='List of temperatures in Kelvin.')
    parser.add_argument('-e', '--ext', type=str, default='endf', help='File extension to filter by.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    processor = FileProcessor(
        dirname=args.dirname,
        output_dir=args.output_dir,
        log_file=args.log_file,
        num_processes=args.num_processes,
        temperatures=args.temperatures,
        ext=args.ext
    )
    processor.process_files()



