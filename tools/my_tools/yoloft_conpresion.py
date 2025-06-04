import os
import sys
import datetime
import yaml
import pandas as pd
from ultralytics.models import YOLOFT, YOLO
from analy_log import extract_best_results
import logging
logger = logging.getLogger("ultralytics")
def process_and_save_results(all_results, final_results_file):
    # move the experiment_name column to the first column and the session column to the second column
    cols = all_results.columns.tolist()
    cols.insert(0, cols.pop(cols.index('experiment_name')))
    cols.insert(1, cols.pop(cols.index('session')))
    all_results = all_results[cols]

    # Group and sort by experiment_name and session
    all_results = all_results.sort_values(by=['experiment_name', 'session', 'AP-ALL', 'AP-0-12', 'AP-12-20'], ascending=[True, True, False, False, False])

    # Save the processed results to a CSV file
    all_results.to_csv(final_results_file, index=False)
    logging.info(f"All experiment results processed and saved to {final_results_file}")


def read_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def get_file_name(path):
    return os.path.basename(path).split(".")[0]

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
        sys.__stdout__.write(buf) # Also write to the original stdout

    def flush(self):
        pass

def train_model(model_config_dir, device, repeats, dataset_config_path, training_config_path, batch_size, epochs, img_size, workers, pretrain_model="yolov8l.pt", log_dir='./runs/logs/'):

    #Experiments vary and need to be modified
    experiment_name = os.path.basename(model_config_dir)
    log_dir = os.path.join(log_dir, experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    final_results_file = os.path.join(log_dir, experiment_name+".csv")

    # Initialize logging
    log_filename = f"{experiment_name}_{get_file_name(dataset_config_path)}_{get_file_name(training_config_path)}_batch{batch_size}_epochs{epochs}_img_size{img_size}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    # logging.basicConfig(level=logging.DEBUG,
    #                     format='%(asctime)s - %(levelname)s - %(message)s',
    #                     handlers=[logging.FileHandler(log_path),
    #                               logging.StreamHandler(sys.__stdout__)]) # Also log to stdout

    # Redirect stdout and stderr to the logger
    # sys.stdout = StreamToLogger(logging.getLogger(), logging.DEBUG)
    # sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)

    batch_size = batch_size * len(device)
    # Initialize an empty DataFrame for storing all the experiment results
    all_results = pd.DataFrame()

    # Iterate over all YAML files in the model configuration directory
    for model_config_path in sorted(os.listdir(model_config_dir)):
        if model_config_path.endswith(".yaml"):
            logging.info(f"!!!!!!!pretrain_model: {pretrain_model}")
            model_config_path = os.path.join(model_config_dir, model_config_path)

            logging.info(f"model_config_path= {model_config_path}")
            logging.info(f"pretrain_model= {pretrain_model}")
            logging.info(f"dataset_config_path= {dataset_config_path}")
            logging.info(f"training_config_path= {training_config_path}")
            logging.info(f"batch_size= {batch_size}")
            logging.info(f"epochs= {epochs}")
            logging.info(f"img_size= {img_size}")
            logging.info(f"device= {device}")
            logging.info(f"workers= {workers}")

            logging.info(f"Starting experiments with the following parameters:")
            logging.info(f"Model Config Path: {model_config_path}")
            logging.info(f"Dataset Config Path: {dataset_config_path}")
            logging.info(f"Training Config Path: {training_config_path}")
            logging.info(f"Batch Size: {batch_size}, Epochs: {epochs}, Image Size: {img_size}, Devices: {device}, Workers: {workers}")

            # Read and print the contents of the configuration file
            model_config = read_yaml(model_config_path)
            dataset_config = read_yaml(dataset_config_path)
            training_config = read_yaml(training_config_path)
            logging.info("Model Config:")
            for key, value in model_config.items():
                logging.info(f"{key}: {value}")

            logging.info("\nDataset Config:")
            for key, value in dataset_config.items():
                logging.info(f"{key}: {value}")

            logging.info("\nTraining Config:")
            for key, value in training_config.items():
                logging.info(f"{key}: {value}")

            for i in range(repeats):
                logging.info(f"\nStarting training session {i+1}")
                try:
                    model = YOLOFT(model_config_path).load(pretrain_model)
                    results = model.train(data=dataset_config_path, cfg=training_config_path, batch=batch_size, epochs=epochs, imgsz=img_size, device=device, workers=workers)
                    logging.info(f"Training session {i+1} completed.")
                except Exception as e:
                    logging.error(f"An error occurred during training session {i+1}: {e}", exc_info=True)

            logging.info(f"All experiments for {model_config_path} completed. Logs saved to {log_path}")

            # Analyze logs and merge results into all_results
            # experiment_results = extract_best_results(log_path, experiment_name=get_file_name(model_config_path), experiment_total=repeats, print_best=True)
            # all_results = pd.concat([all_results, experiment_results], ignore_index=True)
            # process_and_save_results(all_results, final_results_file)

    # Save all experimental results to a file
    # process_and_save_results(all_results, final_results_file)
    # logging.info(f"All experiment results saved to {final_results_file}")

if __name__ == "__main__":
    model_config_dir = "config/yolo_conv"  # Replace it with model configuration directory

    repeats = 2
    dataset_config_path = "config/dataset/XS-VIDv2.yaml"
    training_config_path = "config/train/default.yaml"
    epochs = 9
    img_size = 1024
    workers = 8
    pretrain_model="yolov8s.pt"
    batch_size = 36

    train_model(model_config_dir, [0], repeats, dataset_config_path, training_config_path, batch_size, epochs, img_size, workers, pretrain_model)