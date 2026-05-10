import argparse
from utils.train_config import configs


test_configs = argparse.ArgumentParser()
test_configs.db = 'Aalto_mobile'

if test_configs.db == 'Aalto_mobile':
    test_configs.db_filename = configs.main_db
    test_configs.model_name = 'TypeFormer_pretrained'
    test_configs.results_dir = 'results/' + configs.model_name + "/"
    test_configs.num_test_subjects = 1000
    test_configs.num_validation_subjects = configs.num_validation_subjects
    test_configs.total_num_sessions = 15
    test_configs.enrolment_samples = [1, 2, 5, 7, 10][2]
    test_configs.test_samples = 5
    test_configs.impostor_test_samples = 1
