import numpy as np
import pandas as pd
import time

start = time.time()

file_raw = 'data/raw/keystrokes.csv'
file_users = 'data/raw/test_sections.csv'

def extract_keys_features(session_key):
    Press = np.asarray(session_key.PRESS_TIME)
    Release = np.asarray(session_key.RELEASE_TIME)
    key_code = np.asarray(session_key.KEYCODE) / 255
    key_name = np.asarray(session_key.LETTER)
    hold_time = (Release - Press) / 1000
    inter_press_time = np.append(0, np.diff(Press) / 1000)
    inter_release_time = np.append(0, np.diff(Release) / 1000)
    inter_key_time = np.append(0, (Release[:-1] - Press[1:]) / 1000)
    keys_features = np.array(
        [hold_time.astype(np.float32), inter_press_time.astype(np.float32), inter_release_time.astype(np.float32),
         inter_key_time.astype(np.float32), key_code.astype(np.float32), key_name])
    return keys_features.T

# rows = 10000 # 4*500000
# sessions = 150 # 4*8000

NUM_SESSIONS = 15
keys_db = pd.read_csv(file_raw, sep=",", index_col=False, header=None, encoding_errors='replace',
                      on_bad_lines='skip',
                      names = ['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'TEST_SECTION_ID', 'KEYCODE', 'IKI'])  #, nrows=rows)

other_db = pd.read_csv(file_users, sep=",", index_col=False, header=None, encoding_errors='replace',
                      on_bad_lines='skip',
                      names = ['TEST_SECTION_ID', 'SENTENCE_ID', 'PARTICIPANT_ID', 'USER_INPUT', 'INPUT_TIME', 'EDIT_DISTANCE',
                               'ERROR_RATE', 'WPM', 'INPUT_LENGTH', 'ERROR_LEN', 'POTENTIAL_WPM', 'POTENTIAL_LENGTH', 'DEVICE'])  # , nrows=sessions)

# Map TEST_SECTION_ID -> PARTICIPANT_ID using a merge (much faster than looping)
section_to_participant = other_db[['TEST_SECTION_ID', 'PARTICIPANT_ID']].drop_duplicates('TEST_SECTION_ID')
keys_db = keys_db.merge(section_to_participant, on='TEST_SECTION_ID', how='left')
keys_db['PARTICIPANT_ID'] = keys_db['PARTICIPANT_ID'].fillna(0).astype(int)
print("Mapped participant IDs")




keys_db = keys_db[(keys_db.PARTICIPANT_ID != 0)]

# Optimize filtering (vectorized instead of row-by-row iteration)
print("Filtering participants...")
valid_parts = keys_db.groupby('PARTICIPANT_ID')['TEST_SECTION_ID'].nunique()
valid_parts = valid_parts[valid_parts >= NUM_SESSIONS].index
keys_db = keys_db[keys_db['PARTICIPANT_ID'].isin(valid_parts)]
print(f"Filtered to {len(valid_parts)} valid participants")

keys_db = keys_db.set_index(['PARTICIPANT_ID', 'TEST_SECTION_ID'])
keys_db = keys_db.sort_index()

keys_features_db = []
keys_features_db_users_ids = []
keys_features_db_dict = {}

current_user = None
keys_feature_session = []
keys_feature_session_dict = {}

print("Extracting features (optimized)...")
for index, session_key in keys_db.groupby(level=['PARTICIPANT_ID', 'TEST_SECTION_ID']):
    user_id = index[0]
    session_id = index[1]

    if current_user is None:
        current_user = user_id

    keys_features = extract_keys_features(session_key)

    if current_user != user_id:
        if len(keys_feature_session) >= NUM_SESSIONS:
            keys_features_db.append(keys_feature_session)
            keys_features_db_users_ids.append(current_user)
            keys_features_db_dict[str(current_user)] = keys_feature_session_dict
            
        current_user = user_id
        keys_feature_session = []
        keys_feature_session_dict = {}
        
    keys_feature_session.append(keys_features)
    keys_feature_session_dict[str(session_id)] = keys_features

if len(keys_feature_session) >= NUM_SESSIONS:
    keys_features_db.append(keys_feature_session)
    keys_features_db_users_ids.append(current_user)
    keys_features_db_dict[str(current_user)] = keys_feature_session_dict

end = time.time()
time_elapsed = (end-start)/60
print(f"Time elapsed: {time_elapsed:.2f} minutes")

np.save('data/Mobile_keys_db_6_features.npy', np.array(keys_features_db, dtype=object))
np.save('keystroke_all_dict.npy', keys_features_db_dict)


# file_path = 'D:/Giuseppe/DBs/Mobile_keys_db_6_features.npy'
# keystroke_dataset = list(np.load(file_path, allow_pickle=True))

#
# problematic_users = []
# problems = []
# no_problems = []
# for i in range(len(keys_features_db)):
#     if not(len(keys_features_db[i]) == (len(keystroke_dataset[i]))):
#         problems.append(['dif_num_sess', i, keys_features_db_users_ids[i], np.nan])
#         problematic_users.append(keys_features_db_users_ids[i])
#     else:
#         for j in range(len(keys_features_db[i])):
#             try:
#                 comparison = (keys_features_db[i][j][:, :-1] == keystroke_dataset[i][j][:, :-1])
#                 if np.sum(comparison) / (np.shape(comparison)[0] * np.shape(comparison)[1]) != 1.0:
#                     problems.append(['dif_val', i, keys_features_db_users_ids[i], j])
#                     problematic_users.append(keys_features_db_users_ids[i])
#             except:
#                 problems.append(['dif_ses_len', i, keys_features_db_users_ids[i], j])
#                 problematic_users.append(keys_features_db_users_ids[i])
#             else:
#                 no_problems.append([i, j])
# problematic_users = sorted(list(set(problematic_users)))


# for element in problems:
#     try:
#         print('new ' + str(element[1]) + ' ' + str(element[2]) + ' ' + ''.join(list(keys_features_db[element[1]][element[2]][:, -1])))
#     except Exception as e:
#         print('new ' + str(element[1]) + ' ' + str(element[2]) + ' ' + str(e))
#     try:
#         print('old ' + str(element[1]) + ' ' + str(element[2]) + ' ' + ''.join(list(keystroke_dataset[element[1]][element[2]][:, -1])))
#     except Exception as e:
#         print('old ' + str(element[1]) + ' ' + str(element[2]) + ' ' + str(e))
#     print('\n')
