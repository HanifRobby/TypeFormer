import numpy as np
import pandas as pd
import time

start = time.time()



file_raw = './data/Aalto_mobile/Data_Raw/keystrokes.csv'
file_users = './data/Aalto_mobile/Data_Raw/test_sections.csv'

KEYS_COLUMNS = ['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'TEST_SECTION_ID', 'KEYCODE', 'IKI']
USERS_COLUMNS = ['TEST_SECTION_ID', 'SENTENCE_ID', 'PARTICIPANT_ID', 'USER_INPUT', 'INPUT_TIME', 'EDIT_DISTANCE',
                 'ERROR_RATE', 'WPM', 'INPUT_LENGTH', 'ERROR_LEN', 'POTENTIAL_WPM', 'POTENTIAL_LENGTH', 'DEVICE']


def read_csv_with_bad_line_fallback(path, **kwargs):
    # Fast path: C engine with malformed-line skipping (newer pandas).
    try:
        return pd.read_csv(path, on_bad_lines='skip', **kwargs)
    except (TypeError, ValueError):
        pass
    # Compatibility path: python engine with on_bad_lines (mid pandas versions).
    try:
        return pd.read_csv(path, engine='python', on_bad_lines='skip', **kwargs)
    except TypeError:
        # Legacy compatibility path for older pandas versions.
        return pd.read_csv(path, engine='python', error_bad_lines=False, warn_bad_lines=False, **kwargs)

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
keys_db = read_csv_with_bad_line_fallback(
    file_raw,
    sep=",",
    index_col=False,
    header=None,
    encoding_errors='replace',
    names=KEYS_COLUMNS,
    usecols=['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'LETTER', 'TEST_SECTION_ID', 'KEYCODE']
)  #, nrows=rows)

other_db = read_csv_with_bad_line_fallback(
    file_users,
    sep=",",
    index_col=False,
    header=None,
    encoding_errors='replace',
    names=USERS_COLUMNS,
    usecols=['TEST_SECTION_ID', 'PARTICIPANT_ID']
)  # , nrows=sessions)

# Normalize potentially mixed-type IDs/timestamps from malformed rows.
keys_db['KEYSTROKE_ID'] = pd.to_numeric(keys_db['KEYSTROKE_ID'], errors='coerce')
keys_db['PRESS_TIME'] = pd.to_numeric(keys_db['PRESS_TIME'], errors='coerce')
keys_db['RELEASE_TIME'] = pd.to_numeric(keys_db['RELEASE_TIME'], errors='coerce')
keys_db['TEST_SECTION_ID'] = pd.to_numeric(keys_db['TEST_SECTION_ID'], errors='coerce')
keys_db['KEYCODE'] = pd.to_numeric(keys_db['KEYCODE'], errors='coerce')
keys_db = keys_db.dropna(subset=['KEYSTROKE_ID', 'PRESS_TIME', 'RELEASE_TIME', 'TEST_SECTION_ID', 'KEYCODE']).copy()
keys_db['KEYSTROKE_ID'] = keys_db['KEYSTROKE_ID'].astype(np.int64)
keys_db['PRESS_TIME'] = keys_db['PRESS_TIME'].astype(np.int64)
keys_db['RELEASE_TIME'] = keys_db['RELEASE_TIME'].astype(np.int64)
keys_db['TEST_SECTION_ID'] = keys_db['TEST_SECTION_ID'].astype(np.int64)
keys_db['KEYCODE'] = keys_db['KEYCODE'].astype(np.int64)

other_db['TEST_SECTION_ID'] = pd.to_numeric(other_db['TEST_SECTION_ID'], errors='coerce')
other_db['PARTICIPANT_ID'] = pd.to_numeric(other_db['PARTICIPANT_ID'], errors='coerce')
other_db = other_db.dropna(subset=['TEST_SECTION_ID', 'PARTICIPANT_ID']).copy()
other_db['TEST_SECTION_ID'] = other_db['TEST_SECTION_ID'].astype(np.int64)
other_db['PARTICIPANT_ID'] = other_db['PARTICIPANT_ID'].astype(np.int64)

participant_map = other_db[['TEST_SECTION_ID', 'PARTICIPANT_ID']].drop_duplicates(subset=['TEST_SECTION_ID'])
keys_db = keys_db.merge(participant_map, on='TEST_SECTION_ID', how='left', sort=False)
keys_db = keys_db[keys_db['PARTICIPANT_ID'].notna()].copy()
keys_db['PARTICIPANT_ID'] = keys_db['PARTICIPANT_ID'].astype(int)

participant_sessions = keys_db.groupby('PARTICIPANT_ID')['TEST_SECTION_ID'].nunique()
valid_participants = participant_sessions[participant_sessions >= NUM_SESSIONS].index
keys_db = keys_db[keys_db['PARTICIPANT_ID'].isin(valid_participants)].copy()
keys_db = keys_db.sort_values(['PARTICIPANT_ID', 'TEST_SECTION_ID', 'KEYSTROKE_ID'], kind='mergesort')

end = time.time()

time_elapsed = (end-start)/60
print("time_elapsed:", time_elapsed)


keys_feature_session = []
keys_feature_session_dict = {}
keys_features_db = []
keys_features_db_dict = {}
current_user = None
for (participant_id, test_section_id), session_key in keys_db.groupby(['PARTICIPANT_ID', 'TEST_SECTION_ID'], sort=True):
    if current_user is None:
        current_user = participant_id
    elif current_user != participant_id:
        keys_features_db.append(keys_feature_session)
        keys_features_db_dict[str(current_user)] = keys_feature_session_dict
        keys_feature_session = []
        keys_feature_session_dict = {}
        current_user = participant_id
    keys_features = extract_keys_features(session_key)
    keys_feature_session.append(keys_features)
    keys_feature_session_dict[str(test_section_id)] = keys_features

if current_user is not None:
    keys_features_db.append(keys_feature_session)
    keys_features_db_dict[str(current_user)] = keys_feature_session_dict

# Ragged nested sessions/users require object dtype for stable serialization.
np.save('keystroke_all_list.npy', np.asarray(keys_features_db, dtype=object), allow_pickle=True)
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