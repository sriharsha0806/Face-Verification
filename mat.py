import numpy as np
from scipy.io import loadmat
import pandas as pd 
import datetime as date
from dateutil.relativedelta import relativedelta

cols = ['path', 'class', 'bbox1', 'bbox2', 'bbox2', 'bbox4', 'face_score1', 'face_score2']
imdb_mat = './imdb_crop/imdb.mat'
wiki_mat = './wiki_crop/wiki.mat'

imdb_data = loadmat(imdb_mat)
wiki_data = loadmat(wiki_mat)

del imdb_mat, wiki_mat

imdb = imdb_data['imdb']
wiki = wiki_data['wiki']

imdb_full_path = imdb[0][0][2][0]
imdb_name = imdb[0][0][4][0]
imdb_face_location = imdb[0][0][5][0]
imdb_face_score1 = imdb[0][0][6][0]
imdb_face_score2 = imdb[0][0][7][0]

wiki_full_path = wiki[0][0][2][0]
wiki_name = wiki[0][0][4][0]
wiki_face_location = wiki[0][0][5][0]
wiki_face_score1 = wiki[0][0][6][0]
wiki_face_score2 = wiki[0][0][7][0]

imdb_path = []
wiki_path = []

for path in imdb_full_path:
    imdb_path.append('./imdb_crop/' + path[0])
    
for path in wiki_full_path:
    wiki_path.append('./wiki_crop/'+path[0])
    
imdb_bbox = []
wiki_bbox = []

for box in imdb_face_location:
    imdb_bbox.append(box)
    
for box in wiki_face_location:
    wiki_bbox.append(box)

imdb_bbox = np.asarray(imdb_bbox)
wiki_bbox = np.asarray(wiki_bbox)

imdb_bbox = imdb_bbox.reshape(imdb_bbox.shape[0],(imdb_bbox.shape[1]*imdb_bbox.shape[2]))
wiki_bbox = wiki_bbox.reshape(wiki_bbox.shape[0], (wiki_bbox.shape[1]*wiki_bbox.shape[2]))

final_imdb = np.column_stack((imdb_path, imdb_name, imdb_bbox, imdb_face_score1, imdb_face_score2))
final_wiki = np.column_stack((wiki_path, wiki_name, wiki_bbox, wiki_face_score1, wiki_face_score2))

final_imdb_df = pd.DataFrame(final_imdb)
final_wiki_df = pd.DataFrame(final_wiki)

final_imdb_df.columns = cols
final_wiki_df.columns = cols

meta = pd.concat((final_imdb_df, final_wiki_df))
meta['face_score1'] = meta['face_score1'].astype(str)
meta['face_score2'] = meta['face_score2'].astype(str)
meta = meta[meta['face_score1'] != '-inf']
meta = meta[meta['face_score2'] == 'nan']

meta = meta.drop(['face_score1', 'face_score2'], axis=1)
meta = meta.sample(frac=1)
meta.to_csv('meta.csv', index=False)

print(len(meta))
