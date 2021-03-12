import streamlit as st
import pandas as pd
import numpy as np
import pickle5 as pickle
import urllib
import numpy as np
import pandas as pd
import seaborn as sns
from   sklearn.compose            import *
from   sklearn.ensemble           import GradientBoostingClassifier
from   sklearn.impute             import *
from   sklearn.metrics            import confusion_matrix, accuracy_score
from   sklearn.model_selection    import train_test_split
from   sklearn.pipeline           import Pipeline
from   sklearn.preprocessing      import *

# Full dataset contains pre-play, mid-play, and post-play features
# Filter for only pre-play features
def filter_data(X):
    Xdf_c = X.copy()
    pre_play_features = [
     'posteam', 
     'defteam',
     'quarter_seconds_remaining',
     'half_seconds_remaining',
     'game_seconds_remaining',
     'game_half',
     'qtr',
     'goal_to_go',
     'yrdln',
     'ydstogo',
     'posteam_timeouts_remaining',
     'defteam_timeouts_remaining',
     'score_differential',
     'season'   
     ]
    Xdf_c = Xdf_c[pre_play_features]
    Xdf_c['ydstogo'] = Xdf_c['ydstogo'].astype(float)
    Xdf_c['score_differential'] = pd.cut(Xdf_c['score_differential'],bins=[-100,-17,-12,-9,-4,0,4,9,12,17,100])
    def convert_yd_line_vars(posteam,ydline):
        if type(ydline)==str:
            newydline = ydline.split()
            if ydline == '50':
                return float(ydline)
            elif posteam == newydline[0]:
                return float(newydline[1])
            else:
                return 100 - float(newydline[1])
        else:
            return np.nan
    Xdf_c['yrdln'] = Xdf_c.apply(lambda x: convert_yd_line_vars(x['posteam'], x['yrdln']), axis=1)
    return Xdf_c
categorical_columns = pd.read_pickle("dtypes.pkl")
con_pipe = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True))
                    ])
cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent', add_indicator=True)),
                     ('ohe', OneHotEncoder(handle_unknown='ignore'))
                    ])
preprocessing = ColumnTransformer([('categorical', cat_pipe,  categorical_columns),
                                   ('continuous',  con_pipe, ~categorical_columns),
                                   ])


pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
# Full dataset contains pre-play, mid-play, and post-play features
# Filter for only pre-play features
    
@st.cache
def prediction(user_prediction_data):
    return classifier.predict_proba(user_prediction_data)
# front end elements of the web page 
def main():
    html_temp = """ 
    <div style="background-color:green;padding:13px"> 
    <h1 style ="color:white;text-align:center;">NFL 4th Down Play Prediction</h1>
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 

    # following lines create boxes in which user can enter data required to make prediction 
    columns = [
     'posteam', 
     'defteam',
     'quarter_seconds_remaining',
     'half_seconds_remaining',
     'game_seconds_remaining',
     'game_half',
     'qtr',
     'goal_to_go',
     'yrdln',
     'ydstogo',
     'posteam_timeouts_remaining',
     'defteam_timeouts_remaining',
     'score_differential',
     'season' 
     ]
    teams = sorted(['NE', 'WAS', 'TB', 'NYG', 'GB', 'LV', 'KC', 'CHI', 'CLE', 'SEA',
       'BUF', 'BAL', 'CIN', 'DEN', 'NO', 'DET', 'IND', 'PIT', 'CAR', 'LA',
       'MIN', 'PHI', 'MIA', 'TEN', 'DAL', 'NYJ', 'JAX', 'HOU', 'ARI',
       'ATL', 'SF', 'LAC'])
    posteam = st.selectbox('Team on Offense',teams,index=15)
    negteam = st.selectbox('Team on Defense',teams,index=29)
    sideoffield = st.selectbox("Side Of Field",['OWN',"OPP"])
    if sideoffield == 'OWN':
        side = posteam
    else:
        side = negteam
    ydline = st.slider('Yard Line',min_value=1,max_value=50,value=25)
    ydstogo = st.slider('Yards To Go',min_value=1,max_value=30,value=10)
    quarter = st.selectbox("Quarter",[1,2,3,4])
    if quarter > 2:
        half = 'Half2'
        halfval = 2.0
    else:
        half = "Half1"
        halfval = 1.0
    min_left_in_quarter = st.number_input('Min Left in Quarter', min_value=0.,max_value=15.,value=15.,step=1.)
    min_left_in_half = ((halfval*2)-quarter)*15 + min_left_in_quarter
    min_left_in_game = (2-halfval)*30 + min_left_in_half
    if sideoffield == 'OPP' and ydline < 10:
        goal_to_go = 1
    else:
        goal_to_go = 0
    posteam_timeouts_remaining = st.selectbox("Timeouts",[0,1,2,3],index=3)
    defteam_timeouts_remaining = st.selectbox("Opp. Timeouts",[0,1,2,3],index=3)
    posteam_score = st.slider('Team Points',min_value=0,max_value=50,value=0)
    defteam_score = st.slider('Opp Team Points',min_value=0,max_value=50,value=0)
    arr = [[posteam,
            negteam,
            min_left_in_quarter*60.0,
            min_left_in_half*60.0,
            min_left_in_game*60.0,
            half,
            quarter,
            goal_to_go,
            side +" "+ str(ydline),
            ydstogo,
            posteam_timeouts_remaining*1.0,
            defteam_timeouts_remaining*1.0,
            int(posteam_score-defteam_score),
            2020]]
    user_prediction_data = pd.DataFrame(arr,columns=columns)
    st.write('User Prediction Data:',user_prediction_data)
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(user_prediction_data)[0]
        for playtype,prob in sorted(list(zip(["FIELD_GOAL","PASS","PUNT","RUSH"],result)),key = lambda x: x[1])[::-1]:
            if prob == result.max():
                if playtype != 'PUNT':
                    print (st.write(f'**{playtype}: {prob*100:.1f}%** :sunglasses:'))
                else:
                    print (st.write(f'**{playtype}: {prob*100:.1f}%** :sob:'))
            else:
                print (st.write(f'{playtype}: {prob*100:.1f}%'))
    
if __name__=='__main__': 
    main()  