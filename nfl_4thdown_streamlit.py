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
from   sklearn.impute             import SimpleImputer
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
    st.set_page_config(layout='wide',page_title="Veeral's App 🏈")
    st.title('NFL 4th Down Play Prediction 🏈')
    st.markdown('A Web App by [Veeral Shah](https://veeraldoesdata.com)')

    # display the front end aspect

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
    col01, col02 = st.beta_columns(2)
    col1, col2, col3 = st.beta_columns([1,1,2])
    col11, col22 = st.beta_columns(2)
    col01.subheader("What's the Situation?")
    posteam = col1.selectbox('Team on Offense',teams,index=15)
    negteam = col2.selectbox('Team on Defense',teams,index=29)
    ydstogo = col1.slider('Yards To Go',min_value=1,max_value=30,value=10)
    ydline = col2.slider('Yard Line',min_value=1,max_value=50,value=25)
    sideoffield = col1.selectbox("Side Of Field",['OWN',"OPP"])
    if sideoffield == 'OWN':
        side = posteam
    else:
        side = negteam
    if sideoffield == 'OPP' and ydline < 10:
        goal_to_go = 1
    else:
        goal_to_go = 0
    season = col2.selectbox("Year",(2018,2019,2020),index=2)
    quarter = col1.selectbox("Quarter",[1,2,3,4])
    if quarter > 2:
        half = 'Half2'
        halfval = 2.0
    else:
        half = "Half1"
        halfval = 1.0
    min_left_in_quarter = col2.number_input('Min Left in Quarter', min_value=0.,max_value=15.,value=15.,step=1.)
    min_left_in_half = ((halfval*2)-quarter)*15 + min_left_in_quarter
    min_left_in_game = (2-halfval)*30 + min_left_in_half
    posteam_score = col1.number_input('Team Points', min_value=0,max_value=50,value=0,step=1)
    defteam_score = col2.number_input('Opp Team Points', min_value=0,max_value=50,value=0,step=1)
    posteam_timeouts_remaining = col1.selectbox("Timeouts Left",[0,1,2,3],index=3)
    defteam_timeouts_remaining = col2.selectbox("Opp. Timeouts Left",[0,1,2,3],index=3)
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
            season]]
    user_prediction_data = pd.DataFrame(arr,columns=columns)
    html_temp2 = f"""
    <div style = "border-style: solid;
    background-color: #f0f2f6;
    border-radius: 15px;
    border-width: 1px;
    border-color: #f63366;
    margin-bottom:30px;
    margin-left:5px;
    padding:12px;">
    <h1 style='text-align: center; #262730: black; font-size:45px;'>{posteam} vs. {negteam}</h1>
    <h3 style='text-align: center; #262730: black;'>4th and {ydstogo}</h3>
    <h3 style='text-align: center; #262730: black;'>Ball on the {sideoffield} {ydline}</h3>
    <h3 style='text-align: center; #262730: black;'>{int(min_left_in_quarter)} Minutes Left in Quarter {quarter}</h3>
    <h3 style='text-align: center; color: black;'>Score: {posteam_score}-{defteam_score}</h3>
    </div>
    """
    html_temp3 = f"""
    <div style = "border-style: solid;
    border-radius: 15px;
    background-color: #f0f2f6;
    border-width: 1px;
    border-color: #f63366;
    margin-bottom:30px;
    margin-top:30px;
    margin-left:5px;
    padding:10px;">
    <h4 style='text-align: left; color: #262730;'>1) Choose the matchup</h4>
    <h4 style='text-align: left; color: #262730;'>2) Adjust each variable to set the game situation</h4>
    <h4 style='text-align: left; color: #262730;'>3) Predict what type of play will be run</h4>
    </div>
    """
    col3.markdown(html_temp3,unsafe_allow_html=True)
    col3.markdown(html_temp2,unsafe_allow_html=True)
    # when 'Predict' is clicked, make the prediction and store it 
    giflist = ['https://media.giphy.com/media/FB7yASVBqPiFy/giphy.gif','https://media.giphy.com/media/57G6JvU7SuoNcY9Rs4/giphy.gif','https://media.giphy.com/media/xCyjMEYF9H2ZcLqf7t/giphy.gif','https://media.giphy.com/media/ORUsy5ZwwqZsd5jyd8/giphy.gif']
    if col1.button("Predict"): 
        result = prediction(user_prediction_data)[0]
        
        resultlist = sorted(list(zip(["Field Goal","Pass","Punt","Run"],result,giflist)),key = lambda x: x[1])[::-1]
        for playtype,prob,gif in resultlist:
            if prob == result.max():
                finalgif = gif 
                col11.title(f'{playtype} (in {prob*100:.1f}% of similar situations)')
            else:
                col11.markdown(f'{playtype} ({prob*100:.1f}%)')
        html_temp4 = f"""
    <div style = "margin-left: 8px">
    <div style = "
    text-align:center;
    border-style: solid;
    border-radius: 15px;
    background: url({finalgif});
    background-size: cover;
    background-position: center;
    border-width: 1px;
    border-color: #f63366;
    padding: 5px; 
    width: 100%;
    height: 350px">
    </div>
    </div>
    """
#    <img src={finalgif} width=100%></img>
        col22.markdown(html_temp4,unsafe_allow_html=True)       

if __name__=='__main__': 
    main()  