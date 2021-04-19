import streamlit as st
import pandas as pd
import numpy as np
import pickle5 as pickle
import time as tm
from   datetime import time, timedelta

from   sklearn.compose            import ColumnTransformer
from   sklearn.ensemble           import GradientBoostingClassifier
from   sklearn.impute             import SimpleImputer
from   sklearn.metrics            import accuracy_score
from   sklearn.model_selection    import train_test_split
from   sklearn.pipeline           import Pipeline
from   sklearn.preprocessing      import OneHotEncoder

# Full dataset contains pre-play, mid-play, and post-play features
# Filter for only pre-play features
def convert_yd_line_vars(posteam,ydline):
    """
    Convert yardline feature from form 'NYG 25' to numerical yardline based on (100- yards from endzone)
    """
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
    Xdf_c['yrdln'] = Xdf_c.apply(lambda x: convert_yd_line_vars(x['posteam'], x['yrdln']), axis=1)
    return Xdf_c

pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)
    
@st.cache
def prediction(user_prediction_data):
    return classifier.predict_proba(user_prediction_data)

def main():
    st.set_page_config(layout='wide',page_title="Veeral's App 🏈")
    st.title('NFL 4th Down Play Prediction')
    st.markdown('A Web App by [Veeral Shah](https://veeraldoesdata.com)')
    intro_text = "Hello and welcome! Use the sidebar to customize the game situation.\nKeep track of your changes on the scoreboard before predicting your play!"
    st.text(intro_text)
    #intro = st.empty()
    #for i in range(len(intro_text)):
        #intro.text(intro_text[:i])
        #tm.sleep(.05)
        
        

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
    #teams = sorted(['NE', 'WAS', 'TB', 'NYG', 'GB', 'LV', 'KC', 'CHI', 'CLE', 'SEA',
    #   'BUF', 'BAL', 'CIN', 'DEN', 'NO', 'DET', 'IND', 'PIT', 'CAR', 'LA',
    #   'MIN', 'PHI', 'MIA', 'TEN', 'DAL', 'NYJ', 'JAX', 'HOU', 'ARI',
    #   'ATL', 'SF', 'LAC'])
    
    col01, col02 = st.beta_columns(2)
    col1, col2, col3 = st.beta_columns([1,1,2])
    col11, col22 = st.beta_columns(2)
    teamsdf = pd.read_csv('https://gist.githubusercontent.com/cnizzardini/'+
                    '13d0a072adb35a0d5817/raw/dbda01dcd8c86101e68cbc9fbe05e0aa6ca0305b/nfl_teams.csv')
    teams = sorted(list(teamsdf.Name))
    # Define User Prediction Data
    st.sidebar.subheader("Pick Teams")
    posteam = st.sidebar.selectbox('Team on Offense',teams,index=15)
    negteam = st.sidebar.selectbox('Team on Defense',teams,index=29)
    posteam_abb = teamsdf[teamsdf.Name == posteam].iloc[0,2]
    negteam_abb = teamsdf[teamsdf.Name == negteam].iloc[0,2]
    st.sidebar.subheader("What's the Score?")
    posteam_score = st.sidebar.number_input('Team Points', min_value=0,max_value=50,value=0,step=1)
    defteam_score = st.sidebar.number_input('Opp Team Points', min_value=0,max_value=50,value=0,step=1)
    st.sidebar.subheader("Where's the Ball?")
    sideoffield = st.sidebar.selectbox("Side Of Field",['OWN',"OPP"])
    ydline = st.sidebar.slider('Yard Line',min_value=1,max_value=50,value=25)
    ydstogo = st.sidebar.slider('Yards To Go',min_value=1,max_value=30,value=10)
    if sideoffield == 'OWN':
        side = posteam_abb
    else:
        side = negteam_abb
    if sideoffield == 'OPP' and ydline < 10:
        goal_to_go = 1
    else:
        goal_to_go = 0
    
    st.sidebar.subheader("How much Time is Left?")
    quarter = st.sidebar.selectbox("Quarter",[1,2,3,4])
    if quarter > 2:
        half = 'Half2'
        halfval = 2.0
    else:
        half = "Half1"
        halfval = 1.0
    time_left = st.sidebar.slider("Time Left in Quarter:",value=(time(0,2, 0)),max_value=time(0,15,0),step=timedelta(seconds=5),format='mm:ss')
    sec_left_in_quarter = time_left.minute * 60.0 + time_left.second
    sec_left_in_half = ((halfval*2)-quarter)*15.0*60.0 + sec_left_in_quarter
    sec_left_in_game = (2-halfval)*30*60 + sec_left_in_half
    st.sidebar.subheader("Final Details")
    posteam_timeouts_remaining = st.sidebar.selectbox("Timeouts Left",[0,1,2,3],index=3)
    defteam_timeouts_remaining = st.sidebar.selectbox("Opp. Timeouts Left",[0,1,2,3],index=3)
    season = 2020
    arr = [[posteam_abb,
            negteam_abb,
            sec_left_in_quarter,
            sec_left_in_half,
            sec_left_in_game,
            half,
            quarter,
            goal_to_go,
            side +" "+ str(ydline),
            ydstogo,
            posteam_timeouts_remaining*1.0,
            defteam_timeouts_remaining*1.0,
            int(posteam_score-defteam_score),
            season]]

    teamsdf['Name2'] = teamsdf['Name'].str.replace('NY','New York').str.lower()
    team_str = teamsdf[teamsdf.Name == posteam].iloc[0,-1].replace(' ','-')
    oppteam_str = teamsdf[teamsdf.Name == negteam].iloc[0,-1].replace(' ','-')
    col01.image(f'http://loodibee.com/wp-content/uploads/nfl-{team_str}-team-logo-2-300x300.png',use_column_width='always')
    col02.image(f'http://loodibee.com/wp-content/uploads/nfl-{oppteam_str}-team-logo-2-300x300.png',use_column_width='always')
    user_prediction_data = pd.DataFrame(arr,columns=columns)
    directions_html =f""" 
<div style="border-style: solid;
    border-radius: 5px;
    background-color: #f0f2f6;
    border-width: 2px;
    border-color: #f0f2f6;margin: 0 auto; text-align: center; width: 30%;font-size:4vw; font-weight:bold">{posteam_score}    -   {defteam_score}</div>"""
    directions_html2 = f""" 
    <div style="padding:12px;float: left;font-size:2vw; margin-left:10px;font-weight:bold">4th & {ydstogo}</div>
    <div style="padding:12px;float: right;font-size:2vw; margin-right:10px; font-weight:bold">{sideoffield} {ydline}</div>
    <div style="border-style: solid;
    border-radius: 5px;
    background-color: #f0f2f6;
    border-width: 2px;
    border-color: #f0f2f6; text-align: center;padding:10px;font-size:2vw; font-weight:bold">Q{quarter}  &nbsp;&nbsp;&nbsp;&nbsp; {time_left.minute:02d}:{time_left.second:02d} </div>"""
  

    st.markdown(directions_html,unsafe_allow_html=True)
    st.markdown(directions_html2,unsafe_allow_html=True)
    #st.markdown(summary_html,unsafe_allow_html=True)
    giflist = ['https://media.giphy.com/media/FB7yASVBqPiFy/giphy.gif','https://media.giphy.com/media/57G6JvU7SuoNcY9Rs4/giphy.gif','https://media.giphy.com/media/xCyjMEYF9H2ZcLqf7t/giphy.gif','https://media.giphy.com/media/ORUsy5ZwwqZsd5jyd8/giphy.gif']
    # when 'Predict' is clicked, make the prediction and store it 
    play_class = pd.read_pickle('team_play_freq.pkl')
    if st.sidebar.button("Predict"):
        
        result = prediction(user_prediction_data)[0]
        
        resultlist = sorted(list(zip(["Field Goal","Pass","Punt","Run"],result,giflist)),key = lambda x: x[1])[::-1]
        progress_bar = st.progress(0)
        status_text = st.empty()
        #chart = st.line_chart(np.random.rand(1, 4))

        for i in range(100):
            # Update progress bar.
            progress_bar.progress(i + 1)

            # Update status text.
            if i<25:
                status_text.text('Loading 1.3 million plays...')
            new_rows = np.random.rand(1, 4)

            # Update status text.
            if i >= 25 and i<50:
                status_text.text(f'{posteam_abb} offense vs {negteam_abb} defense? Hmmm...')
            if i >= 50 and i<75:
                if ydstogo > 5:  
                    status_text.text(f"4th and {ydstogo}? That's tough...")
                else:
                    status_text.text(f"4th and {ydstogo}? Seems within reach...")   
                
            if i >= 75:
                if sideoffield == 'OWN':
                    status_text.text("Not even past midfield? Risky... ")
                elif ydline < 51 and ydline > 38:
                    status_text.text("Just out of field goal range...")
                else:
                    status_text.text("No chance of punting, right?...")
             # Append data to the chart.
            #chart.add_rows(new_rows)

            # Pretend we're doing some computation that takes time.
            tm.sleep(0.1)
        
        #status_text.text('Done!')
        
        for play_type,prob,gif in resultlist:
            if prob == result.max():
                finalgif = gif
                if play_type in ['Pass','Run']:
                    df = play_class[(play_class.posteam==posteam_abb)&(play_class.play_type==play_type.lower())&(play_class.play_class.str.contains(play_type.lower()))&(play_class['yd_bucket'].apply(lambda x: ydstogo in x))]
                    play_type = df[df.play_id==df.play_id.max()]['play_class'].iloc[0].title()
                
                st.title(f'{play_type}')
                
                #st.title(f'{playtype} (in {prob*100:.1f}% of similar situations)')
            #else:
            #    st.markdown(f'{playtype} ({prob*100:.1f}%)')
            
        html_temp4 = f"""
        <div style = "margin-left: 8px">
        <div style = "text-align:center;
        border-style: solid;
        border-radius: 15px;
        background: url({finalgif});
        background-size: cover;
        background-position: center;
        border-width: 1px;
        border-color: #f63366;
        padding: 5px; 
        width: 100%;
        height: 350px;">
        </div>
        </div>
                      """
        st.markdown(html_temp4,unsafe_allow_html=True)
    

if __name__=='__main__': 
    main()  
